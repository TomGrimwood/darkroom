"""A lidar's scans as its renderer hands them over, and as the capture keeps them: one LAZ a
lidar (`format.PACKED`), each point in the lidar's own frame as it fired it.

The renderer (ovrtx) hands each scan to its lidar's `ScanWriter` and steps on: on a thread of
its own the writer keeps the whole scans within the window, numbers them by when each began,
names what each point hit (its object ID's prim, a running list) and writes them into the
LAZ a million points or so at a time, with the parallel LAZ backend (the single-core one was
four times slower). The LAZ's index of scans is reserved as it opens, a pair a scan, and
filled in place once the last is written; the first scan is read back and checked.

On a street of three VLS-128s, 10 s of take, the lidars took 12 s where they had taken 71: the
render thread had sorted every point's object ID (58 s rendering), then the scans were
written raw (2.2 GB) and packed after (13 s).

Captures rendered before kept their scans raw until packed (`RAW`, `RawWriter`, `pack_all`):
a file a field (`FIELDS`), indexed as the render ended (`finish`); one carried on is packed so.

Plain Python (numpy, laspy): no USD.
"""

from __future__ import annotations

import contextlib
import gc
import json
import queue
import shutil
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from . import format
from .outcome import Result

#: A lidar's scans as RTX handed them over, in its folder until packed.
RAW = ".raw"
#: The raw files' index, written as the render ends.
INDEX = "index.json"
#: Each field a renderer writes (`ovrtx_lidar`), a file each: its type and how many to a point. The firing
#: offsets are RTX's nanoseconds after the scan's stamp; the objects RTX's 128-bit IDs (only
#: with hits).
FIELDS: dict[str, tuple[Any, int]] = {
    "points": (np.float32, 3),
    "intensity": (np.float32, 1),
    "offsets": (np.int64, 1),
    "ret": (np.uint8, 1),
    "ring": (np.uint32, 1),
    "material": (np.uint32, 1),
    "objects": (np.uint8, 16),
}
#: About this many points a batch, packing.
BATCH = 1_000_000
#: Seconds between a lidar's packing being said.
SAID_EVERY = 0.5
#: Scans handed to a lidar's `ScanWriter` and not yet written before its renderer waits.
QUEUED = 8
#: Says what each of some object IDs (k x 16 bytes) hit: its prim's path, or None.
Names = Callable[[np.ndarray], list[str | None]]


class ScanWriter:
    """Lidar `sensor`'s scans written into its LAZ in capture `folder` as its renderer hands
    them over (`add`), on a thread of its own: those wholly within the window's `span`
    seconds kept, scanned `rate` times a second, each once, numbered by when it began (`start`'s
    `seconds`: a renderer's stamp as seconds of the window). With `names`, what each point hit
    (`format.HITS`), and with `labels` what each of those is labelled (`format.HIT_LABELS`);
    `blind`: a lidar that may see nothing. Written as `.part` until `finish`."""

    def __init__(
        self,
        folder: str | Path,
        sensor: str,
        rate: float,
        span: float,
        names: Names | None = None,
        blind: bool = False,
        labels: Callable[[str], str] | None = None,
    ) -> None:
        self.folder, self.sensor, self.rate, self.span = Path(folder), sensor, rate, span
        self.names, self.blind, self.labels = names, blind, labels
        self.out = self.folder / sensor / format.PACKED
        self.part = self.out.with_name(self.out.name + ".part")
        #: The window's whole scans: as many as the LAZ's index has room for.
        self.wanted = int(span * rate + 1e-9)
        #: Scans handed over; those kept, as (number, when it began, points).
        self.given = 0
        self.kept: list[tuple[int, float, int]] = []
        #: What the points hit, in the order first hit (`format.HITS`).
        self.paths: list[str] = []
        self._hit_of: dict[bytes, int] = {}
        self._path_of: dict[str, int] = {}
        self._seconds: Callable[[int], float] | None = None
        self._poses = np.zeros(0)
        self._batch: list[dict[str, np.ndarray]] = []
        self._batched = 0
        self._first: dict[str, np.ndarray] | None = None
        self._writer: Any = None
        self._queue: queue.Queue[tuple[int, dict[str, np.ndarray], np.ndarray | None] | None] = (
            queue.Queue(QUEUED)
        )
        self._failed: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self, seconds: Callable[[int], float]) -> None:
        """Ready for scans: `seconds` says when in the window one stamped so began."""
        self._seconds = seconds
        self._poses = format.read_poses(self.folder, self.sensor).seconds
        self.part.unlink(missing_ok=True)
        self._thread = threading.Thread(target=self._run, name=f"{self.sensor} scans", daemon=True)
        self._thread.start()

    def add(self, stamp: int, objects: np.ndarray | None = None, **fields: np.ndarray) -> None:
        """A scan stamped `stamp` (in the renderer's nanoseconds): its fields as `FIELDS` says,
        `objects` (what each point hit) named with `names`; arrays handed over, not changed
        after. Waits while `QUEUED` scans are not yet written; raises once writing has failed."""
        if self._thread is None:
            raise RuntimeError(f"{self.sensor}: scans added before the writer started.")
        self._raise()
        self._queue.put((stamp, fields, objects))
        self.given += 1

    def finish(self) -> Result:
        """Every scan written and the LAZ put in place, its index filled and its first scan read
        back. Returns what it made; raises (nothing left in place) with no whole scan in the
        window, scans begun off their poses (more than `OFF_POSE` of a period: their points
        would be placed by the wrong poses), any of the window's whole scans missing, none that
        hit anything (unless `blind`), or a first scan that does not read back as it went in."""
        try:
            self._stop()
            self._raise()
            self._check()
            self._flush()
            if self._writer is None:  # no points at all: the scans' index alone
                self._writer = _open_laz(self.part, np.zeros((self.wanted, 2), "<u4"))
            self._writer.close()
            self._writer = None
            _fill_index(self.part, np.array([(n, c) for n, _, c in self.kept], "<u4"))
            if self._first is not None and not _same(
                format._read_packed(self.part, int(self._first["number"][0])), self._first
            ):
                raise RuntimeError(f"{self.out}: its first scan did not read back as it went in.")
            if self.names is not None:
                format.write_hits(self.folder, self.sensor, self.paths)
                if self.labels is not None:
                    said = [self.labels(path) for path in self.paths]
                    format.write_hit_labels(self.folder, self.sensor, said)
            self.part.replace(self.out)
        except BaseException:
            self.abort()
            raise
        points = sum(c for _, _, c in self.kept)
        if not points:
            return Result(
                len(self.kept), "scans", self.rate, "no points: nothing in reach", timed=False
            )
        return Result(len(self.kept), "scans", self.rate, f"{points / 1e6:.2f} M points")

    def abort(self) -> None:
        """Writing stopped and what was written gone. Safe to call again."""
        self._failed = self._failed or RuntimeError("stopped")
        self._stop()
        if self._writer is not None:
            with contextlib.suppress(Exception):  # going anyway: why it stopped is raised
                self._writer.close()
            self._writer = None
        self.part.unlink(missing_ok=True)

    def _check(self) -> None:
        period = 1 / self.rate
        if not self.kept:
            raise RuntimeError(f"RTX gave no whole scans in {self.span:g} s; it gave {self.given}.")
        off = max(abs(began - number / self.rate) for number, began, _ in self.kept)
        if off > OFF_POSE * period:
            raise RuntimeError(
                f"its scans began {off * 1000:.1f} ms off their poses (a scan every {period:g} s "
                "from the window's start)."
            )
        if len(self.kept) != self.wanted:
            lost = sorted(set(range(self.wanted)) - {number for number, _, _ in self.kept})
            raise RuntimeError(
                f"RTX gave {len(self.kept)} of the window's {self.wanted} whole scans: none for "
                "scan " + ", ".join(str(n) for n in lost[:8]) + (" ..." if len(lost) > 8 else "") + "."
            )  # fmt: skip
        if not self.blind and not any(c for _, _, c in self.kept):
            raise RuntimeError(
                f"none of its {len(self.kept)} scans hit anything: is there a visible mesh in reach?"
            )

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            if self._failed is not None:
                continue  # drained, so `add` never waits on a writer that has stopped
            try:
                self._take(*item)
            except BaseException as exc:
                self._failed = exc

    def _take(self, stamp: int, fields: dict[str, np.ndarray], objects: np.ndarray | None) -> None:
        """A scan kept (whole, in the window) and batched, as `pack` packs one."""
        assert self._seconds is not None
        began = float(self._seconds(stamp))
        if not (began > -1e-6 and began + 1 / self.rate <= self.span + 1e-6):
            return
        number = int(np.floor(began * self.rate + 1e-3))
        if self.kept and number <= self.kept[-1][0]:
            raise RuntimeError(f"scan {number} came after scan {self.kept[-1][0]}.")
        n = len(fields["ring"])
        scan = {
            "number": np.full(n, number, np.uint32),
            "pose": np.full(n, float(self._poses[number])),
            "sizes": np.array([n], np.int64),
            "points": np.asarray(fields["points"], np.float32).reshape(-1, 3),
            "intensity": np.asarray(fields["intensity"], np.float32).reshape(-1),
            # Seconds after its pose: the renderer's stamps are its timeline's, a plain shift.
            "offset": (
                np.asarray(fields["offsets"], np.int64).reshape(-1) / 1e9
                + (began - number / self.rate)
            ).astype(np.float32),
            "ring": np.asarray(fields["ring"], np.uint32).reshape(-1),
            "ret": np.asarray(fields["ret"], np.uint8).reshape(-1),
            "hit": self._hits(objects, n),
            "material": np.asarray(fields["material"], np.uint32).reshape(-1),
        }
        self.kept.append((number, began, n))
        if self._first is None and n:  # read back once written, to check
            self._first = scan
        self._batch.append(scan)
        self._batched += n
        if self._batched >= BATCH:
            self._flush()

    def _hits(self, objects: np.ndarray | None, n: int) -> np.ndarray:
        """Each of a scan's `n` points' prim, an index into `paths` (`format.NO_HIT` for one
        not named); its object IDs never seen before named first."""
        if objects is None or self.names is None:
            return np.full(n, format.NO_HIT, np.uint32)
        unique, back = distinct(np.asarray(objects, np.uint8).reshape(-1, 16))
        keys = [row.tobytes() for row in unique]
        new = [k for k in keys if k not in self._hit_of]
        if new:
            ids = np.frombuffer(b"".join(new), np.uint8).reshape(-1, 16)
            for key, path in zip(new, self.names(ids), strict=True):
                if path and path not in self._path_of:
                    self._path_of[path] = len(self.paths)
                    self.paths.append(path)
                self._hit_of[key] = self._path_of[path] if path else format.NO_HIT
        return np.array([self._hit_of[k] for k in keys], np.uint32)[back]

    def _flush(self) -> None:
        if not self._batch:
            return
        batch = {k: np.concatenate([s[k] for s in self._batch]) for k in self._batch[0]}
        self._batch, self._batched = [], 0
        if self._writer is None:
            self.part.parent.mkdir(parents=True, exist_ok=True)
            self._writer = _open_laz(self.part, np.zeros((self.wanted, 2), "<u4"))
        _write_batch(self._writer, batch)

    def _stop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    def _raise(self) -> None:
        if self._failed is not None:
            raise RuntimeError(f"{self.sensor}: writing its scans failed.") from self._failed


class RawWriter:
    """A lidar's raw files, open for appending scan after scan (`add`), its stamps and point
    counts kept in memory; `with_hits`: the objects too."""

    def __init__(self, raw: Path, with_hits: bool = False) -> None:
        self.raw = raw
        raw.mkdir(parents=True, exist_ok=True)
        names = [n for n in FIELDS if with_hits or n != "objects"]
        self.files: dict[str, BinaryIO] = {n: (raw / f"{n}.bin").open("ab") for n in names}
        self.stamps: list[int] = []
        self.counts: list[int] = []

    def add(self, stamp: int, **fields: np.ndarray) -> None:
        """A scan's fields (as `FIELDS` says), appended."""
        for name, file in self.files.items():
            kind, width = FIELDS[name]
            file.write(np.ascontiguousarray(fields[name], kind).reshape(-1, width).tobytes())
        self.stamps.append(stamp)
        self.counts.append(len(fields["ring"]))

    def close(self) -> None:
        for file in self.files.values():
            file.close()


def write_index(
    raw: Path,
    stamps: list[int],
    counts: list[int],
    kept: list[tuple[int, int, float]],
    rate: float,
    names: dict[int, str] | None = None,
) -> None:
    """The raw files' index: every scan's `stamps` and point `counts` in file order; the `kept`
    ones as (its place in the files, its number, when it began in seconds of the window); the
    scan `rate`; with hits, each object ID's prim (`names`)."""
    index = {
        "stamps": stamps,
        "counts": counts,
        "kept": [list(k) for k in kept],
        "rate": rate,
        "names": {str(k): v for k, v in (names or {}).items()},
    }
    (raw / INDEX).write_text(json.dumps(index) + "\n", encoding="utf-8")


#: How far off its pose a scan may begin and be kept, of a period.
OFF_POSE = 0.01


def finish(
    raws: RawWriter,
    rate: float,
    seconds: Callable[[Any], Any],
    span: float,
    names: dict[int, str] | None = None,
    blind: bool = False,
) -> Result:
    """A lidar's raw scans (`raws`, every one its renderer handed over) closed and indexed
    for packing (`write_index`), whichever renderer wrote them: kept, the whole scans within
    the window's `span` seconds, a period (1 / `rate`) apart, each stamp read as seconds of
    the window by `seconds`, once each. With hits, `names` says which prim each object ID is.
    Returns what it made; raises with no whole scan in the window, scans begun off their poses (more than `OFF_POSE` of a period: their points would be placed
    by the wrong poses), any of the window's whole scans missing (the first or the last
    lost), or none that hit anything (unless `blind`: its empty scans kept)."""
    raws.close()
    period = 1 / rate
    stamps = sorted(
        s for s in set(raws.stamps) if seconds(s) > -1e-6 and seconds(s) + period <= span + 1e-6
    )
    if not stamps:
        raise RuntimeError(f"RTX gave no whole scans in {span:g} s; it gave {len(raws.stamps)}.")
    place = {stamp: k for k, stamp in enumerate(raws.stamps)}
    kept = []
    for stamp in stamps:
        began = float(seconds(stamp))
        kept.append((place[stamp], int(np.floor(began * rate + 1e-3)), began))
    off = max(abs(began - number / rate) for _, number, began in kept)
    if off > OFF_POSE * period:
        raise RuntimeError(
            f"its scans began {off * 1000:.1f} ms off their poses (a scan every {period:g} s "
            "from the window's start)."
        )
    wanted = int(span * rate + 1e-9)
    if len(kept) != wanted:
        lost = sorted(set(range(wanted)) - {number for _, number, _ in kept})
        raise RuntimeError(
            f"RTX gave {len(kept)} of the window's {wanted} whole scans: none for scan "
            + ", ".join(str(n) for n in lost[:8])
            + (" ..." if len(lost) > 8 else "")
            + "."
        )
    write_index(raws.raw, raws.stamps, raws.counts, kept, rate, names)
    points = sum(raws.counts[k] for k, _, _ in kept)
    if not points and blind:
        return Result(len(stamps), "scans", rate, "no points: nothing in reach", timed=False)
    if not points:
        raise RuntimeError(
            f"none of its {len(stamps)} scans hit anything: is there a visible mesh in reach?"
        )
    return Result(len(stamps), "scans", rate, f"{points / 1e6:.2f} M points")


def raw_lidars(folder: Path) -> list[str]:
    """The lidars in capture `folder` whose scans are still raw (indexed: made as the render
    ended)."""
    return sorted(p.parent.parent.name for p in folder.glob(f"*/{RAW}/{INDEX}"))


def pack_all(
    folder: str | Path, on_packing: Callable[[str, float], object] | None = None
) -> tuple[int, int, int]:
    """Every lidar in `folder` whose scans are raw, packed (`pack`). `on_packing` is told each
    lidar's share packed (0 to 1) as it goes, every lidar at 0 first. Returns (scans packed,
    bytes before, bytes after)."""
    folder = Path(folder)
    lidars = raw_lidars(folder)
    tell = on_packing or (lambda sensor, done: None)
    for sensor in lidars:
        tell(sensor, 0.0)
    packed = before = after = 0
    for sensor in lidars:
        scans, was, now = pack(folder, sensor, lambda done, s=sensor: tell(s, done))
        tell(sensor, 1.0)
        packed, before, after = packed + scans, before + was, after + now
    return packed, before, after


def pack(
    folder: Path, sensor: str, on_done: Callable[[float], object] = lambda done: None
) -> tuple[int, int, int]:
    """Lidar `sensor`'s raw scans made into its one LAZ, written beside first and then put in
    place, the first read back and checked; the raw files then gone. With hits, `hits.json`
    first. Raises, the raw files kept, when it does not read back. Returns (scans, bytes
    before, bytes after)."""
    raw = folder / sensor / RAW
    index = json.loads((raw / INDEX).read_text(encoding="utf-8"))
    counts = np.array(index["counts"], np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)])
    total = int(starts[-1])
    columns = {}
    for name, (kind, width) in FIELDS.items():
        file = raw / f"{name}.bin"
        if file.is_file() and total:
            columns[name] = np.memmap(file, kind, "r", shape=(total, width))
        else:
            columns[name] = np.zeros((total, width), kind)
    kept = sorted((int(n), int(k), float(s)) for k, n, s in index["kept"])  # by number
    rate = float(index["rate"])
    hit_of = None
    if (raw / "objects.bin").is_file():
        names = {int(k): v for k, v in index["names"].items()}
        paths, hit_of = _hits(columns["objects"], starts, [k for _, k, _ in kept], names)
        format.write_hits(folder, sensor, paths)
    seconds = format.read_poses(folder, sensor).seconds

    def batches() -> Iterator[dict[str, np.ndarray]]:
        told, batch = time.monotonic(), []
        for done, (number, k, began) in enumerate(kept):
            batch.append((number, k, began))
            if sum(int(counts[b[1]]) for b in batch) >= BATCH or done == len(kept) - 1:
                if time.monotonic() - told >= SAID_EVERY:
                    told = time.monotonic()
                    on_done(done / len(kept))
                yield _batch(batch, columns, starts, counts, rate, seconds, hit_of)
                batch = []

    out = folder / sensor / format.PACKED
    part = out.with_name(out.name + ".part")
    numbers = [n for n, _, _ in kept]
    write_laz(part, numbers, [int(counts[k]) for _, k, _ in kept], batches())
    if kept:
        number, k, began = kept[0]
        first = _batch([(number, k, began)], columns, starts, counts, rate, seconds, hit_of)
        if not _same(format._read_packed(part, number), first):
            part.unlink()
            raise RuntimeError(f"{out}: packed, it did not read back as it was; .raw is kept.")
    part.replace(out)
    before = sum(f.stat().st_size for f in raw.iterdir())
    # The raw files let go of first: Windows does not delete a file that is still mapped.
    columns.clear()
    first = hit_of = None  # the hits' lookup holds the objects file too
    gc.collect()
    shutil.rmtree(raw)
    return len(kept), before, out.stat().st_size


def _batch(
    batch: list[tuple[int, int, float]],
    columns: dict[str, np.ndarray],
    starts: np.ndarray,
    counts: np.ndarray,
    rate: float,
    seconds: np.ndarray,
    hit_of: Callable[[int], np.ndarray] | None,
) -> dict[str, np.ndarray]:
    """Scans `batch` ((number, place in the files, when it began) each) as whole arrays, each
    point's firing time made seconds after its scan's pose: a lidar's renderer's stamps are its
    timeline's, a plain shift, so `offset = ns / 1e9 + (began - number / rate)`."""
    spans = [slice(int(starts[k]), int(starts[k + 1])) for _, k, _ in batch]
    taken = {n: np.concatenate([columns[n][s] for s in spans]) for n in FIELDS}
    sizes = np.array([int(counts[k]) for _, k, _ in batch], np.int64)
    shift = np.repeat([began - number / rate for number, _, began in batch], sizes)
    offset = (taken["offsets"][:, 0] / 1e9 + shift).astype(np.float32)
    hit = (
        np.concatenate([hit_of(k) for _, k, _ in batch])
        if hit_of is not None
        else np.full(int(sizes.sum()), format.NO_HIT, np.uint32)
    )
    return {
        "number": np.repeat([n for n, _, _ in batch], sizes).astype(np.uint32),
        "pose": np.repeat([float(seconds[n]) for n, _, _ in batch], sizes),
        "sizes": sizes,
        "points": np.asarray(taken["points"], np.float32),
        "intensity": np.asarray(taken["intensity"][:, 0], np.float32),
        "offset": offset,
        "ring": np.asarray(taken["ring"][:, 0], np.uint32),
        "ret": np.asarray(taken["ret"][:, 0], np.uint8),
        "hit": hit,
        "material": np.asarray(taken["material"][:, 0], np.uint32),
    }


def write_laz(
    out: Path, numbers: list[int], counts: list[int], batches: Iterable[dict[str, np.ndarray]]
) -> None:
    """A lidar's LAZ `out`: its index (each scan's number and points, in order) and its scans
    in `batches` (`_batch`'s arrays, scans in that order), each written in one call."""
    with _open_laz(out, np.array([numbers, counts], "<u4").T.reshape(-1, 2)) as writer:
        for b in batches:
            _write_batch(writer, b)


def _open_laz(out: Path, pairs: np.ndarray) -> Any:
    """A lidar's LAZ `out` open for writing (a laspy writer), its index `pairs` (each scan's
    number and points, in order) in records of `format._INDEX_PAIRS`."""
    import laspy

    header = format._header(packed=True)
    pairs = np.ascontiguousarray(pairs, "<u4").reshape(-1, 2)
    for k in range(0, max(len(pairs), 1), format._INDEX_PAIRS):
        header.vlrs.append(
            laspy.VLR(
                format._INDEX,
                k // format._INDEX_PAIRS,
                "scan number, points",
                pairs[k : k + format._INDEX_PAIRS].tobytes(),
            )
        )
    return laspy.open(str(out), mode="w", header=header, do_compress=True,
                      laz_backend=laspy.LazBackend.LazrsParallel)  # fmt: skip


def _write_batch(writer: Any, b: dict[str, np.ndarray]) -> None:
    """Scans `b` (`_batch`'s arrays) written in one call."""
    import laspy

    if not len(b["ring"]):
        return
    las = laspy.LasData(format._header(packed=True))
    points = b["points"].astype(np.float64)
    las.x, las.y, las.z = points[:, 0], points[:, 1], points[:, 2]
    las.gps_time = b["pose"] + b["offset"].astype(np.float64)
    ret = b["ret"].astype(np.uint8) + 1
    las.return_number = ret
    # Each scan's own greatest return, every point of it (a scan with none: none).
    most = np.zeros(len(b["sizes"]), np.uint8)
    some = b["sizes"] > 0
    ends = np.cumsum(b["sizes"])[some]
    if len(ends):
        most[some] = np.maximum.reduceat(ret, ends - b["sizes"][some])
    las.number_of_returns = np.repeat(most, b["sizes"])
    intensity = np.clip(b["intensity"].astype(np.float64), 0, 1) * 65535
    las.intensity = intensity.astype(np.uint16)
    las.intensity_raw = b["intensity"]
    las.offset = b["offset"]
    las.ring = b["ring"]
    las.hit = b["hit"]
    las.material = b["material"]
    las.scan = b["number"]
    writer.write_points(las.points)


#: A LAS 1.4 header's bytes, and a variable-length record's own (before its data).
_LAS_HEADER, _VLR_HEADER = 375, 54


def _fill_index(laz: Path, pairs: np.ndarray) -> None:
    """`pairs` (each scan's number and points, in order) written in place over the index
    `_open_laz` reserved in `laz`: its records are as long as they were, so nothing moves."""
    data = np.ascontiguousarray(pairs, "<u4").reshape(-1, 2)
    filled = 0
    with open(laz, "r+b") as file:
        head = file.read(_LAS_HEADER)
        at = int.from_bytes(head[94:96], "little")  # the header's size: the records follow
        for _ in range(int.from_bytes(head[100:104], "little")):
            file.seek(at)
            record = file.read(_VLR_HEADER)
            length = int.from_bytes(record[20:22], "little")
            if record[2:18].rstrip(b"\0") == format._INDEX.encode("ascii"):
                k = int.from_bytes(record[18:20], "little") * format._INDEX_PAIRS
                chunk = data[k : k + format._INDEX_PAIRS].tobytes()
                if len(chunk) != length:
                    raise RuntimeError(f"{laz}: its index has no room for {len(data)} scans.")
                file.write(chunk)
                filled += len(chunk)
            at += _VLR_HEADER + length
    if filled != data.nbytes:
        raise RuntimeError(f"{laz}: its index has room for {filled // 8} scans, not {len(data)}.")


def write_scans(folder: str | Path, sensor: str, scans: Iterable[tuple[int, format.Scan]]) -> Path:
    """`scans` ((number, scan) each, in order) as lidar `sensor`'s one LAZ in capture `folder`,
    each scan's points timed from its pose (`format.read_poses`, else number / 10 s)."""
    folder = Path(folder)
    given = list(scans)
    poses = folder / sensor / format.POSES
    seconds = format.read_poses(folder, sensor).seconds if poses.is_file() else None

    def batch(number: int, scan: format.Scan) -> dict[str, np.ndarray]:
        n = len(scan.points)
        recorded = len(scan.hit) == n
        pose = float(seconds[number]) if seconds is not None else number / 10.0
        return {
            "number": np.full(n, number, np.uint32),
            "pose": np.full(n, pose),
            "sizes": np.array([n], np.int64),
            "points": np.asarray(scan.points, np.float32).reshape(-1, 3),
            "intensity": np.asarray(scan.intensity, np.float32),
            "offset": np.asarray(scan.offset, np.float32),
            "ring": np.asarray(scan.ring, np.uint32),
            "ret": np.asarray(scan.ret, np.uint8),
            "hit": np.asarray(scan.hit, np.uint32) if recorded else np.full(n, format.NO_HIT, np.uint32),
            "material": np.asarray(scan.material, np.uint32) if recorded else np.zeros(n, np.uint32),
        }  # fmt: skip

    out = folder / sensor / format.PACKED
    out.parent.mkdir(parents=True, exist_ok=True)
    numbers = [n for n, _ in given]
    write_laz(out, numbers, [len(s.points) for _, s in given], (batch(n, s) for n, s in given))
    return out


def _same(back: format.Scan, wrote: dict[str, np.ndarray]) -> bool:
    """Whether `back` (read from LAZ) is the scan in `wrote` (`_batch`'s arrays): points to half
    `format.LAZ_SCALE` (and float32's own rounding, 30 um at 300 m), the rest exactly."""
    hit, material = format._recorded(wrote["hit"], wrote["material"])
    near = format.LAZ_SCALE * 0.6 + np.abs(wrote["points"]) * np.finfo(np.float32).eps
    return bool(
        np.all(np.abs(back.points - wrote["points"]) <= near)
        and np.array_equal(back.intensity, wrote["intensity"])
        and np.array_equal(back.offset, wrote["offset"])
        and np.array_equal(back.ring, wrote["ring"])
        and np.array_equal(back.ret, wrote["ret"])
        and np.array_equal(back.hit, hit)
        and np.array_equal(back.material, material)
    )


def _hits(
    objects: np.ndarray, starts: np.ndarray, places: list[int], names: dict[int, str]
) -> tuple[list[str], Callable[[int], np.ndarray]]:
    """The prims the kept scans' points (at `places` in the files) came back from, and for a
    scan's place its points' indices into them (`format.NO_HIT` for an ID RTX did not name)."""
    seen: set[bytes] = set()
    for k in places:
        seen.update(row.tobytes() for row in distinct(objects[starts[k] : starts[k + 1]])[0])
    said = {b: names.get(int.from_bytes(b, "little")) for b in seen}
    paths = sorted({p for p in said.values() if p})
    where = {p: k for k, p in enumerate(paths)}
    index = {b: where[p] if p else format.NO_HIT for b, p in said.items()}
    return paths, lambda k: indices(np.asarray(objects[starts[k] : starts[k + 1]]), index)


def indices(objects: np.ndarray, index: dict[bytes, int]) -> np.ndarray:
    """A scan's points' object IDs (n x 16) as indices into its prims (`index`)."""
    if not len(objects):
        return np.zeros(0, np.uint32)
    unique, back = distinct(objects)
    each = np.array([index[row.tobytes()] for row in unique], np.uint32)
    return each[back]


def distinct(objects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A scan's distinct object IDs (n x 16 bytes: k x 16) and each point's among them, as
    `np.unique(axis=0, return_inverse=True)` gives them, 16 times faster: each ID two 64-bit
    numbers, sorted by both (`np.unique` on rows was three quarters of a lidar pass). Only
    runs are sorted (a point mostly hit what the one fired before it did), by their low half
    alone unless two IDs share one."""
    if not len(objects):
        return objects.reshape(0, 16), np.zeros(0, np.int64)
    pairs = np.ascontiguousarray(objects).view(np.uint64).reshape(-1, 2)
    low, high = pairs[:, 0], pairs[:, 1]
    starts = np.empty(len(pairs), bool)
    starts[0] = True
    np.not_equal(low[1:], low[:-1], out=starts[1:])
    starts[1:] |= high[1:] != high[:-1]
    runs = pairs[starts]
    _, first, of_run = np.unique(runs[:, 0], return_index=True, return_inverse=True)
    unique = runs[first]
    if not np.array_equal(unique[of_run, 1], runs[:, 1]):
        order = np.lexsort((runs[:, 1], runs[:, 0]))
        ordered = runs[order]
        new = np.ones(len(ordered), bool)
        new[1:] = (ordered[1:, 0] != ordered[:-1, 0]) | (ordered[1:, 1] != ordered[:-1, 1])
        unique = ordered[new]
        of_run = np.empty(len(ordered), np.int64)
        of_run[order] = np.cumsum(new) - 1
    return unique.view(np.uint8).reshape(-1, 16), of_run.ravel()[np.cumsum(starts) - 1]
