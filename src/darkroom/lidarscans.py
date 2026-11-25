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

Plain Python (numpy, laspy): no USD.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from . import format
from .outcome import Result

#: About this many points are written to a LAZ in one call.
BATCH = 1_000_000
#: How far off its pose a scan may begin and be kept, of a period.
OFF_POSE = 0.01
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
        """A scan stamped `stamp` (in the renderer's nanoseconds), an array a field, a row a
        point: `points` (x, y, z in the lidar's frame, metres), `intensity`, `offsets` (when
        each was fired, nanoseconds after the stamp), `ret`, `ring`, `material`; and `objects`
        (what each hit: RTX's 128-bit IDs, 16 bytes each), named with `names`. The arrays are
        handed over, not changed after. Waits while `QUEUED` scans are not yet written; raises
        once writing has failed."""
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
        """A scan kept (whole, in the window) and batched, as `_write_batch` takes it."""
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


def write_laz(
    out: Path, numbers: list[int], counts: list[int], batches: Iterable[dict[str, np.ndarray]]
) -> None:
    """A lidar's LAZ `out`: its index (each scan's number and points, in order) and its scans
    in `batches` (as `_write_batch` takes them, scans in that order), each written in one call."""
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
    """Scans `b` written in one call: an array a field, a row a point ("number", its scan's;
    "pose", the seconds its scan's pose is at; "points"; "intensity"; "offset", seconds after
    that pose; "ring"; "ret"; "hit"; "material"), and "sizes", how many points each scan has."""
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
    """Whether `back` (read from LAZ) is the scan in `wrote` (as `_write_batch` takes it): its
    points to the LAZ's grid (half `format.LAZ_SCALE`) and float32's own rounding of what comes
    back (30 um at 300 m), the rest exactly."""
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
