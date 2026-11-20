"""A lidar's scans as its renderer writes them, and as the capture keeps them: one LAZ a lidar
(`format.PACKED`).

The renderer (ovrtx) writes each scan's points as RTX hands them over into the
lidar's `RAW` folder: a file a field (`FIELDS`), each scan's bytes appended (`RawWriter`),
nothing read back. As the render ends it writes an index (`finish`, `INDEX`): every scan's
stamp and points in file order, which are kept (whole, in the window) with their numbers and
when each began, the rate and, with hits, which prim each object ID is. Once all is rendered
the parent makes the LAZ from them
(`pack`): the fields memory-mapped, scans taken a million points or so at a time, each batch
filled with whole-array sums and written in one call, the first scan read back and checked;
the raw files then go.

Measured on two real lidars (28,700 and 8,900 points a scan), against a file a scan read back
and written twice: the render thread 0.13 ms a scan (was 0.61), nothing as the render ends
(was 1.68 ms a scan, a lidar after another), packing 4.5 ms (was 11.2). One process, lidars
one after another, the parallel LAZ backend: the single-core one was four times slower.

Plain Python (numpy, laspy): no USD.
"""

from __future__ import annotations

import gc
import json
import shutil
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
    import laspy

    header = format._header(packed=True)
    pairs = np.array([numbers, counts], "<u4").T.reshape(-1, 2)
    for k in range(0, max(len(pairs), 1), format._INDEX_PAIRS):
        header.vlrs.append(
            laspy.VLR(
                format._INDEX,
                k // format._INDEX_PAIRS,
                "scan number, points",
                pairs[k : k + format._INDEX_PAIRS].tobytes(),
            )
        )
    with laspy.open(str(out), mode="w", header=header, do_compress=True,
                    laz_backend=laspy.LazBackend.LazrsParallel) as writer:  # fmt: skip
        for b in batches:
            if not len(b["ring"]):
                continue
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
    `format.LAZ_SCALE` (and float32's own rounding), the rest exactly."""
    hit, material = format._recorded(wrote["hit"], wrote["material"])
    return bool(
        np.abs(back.points - wrote["points"]).max(initial=0) <= format.LAZ_SCALE * 0.6
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
    numbers, sorted together (`np.unique` on rows was three quarters of a lidar pass)."""
    if not len(objects):
        return objects.reshape(0, 16), np.zeros(0, np.int64)
    pairs = np.ascontiguousarray(objects).view(np.uint64).reshape(-1, 2)
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    ordered = pairs[order]
    new = np.ones(len(ordered), bool)
    new[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    back = np.empty(len(ordered), np.int64)
    back[order] = np.cumsum(new) - 1
    return ordered[new].view(np.uint8).reshape(-1, 16), back
