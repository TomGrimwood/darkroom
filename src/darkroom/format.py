"""A capture folder: what every sensor on a stage measured over a window of time.

Plain files, read by anyone without a renderer or USD (the exporters read only this module):

```
<capture>/
  meta.json                    # Meta: the stage, the window, the site, each sensor
  states.json                  # what was going on, if anyone said: lanes of (start, end, label)
  <sensor>/
    poses.npz                  # truth: seconds (N), position (N x 3, metres), rotation (N x 4, w x y z)
    frames/000000.jpg ...      # camera: a picture per pose
    segments/, normals/, distances/  # camera, when the stage asks (`SEGMENT_TAXONOMY`): what each pixel saw, its normal, how far
    scans.laz                  # lidar: every scan (Scan), one LAZ, each point with its scan's number (`lidarscans` packs it)
    motion.npz                 # lidar: its poses 120 times a second, as poses.npz's: where it was as it fired (`read_motion`)
    hits.json                  # lidar: the prims its points came back from (Scan.hit indexes it)
    imu.npz                    # imu: seconds, accel (N x 3, m/s²), gyro (N x 3, rad/s)
    imu_<noise>.npz ...        # imu: its readings with a noise model's errors added, by its name (`imu_with`)
    gnss.npz                   # gnss: seconds, latitude, longitude, height (+ altitude), its tip's beside them, sky
    sky/000000.png ...         # gnss: the sky its antenna saw at each fix
  <body>/                      # what carries sensors (a pole) and points on it (its tip): `Meta.bodies`
    poses.npz                  # as a sensor's
    model.npz                  # what carries them, as a mesh in its own frame (Mesh), when it has one
  scenery.npz                  # the place near where the sensors went, one mesh (Mesh), when asked for
```

A GNSS antenna's fixes are the truth: where it was, and where its carrier's tip was
(`tip_latitude`, `tip_longitude`, `tip_height`, `tip_altitude`, when what carries it has a
tip), and `sky` (how much of the antenna's view was open sky, 0 to 1, from its sky views). A sky view is 128 px square, white where open sky, grey
where leaves, black where blocked: an equidistant fisheye along the antenna's +Z, its
columns along +X and rows along +Y.

Frames, units and times, every file's (both renderers' alike; exporters convert from these):

- Time: seconds from the window's start (`Meta.start` s into the stage's timeline);
  `Meta.start_utc` says when that was, when the stage says.
- The world: the stage's own frame, metres (its metersPerUnit applied), Z up. With a site
  (`Meta.site`, its `OmniWGS84ReferencePositionAPI`), its origin is there on Earth, its axes
  east, north, up.
- A pose (`POSES`, `MOTION`, a body's): `position` (N x 3, m) and `rotation` (N x 4, w x y z,
  unit) of the thing's own frame in the world: world = R(rotation) p + position, for p in its
  own frame (column vectors; USD's row-vector matrices are turned so, scale taken out:
  `sensorkit.motion.from_usd_matrices`). Rerun and ROS take x y z w: their exporters reorder.
- A sensor's own frame is its prim's: a camera's USD's (x right, y up, looking down -z); a
  lidar's and an IMU's x ahead, y left, z up (RTX's sensor frame); an antenna's z to the sky.
- A camera's frame n is at n / rate s, its pose then (the middle of its exposure; a rolling
  shutter's last row's: `sensorkit.camera`). Its lens (`Sensor.settings`) is OpenCV's model
  with its pixels counted from their corners, as RTX renders it: the top-left pixel spans
  0..1 across and down, so `cx`, `cy` are OpenCV's (and ROS's) + 0.5 (measured: a point
  lands within 0.1 px of fx X / Z + cx, in both renderers, plain or OmniLens).
- A camera's segments (`SEGMENTS`): each pixel the index of its label in `SEGMENT_LABELS`.
  Its normals (`NORMALS`): the world frame, unit, -1..1 an axis as 0..255 (0 itself is half
  a step off: 0.2 degrees at most). Its distances (`DISTANCES`): metres from the camera's
  centre along each pixel's ray (not to its image plane), millimetres in 16 bits (0: nothing
  there, or past 65.5 m): exact in both renderers (a wall 5 m ahead: 0.0 mm). Normals as
  rendered: Isaac Sim's have a material's normal maps in them, ovrtx 0.7.0's are the
  surface's own (a curb stage's ground 13 degrees apart, the median; a plain wall's alike).
- A lidar's scan i (`Scan`): its pose at i / rate s, when the scan began; each point in the
  lidar's own frame where it was as it fired it (not motion-compensated), fired `offset` s
  after; put in the world by the lidar's pose then (`read_motion(...).place(i / rate +
  offset, points)`): a lidar's points and a camera's distances so placed agree to half a
  millimetre (the median, a curb pass, both renderers). `PACKED` holds the same, x y z on a
  `LAZ_SCALE` grid.
- An IMU (`IMU`): `seconds`; `accel`, specific force (m/s², +9.80665 up when still) and
  `gyro`, angular rate (rad/s, right-handed), both in its own frame; the truth, no noise
  (`imu_with` beside it, with a model's errors).
- An antenna (`GNSS`): `seconds`; WGS84 `latitude`, `longitude` (degrees), `height` above
  the ellipsoid and `altitude` above the geoid (m, when the site gives the geoid); the truth.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

#: 3: an IMU's readings with a noise model's errors beside its truth (`imu_with`).
#: 4: a lidar's poses 120 times a second beside its pose a scan (`MOTION`).
VERSION = 4
META = "meta.json"
POSES = "poses.npz"
#: A lidar's poses finer than its pose a scan (`darkroom.truth`'s 120 a second, over the
#: window), as `POSES`' arrays: what a point is put in the world by, at when it was fired
#: (`read_motion`). A lidar sways and turns within a scan; its poses a scan apart, a straight
#: line between them, put a walker's points 15 mm (p90 75 mm) from where they were.
MOTION = "motion.npz"
FRAMES = "frames"
#: A camera's frames segmented, as many as its frames and numbered alike: each pixel the index
#: of what it saw in `SEGMENT_LABELS` beside them (0: nothing labelled). Only when the stage
#: asks for them (`SEGMENT_TAXONOMY`).
SEGMENTS = "segments"
SEGMENT_LABELS = "labels.json"
#: A segmented camera's frames' surface normals, numbered alike: each pixel the world-space
#: normal of what it saw, each axis -1..1 as 0..255 (`encode_normals`, `read_normals`).
NORMALS = "normals"
#: And their distances from the camera's centre along each pixel's ray: millimetres, 16 bits
#: (0: nothing there; to 65.5 m) (`encode_distances`, `read_distances`).
DISTANCES = "distances"
#: The stage's root layer's custom data key naming (comma-separated, as a capture names them) the segmented cameras whose
#: normals and distances are kept too (each costs memory a camera; none, none kept).
GEOMETRY_CAMERAS = "capture:geometry"
#: The stage's root layer's custom data key naming the semantic taxonomy its cameras' frames
#: are segmented by ("lidar", say); none, no segments.
SEGMENT_TAXONOMY = "capture:segment"
#: A lidar's scans in one LAZ (`lidarscans`): each point with its scan's number (`scan`), where
#: each scan starts in an index of variable-length records (`_INDEX`).
PACKED = "scans.laz"
#: A lidar's prims its points came back from (a JSON list of prim paths, `Scan.hit` indexes it);
#: a point whose prim RTX did not name.
HITS, NO_HIT = "hits.json", 0xFFFFFFFF
IMU = "imu.npz"


def imu_with(noise: str) -> str:
    """An IMU's readings with the errors of the noise model called `noise` added, whoever made
    them: the same arrays as `IMU`'s."""
    return f"imu_{noise}.npz"


def imu_errors(folder: str | Path, sensor: str) -> list[str]:
    """The noise models IMU `sensor` has readings with (`imu_with`), by name."""
    found = (Path(folder) / sensor).glob(imu_with("*"))
    return sorted(p.stem.removeprefix("imu_") for p in found)


GNSS = "gnss.npz"
SKY = "sky"
#: A sky view's pixels: what was there, and its grey in the picture (0, 127, 254).
SKY_BLOCKED, SKY_LEAVES, SKY_OPEN = 0, 1, 2
SKY_GREY = 127
#: Written last: a folder without it is a capture that did not finish.
DONE = ".done"
#: In a sensor's folder once its renderer has made all of it: what it said it made. A capture not
#: finished is carried on by rendering only the sensors without one.
MADE = ".made"
#: In a render for several rigs (not a rig's own capture): each sensor rendered, by prim path,
#: with the sensors it stood for (`darkroom.capture_rigs`).
SHARED = "shared.json"
#: In a run's folder (several rigs, a capture folder each, `<run>/<rig>/`): the one render they
#: are made from, hidden (its rigs' folders hold hard links to it).
RENDER = ".render"
#: Lanes of what was going on over the capture (where it was, what was being done), each a
#: list of [start, end, label] in seconds from the window's start; any lane may be missing.
STATES = "states.json"

#: `Meta.extra`'s key for the renderer that made a capture's pictures and scans: its cfg's
#: `renderer_type` ("isaac_rtx", "ovrtx"); a capture made before it was kept has none.
RENDERER = "renderer"

Kind = Literal["camera", "lidar", "imu", "gnss"]


@dataclass
class Site:
    """The stage origin's place on Earth (from its `OmniWGS84ReferencePositionAPI` prim)."""

    latitude: float
    longitude: float
    altitude: float
    geoid: float | None = None


@dataclass
class Sensor:
    """One sensor: its kind, its prim, its rate (Hz), and what its kind says of it.

    camera: `width`, `height`, `fx`, `fy`, `cx`, `cy` (px), `exposure`, `readout` (s), and
    its lens: `model` ("opencvPinhole" or "opencvFisheye") and `distortion` ({k1: …}), a
    fisheye's `circle` (degrees, 0 if not known) and how RTX `rendered` it ("lut" or
    "opencvFisheye": `darkroom.lenses`); its frames' `jpeg_quality`.
    lidar: `model`, `channels`; `fov` (degrees: least and greatest azimuth, elevation) for one
    that does not spin. imu, gnss: none needed.
    """

    kind: Kind
    prim: str
    rate: float
    settings: dict = field(default_factory=dict)


@dataclass
class Meta:
    stage: str
    #: The window, in seconds of the stage's timeline.
    start: float
    end: float
    sensors: dict[str, Sensor]
    site: Site | None = None
    #: What carries the sensors and points on it that are not sensors (a pole, its tip): name ->
    #: prim path; each has its poses (and a carrier its `model.npz`) in `<name>/`.
    bodies: dict[str, str] = field(default_factory=dict)
    #: The window's start in UTC (ISO 8601), when the stage says when it is.
    start_utc: str | None = None
    #: Who made it (`RENDERER`: "isaac_rtx" or "ovrtx"), and anything else worth keeping.
    extra: dict = field(default_factory=dict)
    version: int = VERSION


def write_meta(folder: str | Path, meta: Meta) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / META
    file.write_text(json.dumps(asdict(meta), indent=1) + "\n", encoding="utf-8")
    return file


def read_meta(folder: str | Path) -> Meta:
    data = json.loads((Path(folder) / META).read_text(encoding="utf-8"))
    if data.get("version", VERSION) > VERSION:
        raise ValueError(
            f"{folder}: capture format {data['version']} is newer than this reader ({VERSION})."
        )
    sensors = {name: Sensor(**s) for name, s in data.pop("sensors").items()}
    site = data.pop("site", None)
    return Meta(sensors=sensors, site=Site(**site) if site else None, **data)


@dataclass
class Mesh:
    """Triangles to look at (shown grey): `points` (N x 3, metres), `triangles` (M x 3, indices),
    `normals` (N x 3)."""

    points: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray


MODEL, SCENERY = "model.npz", "scenery.npz"


def write_mesh(file: str | Path, mesh: Mesh) -> Path:
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        file,
        points=np.asarray(mesh.points, np.float32),
        triangles=np.asarray(mesh.triangles, np.uint32),
        normals=np.asarray(mesh.normals, np.float32),
    )
    return file


def read_mesh(file: str | Path) -> Mesh | None:
    """The mesh in `file`, None when there is none."""
    if not Path(file).is_file():
        return None
    with np.load(file) as data:
        return Mesh(data["points"], data["triangles"], data["normals"])


@dataclass
class Poses:
    seconds: np.ndarray
    position: np.ndarray
    rotation: np.ndarray

    def at(self, seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions (N x 3) and rotations (N x 4, w x y z, unit) at `seconds`, each a straight
        line between the poses either side (held at the ends)."""
        seconds = np.asarray(seconds, float)
        rotation = np.asarray(self.rotation, float)
        # Each quaternion the sign nearer the one before, so neighbours blend the short way.
        flips = np.where(np.sum(rotation[1:] * rotation[:-1], axis=1) < 0, -1.0, 1.0)
        rotation = rotation * np.concatenate([[1.0], np.cumprod(flips)])[:, None]
        position = np.stack(
            [np.interp(seconds, self.seconds, self.position[:, i]) for i in range(3)], -1
        )
        turned = np.stack([np.interp(seconds, self.seconds, rotation[:, i]) for i in range(4)], -1)
        return position, turned / np.linalg.norm(turned, axis=-1, keepdims=True)

    def place(self, seconds: np.ndarray, points: np.ndarray) -> np.ndarray:
        """`points` (N x 3, in the sensor's own frame) put in the world, each by the pose at its
        own time in `seconds` (N)."""
        position, rotation = self.at(seconds)
        w, q = rotation[:, :1], rotation[:, 1:]
        t = 2 * np.cross(q, np.asarray(points, float))
        return position + points + w * t + np.cross(q, t)


def write_poses(folder: str | Path, sensor: str, poses: Poses) -> Path:
    file = Path(folder) / sensor / POSES
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(file, seconds=poses.seconds, position=poses.position, rotation=poses.rotation)
    return file


def read_poses(folder: str | Path, sensor: str) -> Poses:
    with np.load(Path(folder) / sensor / POSES) as data:
        return Poses(data["seconds"], data["position"], data["rotation"])


def write_motion(folder: str | Path, sensor: str, poses: Poses) -> Path:
    """Lidar `sensor`'s fine poses (`MOTION`)."""
    file = Path(folder) / sensor / MOTION
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(file, seconds=poses.seconds, position=poses.position, rotation=poses.rotation)
    return file


def has_motion(folder: str | Path, sensor: str) -> bool:
    """Whether lidar `sensor` has its fine poses (`MOTION`): a capture made before them has
    not (`darkroom.add_lidar_motion` adds them, as it was traced, from its stage)."""
    return (Path(folder) / sensor / MOTION).is_file()


def read_motion(folder: str | Path, sensor: str) -> Poses:
    """Where lidar `sensor` was over the window, to put its points in the world by
    (`Poses.place`, at `Scan.offset` after its scan's pose): its fine poses (`MOTION`), else (a
    capture made before them: `has_motion` says) its pose a scan, a straight line between."""
    file = Path(folder) / sensor / MOTION
    if not file.is_file():
        return read_poses(folder, sensor)
    with np.load(file) as data:
        return Poses(data["seconds"], data["position"], data["rotation"])


@dataclass
class Scan:
    """One lidar scan, each point in the sensor's frame where it was when it fired it (as
    fired: not motion compensated). Put it in the world by the lidar's pose then: `offset`
    after the scan's pose (`read_motion(...).place(poses.seconds[i] + offset, points)`).

    `points` (N x 3, metres), `intensity` (N), `offset` (N, seconds after the scan's pose
    each point was fired), `ring` (N, the channel), `ret` (N, the return: 0 first); `hit` (N,
    the prim it came back from: an index into the sensor's `HITS` list, `NO_HIT` unknown) and
    `material` (N, its RTX lidar material ID), both empty when not recorded.
    """

    points: np.ndarray
    intensity: np.ndarray
    offset: np.ndarray
    ring: np.ndarray
    ret: np.ndarray
    hit: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint32))
    material: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint32))


def scan_numbers(folder: str | Path, sensor: str) -> list[int]:
    """The scans lidar `sensor` of capture `folder` has, by number."""
    packed = Path(folder) / sensor / PACKED
    return sorted(_index(packed)) if packed.is_file() else []


#: Points read from a LAZ in one call (`iter_scans`): its parallel decompressor given many
#: chunks at once (a scan a read was 12.8 ms a scan; 64 scans a read, 2.5 ms).
READ_AHEAD = 2_000_000


def iter_scans(folder: str | Path, sensor: str) -> Iterator[tuple[int, Scan]]:
    """Every scan of lidar `sensor`, by number, in order: its one LAZ read through once, about
    `READ_AHEAD` points a read, handed out a scan at a time."""
    import laspy

    packed = Path(folder) / sensor / PACKED
    if not packed.is_file():
        return
    index = sorted(_index(packed).items(), key=lambda item: item[1][0])
    with laspy.open(str(packed), laz_backend=laspy.LazBackend.LazrsParallel) as reader:
        at = 0
        while at < len(index):
            take, points = [], 0
            while at < len(index) and (not take or points + index[at][1][1] <= READ_AHEAD):
                take.append(index[at])
                points += index[at][1][1]
                at += 1
            read = reader.read_points(points)
            first = 0
            for number, (_, count) in take:
                yield number, _scan_of(read[first : first + count])
                first += count


def has_scan(folder: str | Path, sensor: str, index: int) -> bool:
    packed = Path(folder) / sensor / PACKED
    return packed.is_file() and index in _index(packed)


def write_scans(folder: str | Path, sensor: str, scans: Iterable[tuple[int, Scan]]) -> Path:
    """`scans` ((number, scan) each, in order) as lidar `sensor`'s one LAZ (`lidarscans`)."""
    from .lidarscans import write_scans

    return write_scans(folder, sensor, scans)


def frame_path(folder: str | Path, sensor: str, index: int) -> Path:
    return Path(folder) / sensor / FRAMES / f"{index:06d}.jpg"


def normals_path(folder: str | Path, sensor: str, index: int) -> Path:
    return Path(folder) / sensor / NORMALS / f"{index:06d}.png"


def distances_path(folder: str | Path, sensor: str, index: int) -> Path:
    return Path(folder) / sensor / DISTANCES / f"{index:06d}.png"


def encode_distances(metres: np.ndarray) -> np.ndarray:
    """Distances (H x W, metres; inf or NaN for nothing) as 16-bit millimetres (0: nothing)."""
    mm = np.round(
        np.nan_to_num(np.asarray(metres, np.float64).squeeze(), nan=0.0, posinf=0.0) * 1000
    )
    return np.clip(mm, 0, 65535).astype(np.uint16)


def read_distances(folder: str | Path, sensor: str, index: int) -> np.ndarray | None:
    """A frame's distances from the camera (H x W, metres; NaN where nothing), or None."""
    file = distances_path(folder, sensor, index)
    if not file.is_file():
        return None
    from PIL import Image

    mm = np.asarray(Image.open(file), np.float64)
    return np.where(mm > 0, mm / 1000.0, np.nan)


def encode_normals(normals: np.ndarray) -> np.ndarray:
    """World-space normals (H x W x 3, -1..1) as an 8-bit picture (0..255 an axis)."""
    return np.clip(np.round((np.nan_to_num(normals) + 1.0) * 127.5), 0, 255).astype(np.uint8)


def read_normals(folder: str | Path, sensor: str, index: int) -> np.ndarray | None:
    """A frame's surface normals (H x W x 3, unit, world space), or None if not recorded."""
    file = normals_path(folder, sensor, index)
    if not file.is_file():
        return None
    from PIL import Image

    n = np.asarray(Image.open(file).convert("RGB"), np.float32) / 127.5 - 1.0
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)


def segment_path(folder: str | Path, sensor: str, index: int) -> Path:
    return Path(folder) / sensor / SEGMENTS / f"{index:06d}.png"


def read_segments(folder: str | Path, sensor: str, index: int) -> tuple[np.ndarray, list[str]]:
    """A camera's frame `index` segmented: each pixel's index (H x W) and the labels they
    index (0: "", nothing labelled)."""
    from PIL import Image

    labels = json.loads(
        (Path(folder) / sensor / SEGMENTS / SEGMENT_LABELS).read_text(encoding="utf-8")
    )
    return np.asarray(Image.open(segment_path(folder, sensor, index))), labels


def sky_path(folder: str | Path, sensor: str, index: int) -> Path:
    return Path(folder) / sensor / SKY / f"{index:06d}.png"


def write_sky(folder: str | Path, sensor: str, index: int, sky: np.ndarray) -> Path:
    """A sky view (`SKY_BLOCKED`, `SKY_LEAVES` or `SKY_OPEN` a pixel) as a grey picture."""
    from PIL import Image

    file = sky_path(folder, sensor, index)
    file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(sky, np.uint8) * SKY_GREY).astype(np.uint8)).save(file)
    return file


def read_sky(folder: str | Path, sensor: str, index: int) -> np.ndarray:
    """A sky view: `SKY_BLOCKED`, `SKY_LEAVES` or `SKY_OPEN` a pixel."""
    from PIL import Image

    grey = np.asarray(Image.open(sky_path(folder, sensor, index)).convert("L"), float)
    return np.clip(np.rint(grey / SKY_GREY), 0, SKY_OPEN).astype(np.uint8)


def read_scan(folder: str | Path, sensor: str, index: int) -> Scan:
    return _read_packed(Path(folder) / sensor / PACKED, index)


def write_hits(folder: str | Path, sensor: str, paths: list[str]) -> Path:
    file = Path(folder) / sensor / HITS
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(paths, indent=0) + "\n", encoding="utf-8")
    return file


def read_hits(folder: str | Path, sensor: str) -> list[str]:
    """The prims a lidar's points came back from (`Scan.hit` indexes it); empty if not recorded."""
    file = Path(folder) / sensor / HITS
    return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else []


#: A packed scan's points: on a grid this fine (metres; LAS's integer coordinates): at most half of it off.
LAZ_SCALE = 1e-4
#: Its fields kept exactly beside LAS's own (extra bytes): (name, dtype, what it is: 32 characters at most).
#: LAS's own intensity is `intensity_raw` scaled to 0..65535, for viewers.
LAZ_EXTRA = (
    ("intensity_raw", np.float32, "intensity, exactly"),
    ("offset", np.float32, "s after the scan's pose fired"),
    ("ring", np.uint32, "channel"),
    ("hit", np.uint32, "prim it hit, in hits.json"),
    ("material", np.uint32, "RTX lidar material ID"),
)


def _header(packed: bool = False) -> Any:
    """A LAS 1.4 header, point format 6, for scans: x, y, z on a `LAZ_SCALE` grid in the sensor's
    frame, and `LAZ_EXTRA` (and `scan`, the scan's number, when `packed`)."""
    import laspy

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([LAZ_SCALE] * 3)
    header.offsets = np.zeros(3)
    header.system_identifier = "darkroom capture"
    header.generating_software = "darkroom.format"
    extra = [*LAZ_EXTRA, ("scan", np.uint32, "scan's number")] if packed else LAZ_EXTRA
    header.add_extra_dims(
        [laspy.ExtraBytesParams(name, np.dtype(dtype), about) for name, dtype, about in extra]
    )
    return header


#: The packed scans' index: variable-length records of this user, each (number, points) pairs
#: (uint32, little-endian) of as many scans as fit in one (a record holds 65,535 bytes).
_INDEX = "darkroom scans"
#: The index's name in captures packed before it was darkroom's own: read, never written.
_INDEX_BEFORE = "strider scans"
_INDEX_PAIRS = 8000


def _index(packed: Path) -> dict[int, tuple[int, int]]:
    """Each scan's number in `packed`, with its first point and how many it has."""
    import laspy

    with laspy.open(str(packed)) as reader:
        records = [v for v in reader.header.vlrs if v.user_id in (_INDEX, _INDEX_BEFORE)]
    records.sort(key=lambda v: v.record_id)
    pairs = np.frombuffer(b"".join(v.record_data for v in records), "<u4").reshape(-1, 2)
    starts = np.concatenate([[0], np.cumsum(pairs[:, 1].astype(np.int64))])
    return {int(n): (int(starts[k]), int(c)) for k, (n, c) in enumerate(pairs)}


def _read_packed(packed: Path, index: int) -> Scan:
    """Scan `index` out of a lidar's one LAZ: only its own chunks read."""
    import laspy

    found = _index(packed).get(index)
    if found is None:
        raise FileNotFoundError(f"{packed} has no scan {index}.")
    start, count = found
    with laspy.open(str(packed), laz_backend=laspy.LazBackend.Lazrs) as reader:
        reader.seek(start)
        points = reader.read_points(count)
    return _scan_of(points)


def _scan_of(las: Any) -> Scan:
    """A scan from LAZ points (a file's, or some of one's)."""
    names = set(las.point_format.extra_dimension_names)
    hit, material = (np.zeros(0, np.uint32),) * 2  # older scans lack them
    if {"hit", "material"} <= names:
        hit, material = _recorded(
            np.asarray(las.hit, np.uint32), np.asarray(las.material, np.uint32)
        )
    return Scan(
        np.stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)], -1).astype(np.float32),
        np.asarray(las.intensity_raw, np.float32),
        np.asarray(las.offset, np.float32),
        np.asarray(las.ring, np.uint32),
        (np.asarray(las.return_number, np.uint8) - 1).astype(np.uint8),
        hit,
        material,
    )


def _recorded(hit: np.ndarray, material: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A scan's `hit` and `material`, or both empty when they say nothing (no prim named, no
    material: how a packed scan writes them when they were not recorded)."""
    if np.all(hit == NO_HIT) and not np.any(material):
        return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
    return hit.astype(np.uint32), material.astype(np.uint32)


def write_arrays(folder: str | Path, sensor: str, name: str, **arrays: np.ndarray) -> Path:
    """An IMU's or GNSS antenna's samples (`IMU`, `GNSS`)."""
    file = Path(folder) / sensor / name
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(file, allow_pickle=False, **arrays)
    return file


def read_arrays(folder: str | Path, sensor: str, name: str) -> dict[str, np.ndarray]:
    with np.load(Path(folder) / sensor / name) as data:
        return {k: data[k] for k in data.files}


def write_states(folder: str | Path, lanes: dict[str, list[tuple[float, float, str]]]) -> Path:
    """Lanes of what was going on (`STATES`): each a list of (start, end, label), seconds from
    the window's start. Lanes with no spans are left out."""
    path = Path(folder) / STATES
    kept = {
        lane: [[round(float(a), 3), round(float(b), 3), str(label)] for a, b, label in spans]
        for lane, spans in lanes.items()
        if spans
    }
    path.write_text(json.dumps(kept, indent=1) + "\n", encoding="utf-8")
    return path


def read_states(folder: str | Path) -> dict[str, list[tuple[float, float, str]]]:
    """The capture's lanes of what was going on: none when it has no `STATES`."""
    path = Path(folder) / STATES
    if not path.is_file():
        return {}
    return {
        lane: [(float(a), float(b), str(label)) for a, b, label in spans]
        for lane, spans in json.loads(path.read_text(encoding="utf-8")).items()
    }


def done(folder: str | Path) -> None:
    """Mark a capture finished: call last."""
    (Path(folder) / DONE).touch()


def finished(folder: str | Path) -> bool:
    return (Path(folder) / DONE).is_file() and (Path(folder) / META).is_file()


@dataclass(frozen=True)
class WasMade:
    """What a sensor its renderer made all of said it made, and how many frames or scans (None:
    not said)."""

    said: str
    count: int | None = None


def made(folder: str | Path, sensor: str, said: str, count: int | None = None) -> None:
    """Mark sensor `sensor` of capture `folder` made by its renderer, with what it said it made
    and how many frames or scans (`MADE`, JSON)."""
    file = Path(folder) / sensor / MADE
    file.write_text(json.dumps({"said": said, "count": count}) + "\n", encoding="utf-8")


def frames_kept(folder: str | Path, sensor: str) -> set[int]:
    """The numbers of camera `sensor`'s frames in capture `folder` that are whole (a JPEG to its
    end: one whose renderer stopped writing part way is not), part made or all."""
    found = set()
    frames = Path(folder) / sensor / FRAMES
    for file in frames.glob("[0-9]" * 6 + ".jpg") if frames.is_dir() else ():
        with file.open("rb") as f:
            f.seek(0, 2)
            if f.tell() < 4:
                continue
            f.seek(-2, 2)
            if f.read(2) == b"\xff\xd9":
                found.add(int(file.stem))
    return found


def was_made(folder: str | Path, sensor: str) -> WasMade | None:
    """What sensor `sensor` of capture `folder` said it made, if its renderer made all of it."""
    file = Path(folder) / sensor / MADE
    if not file.is_file():
        return None
    text = file.read_text(encoding="utf-8").strip()
    if not text.startswith("{"):  # one made before counts were kept: its sentence alone
        return WasMade(text)
    data = json.loads(text)
    return WasMade(str(data["said"]), data.get("count"))


def read_shared(folder: str | Path) -> dict[str, list[str]] | None:
    """A render's `shared.json` (each sensor rendered, with those it stood for), if it is a
    render for several rigs rather than a rig's own capture; None if not."""
    file = Path(folder) / SHARED
    return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else None
