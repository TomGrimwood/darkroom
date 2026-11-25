"""What needs no renderer: every sensor's pose over the window, and what the IMUs and GNSS
antennas report (the truth, and their carrier's tip's beside it), read from the stage with USD
alone. How much of each antenna's view was sky is `skies`', once its renderer has rendered it.

A sensor's pose is its prim's world transform at each time code it is sampled at: frames
are plain time-sampled Xforms, so USD composes them as the renderer plays them.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Usd, UsdGeom

from darkroom import sensorkit
from darkroom.sensorkit import gnss, imu, motion

from . import format, surroundings
from .rates import update_rate

#: Samples a second a sensor's motion is read at, for its rates.
GRID = 120.0
#: Samples a second a body's (a pole's, its tip's) poses are kept at.
BODY_RATE = 60.0
#: The environment's prim, as omni.kit.environment.core keeps its date, time and place on it.
ENVIRONMENT = "/Environment"


def carriers(stage: Usd.Stage, sensors: list) -> dict[str, str]:
    """What carries the sensors and points on it: each prop (a prim declaring `Frames`) that a
    sensor sits under, by its name, and its `Frames/Tip` as `<name>_tip`."""
    from darkroom.sensorkit.frames import FRAMES, TIP, carrier

    found: dict[str, str] = {}
    for path in sensors:
        prim = carrier(stage, path)
        if prim is None:
            continue
        found.setdefault(prim.GetName(), str(prim.GetPath()))
        tip = prim.GetChild(FRAMES).GetChild(TIP)
        if tip:
            found.setdefault(f"{prim.GetName()}_tip", str(tip.GetPath()))
    return found


def named(stage: Usd.Stage, patterns: Collection[str]) -> dict[str, str]:
    """The prims `patterns` name (paths, or patterns of them as a shell's: `*` within one
    name, "/World/Machine/*" its children), by name. Raises for a pattern that names none,
    or two prims of one name."""
    from fnmatch import fnmatchcase

    found: dict[str, str] = {}
    for pattern in patterns:
        parts = pattern.strip("/").split("/")
        if not any(c in pattern for c in "*?["):
            prims = [stage.GetPrimAtPath(pattern)] if stage.GetPrimAtPath(pattern) else []
        else:
            prims = []
            for prim in stage.Traverse():
                names = str(prim.GetPath()).strip("/").split("/")
                if len(names) == len(parts) and all(map(fnmatchcase, names, parts)):
                    prims.append(prim)
        if not prims:
            raise RuntimeError(f"No prim is {pattern!r}, to keep as a body.")
        for prim in prims:
            was = found.setdefault(prim.GetName(), str(prim.GetPath()))
            if was != str(prim.GetPath()):
                raise RuntimeError(
                    f"Two bodies are called {prim.GetName()}: {was}, {prim.GetPath()}."
                )
    return found


def _fixes(moving: motion.Motion, where: Any, rate: float) -> dict[str, np.ndarray]:
    """`moving`'s true fixes at `rate` a second: seconds, latitude, longitude, height, and
    altitude when the site gives the geoid."""
    fixes = gnss.fixes(moving, where, rate)
    arrays = {
        "seconds": fixes.seconds,
        "latitude": fixes.latitude,
        "longitude": fixes.longitude,
        "height": fixes.height,
    }
    if fixes.altitude is not None:
        arrays["altitude"] = fixes.altitude
    return arrays


def start_utc(stage: Usd.Stage, seconds: float) -> str | None:
    """When `seconds` into the stage's timeline is, in UTC (ISO 8601): from the environment's `date`
    and `time:current` (local solar time, hours) and `location:longitude` (omni.kit.environment.core's),
    solar time less longitude / 15 h. None when the stage does not say."""
    env = stage.GetPrimAtPath(ENVIRONMENT)
    if not env:
        return None
    date, hour = env.GetAttribute("date"), env.GetAttribute("time:current")
    if not (date and date.Get() and hour and hour.Get() is not None):
        return None
    longitude = env.GetAttribute("location:longitude")
    east = float(longitude.Get()) if longitude and longitude.Get() is not None else 0.0
    day = datetime.fromisoformat(str(date.Get())).replace(tzinfo=UTC)
    return (day + timedelta(hours=float(hour.Get()) - east / 15.0, seconds=seconds)).isoformat()


class Window:
    """The seconds of a stage's timeline to capture: from its first time code, by default all of it."""

    def __init__(self, stage: Usd.Stage, window: tuple[float, float] | None = None):
        self.per_second = stage.GetTimeCodesPerSecond()
        self.first = stage.GetStartTimeCode()
        whole = (stage.GetEndTimeCode() - self.first) / self.per_second
        if whole <= 0:
            raise RuntimeError(
                "the stage has no time range: give its layers a start and end time code."
            )
        self.start, self.end = window or (0.0, whole)
        if not 0 <= self.start < self.end <= whole + 1e-9:
            raise RuntimeError(
                f"a window of {self.start:g} to {self.end:g} s is not within the stage's {whole:g} s."
            )

    def code(self, seconds: np.ndarray | float) -> np.ndarray:
        """The time code `seconds` after the stage's first."""
        return self.first + np.asarray(seconds) * self.per_second


def poses(stage: Usd.Stage, path, codes: np.ndarray) -> np.ndarray:
    """A prim's world transforms (N x 4 x 4, USD's row-vector convention) at `codes`."""
    prim = stage.GetPrimAtPath(path)
    xformable = UsdGeom.Xformable(prim)
    return np.array([np.array(xformable.ComputeLocalToWorldTransform(float(c))) for c in codes])


def fine(stage: Usd.Stage, span: Window, path, rate: float = GRID) -> format.Poses:
    """A prim's poses `rate` times a second over `span` (seconds from its start, metres): a
    lidar's, `GRID` a second, to put each point where it was fired (`format.MOTION`)."""
    grid = motion.sample_times(span.start, span.end, rate)
    positions, rotations = motion.from_usd_matrices(poses(stage, path, span.code(grid)))
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    return format.Poses(grid - span.start, positions * metres, rotations)


def traced_rate(stage: Usd.Stage, lidar_rates: list[float]) -> float:
    """How many updates a second a capture made before its lidars kept their fine poses traced
    them at: darkroom's lidar pass ran at a whole multiple of its lidars' rates, the least
    (unless the stage's root layer was animated: its own time codes a second). RTX traced each
    firing from a pose a straight line between the updates' poses."""
    per_second = stage.GetTimeCodesPerSecond()
    animated = bool(stage.GetRootLayer().ListAllTimeSamples())
    return update_rate(lidar_rates, per_second if animated else max(lidar_rates))


def _by_ovrtx(folder: Path, meta: format.Meta) -> bool:
    """Whether ovrtx made capture `folder`: its meta says so, or (one made before it did) its
    `perf.json` holds ovrtx's own profile, as only ovrtx's captures do."""
    import json

    from . import perf

    if format.RENDERER in meta.extra:
        return meta.extra[format.RENDERER] == "ovrtx"
    for file in (folder / perf.FILE, folder.parent / format.RENDER / perf.FILE):
        if file.is_file() and "ovrtx" in json.loads(file.read_text(encoding="utf-8")):
            return True
    return False


def add_lidar_motion(folder: str | Path, traced: float | None = None) -> dict[str, float]:
    """Capture `folder`'s lidars given their fine poses (`format.MOTION`) where they have none
    (made before them), worked out from its stage and window with USD: no render. Such a
    capture's lidars were traced from poses a straight line between those of its lidar pass's
    updates (`traced_rate`: once a scan, unless a faster lidar was in the pass), so that is
    what they are given: its points then lie where they were traced, as they do with a new
    capture's poses 120 times a second (at 60 updates a second, where it traced from). A capture
    traced once a scan stays as far from the true motion as it was (15 mm, p90 75 mm on a walk):
    render its lidars again for that. One ovrtx made (`format.RENDERER`, or ovrtx's profile in
    its `perf.json`) was traced at the stage's frames, as ovrtx's lidars step. `traced`: the
    updates a second its lidars were traced at, where known otherwise. Returns each lidar given
    them, with its poses a second."""
    folder = Path(folder)
    meta = format.read_meta(folder)
    wanting = [
        (name, sensor.prim)
        for name, sensor in meta.sensors.items()
        if sensor.kind == "lidar" and not format.has_motion(folder, name)
    ]
    if not wanting:
        return {}
    if not Path(meta.stage).is_file():
        raise RuntimeError(
            f"{folder}: its stage {meta.stage} is gone; its lidars' poses can't be worked out."
        )
    stage = Usd.Stage.Open(meta.stage)
    span = Window(stage, (meta.start, meta.end))
    # The pass rendered every lidar of the one render (a run of rigs: `format.RENDER` beside).
    render = folder.parent / format.RENDER
    rendered = format.read_meta(render) if (render / format.META).is_file() else meta
    rates = [s.rate for s in rendered.sensors.values() if s.kind == "lidar"]
    if traced is None and _by_ovrtx(folder, meta):
        traced = update_rate(rates, stage.GetTimeCodesPerSecond())
    rate = float(min(traced or traced_rate(stage, rates), GRID))
    made = {}
    for name, prim in wanting:
        if not stage.GetPrimAtPath(prim):
            raise RuntimeError(f"{folder}: its stage has no {prim} ({name}).")
        format.write_motion(folder, name, fine(stage, span, prim, rate))
        made[name] = rate
    return made


def write(
    stage_file: str | Path,
    out: str | Path,
    window: tuple[float, float] | None = None,
    kinds: Collection[str] | None = None,
    place: float | None = surroundings.WITHIN,
    on_line=None,
    cell: float = surroundings.CELL,
    bodies: Collection[str] = (),
) -> format.Meta:
    """Write `meta.json`, each sensor's `poses.npz`, and the IMUs' and antennas' samples into
    `out`; what carries them (`carriers`: poses, and its mesh) and the prims `bodies` names
    (`named`), and the place within `place` m of where it went (`surroundings.nearby`, its points
    merged on a `cell` m grid; None: not). Returns the meta, for the renderer's side to add to."""
    stage = Usd.Stage.Open(str(stage_file))
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError(f"{stage_file} is Y up; sensors are worked out on Z-up stages.")
    span = Window(stage, window)
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    where = sensorkit.site.find(stage)
    found = sensorkit.sensors(stage, kinds)
    if not found:
        raise RuntimeError(
            f"{stage_file} has no sensors{' of ' + ', '.join(kinds) if kinds else ''}."
        )
    meta = format.Meta(
        stage=str(Path(stage_file).resolve()),
        start=span.start,
        end=span.end,
        sensors={},
        site=format.Site(where.latitude, where.longitude, where.altitude, where.geoid)
        if where
        else None,
    )
    grid = motion.sample_times(span.start, span.end, GRID)
    for sensor in found:
        prim = stage.GetPrimAtPath(sensor.path)
        settings = sensorkit.camera.lens(prim).settings() if sensor.kind == "camera" else {}
        debug = prim.GetAttribute(sensorkit.DEBUG)
        if debug and debug.Get():
            settings = {**settings, "debug": True}
        if sensor.kind == "gnss" and where is None:
            raise RuntimeError(f"{sensor.name}: a GNSS antenna needs the stage to have a site.")
        if sensor.kind == "lidar":
            sensorkit.lidar.scan_rate(prim)  # refuses a lidar RTX cannot measure, before it starts
            seen = sensorkit.lidar.field_of_view(prim)
            if seen is not None:  # one that does not spin: what it sees
                settings = {"fov": list(seen)}
        meta.sensors[sensor.name] = format.Sensor(
            sensor.kind, str(sensor.path), sensor.rate, settings
        )

        at = motion.sample_times(span.start, span.end, sensor.rate)
        positions, rotations = motion.from_usd_matrices(poses(stage, sensor.path, span.code(at)))
        format.write_poses(
            out, sensor.name, format.Poses(at - span.start, positions * metres, rotations)
        )
        if sensor.kind == "lidar":
            format.write_motion(out, sensor.name, fine(stage, span, sensor.path))

        if sensor.kind in ("imu", "gnss"):
            matrices = poses(stage, sensor.path, span.code(grid))
            if where is not None:
                matrices = matrices @ where.to_site
            else:
                matrices[:, 3, :3] *= metres
            moving = motion.motion(grid - span.start, *motion.from_usd_matrices(matrices))
            if sensor.kind == "imu":
                got = imu.samples(moving, sensor.rate)
                format.write_arrays(
                    out,
                    sensor.name,
                    format.IMU,
                    seconds=got.seconds,
                    accel=got.accel,
                    gyro=got.gyro,
                )
            else:
                assert where is not None
                arrays = _fixes(moving, where, sensor.rate)
                # Its carrier's tip (a pole's), where a surveyor's fix is taken, beside it.
                from darkroom.sensorkit.frames import FRAMES, TIP, carrier

                held = carrier(stage, sensor.path)
                tip = held.GetChild(FRAMES).GetChild(TIP) if held else None
                if tip:
                    at_tip = poses(stage, tip.GetPath(), span.code(grid)) @ where.to_site
                    tipped = motion.motion(grid - span.start, *motion.from_usd_matrices(at_tip))
                    got = _fixes(tipped, where, sensor.rate)
                    arrays |= {f"tip_{k}": v for k, v in got.items() if k != "seconds"}
                format.write_arrays(out, sensor.name, format.GNSS, **arrays)

    # What carries the sensors and its tip, as sensors are kept; the carrier's mesh; the place about.
    meta.start_utc = start_utc(stage, span.start)
    meta.bodies = carriers(stage, [s.path for s in found]) | named(stage, bodies)
    at = motion.sample_times(span.start, span.end, BODY_RATE)
    way = []
    for name, path in meta.bodies.items():
        matrices = poses(stage, path, span.code(at))
        positions, rotations = motion.from_usd_matrices(matrices)
        format.write_poses(out, name, format.Poses(at - span.start, positions * metres, rotations))
        if not name.endswith("_tip"):
            way.append(positions[:: max(1, int(BODY_RATE))] * metres)
            made = surroundings.model(stage, path, float(span.code(span.start)))
            if made is not None:
                format.write_mesh(Path(out) / name / format.MODEL, made)
    if place and not way:  # nothing carries them (a rig on a stand): where the sensors are
        for sensor in found:
            ats = format.read_poses(out, sensor.name).position
            way.append(ats[:: max(1, len(ats) // 10)])
    if place and way:
        said: list[str] = []
        near = surroundings.nearby(stage, np.concatenate(way), within=place, cell=cell, said=said)
        if near is not None:
            format.write_mesh(Path(out) / format.SCENERY, near)
            said.append(f"scenery: {len(near.triangles):,} triangles within {place:g} m of the way")
        if on_line is not None:
            for line in said:
                on_line(line)
    format.write_meta(out, meta)
    return meta
