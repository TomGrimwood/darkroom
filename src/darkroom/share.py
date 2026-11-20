"""Several rigs on one stage, each distinct sensor rendered once, each rig given its own capture.

Two sensors are the same when everything that decides what they record is the same: their
kind, what carries them, their pose on it, and every attribute value in their prims (their
sensor file's and the rig's over it, rate and resolution among them): a `SensorKey`, read
from the stage as it is, so an edit made to a rig counts.

    rigs = [Rig("A", "/World/Walk/Frames/SensorMount/A"), Rig("B", ".../B"),
            Rig("Drone", ".../Drone", own=False)]          # in every other rig's capture
    folders = capture_rigs("run.usda", "out/run", rigs, Options(window=(10, 12)))
    # {"A": out/run/A, "B": out/run/B}; the one render in out/run/.render

`capture_rigs` renders a stage over the one given with every sensor but the first of each key
turned off (`rendered.usda`, and which stands for which in `shared.json`, in the run folder's
hidden `.render/`), then gives each rig a capture folder in the run's folder: its sensors under
the names they have with it alone, their files hard links to the one render of each. Once every
rig has its folder, `.render/` keeps only what says how it was made (`meta.json`, `shared.json`,
`rendered.usda`, `perf.json`, `.done`): its files are each in a rig's folder, and a run's folder copied or
uploaded has no second copy of them.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from darkroom import process

from . import events, format, perf, sensorkit
from .options import Options
from .run import Said, capture

#: The stage rendered for several rigs: the one given, with the duplicates off.
RENDERED = "rendered.usda"


@dataclass(frozen=True)
class Rig:
    """A rig attached on a stage: its name, the prim it was attached as, and whether it is
    given a capture of its own (else it is in every other's: a review camera, say)."""

    name: str
    path: str
    own: bool = True


#: How finely poses and values are compared: closer than this is the same.
DIGITS = 6


@dataclass(frozen=True)
class SensorKey:
    """What makes a sensor's recording what it is; equal keys record the same."""

    kind: str
    #: The frame it is carried by (a walk's `SensorMount`).
    frame: str
    #: Its pose on the frame, the 4x4 matrix row by row.
    pose: tuple[float, ...]
    #: Every attribute value in its prim and below: (prim path inside it, attribute, value).
    settings: frozenset[tuple[str, str, Any]]


def _plain(value: Any) -> Any:
    """`value` as something hashable that compares as its meaning does."""
    if isinstance(value, float):
        return round(value, DIGITS)
    if isinstance(value, Sdf.AssetPath):
        return value.resolvedPath or value.path
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Gf.Matrix4d | Gf.Matrix3d | Gf.Matrix4f | Gf.Matrix3f):
        return tuple(_plain(float(x)) for row in value for x in row)
    if isinstance(value, Vt.StringArray | Vt.TokenArray | Sdf.AssetPathArray):
        return tuple(_plain(x) for x in value)
    try:  # vectors, quaternions and arrays of numbers or of them, at numpy's speed: a lidar's
        # firing tables are a million numbers
        numbers = np.asarray(value)
    except (TypeError, ValueError):
        return str(value)
    if numbers.dtype.kind == "f":
        numbers = np.round(numbers.astype(np.float64), DIGITS) + 0.0  # -0.0 as 0.0
    elif numbers.dtype.kind not in "iub":
        return str(value)
    return (numbers.dtype.kind, numbers.shape, numbers.tobytes())


def key_of(prim: Usd.Prim, kind: str, frame: Usd.Prim, cache: UsdGeom.XformCache) -> SensorKey:
    """Sensor `prim`'s key, carried by `frame` (`cache` at the time poses are compared)."""
    on_frame = (
        cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(frame).GetInverse()
    )
    settings = set()
    for inside in Usd.PrimRange(prim):
        where = str(inside.GetPath().MakeRelativePath(prim.GetPath()))
        settings.add((where, "type", inside.GetTypeName()))
        settings.add((where, "apiSchemas", tuple(inside.GetAppliedSchemas())))
        for attribute in inside.GetAttributes():
            if attribute.HasValue():
                settings.add((where, attribute.GetName(), _plain(attribute.Get())))
        for relationship in inside.GetRelationships():
            targets = tuple(
                str(t.MakeRelativePath(prim.GetPath())) for t in relationship.GetTargets()
            )
            settings.add((where, relationship.GetName(), targets))
    return SensorKey(kind, str(frame.GetPath()), _plain(on_frame), frozenset(settings))


def sensor_keys(stage: Usd.Stage, rigs: list[str]) -> dict[SensorKey, list[str]]:
    """Every sensor in the attached rigs at paths `rigs` (each a frame's child), by key: the
    sensors (prim paths) that share it, in stage order."""
    cache = UsdGeom.XformCache(stage.GetStartTimeCode())
    found: dict[SensorKey, list[str]] = {}
    for sensor in sensorkit.sensors(stage):
        rig = next((r for r in rigs if sensor.path.HasPrefix(Sdf.Path(r))), None)
        if rig is None:
            continue
        frame = stage.GetPrimAtPath(Sdf.Path(rig).GetParentPath())
        key = key_of(stage.GetPrimAtPath(sensor.path), sensor.kind, frame, cache)
        found.setdefault(key, []).append(str(sensor.path))
    return found


def capture_rigs(
    stage_file: str | Path,
    out: str | Path,
    rigs: Sequence[Rig],
    options: Options = Options(),
    *,
    on_progress: Callable[[process.Progress], object] | None = None,
    on_start: Callable[[Any], object] | None = None,
    on_line: Said | None = print,
) -> dict[str, Path]:
    """`rigs` on stage `stage_file` captured as `options` say into run folder `out`, a capture
    folder in it for each rig that has its own (`<out>/<rig>/`). With several, each distinct
    sensor is rendered once (into `<out>/.render/`, `format.RENDER`) and each rig's folder made from
    it, its files hard links, then only what says how it was made left in `.render/`; with
    one, it is rendered straight into its folder. Before rendering, `on_progress` is told which rigs each sensor is for (`events.Uses`); the
    callbacks are otherwise `capture`'s. Returns each rig's folder."""
    run = Path(out)
    run.mkdir(parents=True, exist_ok=True)
    rigs = list(rigs)
    own = [r for r in rigs if r.own]
    if len(own) > 1:  # the duplicates off, over the stage given
        out = run / format.RENDER
        out.mkdir(exist_ok=True)
        rendered, groups = _render_once(Path(stage_file).resolve(), out, rigs)
    else:  # one capture: the stage as it is, into the rig's folder
        out = run / (own[0].name if own else format.RENDER)
        rendered, groups = Path(stage_file).resolve(), {}
    if on_progress:
        uses = _uses(rendered, rigs, groups)
        on_progress(process.Progress("sensors", None, events.extra(uses)))
    capture(rendered, out, options, on_progress=on_progress, on_start=on_start, on_line=on_line)
    if len(own) <= 1:
        return {r.name: out for r in own}
    if on_progress:
        on_progress(process.Progress("making each rig's folder", None, {}))
    made = _fan_out(out, run, rendered, rigs, groups, options.kinds)
    _leave_only_how(out)
    return made


def _render_once(stage_file: Path, out: Path, rigs: list[Rig]) -> tuple[Path, dict]:
    """A stage over `stage_file` with every sensor but the first of each key off, in `out`,
    and which stands for which (each rendered sensor's path: the sensors it stands for)."""
    file = str(out / RENDERED)  # made again when a capture is carried on
    layer = Sdf.Layer.FindOrOpen(file) if (out / RENDERED).is_file() else None
    if layer is not None:
        layer.Clear()
    else:
        layer = Sdf.Layer.CreateNew(file)
    source = Sdf.Layer.FindOrOpen(str(stage_file))
    layer.subLayerPaths.append(str(stage_file))
    for info in ("upAxis", "metersPerUnit", "defaultPrim"):
        if source.pseudoRoot.HasInfo(info):
            layer.pseudoRoot.SetInfo(info, source.pseudoRoot.GetInfo(info))
    layer.timeCodesPerSecond = source.timeCodesPerSecond
    layer.startTimeCode, layer.endTimeCode = source.startTimeCode, source.endTimeCode
    layer.customLayerData = dict(source.customLayerData)
    stage = Usd.Stage.Open(layer)
    groups = {paths[0]: paths for paths in sensor_keys(stage, [r.path for r in rigs]).values()}
    for paths in groups.values():
        for path in paths[1:]:
            Sdf.CreatePrimInLayer(layer, path).active = False
    layer.Save()
    (out / format.SHARED).write_text(json.dumps(groups, indent=1) + "\n", encoding="utf-8")
    return out / RENDERED, groups


def _uses(rendered: Path, rigs: list[Rig], groups: dict) -> events.Uses:
    """Each sensor rendered (by its name in the capture): its kind, rate and the rigs it is
    for; how many sensors the rigs have, and how many are made once for them."""
    stage = Usd.Stage.Open(str(rendered))

    def rig_of(path: str) -> str:
        return next((r.name for r in rigs if Sdf.Path(path).HasPrefix(Sdf.Path(r.path))), "")

    found: dict[str, events.Use] = {}
    in_rigs = 0
    for sensor in sensorkit.sensors(stage):
        paths = groups.get(str(sensor.path), [str(sensor.path)])
        rigs_of = tuple(dict.fromkeys(r for r in map(rig_of, paths) if r))
        found[sensor.name] = events.Use(sensor.kind, sensor.rate, rigs_of)
        in_rigs += len(paths)
    return events.Uses(found, in_rigs, len(found))


def _fan_out(
    done: Path,
    run: Path,
    rendered: Path,
    rigs: list[Rig],
    groups: dict,
    kinds: Sequence[str] | None,
) -> dict[str, Path]:
    """Capture `done` given out as a capture folder per rig with its own (`<run>/<rig>/`):
    its sensors under the names they have with it alone (the rigs without their own beside
    it), their files hard links to the sensor rendered for them; what carried them, the
    scenery and the states linked too."""
    stands_for = {p: kept for kept, ps in groups.items() for p in ps}
    meta = format.read_meta(done)
    name_of = {s.prim: name for name, s in meta.sensors.items()}
    made: dict[str, Path] = {}
    for rig in [r for r in rigs if r.own]:
        out = run / rig.name
        shutil.rmtree(out, ignore_errors=True)  # one left part made (links only): made again
        out.mkdir()
        sensors = {}
        for sensor in _alone(rendered, rigs, rig, groups, kinds):
            kept = name_of[stands_for.get(str(sensor.path), str(sensor.path))]
            _link(done / kept, out / sensor.name)
            mosaic = done / f"{kept}_mosaic.jpg"  # a camera's frames at a glance
            if mosaic.is_file():
                _link(mosaic, out / f"{sensor.name}_mosaic.jpg")
            sensors[sensor.name] = replace(meta.sensors[kept], prim=str(sensor.path))
        for entry in [*meta.bodies, "scenery.npz", "perf.json", format.STATES]:
            if (done / entry).exists():
                _link(done / entry, out / entry)
        format.write_meta(out, replace(meta, sensors=sensors))
        format.done(out)
        made[rig.name] = out
    return made


#: What `.render/` keeps once every rig has its folder: how it was made (`perf.json` too: which
#: renderer made it, `truth.add_lidar_motion` reads), and that it was.
KEPT = (format.META, format.SHARED, RENDERED, perf.FILE, format.DONE)


def _leave_only_how(done: Path) -> None:
    """The render `done` with only `KEPT` left: everything else is in the rigs' folders, so a
    copy or upload of the run's folder (which cannot keep hard links) has each file once."""
    for entry in done.iterdir():
        if entry.name in KEPT:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _alone(
    rendered: Path, rigs: list[Rig], rig: Rig, groups: dict, kinds: Sequence[str] | None
) -> list[sensorkit.Sensor]:
    """The sensors the stage has with only `rig` of those with their own on (and every one
    of its sensors on, shared or not), named as they would be."""
    stage = Usd.Stage.Open(str(rendered))
    session = stage.GetSessionLayer()
    with Usd.EditContext(stage, session):
        for paths in groups.values():
            for path in paths[1:]:
                Sdf.CreatePrimInLayer(session, path).active = True
        for other in rigs:
            if other.own and other != rig:
                Sdf.CreatePrimInLayer(session, other.path).active = False
    return sensorkit.sensors(stage, kinds)


def count_shared(rigs: dict[str, Path]) -> tuple[int, int]:
    """How many sensors rig files `rigs` (name: file) have between them, and how many are
    rendered when one frame carries them all."""
    import usdsensors

    layer = Sdf.Layer.CreateAnonymous(".usda")
    frame = Sdf.CreatePrimInLayer(layer, "/Frame")
    frame.specifier, frame.typeName = Sdf.SpecifierDef, "Xform"
    for name, file in rigs.items():
        usdsensors.attach(layer, file, "/Frame", name)
    keys = sensor_keys(Usd.Stage.Open(layer), [f"/Frame/{name}" for name in rigs])
    return sum(len(p) for p in keys.values()), len(keys)


def _link(source: Path, to: Path) -> None:
    """`source` (a file or a folder) at `to` as hard links: no copies made, where the drive
    has hard links (not FAT, exFAT, many network shares: copies there)."""
    if source.is_dir():
        shutil.copytree(source, to, copy_function=_link_file)
    else:
        _link_file(source, to)


def _link_file(source: str | Path, to: str | Path) -> None:
    try:
        os.link(source, to)
    except OSError:
        shutil.copy2(source, to)
