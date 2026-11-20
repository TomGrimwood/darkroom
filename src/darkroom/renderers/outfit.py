"""A capture's cameras made ready for a renderer, and what says how far each has got: each
camera's size, frames and board row, what puts it on rendering into its folder (named by its
poses: `frame_store`), what finishes it once off (`outfit`); where a camera is at a time of the
timeline (`poser`); which of its frames it has (`lacking_from`, `covered`). `once` renders them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base_renderer import BaseRenderer
from .simulation_context import SimulationContext

#: Updates played past the window's end, for the last renders to be handed over (at 4, the
#: last frame and scan came after a 2 s window's end at 60 updates a second, and were lost).
#: Every pass's.
TRAILING = 12
#: Seconds between reports of how far a pass has got and readings of the GPU's memory.
PROGRESS = 0.5


def _gap(have: set[int], poses: int) -> tuple[int, int] | None:
    """The longest stretch of `poses` a camera lacks, round the end: its first pose and the
    pose after its last (mod `poses`); None when it has all or none."""
    got = np.zeros(poses, bool)
    got[[k for k in have if 0 <= k < poses]] = True
    if got.all() or not got.any():
        return None
    lacking = np.flatnonzero(~np.concatenate([got, got]))
    runs = np.split(lacking, np.flatnonzero(np.diff(lacking) != 1) + 1)
    longest = max(runs, key=len)
    return int(longest[0]) % poses, int(longest[-1] + 1) % poses


def lacking_from(have: set[int], poses: int) -> int:
    """The first pose of the longest stretch of `poses` a camera lacks (0 when it lacks all)."""
    gap = _gap(have, poses) if poses else None
    return gap[0] if gap else 0


def covered(have: set[int], poses: int) -> tuple[float, float]:
    """Where a camera's bar starts filling (0 to 1) and how full it is: the poses it `have`s of
    `poses`, drawn on from the end of the longest stretch it lacks (`board.covered`)."""
    if not poses:
        return 0.0, 0.0
    share = len([k for k in have if 0 <= k < poses]) / poses
    gap = _gap(have, poses)
    return (gap[1] / poses if gap else 0.0), min(share, 0.99 if share < 1 else 1.0)


#: A camera's own frame (USD's: looking down -Z, +Y up) in the convention Isaac Lab hands
#: `update_camera` (looking along +X, +Z up), as a quaternion (w, x, y, z).
TO_WORLD_CONVENTION = (0.5, -0.5, 0.5, 0.5)


def poser(
    stage: Any, path: Any, lens: Any
) -> Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """What says where the camera at `path` is at a time of the timeline (seconds), for
    `BaseRenderer.update_camera`: its position (1 x 3, metres), its orientation (1 x 4, a
    quaternion x y z w, in Isaac Lab's world convention) and its intrinsic matrix (1 x 3 x 3,
    from `lens`), read from the stage with USD."""
    from pxr import UsdGeom

    from darkroom.sensorkit import motion

    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    intrinsics = np.array(
        [[[lens.fx, 0.0, lens.cx], [0.0, lens.fy, lens.cy], [0.0, 0.0, 1.0]]], np.float32
    )
    cw, cx, cy, cz = TO_WORLD_CONVENTION

    def pose(time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        code = time * stage.GetTimeCodesPerSecond()
        matrix = np.array(xformable.ComputeLocalToWorldTransform(code))[None]
        positions, rotations = motion.from_usd_matrices(matrix)
        w, x, y, z = rotations[0]
        turned = np.array(  # its rotation, then the change of convention (a Hamilton product)
            [
                [
                    w * cx + x * cw + y * cz - z * cy,
                    w * cy - x * cz + y * cw + z * cx,
                    w * cz + x * cy - y * cx + z * cw,
                    w * cw - x * cx - y * cy - z * cz,
                ]
            ],
            np.float32,
        )
        return (positions * metres).astype(np.float32), turned, intrinsics

    return pose


@dataclass
class Outfit:
    """A capture's cameras as a renderer is to render them: how big each is (`size`) and how
    many frames it should make (`frames`), their rows on a board (`order`, biggest first;
    `turn_of`, `planned`), what puts one on the renderer rendering into its folder (`camera`:
    a `frame_store.Camera`) and what finishes one once off (`off`: its outcome)."""

    size: Callable[[Any], tuple[int, int]]
    frames: Callable[[Any], int]
    order: list[Any]
    turn_of: dict[str, int]
    planned: tuple[Any, ...]
    camera: Callable[[Any, Path], Any]
    off: Callable[[Any, Any, float, float, Callable[[Any], Any]], Any]


def outfit(renderer: BaseRenderer, sim: SimulationContext, groups: list[list[Any]]) -> Outfit:
    """The cameras of `groups` (`sensorkit.Sensor`s) as `renderer` is to render them over the
    window of `sim.job`, into its capture folder: each one's frames named by their poses, its
    mosaic made, marked made (`Outfit`)."""
    from darkroom import format, sensorkit
    from darkroom.events import Plan
    from darkroom.outcome import finished

    from .camera_cfg import CameraCfg
    from .camera_render_spec import CameraRenderSpec
    from .frame_store import Camera, FrameStore

    job, stage = sim.job, sim.stage
    out = Path(job.out)
    start, end = job.window
    lenses = {s.name: sensorkit.camera.lens(stage.GetPrimAtPath(s.path)) for g in groups for s in g}

    def size(sensor: Any) -> tuple[int, int]:
        return (lenses[sensor.name].width, lenses[sensor.name].height)

    def frames(sensor: Any) -> int:
        """The frames `sensor` should make: a pose at both ends of the window, a frame each."""
        return int((end - start) * sensor.rate + 1e-9) + 1

    # Each camera its own row on a board, numbered as it may get on: biggest first.
    order = [s for g in groups for s in sorted(g, key=lambda s: math.prod(size(s)), reverse=True)]
    turn_of = {s.name: n for n, s in enumerate(order)}
    planned = tuple(
        Plan(s.name, "camera", s.rate, size(s), n, len(order), frames(s))
        for n, s in enumerate(order)
    )
    # What the stage asks of its cameras beside their pictures: each pixel's segment (by the
    # taxonomy it names), and for the cameras it names, its normal and distance.
    asked = stage.GetRootLayer().customLayerData
    taxonomy = asked.get(format.SEGMENT_TAXONOMY)
    with_geometry = str(asked.get(format.GEOMETRY_CAMERAS) or "").split(",")

    def camera(sensor: Any, folder: Path) -> Camera:
        geometry = bool(taxonomy) and sensor.name in with_geometry
        wanted = ["rgb"] + (["semantic_segmentation"] if taxonomy else [])
        wanted += ["normals", "distance_to_camera"] if geometry else []
        width, height = size(sensor)
        cfg = CameraCfg(
            prim_path=str(sensor.path),
            width=width,
            height=height,
            update_period=1 / sensor.rate,
            data_types=tuple(wanted),
            output_dir=str(folder / format.FRAMES),
        )
        spec = CameraRenderSpec(cfg, sim.device, 1, (str(sensor.path),), 1)
        store = FrameStore(
            out,
            sensor.name,
            sensor.rate,
            (width, height),
            quality=job.quality,
            taxonomy=taxonomy,
            geometry=geometry,
        )
        return Camera(renderer, spec, store, poser(stage, sensor.path, lenses[sensor.name]))

    def off(
        sensor: Any, running: Any, keep_from: float, took: float, seconds: Callable[[Any], Any]
    ) -> Any:
        """A camera off the loop finished: its frames kept and named, its mosaic, marked made."""
        return finished(
            "camera", sensor, running, seconds, end - start, took, out, "cameras", keep_from
        )

    return Outfit(size, frames, order, turn_of, planned, camera, off)
