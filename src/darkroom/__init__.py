"""darkroom: a stage's sensors rendered into a capture folder (`darkroom.format`).

    darkroom.capture("run.usda", "<folder>")                     # every sensor on the stage
    darkroom.capture(..., Options(window=(10, 12), look="path_traced"))   # made so
    darkroom.captures([Job("a.usda", "<folder>"), Job(...)])     # several, in few processes
    darkroom.stills("scene.usd", [Shot(Camera(...), "a.png")])   # pictures from any cameras

    rigs = [darkroom.Rig("A", "<prim>"), darkroom.Rig("B", "<prim>")]
    darkroom.capture_rigs("run.usda", "<folder>", rigs)         # a sensor the same in both: once
    board = darkroom.board.Board()                               # on_progress=board.take: rows

Poses, IMUs and true GNSS fixes are worked out from the stage with USD (`truth`); cameras,
lidars and the antennas' skies are rendered by NVIDIA's ovrtx in a process of its own
(`darkroom.ovrtx`, through `darkroom.process`), its settings `Options.renderer_cfg`
(`darkroom.renderers.OVRTXRendererCfg`). Every kind's update rate is `rates`'. What a
sensor is comes from `darkroom.sensorkit`. Knows nothing of environments, characters or
trajectories.

Importing this package imports no USD and no ovrtx: what needs them loads when first asked.
"""

import importlib
from typing import TYPE_CHECKING, Any

from . import board, format, terminal
from .looks import LOOKS
from .options import Job, Options, Span
from .process import Failed, Progress
from .stills import Camera, Shot, stills

if TYPE_CHECKING:
    from . import sensorkit
    from .run import capture, captures
    from .share import Rig, capture_rigs, count_shared
    from .truth import add_lidar_motion

#: The whole of darkroom's interface; its other modules are its own.
__all__ = [
    "LOOKS",  # how cameras are rendered: presets of RTX settings by name
    "Camera",  # a camera for `stills`
    "Failed",  # raised when the renderer's side fails (the GPU out of memory, say)
    "Job",  # a capture among several, for `captures`
    "Options",  # how a capture is made: window, kinds, look, ...
    "Progress",  # what `on_progress` is handed: a step, how far, what else
    "Rig",  # a rig attached on a stage, for `capture_rigs`
    "Shot",  # a picture `stills` makes
    "Span",  # a span of the timeline in another look (`Options.during`)
    "add_lidar_motion",  # an older capture's lidars given their poses as traced, from its stage
    "board",  # a render's progress as rows, for a window or a terminal to draw
    "capture",  # a stage's sensors into a capture folder
    "capture_rigs",  # rigs on a stage, each distinct sensor once, a capture each
    "captures",  # several, in as few renderer processes as render them
    "count_shared",  # how many sensors rigs have between them, and how many are rendered
    "format",  # capture folders: read and written as plain files
    "sensorkit",  # what the sensors on a stage are; IMU and GNSS maths
    "stills",  # pictures from any cameras at chosen times
    "terminal",  # a render's board drawn live in a terminal (rich)
]


def __getattr__(name: str) -> Any:
    """What loads USD (`capture`, `captures`, `capture_rigs`, `Rig`, `count_shared`,
    `add_lidar_motion`, `sensorkit`): loaded when first asked for, so that importing darkroom
    stays cheap."""
    if name in ("capture", "captures"):
        return getattr(importlib.import_module(f"{__name__}.run"), name)
    if name in ("Rig", "capture_rigs", "count_shared"):
        return getattr(importlib.import_module(f"{__name__}.share"), name)
    if name == "add_lidar_motion":
        return importlib.import_module(f"{__name__}.truth").add_lidar_motion
    if name == "sensorkit":
        return importlib.import_module(f"{__name__}.sensorkit")
    raise AttributeError(f"module 'darkroom' has no attribute {name!r}")
