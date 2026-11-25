"""How a capture is made: its options (`Options`), the one thing `capture`, `captures` and
`capture_rigs` take besides the stage and the folder; a capture to make among several (`Job`);
a span of the timeline rendered another way (`Span`). Plain data: importing this loads no USD.

    Options(window=(10, 12), kinds=["camera"], look="path_traced")
    Options(during=[Span(30, 45, "path_traced")])   # a look for part of the walk only
    Job("run.usda", "out/run", Options(blind=True))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from .renderers.renderer_cfg import RendererCfg

#: The kinds of sensor a capture makes (`format.Kind`).
KINDS = ("camera", "lidar", "imu", "gnss")
#: How many cameras are on at once (`turns.PACKING`, whose keys these are).
Packing = Literal["one by one", "conservative", "normal", "aggressive"]

#: A look (`darkroom.LOOKS`): a preset's name, a preset file (`.settings.usda`), RTX settings
#: of your own (`{"/rtx/pathtracing/spp": 128}`), or a list of them, later ones winning.
Look = str | Path | dict[str, Any] | list[str | Path | dict[str, Any]] | None


@dataclass(frozen=True)
class Span:
    """A span of the timeline rendered in `look` (seconds from the stage's start, as a
    capture's `window`): only the settings its look names change, and only while it lasts
    (inside a building, say, when the rest of the walk is out)."""

    start: float
    end: float
    look: Look

    def __post_init__(self) -> None:
        if not self.start < self.end:
            raise ValueError(f"a span starts before it ends, not {self.start} to {self.end}.")


@dataclass(frozen=True)
class Options:
    """How a capture is made; each left out is as said."""

    #: Seconds of the timeline captured, from the stage's start; None, all of it.
    window: tuple[float, float] | None = None
    #: The kinds of sensor captured ("camera", "lidar", "imu", "gnss"); None, every kind.
    kinds: Sequence[Literal["camera", "lidar", "imu", "gnss"]] | None = None
    #: How the cameras' pictures are rendered over the stage's own settings (`Look`); None,
    #: `looks.DEFAULT`; `{}`, the stage's settings as they are.
    look: Look = None
    #: Other looks for spans of the timeline.
    during: Sequence[Span] = ()
    #: The scenery kept beside the capture (`scenery.npz`) merged on a grid this fine (m): a
    #: street's 0.2 m (`surroundings.CELL`), a room's charts a few millimetres.
    scenery_cell: float = 0.2
    #: A lidar that hits nothing keeps its empty scans (past its reach in a test range, say)
    #: rather than failing.
    blind: bool = False
    #: Each lidar point's object kept: the prim it came back from (its scan's `hit`, listed
    #: in `hits.json`), for the test ranges' charts; else none. Off, it costs nothing (on, RTX's
    #: object ID table of the whole scene is read every update).
    lidar_hits: bool = False
    #: Prims whose poses are kept as bodies besides what carries the sensors (`Meta.bodies`):
    #: paths, or patterns of them (`fnmatch`'s: "/World/Machine/*"). A sensor under one rides
    #: on it (its exports mount it there); one that holds no sensor is a truth of its own.
    bodies: Sequence[str] = ()
    #: How many cameras are on at once (`turns.PACKING`): "one by one", whatever the GPU's
    #: memory; else while they fit in what is free, "conservative", "normal" or "aggressive"
    #: in what a pixel is taken to need and the headroom kept.
    packing: Packing = "normal"
    #: The cameras' frames' JPEG quality, 1 to 95 (PIL's): 80 is 62% of the bytes 92 makes on a
    #: 12 MP frame and 75 56%, each frame encoded once, as it is rendered.
    jpeg_quality: int = 80
    #: Each GNSS antenna's sky rendered (a 128x128 view a fix, its share of open sky kept:
    #: `skies`); False, none rendered, its fixes the truth alone.
    skies: bool = True
    #: The renderer's settings (`darkroom.renderers.OVRTXRendererCfg`: its step rate, its
    #: caches, ...); left as it is ("default"), ovrtx's as they are.
    renderer_cfg: RendererCfg = field(default_factory=RendererCfg)

    def __post_init__(self) -> None:
        """Refused as made, saying what may be: kinds of the four (a list, not one string),
        a packing of `Packing`, a window that starts before it ends and not before 0, a JPEG
        quality of 1 to 95, a scenery cell above 0."""
        kinds = self.kinds
        if kinds is not None and (isinstance(kinds, str) or not set(kinds) <= set(KINDS)):
            raise ValueError(f"kinds are a list of {', '.join(KINDS)}, not {kinds!r}.")
        if self.packing not in get_args(Packing):
            raise ValueError(
                f"no packing {self.packing!r}: there are {', '.join(get_args(Packing))}."
            )
        if self.window is not None and not 0 <= self.window[0] < self.window[1]:
            raise ValueError(
                f"a window starts at 0 or after and before it ends, not {self.window}."
            )
        if not (isinstance(self.jpeg_quality, int) and 1 <= self.jpeg_quality <= 95):
            raise ValueError(f"a JPEG quality is a whole number, 1 to 95, not {self.jpeg_quality}.")
        if not isinstance(self.renderer_cfg, RendererCfg):
            raise ValueError(
                f"a renderer_cfg is a renderer's cfg (darkroom.renderers), not {self.renderer_cfg!r}."
            )
        if not self.scenery_cell > 0:
            raise ValueError(f"a scenery cell is above 0 m, not {self.scenery_cell}.")


@dataclass(frozen=True)
class Job:
    """A capture of stage `stage_file` into the new or empty folder `out`, made as `options`."""

    stage_file: str | Path
    out: str | Path
    options: Options = Options()
