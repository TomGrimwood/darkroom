"""What a capture hands its renderer's process to render (`ovrtx.render_all`), typed: a
`RenderJob`, sent across as plain data (`plain`, `read`), as `events` are. Standard library
only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class RenderJob:
    """A stage's sensors to render into a capture folder whose `meta.json` `darkroom.truth`
    has written."""

    #: The stage, and the capture folder.
    stage_file: str
    out: str
    #: Seconds of the stage's timeline rendered, from its start.
    window: tuple[float, float]
    #: RTX settings for the cameras' pass only, over the stage's own (`darkroom.looks`).
    look: Mapping[str, Any] = field(default_factory=dict)
    #: Camera frames played before the window and not kept (a path tracer's first is black,
    #: auto exposure settles); the other kinds play one sensor period before it.
    warmup: int = 1
    #: (start, end, settings), seconds of the stage's timeline: the cameras' settings over
    #: `look` while it is between them (a walk indoors, say), back to `look`'s after.
    spans: tuple[tuple[float, float, Mapping[str, Any]], ...] = ()
    #: How fisheye cameras are rendered, by prim path (`darkroom.lenses.wanted`).
    lenses: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: A lidar that hits nothing keeps its empty scans, not a failure.
    blind: bool = False
    #: Each lidar point's object kept (`Options.lidar_hits`).
    hits: bool = False
    #: How many cameras ride the loop at once (`darkroom.turns.PACKING`).
    packing: str = "normal"
    #: The cameras' frames' JPEG quality (`Options.jpeg_quality`).
    quality: int = 80
    #: Pixels the cameras on the loop may have at once at most, whatever is free (after running
    #: out of memory, those on then less the smallest: `turns.step_back`); None, no cap.
    most: int | None = None
    #: The renderer the cameras are rendered by, as plain data (`renderers.renderer_cfg.plain`
    #: of the capture's `Options.renderer_cfg`); none, the default renderer's.
    renderer_cfg: Mapping[str, Any] = field(default_factory=dict)


def plain(job: RenderJob) -> dict[str, Any]:
    """`job` as plain data, ready for JSON."""
    return asdict(job)


def read(data: Mapping[str, Any]) -> RenderJob:
    """The job in plain data (`plain`, through JSON). Raises for a key it does not know: the
    two sides out of step."""
    known = {f.name for f in fields(RenderJob)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"a render job has no {', '.join(unknown)}.")
    window, most = data["window"], data.get("most")
    return RenderJob(
        stage_file=str(data["stage_file"]),
        out=str(data["out"]),
        window=(float(window[0]), float(window[1])),
        look=dict(data.get("look", {})),
        warmup=int(data.get("warmup", 1)),
        spans=tuple((float(a), float(b), dict(s)) for a, b, s in data.get("spans", ())),
        lenses={str(k): dict(v) for k, v in data.get("lenses", {}).items()},
        blind=bool(data.get("blind", False)),
        hits=bool(data.get("hits", False)),
        packing=str(data.get("packing", "normal")),
        quality=int(data.get("quality", 80)),
        most=int(most) if most is not None else None,
        renderer_cfg=dict(data.get("renderer_cfg", {})),
    )
