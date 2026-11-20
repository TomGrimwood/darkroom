"""Still pictures of a stage: any cameras, as many as wanted, in one ovrtx process
(`darkroom.ovrtx.stills`).

    from darkroom.stills import Camera, Shot, stills
    above = Camera(position=(10, 5, 50), projection="orthographic", width=20, resolution=(800, 400))
    stills("scene.usd", [Shot(above, "top.png"), Shot(replace(above, near=41.0), "cut_z9.png")])

A camera looks from `position` at `look_at` (straight down when there is none), `up` the way
the picture's top points. Perspective: `focal_length` and `horizontal_aperture` in mm, as USD's
camera has them. Orthographic: `width` across the picture, in stage units. Nothing nearer
than `near` is drawn: a camera at height h with `near` d cuts everything above h - d away.
Its picture's shape is `resolution`'s.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from darkroom import process

XYZ = tuple[float, float, float]


@dataclass(frozen=True)
class Camera:
    position: XYZ
    look_at: XYZ | None = None
    up: XYZ = (0.0, 1.0, 0.0)
    projection: Literal["perspective", "orthographic"] = "perspective"
    focal_length: float = 24.0
    horizontal_aperture: float = 36.0
    width: float | None = None
    near: float = 0.1
    far: float = 100000.0
    resolution: tuple[int, int] = (1280, 720)

    def target(self) -> XYZ:
        """Where it looks: `look_at`, else straight down."""
        x, y, z = self.position
        return self.look_at if self.look_at is not None else (x, y, z - 1.0)


@dataclass(frozen=True)
class Shot:
    camera: Camera
    png: str


def stills(
    stage_file: str | Path,
    shots: Sequence[Shot],
    settle: tuple[int, int] = (8, 4),
    subframes: int = 4,
    on_progress: Callable[[process.Progress], object] | None = None,
    on_start: Callable[[Any], object] | None = None,
    settings: dict[str, Any] | None = None,
    sun: tuple[float, float] | None = None,
    renderer_cfg: Any = None,
) -> list[float]:
    """Render each shot of the stage into its PNG, in one ovrtx process. Returns the
    seconds each took (the first also loads the scene's textures: several seconds).

    `settle`: frames rendered before the one kept, for the first shot and each after it
    (RTX's lighting settles over the first few); `subframes`: renders a frame accumulates.
    Fewer of either is faster, and noisier. `settings`: RTX settings over the scene's own.
    `sun` (elevation, azimuth: degrees up, and clockwise from north): lit by darkroom's own sun
    from there and a soft sky, the scene's lights off, so any scene is lit the same.
    `renderer_cfg`: the renderer's settings (`OVRTXRendererCfg`); none, its own."""
    for shot in shots:
        if shot.camera.projection == "orthographic" and not shot.camera.width:
            raise ValueError(f"{shot.png}: an orthographic camera needs a width.")
    from darkroom.renderers.launchers import OvrtxLauncher

    made = OvrtxLauncher().call(
        "darkroom.ovrtx.stills:stills",
        str(Path(stage_file).resolve()),
        [{"camera": asdict(s.camera), "png": str(Path(s.png).resolve())} for s in shots],
        settle=list(settle),
        subframes=subframes,
        render_settings=dict(settings or {}),
        sun=list(sun) if sun is not None else None,
        on_progress=on_progress,
        on_start=on_start,
        on_line=None,
    )
    return list(made["seconds"])
