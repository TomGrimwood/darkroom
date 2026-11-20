"""How the cameras' pictures are rendered: render presets, put over the stage's own settings.

A preset is a USD layer as Isaac Sim saves and loads one (its Render Settings window, its
viewport's Draft/Medium/Final): `<name>.settings.usda`, its RTX settings in the layer's custom
data (`renderSettings`, keys like `rtx:post:aa:op`), what it is for in its `doc`, and the camera
frames played before the window and not kept in `darkroom:warmup` (a path tracer's first are
black; a camera's auto exposure takes a few to settle). darkroom's own are in `presets/`:

- `real_time`: Isaac Sim's own lighting, no flat fill, no haze (real-time haze darkens the sky
  down to the horizon away from the sun);
- `path_traced`, `path_traced_draft`: 64 or 16 samples a pixel, denoised, a firefly filter (a
  bright sample found by chance is otherwise smeared by the denoiser into a coloured blotch).

Every preset turns RTX's render-wide auto exposure off: auto exposure is a camera's own (RTX's
`OmniRtxCameraAutoExposureAPI` on it), off unless it says so.

    LOOKS["path_traced"].about                       # what it is for, a sentence
    resolve("path_traced")                           # -> (Kit settings, warmup)
    resolve("~/my_preset.settings.usda")             # any preset file, Isaac Sim's own included
    resolve(["path_traced", {"/rtx/pathtracing/spp": 128}])   # presets and settings, later wins

A stage keeps its own settings (its haze, fill light, exposure) in its layers in the same way; a
look changes the cameras' pass only, never the lidars' or the skies'. Read with USD when first
used: importing this module loads none (Kit has to start before `pxr` loads).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .options import Look

#: darkroom's own presets, `<name>.settings.usda` each.
PRESETS = Path(__file__).with_name("presets")
SUFFIX = ".settings.usda"
#: The key in a preset's custom data for the camera frames played first and not kept.
WARMUP = "darkroom:warmup"
#: The look a capture has when it is given none.
DEFAULT = "real_time"


@dataclass(frozen=True)
class Preset:
    about: str
    #: Kit settings, by path ("/rtx/pathtracing/spp").
    settings: dict[str, Any] = field(default_factory=dict)
    #: Camera frames played before the window, not kept.
    warmup: int = 1


def read(file: str | Path) -> Preset:
    """The look of preset `file` (an Isaac Sim `.settings.usd(a)`): its RTX settings as Kit
    paths, its `doc`, its warm-up (one frame when it does not say)."""
    from pxr import Gf, Sdf

    layer = Sdf.Layer.FindOrOpen(str(Path(file).expanduser()))
    if layer is None:
        raise ValueError(f"no render preset at {file}.")
    data = layer.customLayerData

    def plain(value: Any) -> Any:  # a colour as Kit keeps it: a list
        return list(value) if isinstance(value, Gf.Vec3f | Gf.Vec3d) else value

    settings = {
        "/" + key.replace(":", "/"): plain(value)
        for key, value in (data.get("renderSettings") or {}).items()
    }
    return Preset(layer.documentation, settings, int(data.get(WARMUP, 1)))


class _Presets(Mapping[str, Preset]):
    """darkroom's presets by name, read from `PRESETS` when first asked for."""

    def __init__(self) -> None:
        self._read: dict[str, Preset] | None = None

    def _all(self) -> dict[str, Preset]:
        if self._read is None:
            files = sorted(PRESETS.glob(f"*{SUFFIX}"))
            found = {f.name[: -len(SUFFIX)]: read(f) for f in files}
            self._read = {DEFAULT: found.pop(DEFAULT), **found}  # the default first
        return self._read

    def __getitem__(self, name: str) -> Preset:
        return self._all()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._all())

    def __len__(self) -> int:
        return len(self._all())


#: darkroom's presets by name, the default first: "real_time", "path_traced", "path_traced_draft".
LOOKS: Mapping[str, Preset] = _Presets()


def resolve(
    look: Look,
) -> tuple[dict[str, Any], int]:
    """The Kit settings and warm-up frames of a look: a preset's name, a preset file, settings
    of your own, or a list of them (later ones win; the most warm-up asked for); none, the
    `DEFAULT` preset. Raises ValueError for an unknown name."""
    parts = look if isinstance(look, list) else [DEFAULT] if look is None else [look]
    settings: dict[str, Any] = {}
    warmup = 1
    for part in parts:
        if isinstance(part, dict):
            settings.update(part)
            continue
        if isinstance(part, Path) or str(part).endswith((".usd", ".usda", ".usdc")):
            found = read(part)
        elif part in LOOKS:
            found = LOOKS[str(part)]
        else:
            raise ValueError(f"no render look {part!r}: there are {', '.join(LOOKS)}.")
        settings.update(found.settings)
        warmup = max(warmup, found.warmup)
    return settings, warmup


def stage_settings(stage: Any) -> dict[str, Any]:
    """The RTX settings the stage's layers keep for themselves (Kit's `renderSettings` in a layer's
    custom data: `{"rtx:post:histogram:enabled": False, ...}`), strongest layer first. Kit applies
    only its root layer's when it opens a stage, and a run's root layer is the run's own."""
    found: dict[str, Any] = {}
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        for key, value in (layer.customLayerData.get("renderSettings") or {}).items():
            name = "/" + key.replace(":", "/").lstrip("/")
            if name not in found:
                found[name] = (
                    list(value)
                    if hasattr(value, "__len__") and not isinstance(value, str)
                    else value
                )
    return found
