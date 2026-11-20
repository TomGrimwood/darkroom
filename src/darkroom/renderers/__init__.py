"""The renderer: NVIDIA's ovrtx (`darkroom.ovrtx`), in a process of its own, its settings an
`OVRTXRendererCfg` (`Options.renderer_cfg`; none, the default's).

    from darkroom.renderers import OVRTXRendererCfg
    darkroom.capture("run.usda", "out/run", Options(renderer_cfg=OVRTXRendererCfg(step_rate=240)))

The design is Isaac Lab's (`isaaclab.renderers`, and its backends' `isaaclab_physx.renderers`,
`isaaclab_ov.renderers`, `isaaclab_newton.renderers`): a `RendererCfg` names its renderer's
class by where it is and says what it can make without importing it; every renderer is a
`BaseRenderer`; and what renders the cameras (`loop`) talks to that alone, so picking a
renderer is picking a cfg. Copied rather than imported: darkroom depends on neither Isaac
Lab, torch nor Warp. `tests/test_renderers_isaaclab.py` reads Isaac Lab's source beside this
and fails when the two drift apart.

Importing this loads no USD and no ovrtx.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from .base_renderer import BaseRenderer, VisualMaterialBatch
from .camera_cfg import CameraCfg, CameraData
from .camera_render_spec import CameraRenderSpec
from .output_contract import RenderBufferKind, RenderBufferSpec
from .renderer_cfg import BackendCfg, RendererCfg
from .simulation_context import SimulationContext

#: Each renderer's cfg, by name, and the module it is in (beside its renderer): loaded when
#: first asked for.
CFGS = {
    "OVRTXRendererCfg": "darkroom.ovrtx.ovrtx_renderer_cfg",
    "OVRTXBackendCfg": "darkroom.ovrtx.ovrtx_renderer_cfg",
}
#: The renderers there are, by `renderer_type`: where each one's cfg is.
RENDERERS = {"ovrtx": "OVRTXRendererCfg"}

__all__ = [
    "BackendCfg",
    "BaseRenderer",
    "CameraCfg",
    "CameraData",
    "CameraRenderSpec",
    "OVRTXBackendCfg",
    "OVRTXRendererCfg",
    "RenderBufferKind",
    "RenderBufferSpec",
    "RendererCfg",
    "SimulationContext",
    "VisualMaterialBatch",
    "get_default_renderer_cfg",
    "of",
    "resolve",
]


def __getattr__(name: str) -> Any:
    if name in CFGS:
        return getattr(importlib.import_module(CFGS[name]), name)
    raise AttributeError(f"module 'darkroom.renderers' has no attribute {name!r}")


def get_default_renderer_cfg() -> RendererCfg:
    """The renderer a capture has when it names none: ovrtx's, as it is."""
    return __getattr__("OVRTXRendererCfg")()


def of(said: Mapping[str, Any] | None) -> RendererCfg:
    """The cfg a render job carries as plain data (`renderer_cfg.plain`); none, the default's."""
    from .renderer_cfg import read

    return read(said) if said else get_default_renderer_cfg()


def resolve(cfg: RendererCfg | None) -> RendererCfg:
    """The cfg a capture renders with: `cfg`, or for none or a bare `RendererCfg()` (its
    `renderer_type` "default") the default renderer's, as Isaac Lab's camera resolves its own.
    Raises ValueError for a cfg that names no renderer's class."""
    if cfg is None or cfg.renderer_type == "default":
        return get_default_renderer_cfg()
    if cfg.class_type is None:
        raise ValueError(
            f"{type(cfg).__name__} ({cfg.renderer_type!r}) names no class_type to render "
            f"with: there are {', '.join(RENDERERS)}."
        )
    return cfg
