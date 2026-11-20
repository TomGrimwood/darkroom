# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers.

Isaac Lab's `RendererCfg` (`isaaclab/renderers/renderer_cfg.py`) and the `BackendCfg` it rests
on (`isaaclab/sim/simulation_cfg.py`), as frozen dataclasses (darkroom has no `configclass`):
the same fields, names and defaults, so a capture picks its renderer as an Isaac Lab camera
does, by the `renderer_cfg` it is given.

    Options(renderer_cfg=OVRTXRendererCfg(step_rate=240))   # left out, ovrtx's as it is

Each renderer's own cfg is beside its implementation, as Isaac Lab's are in their backends'
packages: `darkroom.ovrtx.ovrtx_renderer_cfg` ("ovrtx"). A cfg names its renderer's class by where it is (`class_type`, "module:Class"),
imported only in the process that renders, and says what it can make
(`supported_output_types`) without it. A cfg crosses to that process as plain data (`plain`,
`read`).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from .output_contract import RenderBufferKind, RenderBufferSpec

if TYPE_CHECKING:
    from .base_renderer import BaseRenderer


@dataclass(frozen=True)
class BackendCfg:
    """Declarative settings and value identity for a resource a simulation owns (a renderer,
    ovrtx's native engine): made once for equal cfgs (`SimulationContext.get_or_create_backend`)."""

    class_type: Callable[..., Any] | str | None = None
    """Constructor of the resource, or where it is ("module:name"); what it makes must
    implement ``close()``. Isaac Lab's is ``MISSING`` until a concrete cfg sets it."""


@dataclass(frozen=True)
class RendererCfg(BackendCfg):
    """Configuration for a renderer."""

    class_type: type[BaseRenderer] | str | None = None
    """Renderer implementation class. Concrete configs must set this field."""

    renderer_type: str = "default"
    """Which renderer: "default" (a capture's own default, ovrtx's), or a renderer's own
    ("isaac_rtx", "ovrtx")."""

    launcher_type: ClassVar[str | None] = None
    """The launcher that starts the runtime this renderer needs, as ``"module:Class"``, or None if none is needed."""

    cloning_contexts: tuple[type | str, ...] = ()
    """Clone contexts that build this renderer's scene representation from the asset plan.
    Isaac Lab's, for its copies of an environment: a capture has one stage, so none."""

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec] | None:
        """Return the camera output layouts supported by this renderer configuration.

        Concrete renderer configurations override this method when their output contract can be
        determined without importing or instantiating the renderer implementation. Returning
        ``None`` defers compatibility validation until the renderer is created.

        Returns:
            Mapping from supported output types to their buffer layouts, or ``None`` when the
            renderer is selected dynamically.
        """
        return None


def string_to_callable(name: str) -> Any:
    """What ``"module:name"`` names, its module imported now (Isaac Lab's helper of the same
    name): how a cfg's `class_type` and `launcher_type` are found only when they are needed,
    so that naming a renderer loads no ovrtx."""
    module, _, attribute = name.partition(":")
    if not attribute:
        raise ValueError(f"{name!r} is not 'module:name'.")
    return getattr(importlib.import_module(module), attribute)


def instantiate(cfg: BackendCfg, **kwargs: Any) -> Any:
    """What `cfg.class_type` makes of `cfg` (and `kwargs`: a renderer's `sim`)."""
    made = cfg.class_type
    if made is None:
        raise ValueError(f"{type(cfg).__name__} names no class_type.")
    return (string_to_callable(made) if isinstance(made, str) else made)(cfg, **kwargs)


def plain(cfg: BackendCfg) -> dict[str, Any]:
    """`cfg` as plain data, ready for JSON: its fields, and "cfg_type" saying which cfg class
    it is ("module:Class"); a `class_type` given as a class is said as where it is."""

    def said(value: Any) -> Any:
        if isinstance(value, type):
            return f"{value.__module__}:{value.__qualname__}"
        if is_dataclass(value) and not isinstance(value, type):
            return {f.name: said(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, Mapping):
            return {str(k): said(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [said(v) for v in value]
        return value

    return {
        "cfg_type": f"{type(cfg).__module__}:{type(cfg).__qualname__}",
        **{f.name: said(getattr(cfg, f.name)) for f in fields(cfg)},
    }


def read(data: Mapping[str, Any]) -> RendererCfg:
    """The cfg in plain data (`plain`, through JSON). Raises for a field its class does not
    have: the two sides out of step."""
    values = dict(data)
    kind = string_to_callable(str(values.pop("cfg_type")))
    if not (isinstance(kind, type) and issubclass(kind, RendererCfg)):
        raise ValueError(f"{kind!r} is not a renderer's cfg.")
    unknown = sorted(set(values) - {f.name for f in fields(kind)})
    if unknown:
        raise ValueError(f"{kind.__name__} has no {', '.join(unknown)}.")
    made = kind()
    for name, value in values.items():
        default = getattr(made, name)
        if is_dataclass(default) and not isinstance(default, type) and isinstance(value, Mapping):
            value = type(default)(**value)  # a cfg within a cfg
        elif isinstance(default, tuple) and isinstance(value, list):
            value = tuple(value)
        values[name] = value
    return kind(**values)
