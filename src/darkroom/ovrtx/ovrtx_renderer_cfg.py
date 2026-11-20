# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for OVRTX Renderer: NVIDIA's ovrtx.

Isaac Lab's `OVRTXRendererCfg` and `OVRTXBackendCfg`
(`isaaclab_ov/renderers/ovrtx_renderer_cfg.py`), field for field.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from darkroom.renderers.output_contract import RenderBufferKind, RenderBufferSpec
from darkroom.renderers.renderer_cfg import BackendCfg, RendererCfg

if TYPE_CHECKING:
    from .ovrtx_renderer import OVRTXBackend, OVRTXRenderer


@dataclass(frozen=True)
class OVRTXBackendCfg(BackendCfg):
    """Native engine settings and the configuration identifying its scene resource."""

    class_type: type[OVRTXBackend] | str = "darkroom.ovrtx.ovrtx_renderer:OVRTXBackend"
    scene_key: OVRTXRendererCfg | None = None
    """None shares the simulation scene; standalone renderers retain their own scene policy."""
    log_file_path: str = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")
    """Native renderer log destination."""
    log_level: str = "error"
    """Native renderer log level."""
    use_ovstage: bool = True
    """Whether the engine borrows an OVStage instead of using its internal stage. darkroom
    renders through one (it plays the stage's USD time): only True."""
    read_gpu_transforms: bool = False
    """Whether OVRTX reads its GPU transform cache, resolved before native construction."""
    device: str = "cuda:0"
    """Normalized simulation device: ``cuda:<index>``."""
    motion_bvh: bool = True
    """darkroom's own, not Isaac Lab's: RTX's motion BVH on, which a rolling shutter needs."""

    cache_root_path: str | None = None
    """darkroom's own, not Isaac Lab's: where OVRTX keeps its caches (compiled shaders: a
    stage's first render with none took 85 s for 6); None, where it keeps them itself (under
    `$XDG_CACHE_HOME`, or `~/.cache`). The first engine made in a process decides."""


@dataclass(frozen=True)
class OVRTXRendererCfg(RendererCfg):
    """Configuration for OVRTX Renderer.

    The OVRTX renderer uses the ovrtx library for high-fidelity RTX-based rendering.
    width, height and data_types are obtained from the `CameraRenderSpec` when
    `BaseRenderer.create_render_data` is called (same pattern as Isaac RTX).
    """

    class_type: type[OVRTXRenderer] | str = "darkroom.ovrtx.ovrtx_renderer:OVRTXRenderer"
    """Renderer implementation class."""

    renderer_type: str = "ovrtx"
    """Type identifier for OVRTX renderer."""

    launcher_type: ClassVar[str | None] = "darkroom.renderers.launchers:OvrtxLauncher"
    """The launcher that starts the process ovrtx renders in."""

    cloning_contexts: tuple[type | str, ...] = ()
    """Isaac Lab's asks for native OVRTX copies of its environments; a capture has none."""

    async_rendering: bool = False
    """Trade one capture of camera latency for pipelined rendering. A capture names each
    frame by the time it shows: only False (synchronous) is rendered."""

    step_rate: float | None = None
    """darkroom's own, not Isaac Lab's: steps of ovrtx's clock a second while a capture's
    cameras render, at least; None, the stage's own frames (its time codes a second).

    ovrtx traces a rolling shutter's rows from where things are in the step they fall in,
    turning each thing between the step's two poses the short way: a thing has to turn well
    under half a turn a step. A propeller at 40 turns a second read over 100 ms is right at
    960 steps a second (15 degrees a step); at a stage's 60 it turns 240 degrees a step, and
    comes out wrong. A step with no camera due costs about 3 ms. A camera with RTX's motion
    blur is refused more than `darkroom.ovrtx.BLUR_STEPS` steps between its frames."""

    motion_bvh: bool | None = None
    """darkroom's own, not Isaac Lab's: RTX's motion BVH (things traced where they were at
    each ray's own time). None, on only for what needs it: lidars, and cameras when one has a
    rolling shutter. ovrtx 0.7.0 path traces a camera 2.4 times slower with it on (a 1280x960
    frame 27 ms for 7 ms on a street: it turns shader execution reordering off)."""

    cache_root_path: str | None = None
    """darkroom's own, not Isaac Lab's: where OVRTX keeps its caches (its compiled shaders);
    None, where it keeps them itself (`OVRTXBackendCfg.cache_root_path`)."""

    temp_usd_dir: str | None = None
    """Directory for temporary USD debug dumps written during OVRTX stage preparation.

    When set, the renderer writes ``ovrtx_renderer_stage.usda`` (the root layer ovrtx opens,
    before any camera is on) and, as each camera is put on, its render product's and its
    eye's layers (``RenderCamera_<n>.usda``, ``..._eye.usda``) under this directory.
    """

    log_level: str = "error"
    """OVRTX carb log level: "verbose", "info", "warn", "error"."""

    log_file_path: str = os.path.join(tempfile.gettempdir(), "ovrtx_renderer.log")
    """Path for OVRTX log file. Defaults to ``<system temp>/ovrtx_renderer.log``."""

    enable_shadows: bool = False
    """Whether lights cast shadows in RTX Minimal mode. Only the ``simple_shading_*`` outputs
    are in that mode; OVRTX's path-traced modes always cast shadows, so this does not change
    ``rgb``."""

    colorize_semantic_segmentation: bool = False
    """Whether to colorize semantic segmentation output. A capture keeps label indices: only
    False."""

    colorize_instance_segmentation: bool = False
    """Whether to colorize instance segmentation output. Not an output of a capture."""

    colorize_instance_id_segmentation: bool = False
    """Whether to colorize instance ID segmentation output. Not an output of a capture."""

    def __post_init__(self) -> None:
        """Refused as made: what a capture cannot render as asked."""
        if self.async_rendering:
            raise ValueError(
                "a capture renders synchronously (each frame named by its time): "
                "async_rendering is False."
            )
        if self.colorize_semantic_segmentation:
            raise ValueError("a capture's segments are label indices: they are not coloured.")

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Return the per-output layouts a capture's OVRTX cameras make: their pictures, and
        where the stage asks, each pixel's semantic class, world normal and distance (metres).
        Isaac Lab's OVRTX renderer also makes depths, albedo, motion vectors and instance
        segmentations; a capture keeps none of those."""
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, np.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, np.uint8),
            RenderBufferKind.DISTANCE_TO_CAMERA: RenderBufferSpec(1, np.float32),
            RenderBufferKind.NORMALS: RenderBufferSpec(3, np.float32),
            RenderBufferKind.SEMANTIC_SEGMENTATION: RenderBufferSpec(1, np.int32),
        }
