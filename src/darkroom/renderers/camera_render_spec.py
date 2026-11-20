# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immutable description of a camera passed to render backends.

Isaac Lab's `CameraRenderSpec` (`isaaclab/renderers/camera_render_spec.py`), field for field.
Its `cfg` is darkroom's `CameraCfg` (`camera_cfg`): the fields of Isaac Lab's a renderer reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from .camera_cfg import CameraCfg


@dataclass(frozen=True)
class CameraRenderSpec:
    """Stable inputs for :meth:`~darkroom.renderers.base_renderer.BaseRenderer.create_render_data`.

    Args:
        cfg: Camera configuration (data types, resolution, rate).
        device: Device string (e.g. ``"cuda:0"``) the renderer renders on.
        num_instances: Number of tiled camera instances (environments). A capture has one
            stage and no copies of it: 1.
        camera_prim_paths: Absolute paths of the authored camera prims: one, the camera's.
        view_count: Number of logical camera instances in the sensor view: 1.
    """

    cfg: CameraCfg
    device: str
    num_instances: int
    camera_prim_paths: tuple[str, ...]
    view_count: int
