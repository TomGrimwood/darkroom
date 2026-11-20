# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Renderer→consumer output contract types.

Copied from Isaac Lab (`isaaclab/renderers/output_contract.py`), so that a renderer written
for one publishes its outputs under the names the other asks for. One change: a buffer's
`dtype` is numpy's here (darkroom has no Warp), where Isaac Lab's is Warp's.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RenderBufferKind(StrEnum):
    """Canonical names for the per-pixel render buffer kinds a renderer can publish.

    String values match the vocabulary used by
    `CameraCfg.data_types`.
    """

    RGB = "rgb"
    RGBA = "rgba"
    RGB_HDR = "rgb_hdr"
    ALBEDO = "albedo"
    DEPTH = "depth"
    DISTANCE_TO_IMAGE_PLANE = "distance_to_image_plane"
    DISTANCE_TO_CAMERA = "distance_to_camera"
    NORMALS = "normals"
    MOTION_VECTORS = "motion_vectors"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    INSTANCE_ID_SEGMENTATION_FAST = "instance_id_segmentation_fast"
    SIMPLE_SHADING_CONSTANT_DIFFUSE = "simple_shading_constant_diffuse"
    SIMPLE_SHADING_DIFFUSE_MDL = "simple_shading_diffuse_mdl"
    SIMPLE_SHADING_FULL_MDL = "simple_shading_full_mdl"


@dataclass(frozen=True)
class RenderBufferSpec:
    """Per-pixel layout (channels + dtype) for one render buffer kind."""

    channels: int
    """Number of per-pixel channels (last dimension of the allocated array)."""

    dtype: type
    """numpy scalar dtype for the buffer (e.g. ``np.float32``, ``np.uint8``)."""
