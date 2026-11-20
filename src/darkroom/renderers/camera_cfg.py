"""What a renderer is told of a camera, and where its pictures land: the fields of Isaac Lab's
`CameraCfg` and `CameraData` (`isaaclab/sensors/camera/`) that a renderer reads and writes,
under the same names. Loads no USD.

A capture's cameras are prims on its stage: their lens, exposure and rolling shutter are read
from the prim, not from here. `CameraCfg` says only what Isaac Lab's does of a camera that
exists already (`spawn=None`): where it is, its picture's size, how often, which outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .output_contract import RenderBufferKind, RenderBufferSpec


@dataclass(frozen=True)
class CameraCfg:
    """A camera on the stage, as a renderer needs it said."""

    prim_path: str
    """The camera's prim."""

    width: int
    """Width of the image in pixels."""

    height: int
    """Height of the image in pixels."""

    update_period: float = 0.0
    """Seconds between its frames (1 / its rate); 0, a frame every step."""

    data_types: tuple[str, ...] = ("rgb",)
    """The outputs wanted, by `RenderBufferKind`'s names."""

    render_settings: dict[str, tuple[str, Any]] = field(default_factory=dict)
    """RTX settings for this camera's render product: attribute name to (USD type name, value),
    as ``{"omni:rtx:post:tonemap:op": ("Token", "aces")}``."""

    background_color: tuple[float, float, float] | None = None
    """What shows where nothing is; None, the stage's dome light."""

    output_dir: str | None = None
    """darkroom's own, not Isaac Lab's: the folder a renderer that writes its frames itself
    writes them into; it says each file's name as it publishes it (`CameraData.info`)."""


class CameraData:
    """Where a camera's renders land: `output`, an array a kind of output (`allocate`), and
    `info`, what the renderer says of each as it publishes it.

    Arrays are numpy's, shaped (instances, height, width, channels) as Isaac Lab's are; "rgb"
    is the first three channels of "rgba", a view of the same memory.
    """

    def __init__(self) -> None:
        self.image_shape: tuple[int, int] | None = None
        """A tuple containing (height, width) of the camera sensor."""
        self.output: dict[str, np.ndarray] = {}
        """Each output's array, by its name."""
        self.info: dict[str, Any] = {}
        """What the renderer said of each output's last publication, by its name: None when
        it published nothing new; else a dict. Under ``"capture"`` (Isaac Lab's key for what
        a picture was taken with), darkroom's renderers list the pictures published since the
        last read, oldest first: each ``{"time": seconds of the simulation's clock it shows}``,
        its pixels in `output` (the last one's), or with ``"file"``, the name of the file in
        `CameraCfg.output_dir` the renderer wrote it as. A segmentation's info has
        ``"labels"`` (Isaac Sim's: the labels its pixels index) or ``"idToLabels"`` (ovrtx's:
        each ID's labels)."""
        self.pos_w: np.ndarray | None = None
        """Position of the camera in the world, (instances, 3), metres."""
        self.quat_w_world: np.ndarray | None = None
        """Its orientation, (instances, 4), a quaternion (x, y, z, w)."""
        self.intrinsic_matrices: np.ndarray | None = None
        """Its intrinsic matrix, (instances, 3, 3), pixels."""

    def allocate(
        self,
        data_types: tuple[str, ...] | list[str],
        height: int,
        width: int,
        num_instances: int,
        specs: dict[RenderBufferKind, RenderBufferSpec],
    ) -> None:
        """Make an array for each of `data_types` as `specs` lays it out (a renderer's
        `supported_output_types`). Raises ValueError for one the renderer does not make."""
        unsupported = [name for name in data_types if name not in {str(k) for k in specs}]
        if unsupported:
            raise ValueError(
                f"the renderer makes no {', '.join(unsupported)}: "
                f"it makes {', '.join(sorted(str(k) for k in specs))}."
            )
        self.image_shape = (height, width)
        self.output, self.info = {}, {}
        wanted = list(data_types)
        if "rgb" in wanted and "rgba" not in wanted:
            wanted.append("rgba")  # rgb is a view of it
        for name in wanted:
            if name == "rgb":
                continue
            spec = specs[RenderBufferKind(name)]
            self.output[name] = np.zeros((num_instances, height, width, spec.channels), spec.dtype)
        if "rgb" in wanted:
            self.output["rgb"] = self.output["rgba"][..., :3]
        self.info = dict.fromkeys(self.output)
