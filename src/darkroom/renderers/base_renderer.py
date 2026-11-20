# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract base class for renderer implementations.

Isaac Lab's `BaseRenderer` (`isaaclab/renderers/base_renderer.py`): the same methods, in the
same order of use, with the same parameters, so that darkroom's renderers and Isaac Lab's are
driven alike (`tests/test_renderers_isaaclab.py` holds the two together). What differs is what
the arguments are made of: arrays are numpy's where Isaac Lab's are Warp's `ProxyArray`,
`sim` is darkroom's `SimulationContext` (a stage played from USD time, not a physics
simulation), `camera_data` its `CameraData`.

The order a camera is rendered in, as Isaac Lab's camera does it:

    renderer = sim.get_or_create_backend(renderer_cfg)       # one for equal cfgs
    renderer.initialize()
    renderer.prepare_stage(stage, num_envs=1)
    data = renderer.create_render_data(CameraRenderSpec(...))
    renderer.set_outputs(data, camera_data.output)
    each step:  sim.step()
                renderer.update_transforms(); renderer.update_geometries()
                renderer.update_camera(data, positions, orientations, intrinsics)
                renderer.render_batch([data, ...])
                renderer.read_output(data, camera_data)        # camera_data.info says what came
    renderer.cleanup(data); renderer.close()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .camera_render_spec import CameraRenderSpec
from .output_contract import RenderBufferKind, RenderBufferSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np

    from .camera_cfg import CameraData
    from .renderer_cfg import RendererCfg
    from .simulation_context import SimulationContext


@dataclass(frozen=True)
class VisualMaterialBatch:
    """One flat material-channel buffer and its aligned backend addresses.

    Isaac Lab's, for randomising materials across its environments; a capture renders its
    stage's materials as they are, so no darkroom renderer takes one."""

    channel: str
    material_paths: tuple[str, ...]
    shader_paths: tuple[str, ...]
    input_names: tuple[str, ...]
    values: Any


class BaseRenderer(ABC):
    """Abstract base class for renderer implementations."""

    def __init__(self, cfg: RendererCfg, *, sim: SimulationContext) -> None:
        """Bind this renderer to its owning simulation.

        Args:
            cfg: Renderer configuration.
            sim: Simulation that creates and owns this renderer through its backend registry.
        """
        self.cfg = cfg
        self._sim = sim

    def initialize(self) -> None:
        """Post-physics one-time initialization hook. Called only once."""
        return

    @property
    def visual_material_writer(self) -> Callable[[tuple[VisualMaterialBatch, ...]], Any] | None:
        """Return the backend's shared material-writer factory, if supported.

        None for every darkroom renderer: materials are the stage's own."""
        return None

    @abstractmethod
    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Per-output layout (channels + dtype) this renderer can produce.

        Outputs absent from the mapping are not produced by this backend.

        Returns:
            Mapping from supported :class:`RenderBufferKind` to its :class:`RenderBufferSpec`.
        """
        pass

    @abstractmethod
    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """Prepare the stage for rendering before :meth:`create_render_data` is called.

        Some renderers need to export or preprocess the USD stage before
        creating render data. This method is called after the renderer is
        instantiated and before :meth:`create_render_data`.

        Args:
            stage: USD stage to prepare, or None if not applicable.
            num_envs: Number of environments. A capture has one stage: 1.
        """
        pass

    @abstractmethod
    def create_render_data(self, spec: CameraRenderSpec) -> Any:
        """Create render data for the given camera :class:`CameraRenderSpec`.

        Args:
            spec: Immutable description of the camera (paths, config, device).

        Returns:
            Renderer-specific data for subsequent :meth:`render` / :meth:`read_output` calls.
        """
        pass

    @abstractmethod
    def set_outputs(self, render_data: Any, output_data: dict[str, np.ndarray]) -> None:
        """Store reference to output buffers for writing during render.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            output_data: Dictionary mapping output names (e.g. ``"rgb"``, ``"depth"``)
                to pre-allocated arrays where rendered data will be written
                (`CameraData.output`).
        """
        pass

    def prepare_capture(  # noqa: B027 - a hook, as Isaac Lab's: empty unless overridden
        self, render_data: Any, camera_data: CameraData, frame: np.ndarray
    ) -> None:
        """Snapshot metadata for the next capture when its image will be delivered asynchronously.

        Synchronous renderers need no snapshot: darkroom's are, and leave this as it is.

        Args:
            render_data: Renderer-owned camera resources.
            camera_data: Current camera pose and calibration.
            frame: Current per-environment capture indices, shape (N,), dtype ``int64``.
        """

    @abstractmethod
    def update_transforms(self) -> None:
        """Update scene transforms before rendering.

        Called to sync where things are (the stage at the simulation's time) into the
        renderer's scene representation.
        """
        pass

    @abstractmethod
    def update_geometries(self) -> None:
        """Update mutable geometry attributes before rendering.

        Called to sync geometry that changes shape (mesh points of what is skinned or
        deformed) into the renderer's scene representation.
        """
        pass

    @abstractmethod
    def update_camera(
        self,
        render_data: Any,
        positions: np.ndarray,
        orientations: np.ndarray,
        intrinsics: np.ndarray,
    ) -> None:
        """Update camera poses and supply initial calibration for the next render.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            positions: Camera positions in world frame, shape ``(N, 3)``.
            orientations: Camera orientations as quaternions ``(x, y, z, w)``, shape ``(N, 4)``.
            intrinsics: Camera intrinsic matrices, shape ``(N, 3, 3)``.
        """
        pass

    def update_camera_intrinsics(
        self, render_data: Any, intrinsics: np.ndarray, parameters: np.ndarray
    ) -> None:
        """Apply a proposed runtime calibration without accessing the authored USD stage.

        A capture's cameras keep the lens their prim has: no darkroom renderer changes one.

        Args:
            render_data: The camera's renderer-owned state.
            intrinsics: Complete proposed calibration, shape (N, 3, 3).
            parameters: Complete physical projection parameters, shape (5, N): focal length,
                horizontal/vertical aperture, and horizontal/vertical aperture offsets.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support runtime camera calibration."
        )

    @abstractmethod
    def render(self, render_data: Any) -> None:
        """Submit a capture; :meth:`read_output` publishes its camera's available observation.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
        """
        pass

    def render_batch(self, render_data: Sequence[Any]) -> None:
        """Submit captures for a collection of cameras.

        All camera poses and shared scene state must be prepared before calling this method.
        An empty sequence is a no-op. Each object must belong to this renderer and appear once.
        The default implementation calls :meth:`render` for each camera; subclasses may override
        this method to submit all cameras together.

        Args:
            render_data: Renderer-specific objects from :meth:`create_render_data`.
        """
        for data in render_data:
            self.render(data)

    @abstractmethod
    def read_output(self, render_data: Any, camera_data: CameraData) -> None:
        """Read rendered outputs from the renderer into the camera data container.

        Publishing one camera must not change another camera's observations.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            camera_data: The `CameraData` instance to populate: its `info` says, for each
                output, what was published (None: nothing new this time).
        """
        pass

    @abstractmethod
    def cleanup(self, render_data: Any) -> None:
        """Release renderer resources associated with the given render data.

        Args:
            render_data: The render data object to clean up, or ``None``.
        """
        pass

    def reset(  # noqa: B027 - a hook, as Isaac Lab's: empty unless overridden
        self, render_data: Any, env_ids: Sequence[int] | None = None
    ) -> None:
        """Reset the renderer-owned state of a camera when its environments reset.

        A renderer implementation drops whatever per-camera state must not survive a reset
        (accumulated temporal render history, say). The default does nothing.

        Args:
            render_data: The render data object from :meth:`create_render_data`.
            env_ids: Environments being reset, or ``None`` for all. A capture has one.
        """

    def close(self) -> None:
        """Release resources owned by the renderer itself rather than by a render data.

        `SimulationContext.clear_instance` calls this once at teardown. The default
        implementation is a no-op. Implementations must be idempotent.
        """
        return
