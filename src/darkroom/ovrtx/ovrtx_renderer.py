# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX Renderer implementation ("ovrtx"): a capture's cameras rendered by NVIDIA's ovrtx,
RTX in a process of its own, behind `BaseRenderer`.

The classes, their names and their methods are Isaac Lab's
(`isaaclab_ov/renderers/ovrtx_renderer.py`: `OVRTXBackend`, `OVRTXCameraRenderData`,
`OVRTXRenderer`); what is in them is darkroom's own. Isaac Lab's is told where every body is
by its physics, through Warp, and tiles its environments into one picture; a capture's stage
is keyed, so here ovstage plays it from USD time itself (`update_transforms`: transforms,
skinned characters and all), a camera is one picture, and arrays are numpy's.

- `prepare_stage`: the stage opened by ovstage (the root layer `ovrtx_usd` builds over the
  stage's own file), the engine attached to it.
- `create_render_data`: the camera's eye (`ovrtx_usd.sensor_eye`: scheduled by ovrtx at the
  camera's rate, with its rolling shutter) and a render product of its own referenced into the
  open stage, the capture's look on them as ovrtx takes settings (`settings`); `cleanup` takes
  both out again, the GPU memory they held given back.
- `render_batch`: one step of ovrtx's clock, every camera on stepped together (a product
  left out of a step loses its history); a camera due in the step has a frame, stamped with
  the time of ovrtx's clock it shows (a rolling shutter's: its last row's).
- `read_output`: each frame published as pixels in the camera's "rgba" array, with its time.
- `create_lidar_data`, `scan_batch`: a capture's lidars too (darkroom's own: Isaac Lab's
  renderers render cameras), each a copy of the lidar with a `PointCloud` render product,
  its scans read as they end (`ovrtx_lidar`).

Measured with ovrtx 0.6.0 and ovstage 0.3.0: a camera scheduled at 15 Hz stepped at 60 has a
frame in the step that ends on each of its periods (its `start_time` and `end_time` the
step's end), showing the stage at the USD time set before that step; an eye referenced in
under an animated parent moves with it.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# The ovrtx C library links to its own version of the USD libraries; with the pxr Python
# package in the same process, this keeps it from checking (and loading) that one.
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"

import ovrtx
import ovstage
from pxr import Usd

from darkroom import eyes, gpu, lidarscans, looks, process, sensorkit
from darkroom.renderers.base_renderer import BaseRenderer
from darkroom.renderers.output_contract import RenderBufferKind, RenderBufferSpec

from . import ovrtx_usd, settings
from .ovrtx_lidar import OVRTXLidarRenderData, Table
from .ovrtx_renderer_cfg import OVRTXBackendCfg, OVRTXRendererCfg
from .stage import OvstageBackend, OvstageBackendCfg, _handle_raw_id

if TYPE_CHECKING:
    from darkroom.renderers.camera_cfg import CameraData
    from darkroom.renderers.camera_render_spec import CameraRenderSpec
    from darkroom.renderers.simulation_context import SimulationContext

_LDR_COLOR_VAR = "LdrColor"
#: ovrtx's image outputs a capture's camera keeps, and the array each is copied into.
_IMAGES = {
    _LDR_COLOR_VAR: RenderBufferKind.RGBA,
    "NormalSD": RenderBufferKind.NORMALS,
    "DistanceToCameraSD": RenderBufferKind.DISTANCE_TO_CAMERA,
    "SemanticSegmentation": RenderBufferKind.SEMANTIC_SEGMENTATION,
}
#: What names each semantic class, and the tensors it comes as.
_SEMANTIC_ID_MAP_VAR = "SemanticIdMap"
_ID_MAP_TENSORS = ("id", "label_offsets", "label_data")
#: The share of the GPU's memory free under which a step that failed ran out of it.
FULL = 0.08
#: Steps the product that names a scene's objects is given to hand its table over.
NAMES_STEPS = 8


def _raise_failed(prim_path: str, frame: Any) -> None:
    """Raises RuntimeError, saying why, when ovrtx says it could not make one of a frame's
    outputs (each RenderVar's `status` and `error_message`: an output it failed to make is
    still listed, and mapping it fails)."""
    failed = [
        f"{name.rpartition('/')[2]}: {output.error_message}"
        for name, output in frame.render_vars.items()
        if output.status == ovrtx.EventStatus.FAILURE
    ]
    if failed:
        raise RuntimeError(f"{prim_path}: ovrtx could not make {'; '.join(failed)}")


def semantic_labels(
    ids: np.ndarray, label_offsets: np.ndarray, label_data: np.ndarray
) -> dict[str, dict[str, str]]:
    """ovrtx's `SemanticIdMap` (each class's four ID words, where each one's labels end in
    the bytes after, and those bytes) as Replicator's `idToLabels`: {"2": {"class": "tree"}}.
    A class's labels are written "class: tree; other: oak;"; a pixel's class is the first
    word of its ID."""
    data = np.asarray(label_data, np.uint8).reshape(-1)
    ends = np.asarray(label_offsets).reshape(-1)
    named: dict[str, dict[str, str]] = {}
    for row, words in enumerate(np.asarray(ids).reshape(-1, 4)):
        said = data[int(ends[row]) : int(ends[row + 1])].tobytes().decode("utf-8")
        labels: dict[str, str] = {}
        for part in said.rstrip("\x00").split(";"):
            kind, colon, label = part.partition(":")
            if colon and kind.strip():
                labels[kind.strip()] = label.strip()
        named[str(int(words[0]))] = labels
    # A pixel that hit nothing is class 0 and one that hit nothing labelled class 1, named
    # or not: as Replicator calls them.
    named.setdefault("0", {"class": "BACKGROUND"})
    named.setdefault("1", {"class": "UNLABELLED"})
    return named


def schema_files() -> list[Path]:
    """The schema files of the ovrtx installed (its USD plugins' `generatedSchema.usda`):
    what says which attribute stands for which Kit setting (`settings.read`)."""
    return sorted(
        (Path(ovrtx.__file__).parent / "bin" / "usd_plugins").glob("*/generatedSchema.usda")
    )


class OVRTXBackend:
    """Own one native renderer, independently of its camera products and its OVStage."""

    def __init__(self, cfg: OVRTXBackendCfg) -> None:
        self.cfg = cfg
        self.attached = False
        self.render_products: set[str] = set()
        # ovrtx sends NVIDIA a report as it first starts in a process, unless told not to
        # before then: a capture sends none (its launcher says so in the environment too).
        ovrtx.library_telemetry_opt_out()
        device = cfg.device.partition(":")[2]
        self.renderer: Any = ovrtx.Renderer(
            ovrtx.RendererConfig(
                cache_root_path=cfg.cache_root_path,
                active_cuda_gpus=device or None,
                log_file_path=cfg.log_file_path,
                log_level=cfg.log_level,
                read_gpu_transforms=cfg.read_gpu_transforms,
                motion_bvh=ovrtx.MotionBvh.ENABLE if cfg.motion_bvh else ovrtx.MotionBvh.DISABLE,
                # ovrtx's own crash reporter uploads crash dumps unless it is off: a capture's
                # process is watched by what started it, and uploads nothing.
                enable_crash_reporter=False,
            )
        )

    def attach(self, stage: Any) -> None:
        """The engine rendering `stage` (an `ovstage.Stage`) from here on."""
        if not self.attached:
            self.renderer.attach_ovstage(stage)
            self.attached = True

    def detach(self) -> None:
        """Release the borrowed stage before it is rebuilt or destroyed."""
        if self.attached and self.renderer is not None:
            self.renderer.detach_ovstage()
        self.attached = False

    def close(self) -> None:
        """Destroy the engine; the simulation's backend registry releases any borrowed stage."""
        if self.renderer is not None:
            self.detach()
            self.renderer.destroy()
            self.renderer = None


class OVRTXCameraRenderData:
    """Owns one camera sensor's native resources and borrowed output buffers."""

    def __init__(
        self, spec: CameraRenderSpec, render_scope_name: str, camera_paths: list[str]
    ) -> None:
        """Create render data for a camera in its assigned render scope.

        Args:
            spec: Camera render specification.
            render_scope_name: Root scope containing this camera's render product and RenderVars.
            camera_paths: The cameras that render: one, the camera's eye.
        """
        self.spec = spec
        self.render_scope_name = render_scope_name
        self.render_product_path = f"/{render_scope_name}/RenderProduct"
        self.camera_paths = camera_paths
        self.render_var_keys = {
            source: f"/{render_scope_name}/Vars/{name}"
            for source, name in ovrtx_usd.render_var_prim_names_by_source().items()
        }
        self.resources = contextlib.ExitStack()
        self.width = spec.cfg.width
        self.height = spec.cfg.height
        self.num_cols = self.num_rows = 1  # a capture's camera is one picture, not tiles
        self.buffers: dict[str, np.ndarray] = {}
        #: The frames rendered since the last read: `CameraData.info`'s "capture" entries.
        self.captures: list[dict[str, Any]] = []
        #: What each semantic class is called, as ovrtx last said (`semantic_labels`).
        self.id_to_labels: dict[str, dict[str, str]] = {}

    def cleanup(self) -> None:
        """Release this camera's native resources and buffers. Safe to call repeatedly."""
        try:
            self.resources.close()
        finally:
            self.captures = []
            self.buffers = {}


class OVRTXRenderer(BaseRenderer):
    """OVRTX Renderer implementation using the ovrtx library."""

    cfg: OVRTXRendererCfg

    def __init__(self, cfg: OVRTXRendererCfg, *, sim: SimulationContext) -> None:
        super().__init__(cfg, sim=sim)
        self._camera_render_data: list[OVRTXCameraRenderData | OVRTXLidarRenderData] = []
        self.scene: OvstageBackend | None = None
        self.backend: OVRTXBackend | None = None
        """Native engine borrowed from the simulation registry as the stage is prepared."""
        self._initialized_scene = False
        self._object_table: Table | None = None
        self._schemas = settings.Schemas()
        self._look = settings.Translated()
        self._spans: list[settings.Timed] = []

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec]:
        """Publish the per-output layout this OVRTX backend writes."""
        return self.cfg.supported_output_types()

    def prepare_stage(self, stage: Any, num_envs: int) -> None:
        """The stage opened by ovstage and the engine attached to it, before any camera is
        registered: each fisheye camera's lens as the job says (`darkroom.eyes.lenses`), the
        capture's look (the stage's own settings, the job's over them) as ovrtx's attributes.
        A capture has one stage: `num_envs` is 1."""
        if num_envs != 1:
            raise ValueError(f"a capture renders one stage, not {num_envs} environments.")
        job = self._sim.job
        look: dict[str, Any] = looks.stage_settings(stage)
        if job is not None:
            eyes.lenses(stage, {k: dict(v) for k, v in job.lenses.items()})
            look.update(job.look)
        self._schemas = settings.read(schema_files())
        self._look = settings.translate(look, self._schemas)
        # Each span's look, in USD time (spans count from the stage's start): its attributes
        # time-sampled on every camera's eye and product, which ovstage sets as it plays.
        self._spans = [
            (self._sim.origin + a, self._sim.origin + b, settings.translate(said, self._schemas))
            for a, b, said in (job.spans if job is not None else ())
        ]
        left = {*self._look.left, *(x for _, _, said in self._spans for x in said.left)}
        for setting in sorted(left):
            process.progress(f"cameras: ovrtx has no setting for {setting}, left out")
        backend_cfg = OVRTXBackendCfg(
            scene_key=self.cfg,
            log_file_path=self.cfg.log_file_path,
            log_level=self.cfg.log_level,
            device=self._sim.device,
            motion_bvh=self.cfg.motion_bvh is not False,
            cache_root_path=self.cfg.cache_root_path,
        )
        self.backend = self._sim.get_or_create_backend(backend_cfg)
        self.scene = self._sim.get_or_create_backend(OvstageBackendCfg(scene_key=backend_cfg))
        usd = ovrtx_usd.build_stage_as_string(stage)
        if self.cfg.temp_usd_dir is not None:
            folder = Path(self.cfg.temp_usd_dir)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "ovrtx_renderer_stage.usda").write_text(usd, encoding="utf-8")
        self.scene.populate(usd)
        if not self.backend.attached:
            self.backend.attach(self.scene.stage)
            # Closed with the stage, before it goes: the cameras, then the engine let go.
            self.scene.resources.callback(self.backend.detach)
        self.scene.resources.callback(self.close)
        self._initialized_scene = True

    def create_render_data(self, spec: CameraRenderSpec) -> OVRTXCameraRenderData:
        """Register a camera's eye, its product and its settings in the prepared scene."""
        if self.scene is None or self.backend is None or not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call prepare_stage() first.")
        scene, stage = self.scene, self._sim.stage
        prim = stage.GetPrimAtPath(spec.camera_prim_paths[0])
        rate = 1 / spec.cfg.update_period
        own = settings.split(ovrtx_usd.camera_settings(prim), self._schemas)
        eye = ovrtx_usd.sensor_eye(stage, prim, rate, spec.cfg.height)
        on_eye = {**self._look.camera, **own.camera}
        ovrtx_usd.author(eye, on_eye)
        per_second = stage.GetTimeCodesPerSecond()
        dt = self._sim.dt
        timed = settings.samples(self._spans, "camera", on_eye, self._schemas, per_second, dt)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            ovrtx_usd.author_samples(eye, timed)
        eye_path = str(eye.GetPath())
        eye_usd = ovrtx_usd.build_sensor_as_string(eye)
        render_data = OVRTXCameraRenderData(
            spec, f"RenderCamera_{scene.next_camera_id}", [eye_path]
        )
        wanted = {**self._look.product, **own.product, **spec.cfg.render_settings}
        usd = ovrtx_usd.build_render_product_as_string(
            replace(spec, cfg=replace(spec.cfg, render_settings=wanted)),
            render_data,
            device_id=int(self._sim.device.partition(":")[2] or 0),
            enable_shadows=self.cfg.enable_shadows,
        )
        usd = ovrtx_usd.with_samples(
            usd,
            render_data.render_product_path,
            settings.samples(self._spans, "product", wanted, self._schemas, per_second, dt),
            per_second,
        )
        if self.cfg.temp_usd_dir is not None:
            folder = Path(self.cfg.temp_usd_dir)
            (folder / f"{render_data.render_scope_name}.usda").write_text(usd, encoding="utf-8")
            (folder / f"{render_data.render_scope_name}_eye.usda").write_text(
                eye_usd, encoding="utf-8"
            )
        try:
            self._register(
                render_data.resources,
                render_data.render_product_path,
                {eye_path: eye_usd, f"/{render_data.render_scope_name}": usd},
                eye_path,
            )
            # Whole frames only: a rolling shutter's readout crosses steps, and its picture
            # while it is part read is the frame before's.
            scene.write(
                [eye_path],
                "omni:sensor:Core:partialOutputs",
                np.full(1, False, dtype=np.bool_),
                is_array=False,
            )
            scene.commit()
        except Exception:
            render_data.cleanup()
            raise
        scene.next_camera_id += 1
        self._camera_render_data.append(render_data)
        self.backend.render_products.add(render_data.render_product_path)
        render_data.resources.callback(
            self.backend.render_products.discard, render_data.render_product_path
        )
        return render_data

    def _register(
        self,
        resources: contextlib.ExitStack,
        render_product_path: str,
        layers: dict[str, str],
        sensor_path: str,
    ) -> None:
        """A sensor and its render product put on the open stage: each of `layers` (a layer's
        text, by the path it goes at) referenced in, taken out again as `resources` close,
        and the product's `camera` pointed at the sensor at `sensor_path`."""
        assert self.scene is not None
        scene = self.scene
        for at, layer in layers.items():
            reference = ovstage.population.add_usd_reference_from_string(scene.stage, layer, at)
            resources.callback(self._remove_usd_reference, reference)
        ovstage.population.apply_usd_changes(scene.stage, ordinal=scene.ordinal)
        # USD references drop external camera targets; author the relationship in Fabric.
        scene.write(
            [render_product_path],
            "camera",
            np.array([_handle_raw_id(scene.paths.intern_path(sensor_path))], dtype=np.uint64),
            is_array=True,
            semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
        )

    def create_lidar_data(
        self, prim_path: str, scans: lidarscans.ScanWriter, hits: bool
    ) -> OVRTXLidarRenderData:
        """Register a lidar's copy (`ovrtx_usd.sensor_lidar`) and its render product in the
        prepared scene: its scans are handed to `scans` as they come (`scan_batch`); with
        `hits`, each point's object too. Not Isaac Lab's: its renderers render cameras."""
        if self.scene is None or self.backend is None or not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call prepare_stage() first.")
        scene, stage = self.scene, self._sim.stage
        prim = stage.GetPrimAtPath(prim_path)
        lidar = ovrtx_usd.sensor_lidar(stage, prim, sensorkit.lidar.scan_rate(prim))
        lidar_path = str(lidar.GetPath())
        lidar_usd = ovrtx_usd.build_sensor_as_string(lidar)
        scope = f"RenderLidar_{scene.next_camera_id}"
        render_data = OVRTXLidarRenderData(prim_path, scope, scans, hits)
        usd = ovrtx_usd.build_lidar_product_as_string(scope, lidar_path, hits)
        try:
            self._register(
                render_data.resources,
                render_data.render_product_path,
                {lidar_path: lidar_usd, f"/{scope}": usd},
                lidar_path,
            )
            scene.commit()
        except Exception:
            render_data.cleanup()
            raise
        scene.next_camera_id += 1
        self._camera_render_data.append(render_data)
        self.backend.render_products.add(render_data.render_product_path)
        render_data.resources.callback(
            self.backend.render_products.discard, render_data.render_product_path
        )
        return render_data

    def scan_batch(self, render_data: Sequence[OVRTXLidarRenderData]) -> None:
        """One step of ovrtx's clock (`render_batch`'s, for lidars): each lidar's scan that
        ended in it, if one did, handed to its writer."""
        if not render_data:
            return
        result = self._step()
        try:
            came = dict(result.items())
            for data in render_data:
                product = came.get(data.render_product_path)
                for frame in product.frames if product is not None else ():
                    _raise_failed(data.prim_path, frame)
                    data.read(frame)
        finally:
            del result

    def object_table(self) -> Table:
        """What every object ID in the scene is called (`ovrtx_lidar.names_of` looks IDs up
        in it): ovrtx's `StableIdMap`, read once, from a render product of a few pixels put
        on the stage for it, which stays there (`ovrtx_usd.build_names_product_as_string`
        says why): asked for before the lidars whose points' objects it names are put on.
        ovrtx's clock stays where it is."""
        if self.scene is None or self.backend is None or not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call prepare_stage() first.")
        if self._object_table is not None:
            return self._object_table
        scene = self.scene
        scope = f"RenderNames_{scene.next_camera_id}"
        scene.next_camera_id += 1
        product, table = f"/{scope}/RenderProduct", f"/{scope}/Vars/StableIdMap"
        reference = ovstage.population.add_usd_reference_from_string(
            scene.stage, ovrtx_usd.build_names_product_as_string(scope), f"/{scope}"
        )
        scene.resources.callback(self._remove_usd_reference, reference)
        ovstage.population.apply_usd_changes(scene.stage, ordinal=scene.ordinal)
        for _ in range(NAMES_STEPS):  # no time passes: the scene's sensors are not due
            result = self.backend.renderer.step(
                render_products={product}, delta_time=0.0, ordinal=scene.commit()
            )
            for frame in getattr(dict(result.items()).get(product), "frames", ()):
                _raise_failed(product, frame)
                output = dict(frame.render_vars.items()).get(table)
                if output is None:
                    continue
                with output.map(device=ovrtx.Device.CPU) as mapped:
                    ids = np.from_dlpack(mapped["id"]).copy()
                    ends = np.from_dlpack(mapped["label_offsets"]).copy()
                    self._object_table = (ids, ends, np.from_dlpack(mapped["label_data"]).copy())
                    return self._object_table
            del result
        raise RuntimeError(f"ovrtx named no objects in {NAMES_STEPS} steps.")

    def _step(self) -> Any:
        """ovrtx stepped once, as long as the simulation's step (`sim.dt`), at the stage's
        sealed state, every registered render product with it (one left out of a step loses
        its history). Returns what came of it, by render product."""
        if self.scene is None or self.backend is None or not self._initialized_scene:
            raise RuntimeError("Scene not initialized. Call prepare_stage() first.")
        ordinal = self.scene.commit()
        try:
            result = self.backend.renderer.step(
                render_products=self.backend.render_products.copy(),
                delta_time=self._sim.dt,
                ordinal=ordinal,
            )
        except RuntimeError as exc:
            # With the GPU's memory full, ovrtx fails the step without saying why (DLSS
            # cannot be made: "NGX CreateFeature failed"): said as what it is, in the words
            # the capture steps back on (`darkroom.process.OUT_OF_GPU`).
            used = gpu.memory()
            if used is not None and used[1] - used[0] < FULL * used[1]:
                raise RuntimeError(
                    f"Out of GPU memory ({used[0]:.1f} of {used[1]:.1f} GB in use): {exc}"
                ) from exc
            raise
        # A frame is stamped with ovrtx's clock, and found by the simulation's.
        if abs(result.simulation_end_time - self._sim.clock) > self._sim.dt / 4:
            raise RuntimeError(
                f"ovrtx's clock is at {result.simulation_end_time:.4f} s, the simulation's "
                f"at {self._sim.clock:.4f} s: a step of one without the other."
            )
        return result

    def set_outputs(
        self, render_data: OVRTXCameraRenderData, output_data: dict[str, np.ndarray]
    ) -> None:
        """Register the pre-allocated output arrays a camera's frames are copied into. ``rgb``
        is excluded: it is a view into ``rgba`` and is updated with it."""
        render_data.buffers = {
            name: array for name, array in output_data.items() if name != str(RenderBufferKind.RGB)
        }

    def update_transforms(self) -> None:
        """The stage taken to the simulation's time: ovstage reads every time-sampled
        attribute at it (transforms, visibility, a skinned character's joints)."""
        if self.scene is None:
            return
        ovstage.population.update_from_usd_time(
            self.scene.stage, self.scene.ordinal, self._sim.time
        )

    def update_geometries(self) -> None:
        """Nothing: ovstage skins and deforms the stage's meshes from its USD time."""

    def update_camera(
        self,
        render_data: OVRTXCameraRenderData,
        positions: np.ndarray,
        orientations: np.ndarray,
        intrinsics: np.ndarray,
    ) -> None:
        """No-op: a camera's eye moves with the stage, under its own parent."""

    def render(self, render_data: OVRTXCameraRenderData) -> None:
        """Submit one camera capture; :meth:`read_output` publishes the available image."""
        self.render_batch((render_data,))

    def render_batch(self, render_data: Sequence[OVRTXCameraRenderData]) -> None:
        """Render all requested camera products in one native submission: one step of
        ovrtx's clock, as long as the simulation's (`sim.dt`), at the stage's sealed state.

        Every registered render product is submitted (`_step`), not just the requested ones.
        Only ``render_data`` is read back.

        Raises:
            RuntimeError: If the scene is uninitialized, ovrtx's clock is not the
                simulation's, or ovrtx says it could not make an output.
        """
        if not render_data:
            return
        result = self._step()
        try:
            self._process_render_products(render_data, result)
        finally:
            del result

    def _process_render_products(
        self, render_data: Sequence[OVRTXCameraRenderData], products: Any
    ) -> None:
        """Each camera's frame of this step, if it had one, copied into its output arrays and
        noted with its time (`OVRTXCameraRenderData.captures`). A product absent from the
        result, or with no frames, had none due."""
        came = dict(products.items())
        for data in render_data:
            product = came.get(data.render_product_path)
            frames = list(product.frames) if product is not None else []
            if len(frames) > 1:
                raise RuntimeError(
                    f"{data.spec.cfg.prim_path}: {len(frames)} frames in one step: its rate is "
                    f"above the simulation's."
                )
            for frame in frames:
                self._process_render_frame(data, frame)

    def _process_render_frame(self, render_data: OVRTXCameraRenderData, frame: Any) -> None:
        """Extract a frame's picture into the camera's "rgba" array, and what else it was
        asked for into theirs: each pixel's normal, its distance, its semantic class (and
        what each class is called, from ovrtx's `SemanticIdMap`)."""
        _raise_failed(render_data.spec.cfg.prim_path, frame)
        outputs = dict(frame.render_vars.items())
        for source, kind in _IMAGES.items():
            output = outputs.get(render_data.render_var_keys[source])
            buffer = render_data.buffers.get(str(kind))
            if output is None or buffer is None:
                continue
            with output.map(device=ovrtx.Device.CPU) as mapped:
                pixels = np.from_dlpack(mapped)
                pixels = pixels.reshape(*pixels.shape[:2], -1)
                buffer[0] = pixels[..., : buffer.shape[-1]]
        named = outputs.get(render_data.render_var_keys[_SEMANTIC_ID_MAP_VAR])
        if named is not None:
            with named.map(device=ovrtx.Device.CPU) as mapped:
                table = tuple(np.from_dlpack(mapped[name]) for name in _ID_MAP_TENSORS)
                render_data.id_to_labels = semantic_labels(*table)
        # The time of ovrtx's clock the frame is at: its end, when a rolling shutter's last
        # row is sampled (`ovrtx_usd.rolling_shutter`).
        render_data.captures.append({"time": float(frame.end_time)})

    def read_output(self, render_data: OVRTXCameraRenderData, camera_data: CameraData) -> None:
        """The frame rendered since the last read published: its pixels (and normals,
        distances and classes, where asked for) are in the camera's arrays already;
        `CameraData.info` says its time under "capture", and what each semantic class is
        called under "idToLabels" (as Isaac Lab's: {id: {type: label}})."""
        for name in camera_data.info:
            camera_data.info[name] = None
        if render_data.captures:
            camera_data.info[str(RenderBufferKind.RGBA)] = {"capture": render_data.captures}
            render_data.captures = []
            if str(RenderBufferKind.SEMANTIC_SEGMENTATION) in render_data.buffers:
                camera_data.info[str(RenderBufferKind.SEMANTIC_SEGMENTATION)] = {
                    "idToLabels": render_data.id_to_labels
                }

    def cleanup(self, render_data: OVRTXCameraRenderData | OVRTXLidarRenderData | None) -> None:
        """Release the render data's resources: its product and its eye (a lidar's: its
        copy) taken off the stage.

        Each camera owns its product and eye. The scene stays, so other cameras can continue
        rendering.
        """
        if render_data is None:
            return
        try:
            render_data.cleanup()
            if self.scene is not None and self.scene.stage is not None:
                self.scene.commit()
        finally:
            if render_data in self._camera_render_data:
                self._camera_render_data.remove(render_data)

    def _remove_usd_reference(self, reference: Any) -> None:
        """Publish removal of a private USD reference at the shared stage's current ordinal."""
        if self.scene is None or self.scene.stage is None:
            return
        ovstage.population.remove_usd(self.scene.stage, reference)
        ovstage.population.apply_usd_changes(self.scene.stage, ordinal=self.scene.ordinal)

    def close(self) -> None:
        """Release every camera's resources; the registry owns the native engine and stage."""
        try:
            for render_data in tuple(self._camera_render_data):
                self.cleanup(render_data)
        finally:
            self._initialized_scene = False
