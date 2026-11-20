"""A capture's renderer, picked by its cfg: the default Isaac Sim's, a cfg sent to the process
that renders as plain data, a renderer named without loading it, a simulation owning what its
renderers share, and each renderer's cameras started through its own launcher."""

import json
import sys

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom

from darkroom import format, renderers, run, sensorkit
from darkroom.options import Job, Options
from darkroom.renderers import launchers, renderer_cfg
from darkroom.renderers.camera_cfg import CameraData
from darkroom.renderers.renderer_cfg import BackendCfg
from darkroom.renderers.simulation_context import SimulationContext


def test_a_capture_that_names_no_renderer_has_ovrtxs():
    assert Options().renderer_cfg.renderer_type == "default"
    assert renderers.resolve(Options().renderer_cfg).renderer_type == "ovrtx"
    assert renderers.resolve(None).launcher_type == "darkroom.renderers.launchers:OvrtxLauncher"
    ovrtx = renderers.resolve(renderers.OVRTXRendererCfg())
    assert ovrtx.renderer_type == "ovrtx"
    assert ovrtx.launcher_type == "darkroom.renderers.launchers:OvrtxLauncher"
    with pytest.raises(ValueError, match="names no class_type"):
        renderers.resolve(renderers.RendererCfg(renderer_type="mine"))
    with pytest.raises(ValueError, match="renderer's cfg"):
        Options(renderer_cfg="ovrtx")  # ty: ignore[invalid-argument-type]


def test_a_cfg_crosses_to_the_process_that_renders_as_plain_data():
    for cfg in (
        renderers.OVRTXRendererCfg(),
        renderers.OVRTXRendererCfg(temp_usd_dir="/tmp/x", enable_shadows=True, step_rate=960.0),
    ):
        said = json.loads(json.dumps(renderer_cfg.plain(cfg)))
        assert renderers.of(said) == cfg
    assert renderers.of({}) == renderers.get_default_renderer_cfg()
    with pytest.raises(ValueError, match="has no nonsense"):
        renderers.of({**renderer_cfg.plain(renderers.OVRTXRendererCfg()), "nonsense": 1})


def test_a_renderer_says_what_it_makes_without_being_loaded():
    ovrtx = renderers.OVRTXRendererCfg().supported_output_types()
    assert {"rgb", "rgba"} <= {str(k) for k in ovrtx}
    assert "darkroom.ovrtx.ovrtx_renderer" not in sys.modules
    assert "ovrtx" not in sys.modules


def test_a_cameras_pictures_land_in_arrays_laid_out_as_its_renderer_says():
    data = CameraData()
    specs = renderers.OVRTXRendererCfg().supported_output_types()
    data.allocate(("rgb",), 4, 6, 1, specs)
    assert data.output["rgba"].shape == (1, 4, 6, 4) and data.output["rgba"].dtype == np.uint8
    data.output["rgba"][0, 1, 2] = (9, 8, 7, 255)
    assert tuple(data.output["rgb"][0, 1, 2]) == (9, 8, 7)  # a view of the same memory
    data.allocate(("rgb", "semantic_segmentation", "normals", "distance_to_camera"), 4, 6, 1, specs)
    assert data.output["semantic_segmentation"].shape == (1, 4, 6, 1)
    assert data.output["normals"].shape == (1, 4, 6, 3)
    assert data.output["distance_to_camera"].dtype == np.float32
    with pytest.raises(ValueError, match="makes no motion_vectors"):
        data.allocate(("rgb", "motion_vectors"), 4, 6, 1, specs)


class _Resource:
    closed: list[str] = []  # noqa: RUF012 - the order every one made was closed in

    def __init__(self, cfg, **bound):
        self.cfg, self.bound = cfg, bound

    def close(self):
        self.closed.append(self.cfg.name)


def test_a_simulation_makes_one_resource_for_equal_cfgs_and_closes_the_last_first():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Cfg(BackendCfg):
        name: str = ""

    sim = SimulationContext(None, dt=1 / 60)
    first = sim.get_or_create_backend(Cfg(class_type=_Resource, name="first"))
    assert sim.get_or_create_backend(Cfg(class_type=_Resource, name="first")) is first
    second = sim.get_or_create_backend(Cfg(class_type=_Resource, name="second"))
    assert second is not first and first.bound == {}  # only a renderer is bound to its sim
    sim.step()
    sim.seek(5.0)
    sim.step()
    assert sim.time == pytest.approx(5.0 + 1 / 60) and sim.clock == pytest.approx(2 / 60)
    _Resource.closed.clear()
    sim.clear_instance()
    assert _Resource.closed == ["second", "first"]


def _stage(folder):
    """A stage of 1 s with a camera and a lidar."""
    folder.mkdir()
    layer = Sdf.Layer.CreateNew(str(folder / "run.usda"))
    layer.timeCodesPerSecond = 60
    layer.startTimeCode, layer.endTimeCode = 0, 60
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Camera.Define(stage, "/Rig/front")
    sensorkit.mark(layer, "/Rig/front", "camera", 30)
    lidar = stage.DefinePrim("/Rig/lidar", "OmniLidar")
    lidar.CreateAttribute("omni:sensor:Core:scanRateBaseHz", Sdf.ValueTypeNames.UInt).Set(10)
    sensorkit.mark(layer, "/Rig/lidar", "lidar", 10)
    layer.Save()
    return folder / "run.usda"


def test_every_capture_is_rendered_by_ovrtxs_launcher_one_process_for_alike_jobs(
    tmp_path, monkeypatch
):
    started = []

    def made(self, target, jobs, **kwargs):  # the process stood in for: every sensor made
        started.append(
            (target, kwargs["kinds"], [j["renderer_cfg"]["renderer_type"] for j in jobs])
        )
        for job in jobs:
            for sensor in format.read_meta(job["out"]).sensors:
                format.made(job["out"], sensor, "30 frames at 30 Hz")
        return {"jobs": [{} for _ in jobs]}

    monkeypatch.setattr(launchers.OvrtxLauncher, "call", made)
    jobs = [
        Job(_stage(tmp_path / "a"), tmp_path / "out_a"),
        Job(
            _stage(tmp_path / "b"),
            tmp_path / "out_b",
            Options(renderer_cfg=renderers.OVRTXRendererCfg()),
        ),
    ]
    made_ = run.captures(jobs, on_line=None)
    assert made_ == [tmp_path / "out_a", tmp_path / "out_b"]
    # A capture naming no renderer gets ovrtx's; both stages in one process, every kind with it.
    assert started == [("darkroom.ovrtx:render_all", ["camera", "lidar"], ["ovrtx", "ovrtx"])]
