"""Which renderer processes a capture's jobs render in."""

from darkroom import run
from darkroom.options import Options
from darkroom.renderjob import RenderJob


def _ready(at, kinds):
    job = RenderJob(stage_file="s.usda", out=f"out{at}", window=(0.0, 1.0))
    return run._Ready(at, "gnss" in kinds, set(kinds), Options(), job)


def test_jobs_rendering_the_same_kinds_share_a_process():
    sims = run._group([_ready(0, {"camera", "lidar"}), _ready(1, {"lidar", "camera"})])
    assert [(s.kinds, [j.at for j in s.jobs]) for s in sims] == [(["camera", "lidar"], [0, 1])]
    sims = run._group([_ready(0, {"camera", "gnss"}), _ready(1, {"lidar"}), _ready(2, set())])
    assert [s.kinds for s in sims] == [["camera", "gnss"], ["lidar"]]


def test_a_fisheye_is_rendered_as_ovrtxs_own_opencv_fisheye():
    from darkroom import format, sensorkit

    fisheye = format.Sensor("camera", "/R/fish", 10.0, {"model": sensorkit.camera.FISHEYE})
    pinhole = format.Sensor("camera", "/R/pin", 10.0, {"model": sensorkit.camera.PINHOLE})
    meta = format.Meta("s.usda", 0.0, 1.0, {"fish": fisheye, "pin": pinhole})
    assert run._fisheyes(meta) == {"/R/fish": {"model": sensorkit.camera.FISHEYE}}
