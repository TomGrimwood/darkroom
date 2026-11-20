"""Several captures at once: each job's outcome where it is in the jobs given."""

import pytest
from pxr import Sdf, Usd, UsdGeom

from darkroom import format, run, sensorkit
from darkroom.options import Job
from darkroom.renderers import launchers


def _stage(folder):
    """A stage of 1 s with one camera."""
    folder.mkdir()
    layer = Sdf.Layer.CreateNew(str(folder / "run.usda"))
    layer.timeCodesPerSecond = 60
    layer.startTimeCode, layer.endTimeCode = 0, 60
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Camera.Define(stage, "/Rig/front")
    sensorkit.mark(layer, "/Rig/front", "camera", 30)
    layer.Save()
    return folder / "run.usda"


def test_a_job_failing_as_it_is_made_ready_leaves_the_next_in_its_own_place(tmp_path, monkeypatch):
    first, second = _stage(tmp_path / "a"), _stage(tmp_path / "b")
    real = run._fisheyes
    calls = []

    def fisheyes(meta):  # fails for the first stage, after its truth is written
        calls.append(meta)
        if len(calls) == 1:
            raise RuntimeError("no lens for the first")
        return real(meta)

    def call(self, target, jobs, **kwargs):  # the renderer stood in for: every camera made
        for job in jobs:
            out = format.read_meta(job["out"])
            for name in out.sensors:
                format.made(job["out"], name, "30 frames at 30 Hz")
        return {"jobs": [{} for _ in jobs]}

    monkeypatch.setattr(run, "_fisheyes", fisheyes)
    monkeypatch.setattr(launchers.OvrtxLauncher, "call", call)
    jobs = [Job(first, tmp_path / "out_a"), Job(second, tmp_path / "out_b")]
    made = run.captures(jobs, on_line=None)
    assert len(made) == 2
    assert isinstance(made[0], RuntimeError) and "first" in str(made[0])
    assert made[1] == tmp_path / "out_b" and format.finished(tmp_path / "out_b")


def test_captures_left_in_order_with_none(tmp_path):
    assert run.captures([], on_line=None) == []
    with pytest.raises(RuntimeError):
        run.capture(tmp_path / "nothing.usda", tmp_path / "out", on_line=None)
