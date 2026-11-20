"""Rigs on one stage: a sensor the same in two rendered once, each rig given its own capture."""

import numpy as np
import usdsensors
from pxr import Sdf, Usd, UsdGeom

from darkroom import format, sensorkit
from darkroom.share import KEPT, Rig, _fan_out, _leave_only_how, _render_once, count_shared

FRAME = "/World/Walk/Frames/SensorMount"


def _rigs(root):
    """Probe (a camera), Twin (the same camera in the same place, and an IMU), Moved (the
    camera 10 cm higher); a stage with all three under one frame."""
    camera = usdsensors.make_sensor(root / "sensors" / "Camera.usda", "camera")
    imu = usdsensors.make_sensor(root / "sensors" / "IMU.usda", "imu")
    files = {}
    for name, z in (("Probe", 0.2), ("Twin", 0.2), ("Moved", 0.3)):
        files[name] = usdsensors.new_rig(root / "rigs" / f"{name}.usda")
        usdsensors.add(files[name], "camera", camera, usdsensors.Pose((0, 0, z)))
    usdsensors.add(files["Twin"], "imu", imu, usdsensors.Pose((0, 0, 0)))
    layer = Sdf.Layer.CreateNew(str(root / "run.usda"))
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.Xform.Define(stage, FRAME)
    for name, file in files.items():
        usdsensors.attach(layer, file, FRAME, name)
    layer.Save()
    return files, root / "run.usda"


def test_a_sensor_the_same_in_two_rigs_is_rendered_once(tmp_path):
    files, stage_file = _rigs(tmp_path)
    rigs = [Rig(name, f"{FRAME}/{name}") for name in files]
    out = tmp_path / "out"
    out.mkdir()
    rendered, _ = _render_once(stage_file, out, rigs)
    stage = Usd.Stage.Open(str(rendered))
    on = {str(p.GetPath()) for p in stage.Traverse() if p.GetName() in ("camera", "imu")}
    assert f"{FRAME}/Probe/camera/camera" in on
    assert f"{FRAME}/Twin/camera/camera" not in on  # Probe's stands for it
    assert f"{FRAME}/Moved/camera/camera" in on  # elsewhere: not the same
    shared = format.read_shared(out)
    assert shared is not None
    assert shared[f"{FRAME}/Probe/camera/camera"] == [
        f"{FRAME}/Probe/camera/camera",
        f"{FRAME}/Twin/camera/camera",
    ]
    assert "active" not in files["Probe"].read_text(encoding="utf-8")  # the rig files only read


def test_any_setting_changed_makes_a_sensor_another(tmp_path):
    files, _ = _rigs(tmp_path)
    pair = {r: files[r] for r in ("Probe", "Twin")}
    assert count_shared(pair) == (3, 2)
    probe = Usd.Stage.Open(str(files["Probe"]))
    focal = probe.GetPrimAtPath("/Probe/camera/camera").GetAttribute("focalLength").Get()
    usdsensors.change(files["Twin"], "camera", "focalLength", focal * 2)
    assert count_shared(pair) == (3, 3)
    usdsensors.change(files["Twin"], "camera", "focalLength", focal)  # back: the same again
    assert count_shared(pair) == (3, 2)


def test_each_rig_is_given_its_own_capture_of_the_one_render(tmp_path):
    files, stage_file = _rigs(tmp_path)
    rigs = [Rig(name, f"{FRAME}/{name}") for name in files]  # a rig not listed: in each
    run = tmp_path / "captures" / "run"
    done = run / ".render"  # the one render, hidden in the run's folder
    done.mkdir(parents=True)
    rendered, groups = _render_once(stage_file, done, rigs)
    stage = Usd.Stage.Open(str(rendered))
    sensors = {}
    for s in sensorkit.sensors(stage):  # what a capture of it would have written
        sensors[s.name] = format.Sensor(s.kind, str(s.path), s.rate)
        poses = format.Poses(np.zeros(1), np.zeros((1, 3)), np.ones((1, 4)))
        format.write_poses(done, s.name, poses)
    format.write_meta(done, format.Meta(str(rendered), 0.0, 1.0, sensors))
    format.write_states(done, {})

    format.done(done)  # as capture marks it
    (done / "perf.json").write_text("{}", encoding="utf-8")  # which renderer made it, kept

    made = _fan_out(done, run, rendered, rigs, groups, None)
    assert made == {"Probe": run / "Probe", "Twin": run / "Twin", "Moved": run / "Moved"}
    probe, twin = format.read_meta(made["Probe"]), format.read_meta(made["Twin"])
    assert set(probe.sensors) == {"camera"} and set(twin.sensors) == {"camera", "imu"}
    assert twin.sensors["camera"].prim == f"{FRAME}/Twin/camera/camera"
    rendered_poses = done / "probe_camera" / format.POSES  # Probe's, rendered for both
    assert (made["Twin"] / "camera" / format.POSES).stat().st_ino == rendered_poses.stat().st_ino
    _leave_only_how(done)  # every file once in the rigs' folders, the render only how it was made
    assert sorted(e.name for e in done.iterdir()) == sorted(k for k in KEPT if (done / k).exists())
    assert {format.META, format.SHARED, "perf.json"} <= {e.name for e in done.iterdir()}
    assert format.finished(done) and format.read_shared(done) is not None
    probe_poses = made["Probe"] / "camera" / format.POSES
    assert (made["Twin"] / "camera" / format.POSES).stat().st_ino == probe_poses.stat().st_ino
    assert format.finished(made["Twin"]) and (made["Twin"] / format.STATES).is_file()
    assert format.read_shared(made["Twin"]) is None  # a rig's own capture, not the render
