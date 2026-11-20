import numpy as np
from pxr import Sdf, Usd, UsdGeom

from darkroom import format, sensorkit, skies, truth

SITE = "OmniWGS84ReferencePositionAPI"


def _stage(tmp_path):
    """A frame going 2 m/s east for 4 s, carrying an IMU and a GNSS antenna 0.1 m above it."""
    layer = Sdf.Layer.CreateNew(str(tmp_path / "run.usda"))
    layer.timeCodesPerSecond = 60
    layer.startTimeCode, layer.endTimeCode = 0, 240
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Scope.Define(stage, "/Thing/Frames")
    frame = UsdGeom.Xform.Define(stage, "/Thing/Frames/Mount")
    op = frame.AddTranslateOp()
    op.Set((0, 0, 1.5), 0)
    op.Set((8, 0, 1.5), 240)
    for name in ("Imu", "Antenna"):
        UsdGeom.Xform.Define(stage, f"/Thing/Frames/Mount/{name}").AddTranslateOp().Set((0, 0, 0.1))
    sensorkit.mark(layer, "/Thing/Frames/Mount/Imu", "imu", 100)
    sensorkit.mark(layer, "/Thing/Frames/Mount/Antenna", "gnss", 10)
    site = stage.DefinePrim("/Site", "Xform")
    site.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=[SITE]))
    site.CreateAttribute(
        "omni:geospatial:wgs84:reference:referencePosition", Sdf.ValueTypeNames.Double3
    ).Set((-36.85, 174.76, 40.0))
    layer.Save()
    return tmp_path / "run.usda"


def test_truth_writes_poses_imu_and_fixes(tmp_path):
    out = tmp_path / "capture"
    meta = truth.write(_stage(tmp_path), out, window=(1.0, 3.0))
    assert set(meta.sensors) == {"imu", "antenna"} and meta.site is not None
    assert format.read_meta(out) == meta

    poses = format.read_poses(out, "antenna")
    assert len(poses.seconds) == 21 and poses.seconds[0] == 0
    # 1 s into the stage, 2 m east, 1.6 m up.
    assert np.allclose(poses.position[0], (2, 0, 1.6))

    imu = format.read_arrays(out, "imu", format.IMU)
    assert len(imu["seconds"]) == 201
    assert np.allclose(imu["accel"], (0, 0, 9.80665), atol=1e-6)

    fixes = format.read_arrays(out, "antenna", format.GNSS)
    east = (fixes["longitude"][-1] - fixes["longitude"][0]) * 111_320 * np.cos(np.radians(-36.85))
    assert abs(east - 4.0) < 0.01 and np.allclose(fixes["height"], 41.6)


def test_an_antenna_keeps_the_truth_its_tips_beside_it_and_how_much_sky_it_saw(tmp_path):
    """The thing carrying the antenna has a tip on the ground at its origin: its fixes go beside
    the antenna's. Open sky for the first second, none after."""
    stage_file = _stage(tmp_path)
    layer = Sdf.Layer.FindOrOpen(str(stage_file))
    stage = Usd.Stage.Open(layer)
    UsdGeom.Xform.Define(stage, "/Thing/Geometry")  # what carries it: Frames and Geometry
    UsdGeom.Xform.Define(stage, "/Thing/Frames/Tip")
    layer.Save()
    out = tmp_path / "capture"
    meta = truth.write(stage_file, out, window=(1.0, 3.0))
    assert meta.sensors["antenna"].settings == {} and skies.wanted(meta) == ["antenna"]
    got = format.read_arrays(out, "antenna", format.GNSS)
    assert np.allclose(got["height"], 41.6) and np.allclose(got["tip_height"], 40.0)
    assert np.allclose(got["tip_latitude"], -36.85) and np.allclose(got["tip_longitude"], 174.76)
    size = sensorkit.sky.SIZE
    for i in range(20):  # the 21st fix has no view of its own
        seen = sensorkit.sky.OPEN if i < 10 else sensorkit.sky.BLOCKED
        format.write_sky(out, "antenna", i, np.full((size, size), seen, np.uint8))
    (line,) = skies.write(stage_file, out)
    assert line == "antenna: 48% of its view open sky on average, 0% at least"
    got = format.read_arrays(out, "antenna", format.GNSS)
    assert (got["sky"][:10] == 1).all() and (got["sky"][10:] == 0).all()
    assert np.allclose(got["tip_height"], 40.0)  # the rest of the fixes kept as they were


def test_a_sensor_marked_debug_says_so_in_its_settings(tmp_path):
    stage_file = _stage(tmp_path)
    layer = Sdf.Layer.FindOrOpen(str(stage_file))
    antenna = layer.GetPrimAtPath("/Thing/Frames/Mount/Antenna")
    Sdf.AttributeSpec(antenna, sensorkit.DEBUG, Sdf.ValueTypeNames.Bool).default = True
    layer.Save()
    meta = truth.write(stage_file, tmp_path / "capture", window=(1.0, 2.0))
    assert meta.sensors["antenna"].settings == {"debug": True}
    assert meta.sensors["imu"].settings == {}


def _with_lidar(tmp_path, turning: bool = True):
    """`_stage` with a lidar scanning 10 times a second on the mount, which turns a quarter
    turn a second about z as it goes."""
    stage_file = _stage(tmp_path)
    layer = Sdf.Layer.FindOrOpen(str(stage_file))
    stage = Usd.Stage.Open(layer)
    if turning:
        turn = UsdGeom.Xform(stage.GetPrimAtPath("/Thing/Frames/Mount")).AddRotateZOp()
        turn.Set(0.0, 0)
        turn.Set(360.0, 240)
    lidar = stage.DefinePrim("/Thing/Frames/Mount/Lidar", "OmniLidar")
    lidar.CreateAttribute("omni:sensor:Core:scanRateBaseHz", Sdf.ValueTypeNames.UInt).Set(10)
    sensorkit.mark(layer, "/Thing/Frames/Mount/Lidar", "lidar", 10)
    layer.Save()
    return stage_file


def test_a_lidar_keeps_its_poses_120_times_a_second_to_place_each_point_by(tmp_path):
    out = tmp_path / "capture"
    truth.write(_with_lidar(tmp_path), out, window=(1.0, 3.0), kinds=["lidar"])
    scans, fine = format.read_poses(out, "lidar"), format.read_motion(out, "lidar")
    assert format.has_motion(out, "lidar")
    assert len(scans.seconds) == 21 and len(fine.seconds) == 241
    # Between two scan poses the fine poses follow the turn; a straight line between scans
    # would cut the corner.
    t = np.array([1.05])
    where, _ = fine.at(t)
    angle = np.radians(90.0 * (1.0 + t[0]))
    assert np.allclose(where[0, :2], 2 * (1.0 + t[0]) * np.array([1, 0]), atol=1e-6)
    ahead = fine.place(t, np.array([[1.0, 0.0, 0.0]]))[0] - where[0]
    assert np.allclose(ahead[:2], [np.cos(angle), np.sin(angle)], atol=1e-4)


def test_an_older_capture_is_given_its_lidars_poses_as_they_were_traced(tmp_path):
    """Made before fine poses: given the poses of its lidar pass's updates, from its stage. This
    stage is animated in its root layer, so the pass ran at its 60 time codes a second."""
    out = tmp_path / "capture"
    truth.write(_with_lidar(tmp_path), out, window=(1.0, 3.0), kinds=["lidar"])
    made = format.read_motion(out, "lidar")
    (out / "lidar" / format.MOTION).unlink()  # as a capture made before them
    assert not format.has_motion(out, "lidar")
    scans = format.read_poses(out, "lidar")
    assert np.array_equal(format.read_motion(out, "lidar").seconds, scans.seconds)
    assert truth.add_lidar_motion(out) == {"lidar": 60.0}
    given = format.read_motion(out, "lidar")
    assert len(given.seconds) == 121 and np.allclose(given.position, made.position[::2])
    assert truth.add_lidar_motion(out) == {}  # had them already


def test_an_older_ovrtx_capture_is_given_its_lidars_poses_at_the_stages_frames(tmp_path):
    """ovrtx's lidars stepped at the stage's frames: an older capture it made, said by its meta
    or by ovrtx's profile in its perf.json, is given its poses at those; Isaac Sim's, of a stage
    animated in a sublayer, once a scan."""
    import json

    animated = _with_lidar(tmp_path)
    layer = Sdf.Layer.CreateNew(str(tmp_path / "root.usda"))  # its animation in a sublayer
    under = Sdf.Layer.FindOrOpen(str(animated))
    layer.timeCodesPerSecond = under.timeCodesPerSecond
    layer.startTimeCode, layer.endTimeCode = under.startTimeCode, under.endTimeCode
    layer.subLayerPaths.append(str(animated))
    UsdGeom.SetStageUpAxis(Usd.Stage.Open(layer), "Z")
    layer.Save()
    for said, rate in (("meta", 60.0), ("perf", 60.0), ("isaac_rtx", 10.0)):
        out = tmp_path / said
        meta = truth.write(tmp_path / "root.usda", out, window=(1.0, 3.0), kinds=["lidar"])
        (out / "lidar" / format.MOTION).unlink()
        if said == "perf":
            (out / "perf.json").write_text(
                json.dumps({"seconds": 9.0, "ovrtx": {}}), encoding="utf-8"
            )
        else:
            meta.extra[format.RENDERER] = "ovrtx" if said == "meta" else said
            format.write_meta(out, meta)
        assert truth.add_lidar_motion(out) == {"lidar": rate}, said


def test_an_older_lidar_pass_ran_at_the_least_multiple_of_its_lidars_rates():
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(60)
    assert truth.traced_rate(stage, [10.0]) == 10  # once a scan
    assert truth.traced_rate(stage, [10.0, 30.0]) == 30  # a VL53L9 beside it
    stage.DefinePrim("/Moving").CreateAttribute("x", Sdf.ValueTypeNames.Float).Set(1.0, 0)
    assert truth.traced_rate(stage, [10.0]) == 60  # its root layer animated: its own rate
