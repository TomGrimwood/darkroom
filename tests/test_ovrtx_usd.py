"""The USD ovrtx is handed: the stage's root layer, each camera's eye scheduled at its rate
(with its rolling shutter), and its render product with the capture's settings on it."""

from types import SimpleNamespace

import pytest
from pxr import Sdf, Usd, UsdGeom

from darkroom import sensorkit
from darkroom.ovrtx import ovrtx_usd
from darkroom.renderers import CameraCfg, CameraRenderSpec


def _stage(tmp_path, readout=None):
    layer = Sdf.Layer.CreateNew(str(tmp_path / "run.usda"))
    layer.timeCodesPerSecond = 60
    layer.startTimeCode, layer.endTimeCode = 30, 90
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    rig = UsdGeom.Xform.Define(stage, "/World/Rig")
    rig.AddTranslateOp().Set((1.0, 2.0, 3.0))
    camera = UsdGeom.Camera.Define(stage, "/World/Rig/front")
    camera.GetFocalLengthAttr().Set(12.0)
    camera.AddTranslateOp().Set((0.0, 0.0, 0.5))
    prim = camera.GetPrim()
    prim.CreateAttribute("omni:rtx:post:motionblur:enabled", Sdf.ValueTypeNames.Bool).Set(True)
    if readout is not None:
        prim.CreateAttribute(sensorkit.camera.READOUT, Sdf.ValueTypeNames.Double).Set(readout)
        prim.CreateAttribute(ovrtx_usd.SHUTTER_TEXTURE, Sdf.ValueTypeNames.Asset).Set("rows.exr")
    layer.Save()
    return stage, prim


def _schemas(prim):
    """The API schemas authored on `prim` (ovrtx's are not registered with this USD, which
    lists only those it knows as applied)."""
    return list(prim.GetMetadata("apiSchemas").GetAppliedItems())


def test_an_eye_is_scheduled_at_its_cameras_rate_beside_it(tmp_path):
    stage, prim = _stage(tmp_path)
    eye = ovrtx_usd.sensor_eye(stage, prim, 15.0, 100)
    assert str(eye.GetPath()) == "/World/Rig/front_CaptureEye"
    assert "OmniSensorAPI" in _schemas(eye)
    assert tuple(eye.GetAttribute("omni:sensor:frameRate").Get()) == (15, 1)
    assert eye.GetAttribute("focalLength").Get() == 12.0
    assert "OmniRtxRollingShutterAPI" not in _schemas(eye)
    assert not eye.HasAttribute("omni:rtx:post:motionblur:enabled")  # its render product's
    assert not stage.GetRootLayer().GetPrimAtPath(eye.GetPath())  # the stage's file as it was


def test_a_readout_is_a_rolling_shutter_whose_last_row_is_read_as_the_frame_is_due(tmp_path):
    stage, prim = _stage(tmp_path, readout=0.02)
    eye = ovrtx_usd.sensor_eye(stage, prim, 30.0, 101)
    assert "OmniRtxRollingShutterAPI" in _schemas(eye)
    assert eye.GetAttribute("omni:rtx:rollingShutter:scanOrder").Get() == "topToBottom"
    top, bottom = eye.GetAttribute("omni:rtx:rollingShutter:lineOffset").Get()
    assert top / bottom == pytest.approx(0.02 / 100)  # 101 rows: 100 steps over the readout
    exposed, _ = eye.GetAttribute("omni:rtx:rollingShutter:lineExposure").Get()
    assert exposed == 0  # no shutter time on the camera: a row is the instant it starts
    top, bottom = eye.GetAttribute("omni:sensor:startTime").Get()
    assert top / bottom == pytest.approx(1 / 30 - 0.02)  # the last row on each period
    assert not eye.HasAttribute(ovrtx_usd.SHUTTER_TEXTURE)  # Isaac Sim's way, not ovrtx's
    ovrtx_usd.build_sensor_as_string(eye)  # handed over: the stage without it again
    # ovrtx keeps two exposures at once: a readout of up to two frames, no longer.
    ovrtx_usd.build_sensor_as_string(ovrtx_usd.sensor_eye(stage, prim, 100.0, 101))
    with pytest.raises(ValueError, match="more than two frames"):
        ovrtx_usd.sensor_eye(stage, prim, 120.0, 101)


def test_a_row_is_sampled_at_the_middle_of_its_exposure_the_last_at_the_frames_time():
    from fractions import Fraction as F

    rate, rows, readout, exposure = F(30), 541, F(30, 1000), F(10, 1000)
    offset, exposed, start = ovrtx_usd.rolling_shutter(rate, rows, readout, exposure)
    assert offset == readout / 540 and exposed == exposure / 2

    def sampled(row):  # ovrtx samples a line at the end of the interval it is given
        return start + row * offset + exposed

    # The last row on a frame's time (a whole number of periods), the first a readout before;
    # each the middle of an exposure that opened half the shutter time before it.
    assert sampled(540) * rate == 2
    assert sampled(540) - sampled(0) == readout
    assert start >= 0 and sampled(0) - exposure / 2 == start
    # With no shutter time, as before: a line is the instant it starts.
    assert ovrtx_usd.rolling_shutter(rate, rows, readout, F(0)) == (
        readout / 540,
        0,
        1 / rate - readout,
    )
    # A readout of a whole frame starts with the frame before.
    assert ovrtx_usd.rolling_shutter(F(10), rows, F(1, 10), F(0))[2] == 0


def test_a_cameras_shutter_time_is_its_usd_shutter_and_its_readout_usdsensors(tmp_path):
    stage, prim = _stage(tmp_path, readout=0.02)
    camera = UsdGeom.Camera(prim)
    camera.GetShutterOpenAttr().Set(-0.15)  # time codes, 60 a second: 5 ms open in all
    camera.GetShutterCloseAttr().Set(0.15)
    lens = sensorkit.camera.lens(prim)
    assert lens.readout == pytest.approx(0.02) and lens.exposure == pytest.approx(0.005)
    assert lens.settings()["readout"] == pytest.approx(0.02)  # kept with the capture
    eye = ovrtx_usd.sensor_eye(stage, prim, 30.0, 101)
    top, bottom = eye.GetAttribute("omni:rtx:rollingShutter:lineExposure").Get()
    assert top / bottom == pytest.approx(0.0025)
    top, bottom = eye.GetAttribute("omni:sensor:startTime").Get()
    assert top / bottom == pytest.approx(1 / 30 - 0.02 - 0.0025)


def test_an_eye_is_handed_over_as_a_layer_of_its_own_and_leaves_the_stage_as_it_was(tmp_path):
    stage, prim = _stage(tmp_path)
    eye = ovrtx_usd.sensor_eye(stage, prim, 15.0, 100)
    ovrtx_usd.author(eye, {"exposure:iso": ("float", 200.0)})
    path = eye.GetPath()
    text = ovrtx_usd.build_sensor_as_string(eye)
    assert not stage.GetPrimAtPath(path)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(text)
    opened = Usd.Stage.Open(layer)
    made = opened.GetDefaultPrim()
    assert "OmniSensorAPI" in _schemas(made)
    assert made.GetName() == "Sensor" and made.IsA(UsdGeom.Camera)
    assert made.GetAttribute("exposure:iso").Get() == 200.0
    assert tuple(made.GetAttribute("omni:sensor:frameRate").Get()) == (15, 1)
    assert UsdGeom.Xformable(made).GetLocalTransformation().ExtractTranslation() == (0, 0, 0.5)


def test_a_render_product_points_at_the_eye_with_the_settings_authored_on_it(tmp_path):
    cfg = CameraCfg(
        prim_path="/World/Rig/front",
        width=640,
        height=480,
        data_types=("rgb",),
        render_settings={
            "omni:rtx:rtpt:maxBounces": ("uint", 6),
            "omni:rtx:post:tonemap:op": ("Token", "aces"),  # a type named as Isaac Lab does
        },
    )
    spec = CameraRenderSpec(cfg, "cuda:0", 1, ("/World/Rig/front",), 1)
    data = SimpleNamespace(
        render_scope_name="RenderCamera_3",
        render_product_path="/RenderCamera_3/RenderProduct",
        camera_paths=["/World/Rig/front_CaptureEye"],
        num_cols=1,
        num_rows=1,
        width=640,
        height=480,
    )
    text = ovrtx_usd.build_render_product_as_string(spec, data, device_id=0)  # ty: ignore[invalid-argument-type]
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(text)
    stage = Usd.Stage.Open(layer)
    assert stage.GetDefaultPrim().GetName() == "RenderCamera_3"
    product = stage.GetPrimAtPath("/RenderCamera_3/RenderProduct")
    assert product.GetRelationship("camera").GetTargets() == ["/World/Rig/front_CaptureEye"]
    assert tuple(product.GetAttribute("resolution").Get()) == (640, 480)
    assert product.GetAttribute("omni:rtx:rendermode").Get() == "RealTimePathTracing"
    assert product.GetAttribute("omni:rtx:rtpt:maxBounces").Get() == 6
    assert product.GetAttribute("omni:rtx:post:tonemap:op").Get() == "aces"
    assert product.GetRelationship("orderedVars").GetTargets() == ["/RenderCamera_3/Vars/LdrColor"]
    source = stage.GetPrimAtPath("/RenderCamera_3/Vars/LdrColor").GetAttribute("sourceName")
    assert source.Get() == "LdrColor"


def test_the_root_layer_ovrtx_opens_has_the_stages_file_under_it_and_its_times(tmp_path):
    stage, _ = _stage(tmp_path)
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(ovrtx_usd.build_stage_as_string(stage))
    assert list(layer.subLayerPaths) == [str(tmp_path / "run.usda")]
    assert (layer.startTimeCode, layer.endTimeCode, layer.timeCodesPerSecond) == (30, 90, 60)
    opened = Usd.Stage.Open(layer)
    assert opened.GetPrimAtPath("/World/Rig/front").IsA(UsdGeom.Camera)


def test_a_cameras_own_rtx_settings_are_read_with_their_types(tmp_path):
    _, prim = _stage(tmp_path)
    assert ovrtx_usd.camera_settings(prim) == {"omni:rtx:post:motionblur:enabled": ("bool", True)}


def test_outputs_are_isaac_labs_render_vars():
    assert ovrtx_usd.get_render_var_configs(["rgb", "rgba"], "S") == [
        ("/S/Vars/LdrColor", "LdrColor", "LdrColor")
    ]
    with pytest.raises(ValueError, match="Unsupported"):
        ovrtx_usd.get_render_var_configs(["x_ray"], "S")
    with pytest.raises(ValueError, match="at least one"):
        ovrtx_usd.get_render_var_configs([], "S")


def _lidar(tmp_path, spins=False):
    """A lidar as a capture's stage has one: Isaac Sim's attributes, its maker's version."""
    layer = Sdf.Layer.CreateNew(str(tmp_path / "lidar.usda"))
    stage = Usd.Stage.Open(layer)
    UsdGeom.Xform.Define(stage, "/World/Rig")
    prim = stage.DefinePrim("/World/Rig/lidar", "OmniLidar")
    prim.SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(["SensorLidarAPI", "OmniSensorGenericLidarCoreAPI"]),
    )
    values = {
        "omni:sensor:Core:scanRateBaseHz": (Sdf.ValueTypeNames.UInt, 10),
        "omni:sensor:Core:patternFiringRateHz": (Sdf.ValueTypeNames.UInt, 20000 if spins else 10),
        "omni:sensor:Core:reportRateBaseHz": (Sdf.ValueTypeNames.UInt, 10),
        "omni:sensor:Core:scanType": (
            Sdf.ValueTypeNames.Token,
            "ROTARY" if spins else "SOLID_STATE",
        ),
        "omni:sensor:Core:accumulateOutputs": (Sdf.ValueTypeNames.Bool, True),
        "omni:sensor:Core:skipDroppingInvalidPoints": (Sdf.ValueTypeNames.Bool, False),
        "omni:sensor:Core:elementsCoordsType": (Sdf.ValueTypeNames.Token, "SPHERICAL"),
        "omni:sensor:Core:farRangeM": (Sdf.ValueTypeNames.Float, 70.0),
        "omni:sensor:Core:emitterState:s001:fireTimeNs": (Sdf.ValueTypeNames.UIntArray, [0, 5, 9]),
        "omni:sensor:tickRate": (Sdf.ValueTypeNames.Float, 10.0),
        "omni:sensor:modelVersion": (Sdf.ValueTypeNames.String, "0.1.0"),
        "sensor:lidar:note": (Sdf.ValueTypeNames.String, "not RTX's"),
    }
    for name, (kind, value) in values.items():
        prim.CreateAttribute(name, kind).Set(value)
    return stage, prim


def test_a_lidar_that_measures_is_its_lidar_as_ovrtxs_schema_has_one(tmp_path):
    stage, prim = _lidar(tmp_path)
    made = ovrtx_usd.sensor_lidar(stage, prim, 10.0)
    assert (
        str(made.GetPath()) == "/World/Rig/lidar_CaptureLidar" and made.GetTypeName() == "OmniLidar"
    )
    assert _schemas(made) == ["OmniSensorGenericLidarCoreAPI"]
    names = {a.GetName() for a in made.GetAttributes() if a.HasAuthoredValue()}
    # Isaac Sim's own attributes are not handed on; the scan rate is the sensor's frame rate.
    assert not names & {
        "omni:sensor:Core:scanRateBaseHz",
        "omni:sensor:Core:patternFiringRateHz",
        "omni:sensor:Core:accumulateOutputs",
        "omni:sensor:Core:skipDroppingInvalidPoints",
        "omni:sensor:tickRate",
        "omni:sensor:modelVersion",
        "sensor:lidar:note",
        ovrtx_usd.LIDAR_FIRING_ANGLE,
    }
    assert tuple(made.GetAttribute("omni:sensor:frameRate").Get()) == (10, 1)
    assert made.GetAttribute("omni:sensor:Core:includeInvalidPoints").Get() is False
    assert made.GetAttribute("omni:sensor:Core:farRangeM").Get() == 70.0
    assert list(made.GetAttribute("omni:sensor:Core:emitterState:s001:fireTimeNs").Get()) == [
        0,
        5,
        9,
    ]
    # Whole scans, as x, y, z in its own axes, whatever its lidar says.
    assert made.GetAttribute("omni:sensor:Core:elementsCoordsType").Get() == "CARTESIAN"
    assert made.GetAttribute("omni:sensor:Core:outputFrameOfReference").Get() == "SENSOR"
    assert made.GetAttribute("omni:sensor:Core:partialOutputs").Get() is False
    # Turned so ovrtx's sensor axes (ahead down the prim's -z, its +y up) are the lidar's own.
    assert tuple(made.GetAttribute("xformOp:rotateXYZ").Get()) == (90, 0, -90)
    assert list(made.GetAttribute("xformOpOrder").Get()) == ["xformOp:rotateXYZ"]
    assert not stage.GetRootLayer().GetPrimAtPath(made.GetPath())  # the stage's file as it was


def test_a_spinning_lidars_firing_rate_is_the_angle_it_turns_between_firings(tmp_path):
    stage, prim = _lidar(tmp_path, spins=True)
    made = ovrtx_usd.sensor_lidar(stage, prim, 10.0)
    # 20,000 firings a second, 10 turns a second: 0.18 degrees a firing.
    assert made.GetAttribute(ovrtx_usd.LIDAR_FIRING_ANGLE).Get() == pytest.approx(0.18)


@pytest.mark.parametrize("hits", [False, True])
def test_a_lidars_render_product_is_a_point_cloud_of_what_a_capture_keeps(hits):
    text = ovrtx_usd.build_lidar_product_as_string(
        "RenderLidar_2", "/World/Rig/lidar_CaptureLidar", hits
    )
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(text)
    stage = Usd.Stage.Open(layer)
    product = stage.GetPrimAtPath("/RenderLidar_2/RenderProduct")
    assert product.GetRelationship("camera").GetTargets() == ["/World/Rig/lidar_CaptureLidar"]
    cloud = stage.GetPrimAtPath("/RenderLidar_2/Vars/PointCloud")
    assert cloud.GetAttribute("sourceName").Get() == "PointCloud"
    channels = list(cloud.GetAttribute("channels").Get())
    assert channels[:3] == ["Coordinates", "Intensity", "TimeOffsetNs"]
    assert ("ObjectId" in channels) == hits
    ordered = [str(t) for t in product.GetRelationship("orderedVars").GetTargets()]
    assert ordered == ["/RenderLidar_2/Vars/PointCloud"]  # the IDs' names are another product's


def test_a_scenes_objects_are_named_by_a_render_product_of_its_own():
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(ovrtx_usd.build_names_product_as_string("RenderNames_5"))
    stage = Usd.Stage.Open(layer)
    product = stage.GetPrimAtPath("/RenderNames_5/RenderProduct")
    assert product.GetRelationship("camera").GetTargets() == ["/RenderNames_5/Camera"]
    assert stage.GetPrimAtPath("/RenderNames_5/Camera").IsA(UsdGeom.Camera)
    source = stage.GetPrimAtPath("/RenderNames_5/Vars/StableIdMap").GetAttribute("sourceName")
    assert source.Get() == "StableIdMap"


def test_an_eye_has_its_cameras_lens_schema_though_this_usd_does_not_know_it(tmp_path):
    from darkroom import eyes

    stage, prim = _stage(tmp_path)
    prim.AddAppliedSchema("OmniLensDistortionOpenCvFisheyeAPI")
    prim.AddAppliedSchema("PhysxRigidBodyAPI")  # physics is not a camera's eye's
    made = eyes.eye(stage, prim)
    assert "OmniLensDistortionOpenCvFisheyeAPI" in _schemas(made)
    assert "PhysxRigidBodyAPI" not in _schemas(made)


def test_an_eye_clips_as_near_as_its_camera_down_to_a_few_centimetres_and_a_sky_further(tmp_path):
    from pxr import Gf

    from darkroom import eyes

    stage, prim = _stage(tmp_path)
    UsdGeom.Camera(prim).GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 500.0))
    near, far = UsdGeom.Camera(eyes.eye(stage, prim)).GetClippingRangeAttr().Get()
    assert (near, far) == (pytest.approx(eyes.NEAR), 500.0) and eyes.NEAR == 0.05
    sky = eyes.sky(stage, UsdGeom.Xform.Define(stage, "/World/Rig/antenna").GetPrim())
    assert UsdGeom.Camera(sky).GetClippingRangeAttr().Get()[0] == pytest.approx(eyes.SKY_NEAR)


def test_a_sky_camera_is_a_fisheye_of_half_a_sphere_looking_along_its_antennas_z(tmp_path):
    from darkroom import eyes

    stage, _ = _stage(tmp_path)
    antenna = UsdGeom.Xform.Define(stage, "/World/Rig/antenna").GetPrim()
    made = eyes.sky(stage, antenna)
    assert str(made.GetPath()) == "/World/Rig/antenna_CaptureSky"
    assert "OmniLensDistortionOpenCvFisheyeAPI" in _schemas(made)
    lens = sensorkit.camera.lens(made)
    assert (lens.width, lens.height) == (sensorkit.sky.SIZE, sensorkit.sky.SIZE)
    assert lens.fx == pytest.approx(sensorkit.sky.FOCAL)  # 90 degrees off its axis at the edge
    eye = ovrtx_usd.sensor_eye(stage, made, 10.0, sensorkit.sky.SIZE)
    assert "OmniLensDistortionOpenCvFisheyeAPI" in _schemas(eye)  # as ovrtx renders it


def test_a_camera_blurring_motion_is_refused_more_steps_between_frames_than_ovrtx_blurs_right(
    tmp_path,
):
    from darkroom.ovrtx import BLUR_STEPS, refuse_blur

    stage, prim = _stage(tmp_path)  # its motion blur on
    front = sensorkit.Sensor(prim.GetPath(), "camera", "front", 30.0)
    refuse_blur(stage, [front], 30.0 * BLUR_STEPS)
    with pytest.raises(RuntimeError, match=r"front blurs motion.*480 steps a second put 16"):
        refuse_blur(stage, [front], 480.0)
    prim.GetAttribute(sensorkit.camera.MOTION_BLUR).Set(False)
    refuse_blur(stage, [front], 480.0)  # no blur: as many steps as a rolling shutter wants
