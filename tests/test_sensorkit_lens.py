"""A fisheye camera's image circle, and its lens as LUT textures."""

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

from darkroom.sensorkit import camera


def _fisheye(stage: Usd.Stage, path: str = "/Rig/Cam", model: str = "opencvFisheye") -> Usd.Prim:
    """A 64 x 48 fisheye camera, 120 deg across, as Lila's assemblies write them."""
    prim = UsdGeom.Camera.Define(stage, path).GetPrim()
    cam = UsdGeom.Camera(prim)
    f = 32 / np.radians(60)  # 60 deg at the edge, 32 px out
    cam.GetFocalLengthAttr().Set(1.0)
    cam.GetHorizontalApertureAttr().Set(64 / f)
    cam.GetVerticalApertureAttr().Set(48 / f)
    at = "omni:lensdistortion:opencvFisheye:"
    prim.CreateAttribute(camera.MODEL, Sdf.ValueTypeNames.Token).Set(model)
    prim.CreateAttribute(at + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(64, 48))
    for name, value in {
        "fx": f,
        "fy": f,
        "cx": 32,
        "cy": 24,
        "k1": 0,
        "k2": 0,
        "k3": 0,
        "k4": 0,
    }.items():
        prim.CreateAttribute(at + name, Sdf.ValueTypeNames.Float).Set(value)
    return prim


def test_a_fisheyes_image_circle_is_its_attribute():
    stage = Usd.Stage.CreateInMemory()
    prim = _fisheye(stage)
    assert camera.lens(prim).circle == 0.0  # not said
    prim.CreateAttribute(camera.CIRCLE, Sdf.ValueTypeNames.Float).Set(100.0)
    assert camera.lens(prim).circle == 100.0


def test_a_lut_camera_is_read_as_the_opencv_fisheye_it_carries():
    stage = Usd.Stage.CreateInMemory()
    lens = camera.lens(_fisheye(stage, model="lut"))
    assert lens.model == camera.FISHEYE and (lens.width, lens.height) == (64, 48)


def test_a_cameras_auto_exposure_is_read_as_the_render_wide_settings_it_stands_for():
    from pxr import Sdf, Usd, UsdGeom

    from darkroom.sensorkit import camera as cameras

    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Camera.Define(stage, "/Camera").GetPrim()
    assert cameras.auto_exposure(prim) is None  # not asked for
    prim.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(True)
    low = prim.CreateAttribute(
        "omni:rtx:autoExposure:histogramClamping:minLuminance", Sdf.ValueTypeNames.Float
    )
    low.Set(10.0)
    assert cameras.auto_exposure(prim) == {
        "/rtx/post/histogram/enabled": True,
        "/rtx/post/histogram/minEV": 10.0,
    }
    prim.GetAttribute("omni:rtx:autoExposure:enabled").Set(False)
    assert cameras.auto_exposure(prim) is None
