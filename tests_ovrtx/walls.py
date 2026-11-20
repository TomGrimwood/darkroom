"""A wall 5 m ahead of three cameras, to check a camera's distances are from the camera, along
each pixel's ray, in metres, whatever its lens and near clip: a plain camera, an OmniLens
pinhole and an OmniLens fisheye (`test_ovrtx_capture`)."""

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdSemantics

from darkroom import format, sensorkit

AHEAD, NEAR = 5.0, 0.5
WIDTH, HEIGHT, FOCAL, ACROSS = 480, 270, 24.0, 36.0


def _lens(prim: Usd.Prim, model: str) -> None:
    fx = FOCAL * WIDTH / ACROSS
    prim.AddAppliedSchema(
        "OmniLensDistortionOpenCvPinholeAPI"
        if model == "opencvPinhole"
        else "OmniLensDistortionOpenCvFisheyeAPI"
    )
    prim.CreateAttribute(sensorkit.camera.MODEL, Sdf.ValueTypeNames.Token).Set(model)
    at = f"omni:lensdistortion:{model}:"
    prim.CreateAttribute(at + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(WIDTH, HEIGHT))
    for k, v in {"fx": fx, "fy": fx, "cx": WIDTH / 2, "cy": HEIGHT / 2}.items():
        prim.CreateAttribute(at + k, Sdf.ValueTypeNames.Float).Set(v)
    for k in sensorkit.camera.DISTORTION[model]:
        prim.CreateAttribute(at + k, Sdf.ValueTypeNames.Float).Set(0.0)


def wall(folder: Path) -> Path:
    """A wall 5 m ahead of three cameras looking along +x, each clipped at `NEAR`."""
    layer = Sdf.Layer.CreateNew(str(folder / "wall.usda"))
    layer.timeCodesPerSecond = 60
    layer.startTimeCode, layer.endTimeCode = 0, 30
    layer.customLayerData = {
        format.SEGMENT_TAXONOMY: "class",
        format.GEOMETRY_CAMERAS: "plain,pinhole,fisheye",
    }
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1000)
    cube = UsdGeom.Cube.Define(stage, "/World/Wall")
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(AHEAD + 0.5, 0, 1.5))
    cube.AddScaleOp().Set(Gf.Vec3f(1.0, 60.0, 60.0))
    UsdSemantics.LabelsAPI.Apply(cube.GetPrim(), "class").CreateLabelsAttr(["wall"])
    UsdGeom.Xform.Define(stage, "/World/Rig").AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.5))
    for name in ("plain", "pinhole", "fisheye"):
        camera = UsdGeom.Camera.Define(stage, f"/World/Rig/{name}")
        camera.CreateFocalLengthAttr(FOCAL)
        camera.CreateHorizontalApertureAttr(ACROSS)
        camera.CreateVerticalApertureAttr(ACROSS * HEIGHT / WIDTH)
        camera.CreateClippingRangeAttr(Gf.Vec2f(NEAR, 1000.0))
        camera.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, -90))  # down its -z, +y up: along +x, z up
        if name != "plain":
            _lens(camera.GetPrim(), "opencvPinhole" if name == "pinhole" else "opencvFisheye")
        sensorkit.mark(layer, f"/World/Rig/{name}", "camera", 10.0)
    layer.Save()
    return folder / "wall.usda"


def check(out: Path) -> None:
    """Each camera's distance at the middle of its picture is the wall's, to a couple of
    millimetres; a pinhole's across all of it, along each ray."""
    meta = format.read_meta(out)
    for name in ("plain", "pinhole", "fisheye"):
        s = meta.sensors[name].settings
        far = format.read_distances(out, name, 1)
        assert far is not None, name
        row, column = int(s["cy"]), int(s["cx"])
        middle = np.median(far[row - 3 : row + 3, column - 3 : column + 3])
        assert abs(middle - AHEAD) < 0.002, (name, middle)
        if name != "fisheye":
            v, u = np.mgrid[: s["height"], : s["width"]] + 0.5
            along = np.sqrt(1 + ((u - s["cx"]) / s["fx"]) ** 2 + ((v - s["cy"]) / s["fy"]) ** 2)
            assert np.abs(far - AHEAD * along).max() < 0.003, name
