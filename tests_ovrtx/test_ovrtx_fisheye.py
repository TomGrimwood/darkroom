"""A fisheye by ovrtx for real (the GPU, seconds): ovrtx's own OpenCV fisheye fills its frame,
corners and all, where OpenCV's model puts every point; no lookup-table lens needed.

    uv run --extra ovrtx pytest tests_ovrtx
"""

import math
from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdSemantics

pytest.importorskip("ovrtx")

import darkroom
from darkroom import Options, format, sensorkit
from darkroom.renderers import OVRTXRendererCfg


def _ground(folder: Path, name: str) -> tuple[Sdf.Layer, Usd.Stage]:
    layer = Sdf.Layer.CreateNew(str(folder / f"{name}.usda"))
    layer.timeCodesPerSecond = 60
    layer.startTimeCode, layer.endTimeCode = 0, 120
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1000)
    return layer, stage


W, H, FX = 640, 480, 140.0  # corners 164 degrees off the axis


def _fisheye(folder: Path) -> tuple[Path, list[tuple[str, float, float]]]:
    """Balls 10 m off at known angles round an equidistant OpenCV fisheye looking down,
    each labelled; past 90 degrees off its axis and into the frame's corners."""
    layer, stage = _ground(folder, "fisheye")
    layer.customLayerData = {format.SEGMENT_TAXONOMY: "class"}
    balls = []
    for t in (0, 45, 90, 120, 150, 160):
        for j, p in enumerate((36.87, 143.13, 216.87, 323.13) if t else (0.0,)):
            name = f"b{t}_{j}"
            th, ph = math.radians(t), math.radians(p)
            ball = UsdGeom.Sphere.Define(stage, f"/World/{name}")
            ball.CreateRadiusAttr(0.12)
            ball.AddTranslateOp().Set(
                Gf.Vec3d(
                    10 * math.sin(th) * math.cos(ph),
                    10 * math.sin(th) * math.sin(ph),
                    -10 * math.cos(th),
                )
            )
            UsdSemantics.LabelsAPI.Apply(ball.GetPrim(), "class").CreateLabelsAttr([name])
            balls.append((name, W / 2 + FX * th * math.cos(ph), H / 2 - FX * th * math.sin(ph)))
    camera = UsdGeom.Camera.Define(stage, "/World/fisheye")
    camera.CreateFocalLengthAttr(FX * 36.0 / W)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(36.0 * H / W)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    prim = camera.GetPrim()
    prim.AddAppliedSchema("OmniLensDistortionOpenCvFisheyeAPI")
    prim.CreateAttribute(sensorkit.camera.MODEL, Sdf.ValueTypeNames.Token).Set("opencvFisheye")
    at = "omni:lensdistortion:opencvFisheye:"
    prim.CreateAttribute(at + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(W, H))
    for key, value in {
        "fx": FX,
        "fy": FX,
        "cx": W / 2,
        "cy": H / 2,
        "k1": 0,
        "k2": 0,
        "k3": 0,
        "k4": 0,
    }.items():
        prim.CreateAttribute(at + key, Sdf.ValueTypeNames.Float).Set(float(value))
    sensorkit.mark(layer, "/World/fisheye", "camera", 10.0)
    layer.Save()
    return folder / "fisheye.usda", balls


def test_a_fisheye_puts_every_point_where_opencvs_model_does_corners_and_all(tmp_path):
    """ovrtx's own OpenCV fisheye, not cut to a circle as Isaac Sim's (IsaacSim issue #838):
    every ball where the model puts it, to 164 degrees off the axis in the frame's corners."""
    stage_file, balls = _fisheye(tmp_path)
    out = darkroom.capture(
        stage_file,
        tmp_path / "capture",
        Options(window=(0.2, 0.4), kinds=["camera"], renderer_cfg=OVRTXRendererCfg()),
        on_line=None,
    )
    assert format.read_meta(out).sensors["fisheye"].settings["rendered"] == "opencvFisheye"
    labels, names = format.read_segments(out, "fisheye", 1)
    checked = 0
    for name, eu, ev in balls:
        mask = np.isin(labels, [i for i, n in enumerate(names) if n.endswith(name)])
        if not (8 <= eu < W - 8 and 8 <= ev < H - 8):
            continue  # cut by the frame's edge: its centre is not the ball's
        assert mask.sum() >= 2, name
        v, u = np.nonzero(mask)
        assert math.hypot(u.mean() + 0.5 - eu, v.mean() + 0.5 - ev) < 1.0, name
        checked += 1
    assert checked >= 17  # out to 160 degrees off the axis
