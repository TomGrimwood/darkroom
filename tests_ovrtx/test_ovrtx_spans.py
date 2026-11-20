"""A span of the window in another look, by ovrtx for real (the GPU, seconds): only its frames
in its look, against a camera of fixed exposure on grey ground.

    uv run --extra ovrtx pytest tests_ovrtx
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

pytest.importorskip("ovrtx")

import darkroom
from darkroom import Options, Span, format, sensorkit
from darkroom.renderers import OVRTXRendererCfg

RATE = 60.0


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


def _spans(folder: Path) -> Path:
    """A camera of fixed exposure looking down at grey ground, a frame every step."""
    layer, stage = _ground(folder, "spans")
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.18, 0.18)])
    ground.AddScaleOp().Set(Gf.Vec3f(100, 100, 0.05))
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.5))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
    for key, value in {"time": 0.005, "iso": 100.0, "fStop": 8.0}.items():
        camera.GetPrim().CreateAttribute(f"exposure:{key}", Sdf.ValueTypeNames.Float).Set(value)
    sensorkit.mark(layer, "/World/Camera", "camera", RATE)
    layer.Save()
    return folder / "spans.usda"


def test_a_span_renders_in_its_look_and_only_its_frames_do(tmp_path):
    """A span at ISO 400 (the camera's attribute) and one with the tone mapper "raw" (its
    render product's: white here): the frames in each, and none either side, have its look."""
    during = [
        Span(0.6, 0.65, {"/rtx/post/tonemap/filmIso": 400.0}),
        Span(0.7, 0.75, {"/rtx/post/tonemap/op": "raw"}),
    ]
    out = darkroom.capture(
        _spans(tmp_path),
        tmp_path / "capture",
        Options(
            window=(0.5, 0.8), kinds=["camera"], during=during, renderer_cfg=OVRTXRendererCfg()
        ),
        on_line=None,
    )
    name = next(n for n, s in format.read_meta(out).sensors.items() if s.kind == "camera")
    seconds = format.read_poses(out, name).seconds + 0.5
    means = [
        np.asarray(Image.open(format.frame_path(out, name, k)).convert("L"), float).mean()
        for k in range(len(seconds))
    ]
    base = means[0]
    for t, mean in zip(seconds, means, strict=True):
        if 0.6 - 1e-6 <= t < 0.65 - 1e-6:
            assert mean > base * 1.8, (t, mean, base)  # 4 times the exposure
        elif 0.7 - 1e-6 <= t < 0.75 - 1e-6:
            assert mean > 250, (t, mean)
        else:
            assert abs(mean - base) < 1.0, (t, mean, base)
