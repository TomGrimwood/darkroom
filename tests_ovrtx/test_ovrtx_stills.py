"""Stills by ovrtx for real (the GPU, seconds): `darkroom.stills` with an `OVRTXRendererCfg`,
no Isaac Sim, against where a known square must be.

    uv run --extra ovrtx pytest tests_ovrtx
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

pytest.importorskip("ovrtx")

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


def test_stills_are_where_the_camera_puts_them_orthographic_or_not(tmp_path):
    """A red square 0.2 m across at (1, 0.5) on the ground: from 10 m up through 24 mm over
    36 mm (960x540) its centre at (544, 238); from 20 m up orthographic 4 m across (400x400)
    at (300, 150), 20 pixels across; with the near clip past it, gone."""
    from dataclasses import replace

    from darkroom.stills import Camera, Shot, stills

    layer, stage = _ground(tmp_path, "square")
    square = UsdGeom.Cube.Define(stage, "/World/Square")
    square.CreateDisplayColorAttr([Gf.Vec3f(1, 0, 0)])
    square.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.5, 0))
    square.AddScaleOp().Set(Gf.Vec3f(0.1, 0.1, 0.01))
    layer.Save()
    persp = Camera(position=(0, 0, 10), resolution=(960, 540))
    ortho = Camera(position=(0, 0, 20), projection="orthographic", width=4.0, resolution=(400, 400))
    shots = [
        Shot(persp, str(tmp_path / "persp.png")),
        Shot(ortho, str(tmp_path / "ortho.png")),
        Shot(replace(ortho, near=20.5), str(tmp_path / "cut.png")),
    ]
    stills(tmp_path / "square.usda", shots, renderer_cfg=OVRTXRendererCfg())

    def red(name):
        image = np.asarray(Image.open(tmp_path / name).convert("RGB"), float)
        return np.nonzero(image[..., 0] - image[..., 1] > 60)

    v, u = red("persp.png")
    assert abs(u.mean() + 0.5 - 544) < 0.5 and abs(v.mean() + 0.5 - 238) < 0.5
    v, u = red("ortho.png")
    assert abs(u.mean() + 0.5 - 300) < 0.5 and abs(v.mean() + 0.5 - 150) < 0.5
    assert u.max() - u.min() + 1 == 20
    assert len(red("cut.png")[0]) == 0
