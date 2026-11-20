"""A GNSS antenna's sky rendered by ovrtx for real (the GPU, seconds), held to
known answers: a box to the antenna's east and one to its north block
the sky where `darkroom.sensorkit.sky` says they are, one to its west labelled vegetation is
leaves, and a sky dome labelled sky around it all is open sky. No Isaac Sim is started.

    uv run --extra ovrtx pytest tests_ovrtx
"""

from pathlib import Path

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom

pytest.importorskip("ovrtx")

from darkroom import Options, capture, format, sensorkit
from darkroom.renderers import OVRTXRendererCfg
from darkroom.sensorkit import sky

SITE = "OmniWGS84ReferencePositionAPI"


def _stage(folder: Path) -> Path:
    """An antenna 1.5 m up for half a second; 6 m boxes 8 m up and 8 m east, north and west
    of it (the west one a tree), under a sky dome 400 m out."""
    layer = Sdf.Layer.CreateNew(str(folder / "sky.usda"))
    layer.timeCodesPerSecond = 60
    layer.startTimeCode, layer.endTimeCode = 0, 30
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Scope.Define(stage, "/Thing/Frames")
    UsdGeom.Xform.Define(stage, "/Thing/Frames/Mount").AddTranslateOp().Set((0, 0, 1.5))
    UsdGeom.Xform.Define(stage, "/Thing/Frames/Mount/Antenna")
    sensorkit.mark(layer, "/Thing/Frames/Mount/Antenna", "gnss", 10)
    for name, where in (("East", (8, 0, 9.5)), ("North", (0, 8, 9.5)), ("Tree", (-8, 0, 9.5))):
        cube = UsdGeom.Cube.Define(stage, f"/World/{name}")
        cube.CreateSizeAttr(6.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*where))
    UsdGeom.Sphere.Define(stage, "/Environment/sky").CreateRadiusAttr(400.0)
    for path, label in (("/World/Tree", "vegetation"), ("/Environment/sky", "sky")):
        prim = stage.GetPrimAtPath(path)
        prim.AddAppliedSchema("SemanticsLabelsAPI:class")
        prim.CreateAttribute("semantics:labels:class", Sdf.ValueTypeNames.TokenArray).Set([label])
    site = stage.DefinePrim("/Site", "Xform")
    site.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=[SITE]))
    site.CreateAttribute(
        "omni:geospatial:wgs84:reference:referencePosition", Sdf.ValueTypeNames.Double3
    ).Set((-36.85, 174.76, 40.0))
    layer.Save()
    return folder / "sky.usda"


def test_boxes_block_the_sky_where_they_are_and_no_isaac_sim_is_started(tmp_path):
    started = []
    out = capture(
        _stage(tmp_path),
        tmp_path / "capture",
        Options(renderer_cfg=OVRTXRendererCfg()),
        on_line=None,
        on_progress=lambda p: started.append(p.step) if p.step.startswith("starting") else None,
    )
    assert set(started) == {"starting ovrtx"}
    name = next(n for n, s in format.read_meta(out).sensors.items() if s.kind == "gnss")
    view = format.read_sky(out, name, 0)
    towards = (
        np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1], [0, 0, 1]])
        / np.sqrt([2, 2, 2, 2, 1])[:, None]
    )
    seen = sky.seen(view, towards)
    assert list(seen) == [sky.BLOCKED, sky.BLOCKED, sky.LEAVES, sky.OPEN, sky.OPEN], seen
    assert 0.5 < sky.sky_share(view, np.eye(3), through=0.0) < 0.9
    got = format.read_arrays(out, name, format.GNSS)
    assert len(got["sky"]) == 6 and all(format.sky_path(out, name, k).is_file() for k in range(6))
    assert (got["sky"] > 0).all() and (got["sky"] < 0.9).all()  # boxes and a tree in the way
    # Past 90 degrees off its axis a view is nothing hit, as Isaac Sim's is.
    assert (view[sky.directions()[..., 2] <= 0] == sky.OPEN).all()


def test_every_view_is_the_sky_at_its_fix_the_first_too(tmp_path):
    """An antenna moving along x at 5 m/s out from under a roof 3 m over it (a 1 m slab over
    x < 0): each pixel is blocked exactly where its ray meets the roof, worked out at the
    antenna's place at each fix. A sky camera's first view showed the stage as it was loaded."""
    layer = Sdf.Layer.CreateNew(str(tmp_path / "roof.usda"))
    layer.timeCodesPerSecond = 60
    layer.startTimeCode, layer.endTimeCode = 0, 120
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Scope.Define(stage, "/Thing/Frames")
    move = UsdGeom.Xform.Define(stage, "/Thing/Frames/Mount").AddTranslateOp()
    for code in (0, 120):
        move.Set(Gf.Vec3d(5.0 * (code / 60 - 1.0), 0, 1.5), code)
    UsdGeom.Xform.Define(stage, "/Thing/Frames/Mount/Antenna")
    sensorkit.mark(layer, "/Thing/Frames/Mount/Antenna", "gnss", 10)
    roof = UsdGeom.Cube.Define(stage, "/World/Roof")
    roof.AddTranslateOp().Set(Gf.Vec3d(-200, 0, 5.0))
    roof.AddScaleOp().Set(Gf.Vec3f(200, 200, 0.5))
    site = stage.DefinePrim("/Site", "Xform")
    site.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=[SITE]))
    site.CreateAttribute(
        "omni:geospatial:wgs84:reference:referencePosition", Sdf.ValueTypeNames.Double3
    ).Set((-36.85, 174.76, 40.0))
    layer.Save()
    out = capture(
        tmp_path / "roof.usda",
        tmp_path / "capture",
        Options(window=(0.5, 1.5), kinds=["gnss"], renderer_cfg=OVRTXRendererCfg()),
        on_line=None,
    )
    d = sky.directions()
    up = d[..., 2] > 0
    name = next(n for n, s in format.read_meta(out).sensors.items() if s.kind == "gnss")
    got = format.read_arrays(out, name, format.GNSS)
    assert len(got["seconds"]) == 11
    alike = []
    for k, t in enumerate(got["seconds"]):
        x = 5.0 * (0.5 + float(t) - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            hit = x + 3.0 * d[..., 0] / d[..., 2]  # where each ray meets the roof's underside
            face = x * d[..., 2] / -d[..., 0]  # how high it passes the roof's edge (x = 0)
            across = 3.0 * d[..., 1] / d[..., 2]
        blocked = up & (hit < 0) & (hit > -400) & (np.abs(across) < 200)
        if x > 0:
            blocked |= up & (d[..., 0] < 0) & (face >= 3.0) & (face <= 4.0)
        view = format.read_sky(out, name, k)
        alike.append(float(((view == sky.BLOCKED) == blocked)[up].mean()))
    assert min(alike) > 0.999, alike
