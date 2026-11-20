"""darkroom.surroundings and what truth.write records besides sensors: the carrier and its tip, its mesh,
the place near where it went, and the UTC start."""

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

from darkroom import surroundings, truth


def _quad(stage, path, corners, at=(0.0, 0.0, 0.0)):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*at))
    return mesh


def _stage():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(48)
    stage.SetTimeCodesPerSecond(24)
    # The place: ground along the way, a shed 300 m off, a sky's dome a kilometre up.
    _quad(stage, "/World/Ground", [(-20, -5, 0), (20, -5, 0), (20, 5, 0), (-20, 5, 0)])
    _quad(stage, "/World/Shed", [(0, 0, 0), (3, 0, 0), (3, 0, 3), (0, 0, 3)], at=(300, 0, 0))
    _quad(
        stage,
        "/World/Dome",
        [(-1e5, -1e5, 1e3), (1e5, -1e5, 1e3), (1e5, 1e5, 1e3), (-1e5, 1e5, 1e3)],
    )
    # Something moving through it (not the place), and the pole carried along the way.
    cart = UsdGeom.Xform.Define(stage, "/World/Cart")
    op = cart.AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, 0), 0)
    op.Set(Gf.Vec3d(5, 0, 0), 48)
    _quad(stage, "/World/Cart/Body", [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])
    pole = UsdGeom.Xform.Define(stage, "/World/Walk/Props/Pole")
    move = pole.AddTranslateOp()
    move.Set(Gf.Vec3d(-10, 0, 1.3), 0)
    move.Set(Gf.Vec3d(10, 0, 1.3), 48)
    UsdGeom.Xform.Define(stage, "/World/Walk/Props/Pole/Geometry")
    _quad(
        stage,
        "/World/Walk/Props/Pole/Geometry/Rod",
        [(0, 0, -1.3), (0.02, 0, -1.3), (0.02, 0, 0.6), (0, 0, 0.6)],
    )
    UsdGeom.Scope.Define(stage, "/World/Walk/Props/Pole/Frames")
    UsdGeom.Xform.Define(stage, "/World/Walk/Props/Pole/Frames/Mount")
    tip = UsdGeom.Xform.Define(stage, "/World/Walk/Props/Pole/Frames/Tip")
    tip.AddTranslateOp().Set(Gf.Vec3d(0, 0, -1.3))
    UsdGeom.Xform.Define(stage, "/World/Walk/Props/Pole/Frames/Mount/Antenna")
    env = stage.DefinePrim("/Environment", "Xform")
    env.CreateAttribute("date", Sdf.ValueTypeNames.String).Set("2025-10-28")
    env.CreateAttribute("time:current", Sdf.ValueTypeNames.Float).Set(14.5)
    env.CreateAttribute("location:longitude", Sdf.ValueTypeNames.Double).Set(-105.0)
    return stage


def test_the_carrier_its_tip_its_mesh_and_when():
    stage = _stage()
    bodies = truth.carriers(stage, ["/World/Walk/Props/Pole/Frames/Mount/Antenna"])
    assert bodies == {
        "Pole": "/World/Walk/Props/Pole",
        "Pole_tip": "/World/Walk/Props/Pole/Frames/Tip",
    }
    rod = surroundings.model(stage, bodies["Pole"], 24.0)
    assert rod is not None and abs(rod.points[:, 2].min() + 1.3) < 1e-3  # in the pole's own frame
    # 14:30 solar time at 105 W is 21:30 UTC.
    assert truth.start_utc(stage, 60.0) == "2025-10-28T21:31:00+00:00"


def test_the_place_near_the_way_is_still_meshes_only():
    stage = _stage()
    way = np.array([[x, 0.0, 1.3] for x in np.linspace(-10, 10, 21)])
    said = []
    place = surroundings.nearby(stage, way, within=8.0, cell=0.5, said=said)
    assert place is not None and not said
    lo, hi = place.points.min(0), place.points.max(0)
    assert (
        lo[0] >= -20.1 and hi[0] <= 20.1 and hi[2] < 5
    )  # the ground: not the shed, the dome, the cart
    assert len(place.normals) == len(place.points) and np.allclose(
        np.abs(place.normals[:, 2]), 1, atol=1e-3
    )


def test_thinning_merges_points_and_drops_collapsed_triangles():
    x, y = np.meshgrid(np.linspace(0, 1, 11), np.linspace(0, 1, 11))
    points = np.stack([x.ravel(), y.ravel(), np.zeros(121)], 1)
    i = np.arange(10)[:, None] * 11 + np.arange(10)[None]
    i = i.ravel()
    faces = np.concatenate([np.stack([i, i + 1, i + 12], 1), np.stack([i, i + 12, i + 11], 1)])
    p, f = surroundings.thin(points, faces, 0.25)
    assert len(p) < 40 and 0 < len(f) < len(faces)
    assert (f[:, 0] != f[:, 1]).all() and (f[:, 1] != f[:, 2]).all()
