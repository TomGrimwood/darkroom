from pxr import Sdf, Usd, UsdGeom

from darkroom import sensorkit


def _moving_frame(layer):
    stage = Usd.Stage.Open(layer)
    UsdGeom.Scope.Define(stage, "/Drone/Frames")
    frame = UsdGeom.Xform.Define(stage, "/Drone/Frames/Gimbal")
    op = frame.AddTranslateOp()
    op.Set((0, 0, 0), 0)
    op.Set((4, 0, 0), 10)
    UsdGeom.Xform.Define(stage, "/Drone/NotAFrame")
    return stage


def test_frames_are_xforms_in_a_frames_scope():
    stage = _moving_frame(Sdf.Layer.CreateAnonymous(".usda"))
    assert sensorkit.frames(stage) == [Sdf.Path("/Drone/Frames/Gimbal")]
