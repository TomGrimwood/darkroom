"""Darkroom's own light, for stills lit the same whatever the scene (`darkroom.stills`' `sun`):
by `ovrtx.stills`."""

from __future__ import annotations

#: Darkroom's own light for stills that are lit the same whatever the scene (maps from above):
#: a sun this bright, this wide (degrees), and a soft sky.
SUN_INTENSITY, SUN_ANGLE, SKY_INTENSITY = 3000.0, 0.53, 600.0


def own_light(stage, elevation: float, azimuth: float) -> None:
    """The scene's lights off and darkroom's own light on, for these stills only (the session layer): a sun
    `elevation` degrees up from `azimuth` (degrees clockwise from north, +Y; Z up), casting
    shadows, and a soft white sky."""
    import math

    from pxr import Gf, Usd, UsdGeom, UsdLux

    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                prim.SetActive(False)
        UsdGeom.Xform.Define(stage, "/CaptureStillsLight")
        sky = UsdLux.DomeLight.Define(stage, "/CaptureStillsLight/Sky")
        sky.CreateIntensityAttr(SKY_INTENSITY)
        sun = UsdLux.DistantLight.Define(stage, "/CaptureStillsLight/Sun")
        sun.CreateIntensityAttr(SUN_INTENSITY)
        sun.CreateAngleAttr(SUN_ANGLE)
        e, a = math.radians(elevation), math.radians(azimuth)
        towards_sun = Gf.Vec3d(math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))
        # A distant light shines down its -Z: turned so it shines away from the sun.
        turn = Gf.Rotation(Gf.Vec3d(0, 0, -1), -towards_sun)
        UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf(turn.GetQuat()))
