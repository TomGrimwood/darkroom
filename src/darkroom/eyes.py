"""The camera that renders for a camera on a stage: an `eye`.

A copy of the camera beside it (same parent, same transform), in the session layer, with
everything authored on it but its visibility. Not under it: a camera asset is often authored
invisible, and a camera under an invisible prim renders black. RTX's own settings for it
(`omni:rtx:*`) go on its render product instead: authored with their API schema on a Camera
they crash RTX. The stage's files are never changed.

And how each fisheye camera is rendered (`lenses`), set on the camera before its eye is made.
"""

from __future__ import annotations

from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom

from darkroom import sensorkit

#: What RTX's own settings on a camera start with.
RTX = "omni:rtx:"
#: What an eye is named: its camera's name and this.
EYE = "_CaptureEye"
#: The nearest a camera renders, in metres: nearer, a camera inside the kit it is mounted on
#: renders a black frame.
NEAR = 0.05


def eye(stage: Usd.Stage, prim: Usd.Prim, leave: tuple[str, ...] = ()) -> Usd.Prim:
    """The camera that renders for `prim`: a copy of it beside it, in the session layer, with
    everything authored on it but its transform ops, its RTX settings, its visibility and the
    attributes whose names start with one of `leave`."""
    path = prim.GetParent().GetPath().AppendChild(f"{prim.GetName()}{EYE}")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        made = UsdGeom.Camera.Define(stage, path).GetPrim()
        # Every API schema applied, as authored: `GetAppliedSchemas` leaves out those this
        # process's USD has no definition of (NVIDIA's lens schemas, where they are not registered), and a
        # fisheye without its schema renders as a pinhole.
        for api in prim.GetPrimTypeInfo().GetAppliedAPISchemas():
            if not api.startswith("Physx"):
                made.AddAppliedSchema(api)
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if (
                name.startswith(("xformOp", RTX, "physx", "sensorkit:", "visibility", *leave))
                or not attribute.HasAuthoredValue()
            ):
                continue
            value = attribute.Get()
            if isinstance(value, Sdf.AssetPath):
                value = Sdf.AssetPath(value.resolvedPath or value.path)
            made.CreateAttribute(name, attribute.GetTypeName()).Set(value)
        # Where the camera is in its parent, as one transform op made by USD's own API.
        local = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
        UsdGeom.Xformable(made).AddTransformOp().Set(local)
        clipping = UsdGeom.Camera(made).GetClippingRangeAttr()
        near, far = clipping.Get()
        clipping.Set(Gf.Vec2f(max(near, NEAR), far))
    return made


def lenses(stage: Usd.Stage, lenses: dict[str, dict[str, Any]]) -> None:
    """Each fisheye camera rendered as `lenses` says, by prim path (`run`): as ovrtx's own
    OpenCV fisheye, whatever lens model it is authored with (a library camera may say "lut",
    its OpenCV fisheye attributes beside). Set in the session layer: the stage's files are
    left as they are."""
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path, lens in lenses.items():
            prim = stage.GetPrimAtPath(path)
            prim.CreateAttribute(sensorkit.camera.MODEL, Sdf.ValueTypeNames.Token).Set(
                lens["model"]
            )


#: The nearest a sky camera sees, in metres: past the antenna's own housing and the receiver
#: it sits on, which signals pass through or come from above.
SKY_NEAR = 0.25
#: The farthest a sky camera sees, in metres.
SKY_FAR = 1.0e7
#: Any aperture: the focal length is made to agree with it.
APERTURE = 20.955


def sky(stage: Usd.Stage, prim: Usd.Prim) -> Usd.Prim:
    """The camera that renders the sky above the GNSS antenna `prim`, in the session layer:
    beside it (same parent), the antenna's transform turned to look along its +Z, with the
    equidistant fisheye lens `darkroom.sensorkit.sky` says, as OmniLens's OpenCV fisheye with
    no distortion."""
    path = prim.GetParent().GetPath().AppendChild(f"{prim.GetName()}_CaptureSky")
    size, lens = sensorkit.sky.SIZE, sensorkit.sky.LENS
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        camera = UsdGeom.Camera.Define(stage, path)
        made = camera.GetPrim()
        camera.CreateFocalLengthAttr(lens["fx"] * APERTURE / size)
        camera.CreateHorizontalApertureAttr(APERTURE)
        camera.CreateVerticalApertureAttr(APERTURE)
        camera.CreateClippingRangeAttr(Gf.Vec2f(SKY_NEAR, SKY_FAR))
        made.AddAppliedSchema("OmniLensDistortionOpenCvFisheyeAPI")
        made.CreateAttribute(sensorkit.camera.MODEL, Sdf.ValueTypeNames.Token).Set(
            sensorkit.camera.FISHEYE
        )
        at = f"omni:lensdistortion:{sensorkit.camera.FISHEYE}:"
        made.CreateAttribute(at + "imageSize", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(size, size))
        for k, value in lens.items():
            made.CreateAttribute(at + k, Sdf.ValueTypeNames.Float).Set(float(value))
        for k in sensorkit.camera.DISTORTION[sensorkit.camera.FISHEYE]:
            made.CreateAttribute(at + k, Sdf.ValueTypeNames.Float).Set(0.0)
        # A camera looks down its -Z with +Y up: half a turn about X looks along the antenna's
        # +Z, its image's columns along the antenna's +X and rows along its +Y.
        turn = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180))
        local = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
        UsdGeom.Xformable(made).AddTransformOp().Set(turn * local)
    return made
