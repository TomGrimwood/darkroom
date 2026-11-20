"""A camera's lens: its image in pixels, its pinhole and its distortion, from its OmniLens
attributes.

A Camera with no OmniLens lens is the undistorted pinhole its focal length and apertures say,
`WIDTH` pixels across, square pixels. One with an OmniLens lens is an OpenCV pinhole
(k1-k6, p1, p2) or an OpenCV fisheye (k1-k4), its intrinsics agreeing with its apertures;
RTX renders it distorted as its coefficients say.

When a frame is, and each of its rows (the one place that says so; renderers and readers go
by it):

- Frame n of a camera is at n / rate seconds of the capture's window: its pose in the
  capture (`format.Poses`) is the camera's then.
- A shutter open `exposure` seconds (`Lens.exposure`: the Camera's `shutter:open` to
  `shutter:close`, USD's own attributes; darkroom has no other) is open about that time: a
  frame's time is the middle of its exposure, the instant a picture that gathers no light
  over the exposure (RTX's are one instant a row) is least wrong for.
- A rolling shutter (`Lens.readout` seconds: `READOUT`) reads its rows top to bottom, the
  last at the frame's time: row i of `height` is at
  `n / rate - readout * (1 - i / (height - 1))`, the middle of that row's exposure, which
  opens `exposure / 2` before and closes `exposure / 2` after.

ovrtx is told a readout as line timings (`darkroom.ovrtx.ovrtx_usd.rolling_shutter`); a rig's
texture of row times (`SHUTTER_TEXTURE`, made by `usdsensors` for Isaac Sim) is left off the
eye. It does not blur by the exposure: RTX's motion blur is a post-process of its own
(`MOTION_BLUR`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from pxr import Usd, UsdGeom

WIDTH = 960
MODEL = "omni:lensdistortion:model"
PINHOLE = "opencvPinhole"
FISHEYE = "opencvFisheye"
#: RTX's lookup-table lens: rendered through two textures, described here by its OpenCV
#: fisheye attributes (a fisheye RTX's own OpenCV model would clip to a circle).
LUT = "lut"
#: A fisheye camera's image circle, the full angle (degrees) its lens gathers light over, as an
#: attribute on the camera (usdsensors' `sensor:camera:` namespace).
CIRCLE = "sensor:camera:imageCircleDegrees"
LENS = f"omni:lensdistortion:{PINHOLE}:"
#: Each OmniLens model rendered, and its distortion coefficients.
DISTORTION: dict[str, tuple[str, ...]] = {
    PINHOLE: ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"),
    FISHEYE: ("k1", "k2", "k3", "k4"),
}
#: A rolling shutter's readout, seconds (usdsensors' attribute): 0, a global shutter.
READOUT = "sensor:camera:readoutSeconds"
#: A rolling shutter's texture of row times (RTX's), and whether RTX blurs motion.
SHUTTER_TEXTURE = "shutterTimeTexturePath"
MOTION_BLUR = "omni:rtx:post:motionblur:enabled"
#: A camera's own auto exposure (`OmniRtxCameraAutoExposureAPI_1`), each attribute with the
#: render-wide setting it stands for (its doc's "deprecated carb setting path"). ovrtx takes
#: them per camera, on its render product, all in one pass.
AUTO_EXPOSURE = {
    "omni:rtx:autoExposure:enabled": "/rtx/post/histogram/enabled",
    "omni:rtx:autoExposure:adaptationSpeed": "/rtx/post/histogram/tau",
    "omni:rtx:autoExposure:histogramClamping:enabled": "/rtx/post/histogram/useExposureClamping",
    "omni:rtx:autoExposure:histogramClamping:maxLuminance": "/rtx/post/histogram/maxEV",
    "omni:rtx:autoExposure:histogramClamping:minLuminance": "/rtx/post/histogram/minEV",
    "omni:rtx:autoExposure:histogramFilter": "/rtx/post/histogram/filterType",
    "omni:rtx:autoExposure:whitePointScale": "/rtx/post/histogram/whiteScale",
}
#: How far, in pixels, the OmniLens intrinsics may be from what the apertures say.
AGREE = 0.5


@dataclass(frozen=True)
class Lens:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    #: Seconds the shutter is open (the Camera's shutter:open to shutter:close).
    exposure: float
    #: The OmniLens model (`PINHOLE` or `FISHEYE`), and its coefficients by name.
    model: str = PINHOLE
    distortion: dict[str, float] = field(default_factory=dict)
    #: A fisheye's image circle (`CIRCLE`), degrees; 0 when not known.
    circle: float = 0.0
    #: Seconds a rolling shutter takes to read its rows, top to bottom (`READOUT`); 0, a
    #: global shutter.
    readout: float = 0.0

    def settings(self) -> dict[str, Any]:
        return asdict(self)


def lens(prim: Usd.Prim) -> Lens:
    """`prim`'s lens. Raises `RuntimeError` unless it is a centred Camera and, with an
    OmniLens lens, an OpenCV pinhole or fisheye whose intrinsics agree with its apertures."""
    where = prim.GetPath()
    if not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"{where} is not a Camera.")
    camera = UsdGeom.Camera(prim)
    if (
        camera.GetHorizontalApertureOffsetAttr().Get()
        or camera.GetVerticalApertureOffsetAttr().Get()
    ):
        raise RuntimeError(f"{where} has an aperture offset; its pinhole has to be centred.")
    focal = camera.GetFocalLengthAttr().Get()
    across, tall = camera.GetHorizontalApertureAttr().Get(), camera.GetVerticalApertureAttr().Get()
    exposure = (camera.GetShutterCloseAttr().Get() - camera.GetShutterOpenAttr().Get()) / (
        prim.GetStage().GetTimeCodesPerSecond()
    )
    readout = float(prim.GetAttribute(READOUT).Get() or 0.0) if prim.HasAttribute(READOUT) else 0.0
    if prim.GetAttribute(MODEL).Get() is None:
        width, height = WIDTH, 2 * round(WIDTH * tall / across / 2)
        f = focal * width / across
        return Lens(width, height, f, f, width / 2, height / 2, exposure, readout=readout)

    def value(name: str) -> Any:
        found = prim.GetAttribute(name).Get()
        if found is None:
            raise RuntimeError(f"{where} has no {name}: it needs every OmniLens pinhole attribute.")
        return found

    model = value(MODEL)
    if model == LUT and prim.GetAttribute(f"omni:lensdistortion:{FISHEYE}:fx").Get() is not None:
        model = FISHEYE  # rendered through its textures, as its OpenCV fisheye describes it
    if model not in DISTORTION:
        raise RuntimeError(
            f"{where} has a {model} lens; one of {', '.join(DISTORTION)} is rendered."
        )
    at = f"omni:lensdistortion:{model}:"
    width, height = (int(n) for n in value(at + "imageSize"))
    fx, fy, cx, cy = (float(value(at + k)) for k in ("fx", "fy", "cx", "cy"))
    distortion = {k: float(value(at + k)) for k in DISTORTION[model]}
    pinhole = {
        "fx": focal * width / across,
        "fy": focal * height / tall,
        "cx": width / 2,
        "cy": height / 2,
    }
    said = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    off = {k: v for k, v in pinhole.items() if abs(said[k] - v) > AGREE}
    if off:
        raise RuntimeError(
            f"{where}: its intrinsics disagree with its focal length and apertures: "
            + ", ".join(f"{k} is {said[k]:g}, its pinhole's {v:g}" for k, v in off.items())
        )
    circle = image_circle(prim) if model == FISHEYE else 0.0
    return Lens(width, height, fx, fy, cx, cy, exposure, model, distortion, circle, readout)


def image_circle(prim: Usd.Prim) -> float:
    """Fisheye camera `prim`'s image circle, degrees (`CIRCLE`); 0 when it does not say."""
    attribute = prim.GetAttribute(CIRCLE)
    value = attribute.Get() if attribute else None
    return float(value) if value else 0.0


def outline(lens: Lens, reach: float = 0.5, steps: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """What camera `lens` sees, as line segments (starts, ends: n x 3) in the camera's own frame
    (USD's: looking down -Z, +Y up): a ray to each corner of its picture and its edge `reach`
    metres out. Its distortion is left out: the shape of its field, not its every pixel."""
    w, h = lens.width, lens.height
    edge = np.linspace(0.0, 1.0, steps + 1)
    border = np.concatenate(
        [
            np.stack([edge * w, np.zeros_like(edge)], -1),
            np.stack([np.full_like(edge, w), edge * h], -1),
            np.stack([(1 - edge) * w, np.full_like(edge, h)], -1),
            np.stack([np.zeros_like(edge), (1 - edge) * h], -1),
        ]
    )
    x, y = (border[:, 0] - lens.cx) / lens.fx, (border[:, 1] - lens.cy) / lens.fy
    if lens.model == FISHEYE:  # equidistant: the angle off its axis is the distance off centre
        r = np.hypot(x, y)
        theta = np.minimum(r, np.pi * 0.99)
        scale = np.where(r > 0, np.sin(theta) / np.maximum(r, 1e-12), 0.0)
        rays = np.stack([x * scale, y * scale, np.cos(theta)], -1)
    else:
        rays = np.stack([x, y, np.ones_like(x)], -1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    far = rays * reach * np.array([1.0, -1.0, -1.0])  # OpenCV's (+y down, +z ahead) to USD's
    corners = far[[0, steps + 1, 2 * (steps + 1), 3 * (steps + 1)]]
    starts = np.concatenate([np.zeros((4, 3)), far[:-1]])
    ends = np.concatenate([corners, far[1:]])
    return starts, ends


def auto_exposure(prim: Usd.Prim) -> dict[str, Any] | None:
    """The render-wide settings camera `prim`'s own auto exposure stands for (`AUTO_EXPOSURE`,
    those it authors); None when it does not ask for auto exposure."""
    enabled = prim.GetAttribute("omni:rtx:autoExposure:enabled")
    if not (enabled and enabled.Get()):
        return None
    found = {}
    for name, setting in AUTO_EXPOSURE.items():
        attribute = prim.GetAttribute(name)
        if attribute and attribute.HasAuthoredValue():
            found[setting] = attribute.Get()
    return found
