"""A lidar's scan rate, and the RTX attributes it is measured by."""

from __future__ import annotations

import numpy as np
from pxr import Gf, Usd, UsdGeom

LIDAR = "OmniLidar"
ATTRIBUTES = "omni:sensor:"
TICK_RATE = "omni:sensor:tickRate"
SCAN_RATE = "omni:sensor:Core:scanRateBaseHz"


def authored(prim: Usd.Prim) -> list[Usd.Attribute]:
    """A lidar's `omni:sensor:*` attributes with values: what RTX measures it by."""
    return [
        a
        for a in prim.GetAttributes()
        if a.GetName().startswith(ATTRIBUTES) and a.HasAuthoredValue()
    ]


#: Endings of the emitter attributes that say where each emitter points (degrees).
AZIMUTH, ELEVATION = ":azimuthDeg", ":elevationDeg"


#: RTX's word for how a lidar scans: ROTARY (it spins) or SOLID_STATE.
SCAN_TYPE = "omni:sensor:Core:scanType"
#: What the copy of a lidar that measures for it is named, whichever renderer: its name and this.
CAPTURE = "_CaptureLidar"


def field_of_view(prim: Usd.Prim) -> tuple[float, float, float, float] | None:
    """What a lidar that does not spin sees, from its emitters: the least and greatest azimuth and
    the least and greatest elevation, degrees. None for one that spins (RTX's profile says so:
    `omni:sensor:Core:scanType` ROTARY; else its emitters all at one azimuth, turned together) or
    that says none."""
    if prim.GetAttribute(SCAN_TYPE).Get() == "ROTARY":
        return None
    angles = {
        ending: np.concatenate(
            [
                np.atleast_1d(np.asarray(a.Get(), float))
                for a in authored(prim)
                if a.GetName().endswith(ending)
            ]
            or [np.zeros(0)]
        )
        for ending in (AZIMUTH, ELEVATION)
    }
    across, up = angles[AZIMUTH], angles[ELEVATION]
    if not len(across) or not len(up) or np.ptp(across) < 1:
        return None
    return float(across.min()), float(across.max()), float(up.min()), float(up.max())


def scan_rate(prim: Usd.Prim) -> float:
    """A lidar's scans a second. Raises `RuntimeError` unless it is an `OmniLidar` whose tick
    rate is its scan rate (RTX gives a scan a tick) and which its own transform does not move
    (USD does not see an `OmniLidar` as placed without Isaac's schemas: move its parent)."""
    where = prim.GetPath()
    if prim.GetTypeName() != LIDAR:
        raise RuntimeError(f"{where} is not an {LIDAR}.")
    own = Gf.Matrix4d(1.0)
    for name in prim.GetAttribute("xformOpOrder").Get() or []:
        if name != "!resetXformStack!":
            op = UsdGeom.XformOp(prim.GetAttribute(name.removeprefix("!invert!")))
            own = op.GetOpTransform(Usd.TimeCode.Default()) * own
    if np.abs(np.array(own) - np.eye(4)).max() > 1e-9:
        raise RuntimeError(f"{where} is moved by its own transform: move its parent instead.")
    tick, scan = (prim.GetAttribute(n).Get() for n in (TICK_RATE, SCAN_RATE))
    if not tick or not scan:
        raise RuntimeError(f"{where} needs {TICK_RATE} and {SCAN_RATE}.")
    if abs(tick - scan) > 1e-6:
        raise RuntimeError(
            f"{where} ticks at {tick:g} Hz but scans at {scan:g}: {TICK_RATE} has to be {scan:g}."
        )
    return float(tick)


def outline(prim: Usd.Prim, reach: float = 0.5, steps: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """What a lidar sees, as line segments (starts, ends: n x 3) in its own frame (+X ahead, +Y
    left, +Z up): the edges of its field `reach` metres out, rays to its corners. One that spins
    sees all round, between its emitters' least and greatest elevation."""
    up = np.concatenate(
        [
            np.atleast_1d(np.asarray(a.Get(), float))
            for a in authored(prim)
            if a.GetName().endswith(ELEVATION)
        ]
        or [np.zeros(1)]
    )
    box = field_of_view(prim)
    right, left = (box[0], box[1]) if box else (-180.0, 180.0)
    low, high = (box[2], box[3]) if box else (float(up.min()), float(up.max()))

    def at(az: np.ndarray, el: np.ndarray) -> np.ndarray:
        a, e = np.radians(az), np.radians(el)
        return reach * np.stack([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)], -1)

    across = np.linspace(right, left, steps + 1)
    starts, ends = [], []
    for el in (low, high):  # its top and bottom edges
        line = at(across, np.full_like(across, el))
        starts.append(line[:-1])
        ends.append(line[1:])
    if box:  # its sides, and rays to its corners
        updown = np.linspace(low, high, steps // 2 + 1)
        for az in (right, left):
            line = at(np.full_like(updown, az), updown)
            starts.append(line[:-1])
            ends.append(line[1:])
        corners = at(np.array([right, right, left, left]), np.array([low, high, low, high]))
        starts.append(np.zeros((4, 3)))
        ends.append(corners)
    else:  # spinning: a ray ahead, at its top and bottom
        ahead = at(np.zeros(2), np.array([low, high]))
        starts.append(np.zeros((2, 3)))
        ends.append(ahead)
    return np.concatenate(starts), np.concatenate(ends)
