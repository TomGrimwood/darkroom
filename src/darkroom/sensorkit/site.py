"""Where a stage is on Earth, from USD's standard for it: the one prim with
`OmniWGS84ReferencePositionAPI`, whose frame is the tangent plane at its origin
(+X east, +Y north, +Z up)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pxr import Gf, Usd, UsdGeom

API = "OmniWGS84ReferencePositionAPI"
POSITION = "omni:geospatial:wgs84:reference:referencePosition"
TANGENT_PLANE = "omni:geospatial:wgs84:reference:tangentPlane"
ORIENTATION = "omni:geospatial:wgs84:reference:orientation"
GEOID = "wgs84:geoidHeight"


@dataclass(frozen=True)
class Site:
    latitude: float
    longitude: float
    #: Ellipsoidal height of the origin, metres.
    altitude: float
    #: Stage space to the site's (east, north, up in metres), a row-vector transform.
    to_site: np.ndarray
    geoid: float | None = None


def _applied(prim: Usd.Prim) -> list[str]:
    listed = prim.GetMetadata("apiSchemas")
    return list(listed.GetAddedOrExplicitItems()) if listed else []


def find(stage: Usd.Stage) -> Site | None:
    """The stage's site, None without one. Raises `RuntimeError` for more than one."""
    found = [prim for prim in stage.Traverse() if API in _applied(prim)]
    if not found:
        return None
    if len(found) > 1:
        raise RuntimeError(
            f"a stage has one site; this has {', '.join(str(p.GetPath()) for p in found)}."
        )
    prim = found[0]
    if (prim.GetAttribute(TANGENT_PLANE).Get() or "ENU") != "ENU" or Gf.Vec3d(
        prim.GetAttribute(ORIENTATION).Get() or Gf.Vec3d(0)
    ) != Gf.Vec3d(0):
        raise RuntimeError(f"{prim.GetPath()}: only an unrotated ENU tangent plane is read.")
    latitude, longitude, altitude = prim.GetAttribute(POSITION).Get()
    frame = UsdGeom.XformCache().GetLocalToWorldTransform(prim).RemoveScaleShear()
    metres = Gf.Matrix4d(1.0).SetScale(UsdGeom.GetStageMetersPerUnit(stage))
    geoid = prim.GetAttribute(GEOID).Get()
    return Site(latitude, longitude, altitude, np.array(frame.GetInverse() * metres), geoid)


def to_local(site: Site, points: np.ndarray) -> np.ndarray:
    """Stage points (N x 3) as east, north and up from the site's origin, in metres."""
    return np.asarray(points) @ site.to_site[:3, :3] + site.to_site[3, :3]


def to_geodetic(site: Site, local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """East, north, up (N x 3) as latitude, longitude (degrees) and ellipsoidal height (m).

    Up is added to the origin's height, not measured from the tangent plane, which falls
    away from the ground by 5 cm at 800 m: a scene is a map projection about its origin.
    """
    import pymap3d

    local = np.asarray(local)
    lat, lon, _ = pymap3d.enu2geodetic(
        local[:, 0], local[:, 1], np.zeros(len(local)), site.latitude, site.longitude, 0.0
    )
    return np.asarray(lat), np.asarray(lon), site.altitude + local[:, 2]
