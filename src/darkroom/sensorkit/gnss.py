"""A GNSS antenna's fixes: where it truly was, as WGS84 latitude, longitude and height, at its
own rate. No receiver is simulated (no satellites, corrections or errors): a capture keeps the
truth, and how much of the antenna's view was open sky (`sky.sky_share`, `darkroom.skies`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .motion import Motion, sample_times
from .site import Site, to_geodetic


@dataclass(frozen=True)
class Fixes:
    seconds: np.ndarray
    #: WGS84 degrees.
    latitude: np.ndarray
    longitude: np.ndarray
    #: Metres above the ellipsoid.
    height: np.ndarray
    #: Metres above the geoid; None when the site does not give the geoid.
    altitude: np.ndarray | None


def fixes(antenna: Motion, site: Site, rate: float) -> Fixes:
    """An antenna's fixes at `rate` a second; `antenna` in the site's east, north, up."""
    seconds = sample_times(antenna.grid[0], antenna.grid[-1], rate)
    local, _ = antenna.at(seconds)
    latitude, longitude, height = to_geodetic(site, local)
    altitude = None if site.geoid is None else height - site.geoid
    return Fixes(seconds, latitude, longitude, height, altitude)
