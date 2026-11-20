"""The rates a renderer steps at for sensors of several rates: the one rule for every kind.
Plain Python.

How often each kind is updated (both renderers), by `update_rate`:

- cameras: a whole multiple of every camera's rate, at least the stage's own frames (its time
  codes a second; ovrtx: or its cfg's `step_rate`): at their own rate a camera's pictures came
  out a part of a frame apart in time;
- lidars: a whole multiple of every lidar's scan rate, so a scan begins on an update, and at
  least the stage's own frames: RTX traces a firing from the lidar's pose a straight line
  between the updates either side of it, so a lidar updated once a scan is traced as if it
  went straight through each scan (a walker's points 15 mm off, p90 75 mm; one swaying in a
  room 98 mm), and at the stage's frames as it moved (3.4 mm, ovrtx's too);
- skies: a whole multiple of every antenna's rate (a sky view needs nothing in between).

Each pass starts a whole number of every sensor's periods (`common_period`) before the
window, so every sensor's frames and scans begin on its own poses; a lidar's scans not on
them, or any of the window's whole scans missing, are refused (`lidarscans.finish`).

Measured (Isaac Sim 6.2, ovrtx 0.7.0): a lidar scanning at a rate other than its model's own
(ST's VL53L9, made for 30 Hz, at 15 or 60 Hz, with its report and pattern rates to match)
lost one scan in three in Isaac Sim, at 60 updates a second and at 120 alike, alone or
beside another lidar; ovrtx lost none. A lidar as fast as the updates is not the cause.
"""

from __future__ import annotations

import math
from fractions import Fraction


def _fractions(rates: list[float]) -> list[Fraction]:
    return [Fraction(rate).limit_denominator(1000) for rate in rates]


def update_rate(rates: list[float], at_least: float) -> float:
    """The smallest whole multiple of every rate in `rates` that is at least `at_least`."""
    if not rates:
        return at_least
    fractions = _fractions(rates)
    common = Fraction(
        math.lcm(*(f.numerator for f in fractions)), math.gcd(*(f.denominator for f in fractions))
    )
    return float(common * max(1, math.ceil(at_least / common - 1e-9)))


def common_period(rates: list[float]) -> float:
    """The seconds after which the periods of every rate in `rates` (Hz) end together again:
    the least whole multiple of each (10 and 15 Hz: 0.2 s). 0 with no rates."""
    if not rates:
        return 0.0
    periods = [1 / f for f in _fractions(rates)]
    return float(
        Fraction(
            math.lcm(*(p.numerator for p in periods)), math.gcd(*(p.denominator for p in periods))
        )
    )
