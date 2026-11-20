"""What an IMU reports: specific force and turn rate in its own axes. The truth, without noise.

- axes are the IMU prim's own;
- `accel` is specific force, Rᵀ(a - g) with g = (0, 0, -9.80665) in the site's axes:
  +9.81 along the IMU's up when it is still;
- `gyro` is the angular rate in the IMU's axes, rad/s, right-handed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .motion import Motion, rotation_matrices, sample_times

GRAVITY = np.array([0.0, 0.0, -9.80665])


@dataclass(frozen=True)
class Samples:
    seconds: np.ndarray
    accel: np.ndarray
    gyro: np.ndarray


def samples(moving: Motion, rate: float) -> Samples:
    """An IMU's samples at `rate` a second over the whole of `moving` (in metres, Z up)."""
    seconds = sample_times(moving.grid[0], moving.grid[-1], rate)
    _, quaternions = moving.at(seconds)
    # Rᵀ v, for each sample's rotation R.
    own = np.swapaxes(rotation_matrices(quaternions), 1, 2)
    accel = np.einsum("nij,nj->ni", own, moving.acceleration(seconds) - GRAVITY)
    gyro = np.einsum("nij,nj->ni", own, moving.angular_velocity(seconds))
    return Samples(seconds, accel, gyro)
