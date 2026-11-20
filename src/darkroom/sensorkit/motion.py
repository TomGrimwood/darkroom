"""How a sensor moves, from where it was at a run of times: velocity, acceleration, turn rate.

Quaternions here are (w, x, y, z), as USD's `Gf.Quatd` and the capture format have them.
The poses are taken as they are: a trajectory's frames are smoothed when baked, so what a
camera renders and what an IMU feels come from the same motion. `smoothing` is there for
poses that were not (a Gaussian, in seconds).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sample_times(start: float, end: float, rate: float) -> np.ndarray:
    """The seconds something reporting `rate` times a second reports at, from `start`,
    never past `end`."""
    return start + np.arange(int((end - start) * rate + 1e-9) + 1) / rate


def continuous(quaternions: np.ndarray) -> np.ndarray:
    """Quaternions with each the sign nearer the one before, so neighbours can be averaged."""
    q = np.asarray(quaternions, dtype=float)
    flips = np.where(np.sum(q[1:] * q[:-1], axis=1) < 0, -1.0, 1.0)
    return q * np.concatenate([[1.0], np.cumprod(flips)])[:, None]


def smoothed(values: np.ndarray, sigma: float) -> np.ndarray:
    """The rows of `values` smoothed down the columns by a Gaussian `sigma` rows wide,
    reflected oddly about each end so a straight line comes through as it was."""
    reach = min(int(np.ceil(4 * sigma)), len(values) - 1)
    if sigma <= 0 or reach < 1:
        return np.array(values, dtype=float)
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate(
        [2 * values[0] - values[reach:0:-1], values, 2 * values[-1] - values[-2 : -reach - 2 : -1]]
    )
    return np.stack(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(values.shape[1])], axis=1
    )


def rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Each (w, x, y, z) as a column-vector rotation matrix (N x 3 x 3): world = R · own."""
    w, x, y, z = np.asarray(quaternions, dtype=float).T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        1,
    )


def quaternions_of(rotations: np.ndarray) -> np.ndarray:
    """Column-vector rotation matrices (N x 3 x 3) as (w, x, y, z), sign kept continuous."""
    m = np.asarray(rotations, dtype=float)
    out = np.empty((len(m), 4))
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    big = np.argmax(np.stack([trace, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], 1), 1)
    for k in range(4):
        c = m[big == k]
        if not len(c):
            continue
        if k == 0:
            s = np.sqrt(1 + c[:, 0, 0] + c[:, 1, 1] + c[:, 2, 2]) * 2
            q = [
                0.25 * s,
                (c[:, 2, 1] - c[:, 1, 2]) / s,
                (c[:, 0, 2] - c[:, 2, 0]) / s,
                (c[:, 1, 0] - c[:, 0, 1]) / s,
            ]
        elif k == 1:
            s = np.sqrt(1 + c[:, 0, 0] - c[:, 1, 1] - c[:, 2, 2]) * 2
            q = [
                (c[:, 2, 1] - c[:, 1, 2]) / s,
                0.25 * s,
                (c[:, 0, 1] + c[:, 1, 0]) / s,
                (c[:, 0, 2] + c[:, 2, 0]) / s,
            ]
        elif k == 2:
            s = np.sqrt(1 + c[:, 1, 1] - c[:, 0, 0] - c[:, 2, 2]) * 2
            q = [
                (c[:, 0, 2] - c[:, 2, 0]) / s,
                (c[:, 0, 1] + c[:, 1, 0]) / s,
                0.25 * s,
                (c[:, 1, 2] + c[:, 2, 1]) / s,
            ]
        else:
            s = np.sqrt(1 + c[:, 2, 2] - c[:, 0, 0] - c[:, 1, 1]) * 2
            q = [
                (c[:, 1, 0] - c[:, 0, 1]) / s,
                (c[:, 0, 2] + c[:, 2, 0]) / s,
                (c[:, 1, 2] + c[:, 2, 1]) / s,
                0.25 * s,
            ]
        out[big == k] = np.stack(q, 1)
    return continuous(out)


def from_usd_matrices(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """USD row-vector transforms (N x 4 x 4) as positions (N x 3) and quaternions (N x 4),
    scale and shear taken out."""
    matrices = np.asarray(matrices, dtype=float)
    u, _, vt = np.linalg.svd(np.swapaxes(matrices[:, :3, :3], 1, 2))
    r = u @ vt
    r[np.linalg.det(r) < 0] *= -1
    return matrices[:, 3, :3].copy(), quaternions_of(r)


def _at(grid: np.ndarray, values: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(seconds, grid, values[:, i]) for i in range(values.shape[1])], 1)


@dataclass(frozen=True)
class Motion:
    """A pose at each second of `grid`, and its rates: what a sensor moving so feels.

    Positions in metres in the site's (or stage's) axes; quaternions (w, x, y, z) taking the
    sensor's axes to those; velocities and accelerations in m/s and m/s²; angular velocity
    in rad/s about the site's axes (turn it by Rᵀ for the sensor's own).
    """

    grid: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    angular_velocities: np.ndarray

    def at(self, seconds: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        seconds = np.atleast_1d(np.asarray(seconds, dtype=float))
        q = _at(self.grid, self.quaternions, seconds)
        return _at(self.grid, self.positions, seconds), q / np.linalg.norm(q, axis=1, keepdims=True)

    def velocity(self, seconds: np.ndarray | float) -> np.ndarray:
        return _at(self.grid, self.velocities, np.atleast_1d(seconds))

    def acceleration(self, seconds: np.ndarray | float) -> np.ndarray:
        return _at(self.grid, self.accelerations, np.atleast_1d(seconds))

    def angular_velocity(self, seconds: np.ndarray | float) -> np.ndarray:
        return _at(self.grid, self.angular_velocities, np.atleast_1d(seconds))


def motion(
    seconds: np.ndarray, positions: np.ndarray, quaternions: np.ndarray, smoothing: float = 0.0
) -> Motion:
    """The motion of a pose sampled at `seconds` (evenly, two at least)."""
    grid = np.asarray(seconds, dtype=float)
    if len(grid) < 2:
        raise ValueError("a motion needs two poses at least.")
    positions = np.asarray(positions, dtype=float)
    quaternions = continuous(quaternions)
    if smoothing > 0:
        sigma = smoothing / (grid[1] - grid[0])
        positions = smoothed(positions, sigma)
        quaternions = smoothed(quaternions, sigma)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    velocities = np.gradient(positions, grid, axis=0)
    turning = np.gradient(quaternions, grid, axis=0)
    # ω = 2 q̇ q*: its vector part, in the world's axes.
    w, v = quaternions[:, :1], quaternions[:, 1:]
    w_rate, v_rate = turning[:, :1], turning[:, 1:]
    angular = 2 * (w * v_rate - w_rate * v - np.cross(v_rate, v))
    return Motion(
        grid,
        positions,
        quaternions,
        velocities,
        np.gradient(velocities, grid, axis=0),
        angular,
    )
