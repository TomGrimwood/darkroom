"""The sky a GNSS antenna sees: its sky view, and how much of it is sky (`sky_share`).

The sky view is what the renderer renders at an antenna: an equidistant fisheye looking along
the antenna's +Z, `SIZE` pixels square, 90° from its axis at the edge of the circle (`LENS`):
the whole 180° the antenna sees. Each pixel is `OPEN` (its ray hit nothing, or what is
labelled sky), `LEAVES` (vegetation) or `BLOCKED` (anything else, and a tree's wood: a trunk
or limb labelled `wood` under its `vegetation`), from Replicator's segmentation of the
scene's semantic `class` labels. Its image axes are the antenna's: column along +X, row
along +Y (the view from below, so a map from above looks mirrored).

Plain numpy.
"""

from __future__ import annotations

import numpy as np

#: The sky view's width and height in pixels: about 1.4° a pixel.
SIZE = 128
#: Pixels a radian from the view's axis: 90° is the circle's edge.
FOCAL = (SIZE / 2) / (np.pi / 2)
#: The sky view's OpenCV fisheye intrinsics (no distortion: equidistant).
LENS = {"fx": FOCAL, "fy": FOCAL, "cx": SIZE / 2, "cy": SIZE / 2}
#: The key in a stage layer's custom data that names its GNSS layer: a USD file put over the
#: stage only while the sky views are rendered, taking out what blocks no signal (grass,
#: small plants, the sky dome) so that each view renders faster. An absolute path.
LAYER = "gnssLayer"
#: What a sky view's pixel saw.
BLOCKED, LEAVES, OPEN = 0, 1, 2
#: The semantic `class` labels that are sky, vegetation and a plant's wood (a word of a
#: pixel's labels). Wood blocks a signal, though it is vegetation too: grim's trees label
#: their trunks and twigs `wood` under the tree's `vegetation`.
SKY_LABELS = frozenset({"sky"})
LEAVES_LABELS = frozenset({"vegetation", "tree", "plant", "foliage"})
WOOD_LABELS = frozenset({"wood", "trunk", "branch", "bark"})


def pixels(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where directions in the antenna's axes (... x 3) fall in its sky view: (row, column),
    as whole pixels, which may be outside it."""
    d = np.asarray(directions, float)
    off = np.arccos(np.clip(d[..., 2], -1, 1))
    around = np.arctan2(d[..., 1], d[..., 0])
    column = LENS["cx"] + FOCAL * off * np.cos(around)
    row = LENS["cy"] + FOCAL * off * np.sin(around)
    return np.floor(row).astype(int), np.floor(column).astype(int)


def seen(view: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """What the sky view `view` saw towards each of `directions` (... x 3, the antenna's
    axes): `OPEN`, `LEAVES` or `BLOCKED`. Behind the antenna is always `BLOCKED`."""
    row, column = pixels(directions)
    inside = (row >= 0) & (row < view.shape[0]) & (column >= 0) & (column < view.shape[1])
    got = np.full(row.shape, BLOCKED, np.uint8)
    got[inside] = view[row[inside], column[inside]]
    return np.where(np.asarray(directions)[..., 2] > 0, got, BLOCKED)


def classify(labels: str) -> int:
    """A pixel's class from its comma-separated semantic labels, as Replicator gives them:
    `OPEN` for nothing hit ("BACKGROUND") or sky, `LEAVES` for vegetation but its wood, else
    `BLOCKED`."""
    if labels.strip() == "BACKGROUND":  # Replicator's own, for nothing hit: not a label's
        return OPEN
    words = {w.strip().lower() for w in labels.split(",")}
    if words & SKY_LABELS:
        return OPEN
    if words & WOOD_LABELS:
        return BLOCKED
    return LEAVES if words & LEAVES_LABELS else BLOCKED


def directions(shape: tuple[int, int] = (SIZE, SIZE)) -> np.ndarray:
    """Each sky view pixel's direction in the antenna's axes (rows x columns x 3), from its
    middle: `pixels` the other way."""
    rows, columns = np.indices(shape) + 0.5
    off = np.hypot(rows - LENS["cy"], columns - LENS["cx"]) / FOCAL
    around = np.arctan2(rows - LENS["cy"], columns - LENS["cx"])
    return np.stack([np.sin(off) * np.cos(around), np.sin(off) * np.sin(around), np.cos(off)], -1)


def sky_share(view: np.ndarray, rotation: np.ndarray, through: float = 0.5) -> float:
    """How much of the sky view (the whole 180° circle) is sky, 0 to 1: an open pixel counts
    whole, one of leaves `through` of one, a blocked one nothing. `rotation` (3 x 3) turns the
    antenna's axes into east, north, up (v_enu = R v_antenna): a tilted antenna's view dips
    under the horizon, and nothing down there is sky, open or not."""
    own = directions(view.shape)
    within = own[..., 2] >= 0
    above = own @ np.asarray(rotation, float)[2] > 0  # each pixel's direction's up: (R d)_z
    part = np.where(view == OPEN, 1.0, np.where(view == LEAVES, through, 0.0))
    return float(np.where(above, part, 0.0)[within].mean())
