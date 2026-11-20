"""A GNSS antenna's sky view: its lens, and how much of it is sky."""

import numpy as np

from darkroom.sensorkit import sky

SECONDS = np.arange(0, 30, 0.1)


def _view(open_where) -> np.ndarray:
    """A sky view open where `open_where(east, north, up)` is, for each pixel's direction."""
    rows, columns = np.indices((sky.SIZE, sky.SIZE)) + 0.5
    off = np.hypot(rows - sky.LENS["cy"], columns - sky.LENS["cx"]) / sky.FOCAL
    around = np.arctan2(rows - sky.LENS["cy"], columns - sky.LENS["cx"])
    e, n, u = np.sin(off) * np.cos(around), np.sin(off) * np.sin(around), np.cos(off)
    return np.where((off <= np.pi / 2) & open_where(e, n, u), sky.OPEN, sky.BLOCKED).astype(
        np.uint8
    )


OPEN = _view(lambda e, n, u: np.ones_like(e, bool))
SHUT = np.full((sky.SIZE, sky.SIZE), sky.BLOCKED, np.uint8)
#: Under a tree: leaves everywhere the sky would be.
TREE = np.where(OPEN == sky.OPEN, sky.LEAVES, sky.BLOCKED).astype(np.uint8)
#: A street running north: walls east and west, up to 60° above the horizon.
STREET = _view(lambda e, n, u: u > np.sin(np.radians(60)) * np.hypot(e, u))
#: A wall to the east: the western half of the view open.
WALL = _view(lambda e, n, u: e < 0)


def _tilted(degrees: float) -> np.ndarray:
    """An antenna's axes tipped `degrees` from up towards north (v_enu = R v_antenna)."""
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[1.0, 0, 0], [0, c, s], [0, -s, c]])


def test_the_sky_views_lens_puts_the_zenith_in_the_middle_and_the_horizon_at_the_edge():
    row, column = sky.pixels(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    assert (row[0], column[0]) == (sky.SIZE // 2, sky.SIZE // 2)
    assert column[1] == sky.SIZE and row[1] == sky.SIZE // 2  # east: just past the right
    assert row[2] == sky.SIZE and column[2] == sky.SIZE // 2  # north: just past the bottom
    assert sky.seen(OPEN, np.array([[0, 0, -1.0]]))[0] == sky.BLOCKED  # never behind
    assert np.allclose(
        sky.directions()[sky.pixels(np.array([[0.6, 0.0, 0.8]]))], (0.6, 0, 0.8), atol=0.02
    )


def test_the_sky_share_is_of_the_whole_180_degree_view():
    upright = np.eye(3)
    assert sky.sky_share(OPEN, upright) == 1 and sky.sky_share(SHUT, upright) == 0
    assert abs(sky.sky_share(TREE, upright, through=0.3) - 0.3) < 1e-9
    assert sky.sky_share(WALL, upright) == 0.5
    # Tilted, what the view sees under the horizon is not sky, whatever it rendered as.
    assert 0.7 < sky.sky_share(OPEN, _tilted(45)) < 0.8
    assert 0.5 < sky.sky_share(OPEN, _tilted(85)) < 0.55


def test_pixels_are_classed_by_their_semantic_labels():
    assert sky.classify("BACKGROUND") == sky.OPEN  # nothing hit
    assert sky.classify("prop_general, sky") == sky.OPEN
    assert sky.classify("other,vegetation") == sky.LEAVES
    assert sky.classify("vegetation,wood") == sky.BLOCKED  # a tree's trunk or limb
    assert sky.classify("UNLABELLED") == sky.classify("background") == sky.BLOCKED
