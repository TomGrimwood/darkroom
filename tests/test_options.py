"""Options and spans refused as they are made when they cannot be captured, saying what may."""

from typing import get_args

import pytest

from darkroom import turns
from darkroom.options import Options, Packing, Span


@pytest.mark.parametrize(
    ("given", "said"),
    [
        ({"kinds": "camera"}, "kinds are a list"),  # one string, not a list
        ({"kinds": ["cameras"]}, "kinds are a list of camera, lidar, imu, gnss"),
        ({"packing": "tight"}, "no packing 'tight'"),
        ({"window": (5.0, 5.0)}, "a window starts"),
        ({"window": (-1.0, 2.0)}, "a window starts"),
        ({"scenery_cell": 0.0}, "a scenery cell"),
        ({"jpeg_quality": 0}, "a JPEG quality"),
        ({"jpeg_quality": 100}, "a JPEG quality"),
        ({"jpeg_quality": 80.5}, "a JPEG quality"),
    ],
)
def test_options_that_cannot_be_captured_are_refused_as_made(given, said):
    with pytest.raises(ValueError, match=said):
        Options(**given)


def test_options_that_can_be_are_made():
    Options(kinds=["camera", "lidar"], packing="one by one", window=(0.0, 1.0))
    assert Options().jpeg_quality == 80 and Options(jpeg_quality=92).jpeg_quality == 92


def test_a_span_ends_after_it_starts():
    Span(1.0, 2.0, "path_traced")
    with pytest.raises(ValueError, match="starts before it ends"):
        Span(2.0, 1.0, "path_traced")


def test_the_packings_are_the_loops():
    assert set(get_args(Packing)) == set(turns.PACKING)
