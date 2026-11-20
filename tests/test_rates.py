"""How often each kind is updated, and the rule for lidars (`darkroom.rates`)."""

import pytest

from darkroom.rates import common_period, update_rate


def test_cameras_update_at_a_whole_multiple_of_every_rate_and_the_stages_frames_at_least():
    assert update_rate([15.0], 60.0) == 60.0
    assert update_rate([15.0, 25.0], 60.0) == 75.0
    assert update_rate([], 60.0) == 60.0


@pytest.mark.parametrize(
    ("rates", "per_second", "updates"),
    [
        ([10.0], 60.0, 60.0),  # six a scan, as the stage is keyed
        ([20.0, 10.0], 60.0, 60.0),
        ([15.0, 10.0], 60.0, 60.0),
        ([25.0], 24.0, 25.0),
        ([7.0], 60.0, 63.0),
    ],
)
def test_lidars_update_at_every_ones_rate_and_the_stages_frames_at_least(
    rates, per_second, updates
):
    got = update_rate(rates, per_second)
    assert got == updates
    for rate in rates:
        assert (got / rate) == pytest.approx(round(got / rate))


def test_sensors_periods_end_together_again_after_their_common_period():
    assert common_period([10.0]) == pytest.approx(0.1)
    assert common_period([10.0, 20.0]) == pytest.approx(0.1)
    assert common_period([10.0, 15.0]) == pytest.approx(0.2)
    assert common_period([30.0, 7.5]) == pytest.approx(2 / 15)
    assert common_period([]) == 0.0
