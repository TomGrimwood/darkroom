"""What a sensor made, in the same words as ever (`outcome.said`); and kept so (`.made`)."""

from darkroom import format
from darkroom.outcome import Result, said


def test_results_are_worded_as_before():
    camera = Result(30, "frames", 15.0, "1280x960")
    assert said(camera, 2.4) == "30 frames at 15 Hz, 1280x960, in 2 s"
    off = Result(28, "frames", 30.0, "4096x3000", ("3 rendered off their poses, not kept",))
    assert said(off, 61) == (
        "28 frames at 30 Hz, 4096x3000, in 61 s (3 rendered off their poses, not kept)"
    )
    lidar = Result(20, "scans", 10.0, "1.42 M points")
    assert said(lidar, 3) == "20 scans at 10 Hz, 1.42 M points, in 3 s"
    blind = Result(20, "scans", 10.0, "no points: nothing in reach", timed=False)
    assert said(blind, 3) == "20 scans at 10 Hz, no points: nothing in reach"


def test_a_sensor_made_keeps_what_it_said_and_how_many(tmp_path):
    (tmp_path / "front").mkdir()
    assert format.was_made(tmp_path, "front") is None
    format.made(tmp_path, "front", "30 frames at 15 Hz, 1280x960, in 2 s", 30)
    assert format.was_made(tmp_path, "front") == format.WasMade(
        "30 frames at 15 Hz, 1280x960, in 2 s", 30
    )
    (tmp_path / "back").mkdir()
    (tmp_path / "back" / format.MADE).write_text(
        "12 frames at 30 Hz\n", encoding="utf-8"
    )  # before counts were kept
    assert format.was_made(tmp_path, "back") == format.WasMade("12 frames at 30 Hz")
