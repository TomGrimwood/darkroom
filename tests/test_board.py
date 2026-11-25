"""A render's progress as a board: a row a sensor, from the events it reports."""

import json
from dataclasses import replace

from darkroom import events
from darkroom.board import Board
from darkroom.events import (
    Made,
    Plan,
    Planned,
    Rendering,
    Riding,
    Step,
    Use,
    Uses,
    Writing,
)
from darkroom.process import Progress


def _sent(step, fraction, event):
    """`event` as Isaac Sim's side sends it: a progress's extra, through JSON."""
    return Progress(step, fraction, json.loads(json.dumps(events.extra(event))))


def test_each_event_reads_back_as_it_was_sent():
    for event in (
        Uses({"gnss": Use("gnss", 10.0, ("A", "B"))}, 4, 3),
        Planned("camera", (Plan("front", "camera", 15.0, (1280, 960), 0, 2, 75),)),
        Rendering("camera", 1, 0.25),
        Rendering("camera", 2, 0.25, 3.0, 0.5),
        Riding("camera", ("front", "sky")),
        Writing("lidar", 0),
        Made("camera", "front", "75 frames at 15 Hz"),
        Step("A.rrd", "rrd", ("A",), 0.3, "mapping ml30s's scans", 12.0),
    ):
        assert events.read(_sent("x", None, event).extra) == event
    assert events.read({"kind": "camera"}) is None


def test_a_jobs_sensors_are_a_board_from_what_its_isaac_sims_report():
    board = Board()
    planned = (
        Plan("front", "camera", 15, (1280, 960), 0, 2, 75),
        Plan("fwd", "camera", 30, (4096, 3000), 1, 2, 150),
    )
    assert board.take(_sent("cameras: planned", None, Planned("camera", planned)))
    assert [r.state for r in board.rows()] == ["waiting", "waiting"]
    board.take(_sent("rendering cameras (1 of 2)", 0.4, Rendering("camera", 0, 0.4)))
    front, fwd = board.rows()
    assert (front.state, front.fraction, fwd.state) == ("rendering", 0.4, "waiting")
    board.take(_sent("cameras (1 of 2): writing", None, Writing("camera", 0)))
    board.take(_sent("x", None, Made("camera", "front", "75 frames at 15 Hz", count=75)))
    board.take(_sent("rendering cameras (2 of 2)", 0.5, Rendering("camera", 1, 0.5)))
    front, fwd = board.rows()
    assert (front.state, front.said, front.made) == ("done", "75 frames at 15 Hz", 75)
    assert (fwd.state, fwd.fraction, fwd.size) == ("rendering", 0.5, (4096, 3000))
    assert not board.take(Progress("opening the stage"))  # not a sensor's: said as a line


def test_rigs_sharing_a_sensor_are_on_its_row_and_worked_out_ones_done_when_isaac_sim_starts():
    board = Board()
    uses = {"imu": Use("imu", 100.0, ("A", "B")), "front": Use("camera", 15.0, ("A",))}
    board.take(_sent("sensors", None, Uses(uses, 3, 2)))
    assert board.sharing == (3, 2)
    plan = (Plan("front", "camera", 15, (1280, 960), 0, 1, 75),)
    board.take(_sent("cameras: planned", None, Planned("camera", plan)))
    rows = {r.name: r for r in board.rows()}
    assert rows["imu"].state == "done" and rows["imu"].rigs == ("A", "B")
    assert rows["front"].rigs == ("A",) and rows["front"].state == "waiting"
    board.take(_sent("x", None, Made("camera", "front", "it broke", failed=True)))
    assert {r.name: r.state for r in board.rows()}["front"] == "failed"


def test_no_event_has_a_field_named_as_progress_arguments():
    # Sent as `process.progress(step, fraction, **extra(event))`: a field so named would clash.
    import dataclasses

    for kind in (Uses, Planned, Rendering, Riding, Writing, Made):
        assert not {"step", "fraction"} & {f.name for f in dataclasses.fields(kind)}


def test_sensors_rendering_together_each_show_the_turns_time_and_their_eta_from_it():
    from darkroom.board import duration, row_view

    board = Board()
    planned = (  # one turn: a small camera at half the rate of a big one
        Plan("drone", "camera", 15, (960, 540), 0, 1, 10),
        Plan("fwd", "camera", 30, (4096, 3000), 0, 1, 20),
    )
    board.take(_sent("cameras: planned", None, Planned("camera", planned)))
    board.take(_sent("rendering cameras", 0.0, Rendering("camera", 0, 0.0, 0.1)))
    board.take(_sent("rendering cameras", 0.5, Rendering("camera", 0, 0.5, 100.0)))
    drone, fwd = board.rows()
    assert drone.seconds == fwd.seconds == 100.0  # wall-clock: they render together
    cells = dict(zip(("frames", "elapsed", "eta", "fps"), row_view(fwd).cells[3:7], strict=True))
    assert cells == {
        "frames": "10/20",
        "elapsed": duration(fwd.seconds),
        "eta": duration(100.0),  # half done in 100 s: as long again
        "fps": "0.1",
    }
    board.take(_sent("x", None, Made("camera", "fwd", "20 frames at 30 Hz", seconds=190.0)))
    assert board.rows()[1].seconds == 190.0
    done = row_view(board.rows()[1])
    assert done.cells[4:7] == ("3 min", "", "0.1")  # 20 frames in its 190 s; done: no eta


def test_a_capture_carried_on_shows_what_was_made_before_as_done():
    from darkroom.board import row_view

    board = Board()
    said = "1469 frames at 15 Hz, 960x540, in 547 s"
    before = Made("camera", "drone", said, rate=15.0, size=(960, 540), count=1469)
    board.take(_sent("drone made before", None, before))
    plan = (Plan("lidar_a", "lidar", 10, None, 0, 1, 980),)
    board.take(_sent("lidars: planned", None, Planned("lidar", plan)))
    rows = {r.name: r for r in board.rows()}
    drone = rows["drone"]
    assert (drone.state, drone.fraction, drone.made, drone.said) == ("done", 1.0, 1469, said)
    assert row_view(drone).cells[1] == "960x540"  # its size, not "camera"
    assert rows["lidar_a"].state == "waiting"


def test_a_meter_reads_the_gpu_and_the_pixels_rendered_a_second():
    from darkroom.board import Meter, SensorRow, meter_charts

    meter = Meter(every=1.0)
    row = SensorRow("fwd", "camera", 30, (100, 10), 0, 1, 30, state="rendering", fraction=0.0)
    assert meter.read((row,), now=10.0)
    assert not meter.read((row,), now=10.5)  # once a second at most
    row = replace(row, fraction=0.5)  # 15 frames of 1000 pixels in 2 s
    assert meter.read((row,), now=12.0)
    assert meter.pixels[-1] == 15 * 1000 / 2
    assert len(meter.busy) == len(meter.used) == 2
    assert meter_charts(Meter(), 40) == []  # nothing read yet
    assert [t for t, _ in meter_charts(meter, 40)][2].endswith("Mpx/s")


def test_the_pixels_a_second_are_steady_when_frames_are_counted_in_bursts():
    from darkroom.board import Meter, SensorRow

    meter = Meter(every=1.0)
    row = SensorRow("fwd", "camera", 30, (100, 10), 0, 1, 300, state="rendering", fraction=0.0)
    for second in range(30):  # 20 frames every other second: 10 a second
        made = 20 * ((second + 1) // 2)
        meter.read((replace(row, fraction=made / 300),), now=float(second))
    late = meter.pixels[12:]
    assert max(late) - min(late) <= 0.15 * 10 * 1000  # within 15% of 10 frames a second


def test_a_chart_scrolls_at_a_column_every_few_readings_never_rescaled():
    from darkroom.board import chart, window

    assert window([1, 9, 2, 3, 8], 4, per=2) == [None, 9, 3, 8]  # newest (filling) at the right
    assert window([1, 9, 2, 3, 8, 4], 4, per=2) == [None, 9, 3, 8]  # its column filled: no move
    assert window([1, 9, 2, 3, 8, 4, 7], 4, per=2) == [9, 3, 8, 7]  # a new one: one column left
    assert window(list(range(5000)), 60, per=5)[-1] == 4999  # a long run: its last five minutes
    assert chart([100], 100, 3, rows=1, per=1) == ["  █"]  # blank before the first reading


def test_a_camera_that_got_on_half_way_fills_its_bar_from_there_round_to_the_start():
    from darkroom.board import covered
    from darkroom.terminal import bar

    assert covered(0.0, 0.3) == ((0.0, 0.3),)
    assert covered(0.5, 0.25) == ((0.5, 0.75),)
    assert covered(0.5, 0.75) == ((0.5, 1.0), (0.0, 0.25))
    board = Board()
    board.take(
        _sent("p", None, Planned("camera", (Plan("front", "camera", 15.0, (1280, 960), 0, 1, 75),)))
    )
    board.take(_sent("r", 0.75, Rendering("camera", 0, 0.75, 2.0, 0.5)))
    (row,) = board.rows()
    assert (row.begin, row.fraction) == (0.5, 0.75)
    drawn = bar(row.begin, row.fraction, 8, "green")
    filled = {
        i for span in drawn.spans if span.style == "green" for i in range(span.start, span.end)
    }
    assert filled == {4, 5, 6, 7, 0, 1}  # 0.5 to 1, then 0 to 0.25


def test_a_camera_carried_on_says_its_eta_from_what_it_has_rendered_since():
    from darkroom.board import duration, row_view

    board = Board()
    plan = Plan("fwd", "camera", 30, (4096, 3000), 0, 1, 1000)
    board.take(_sent("p", None, Planned("camera", (plan,))))
    board.take(_sent("r", 0.5, Rendering("camera", 0, 0.5, 1.0, 0.0)))  # had half from before
    board.take(_sent("r", 0.6, Rendering("camera", 0, 0.6, 600.0, 0.0)))
    (row,) = board.rows()
    assert row_view(row).cells[5] == duration(600.0 * 0.4 / 0.1) == "40 min"
    assert (duration(40), duration(5400), duration(3 * 3600 + 5 * 60)) == (
        "40 s",
        "1 h 30",
        "3 h 05",
    )


def test_a_callers_steps_wait_under_the_sensors_from_the_start_then_fill_in_turn():
    from darkroom.board import headline, row_view

    board = Board()
    for name, kind in (("A.rrd", "rrd"), ("A.mcap", "mcap")):
        assert board.take(_sent("", None, Step(name, kind, ("A",))))
    planned = (Plan("front", "camera", 15, (1280, 960), 0, 1, 75),)
    board.take(_sent("cameras: planned", None, Planned("camera", planned)))
    assert [(r.name, r.state) for r in board.rows()] == [
        ("front", "waiting"),
        ("A.rrd", "waiting"),
        ("A.mcap", "waiting"),
    ]
    board.take(_sent("x", None, Made("camera", "front", "75 frames", count=75)))
    board.take(_sent("x", None, Step("A.rrd", "rrd", ("A",), 0.2, "encoding front's video", 10)))
    board.take(_sent("x", None, Step("A.rrd", "rrd", ("A",), 0.6, "mapping its scans", 30)))
    _, rrd, mcap = board.rows()
    view = row_view(rrd)
    assert (rrd.state, rrd.fraction, view.label) == ("making", 0.6, "mapping its scans")
    assert view.cells == ("A.rrd", "rrd", "", "", "30 s", "20 s", "", "A")  # eta: 0.4 at 0.4/20 s
    assert mcap.state == "waiting"
    said = headline("capture", board.rows(), None, 75.0, "A's Rerun recording: x 60%")
    assert (
        said
        == "Capture running: 1 of 1 sensors, rrd 0 of 1, mcap 0 of 1, 75 s, A's Rerun recording: x 60%"
    )
    board.take(
        _sent("x", None, Step("A.rrd", "rrd", ("A",), seconds=40, said="2.1 GB", finished=True))
    )
    board.take(_sent("x", None, Step("A.mcap", "mcap", ("A",), said="disk full", failed=True)))
    _, rrd, mcap = board.rows()
    assert (rrd.state, row_view(rrd).label, row_view(rrd).cells[4]) == ("done", "2.1 GB", "40 s")
    assert (mcap.state, mcap.said) == ("failed", "disk full")
    said = headline("capture", board.rows(), None, 1266.0, "A: checking", "done")
    assert said == "Capture done: 1 of 1 sensors, rrd 1 of 1, mcap 1 of 1, 21 min"  # no step left


def test_a_callers_steps_left_waiting_are_stopped_with_their_job():
    board = Board()
    board.take(_sent("", None, Step("A.rrd", "rrd", ("A",))))
    assert [r.state for r in board.ended("cancelled")] == ["stopped"]
