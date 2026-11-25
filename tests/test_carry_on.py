"""A capture that did not finish, carried on: what Isaac Sim made all of kept; a camera part
made keeps its whole frames; the rest cleared."""

import pytest

from darkroom import format
from darkroom.run import _carry_on


def _unfinished(folder, stage):
    sensors = {n: format.Sensor("camera", f"/Rig/{n}", 30.0) for n in ("front", "back")}
    format.write_meta(folder, format.Meta(str(stage), 0.0, 1.0, sensors))
    for name in sensors:
        (folder / name / format.FRAMES).mkdir(parents=True)
        (folder / name / format.FRAMES / "0000.png").write_bytes(b"x")
        (folder / f"{name}_mosaic.jpg").write_bytes(b"x")
    format.made(folder, "front", "30 frames at 30 Hz, 64x48, in 2 s")  # back stopped part way


def test_an_unfinished_capture_keeps_the_sensors_made_and_clears_the_rest(tmp_path):
    stage = tmp_path / "run.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    out = tmp_path / "capture"
    _unfinished(out, stage)
    said = []
    _carry_on(out, stage.resolve(), said.append)
    assert format.was_made(out, "front") == format.WasMade("30 frames at 30 Hz, 64x48, in 2 s")
    assert (out / "front" / format.FRAMES / "0000.png").is_file()
    assert not (out / "back" / format.FRAMES / "0000.png").exists()  # not a frame named by its time
    assert not (out / "back_mosaic.jpg").exists()
    assert said == [f"carrying on {out}: 1 of 2 sensors made before, kept"]


def test_a_lidar_made_without_its_laz_is_rendered_again(tmp_path):
    """As darkroom 0.1.2 left one whose render stopped before its scans were packed."""
    stage = tmp_path / "run.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    out = tmp_path / "capture"
    sensors = {n: format.Sensor("lidar", f"/Rig/{n}", 10.0) for n in ("packed", "raw")}
    format.write_meta(out, format.Meta(str(stage), 0.0, 1.0, sensors))
    for name in sensors:
        (out / name).mkdir()
        format.made(out, name, "10 scans at 10 Hz, 0.50 M points, in 2 s", 10)
    (out / "packed" / format.PACKED).write_bytes(b"x")
    (out / "raw" / ".raw").mkdir()
    (out / "raw" / ".raw" / "points.bin").write_bytes(b"x")
    said = []
    _carry_on(out, stage.resolve(), said.append)
    assert format.was_made(out, "packed") is not None and not (out / "raw").exists()
    assert said == [f"carrying on {out}: 1 of 2 sensors made before, kept"]


def test_a_camera_part_made_keeps_its_whole_frames_and_drops_a_cut_off_one(tmp_path):
    stage = tmp_path / "run.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    out = tmp_path / "capture"
    _unfinished(out, stage)
    frames = out / "back" / format.FRAMES
    (frames / "000000.jpg").write_bytes(b"\xff\xd8 a frame \xff\xd9")
    (frames / "000001.jpg").write_bytes(b"\xff\xd8 cut off")  # Isaac Sim stopped writing it
    (frames / "render_000007.jpg").write_bytes(b"\xff\xd8 x \xff\xd9")  # not named yet
    said = []
    _carry_on(out, stage.resolve(), said.append)
    assert sorted(f.name for f in frames.iterdir()) == ["000000.jpg"]
    assert format.frames_kept(out, "back") == {0}
    assert said[-1].endswith("1 cameras part made, their frames kept")


def test_an_unfinished_capture_of_another_stage_is_not_carried_on(tmp_path):
    out = tmp_path / "capture"
    _unfinished(out, tmp_path / "one.usda")
    with pytest.raises(RuntimeError, match="unfinished capture of"):
        _carry_on(out, (tmp_path / "other.usda").resolve(), None)
    assert (out / "back").is_dir()  # nothing cleared


def test_out_of_memory_steps_back_the_cameras_on_at_once(tmp_path):
    from darkroom import events, format, run, turns
    from darkroom.options import Options
    from darkroom.renderjob import RenderJob

    big, small = (4096, 3000), (640, 480)
    sensors = {n: format.Sensor("camera", f"/Rig/{n}", 30.0) for n in ("a", "b", "c", "d")}
    format.write_meta(tmp_path, format.Meta("stage.usda", 0.0, 1.0, sensors))
    (tmp_path / "d").mkdir()
    format.made(tmp_path, "d", "made, off the loop before")
    plan = {
        n: events.Plan(n, "camera", 30.0, size, turn, 4, 30)
        for turn, (n, size) in enumerate((("a", big), ("b", big), ("c", small), ("d", big)))
    }

    def ready():
        job = RenderJob(str(tmp_path / "run.usda"), str(tmp_path), (0.0, 1.0))
        return run._Ready(0, False, {"camera"}, Options(), job)

    one, said, steps = ready(), [], []
    # a, b and c on the loop when it ran out
    assert run._step_back(
        one, plan, ["a", "b", "c"], ["a", "b", "c", "d"], "ovrtx", said.append, steps.append
    )
    assert one.job.most == turns.step_back([big, big, small])
    assert "a, b, c at once" in said[0] and "stepping back" in steps[0].step
    assert said[0].endswith("in a new ovrtx")
    lone = ready()
    # a camera alone ran out with nothing made: no further back
    assert not run._step_back(lone, plan, ["a"], ["a", "b", "c"], "ovrtx", None, None)
    # with another made first (d, as this Isaac Sim started unmade): again in a new one
    assert run._step_back(lone, plan, ["a"], ["a", "b", "c", "d"], "ovrtx", said.append, None)
    assert "after other cameras" in said[-1] and lone.job.most is None
    assert "carrying on in a new ovrtx" in said[-1]  # its renderer's process, by name
