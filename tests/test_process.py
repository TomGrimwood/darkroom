"""A function run in a process of its own: its result, progress and lines handed back, its
failing said, killed when the GPU runs out, one at a time on the machine."""

import sys
import textwrap
import time
from pathlib import Path

import pytest

from darkroom import process


@pytest.fixture(autouse=True)
def _own_logs(tmp_path, monkeypatch):
    """Logs and the machine's renderer lock in the test's own folder: a renderer running on
    the machine does not hold the tests up."""
    monkeypatch.setattr(process, "LOGS", tmp_path / "logs")


def _stand_in(tmp_path, script: str, monkeypatch):
    """A Python whose process runs `script` instead of the call (a renderer's output, made up)."""
    (tmp_path / "stand_in.py").write_text(textwrap.dedent(script), encoding="utf-8")
    monkeypatch.setattr(process, "ENTRY", f"exec(open({str(tmp_path / 'stand_in.py')!r}).read())")


def run(target, *args, **kwargs):
    kwargs.setdefault("on_line", None)
    return process.call(target, *args, python=sys.executable, name="ovrtx", **kwargs)


def test_a_call_hands_back_its_result_and_progress_and_its_log_says_how_to_run_it_again():
    progress = []
    result = run("os.path:basename", "out/frames/000001.jpg", on_progress=progress.append)
    assert result == "000001.jpg"
    assert progress[0].step == "starting ovrtx"
    assert progress[0].extra["target"] == "os.path:basename"
    again = Path(progress[0].extra["log"]).read_text(encoding="utf-8").splitlines()[1]
    assert "darkroom.process.child" in again and ".call.json" in again


def test_only_the_newest_calls_logs_are_kept(monkeypatch):
    monkeypatch.setattr(process, "KEPT", 2)
    logs = []
    for name in ("a.txt", "b.txt", "c.txt"):
        progress = []
        run("os.path:basename", name, on_progress=progress.append)
        logs.append(Path(progress[0].extra["log"]))
    assert [log.is_file() for log in logs] == [False, True, True]
    assert sorted(f.name for f in process.LOGS.glob("*.call.json")) == sorted(
        log.with_suffix(".call.json").name for log in logs[1:]
    )


def test_lines_said_and_progress_come_back_as_they_are_said(tmp_path, monkeypatch):
    _stand_in(
        tmp_path,
        """
        import json
        print("2026 [12ms] [Info] a renderer is noisy")
        print("@@ progress " + json.dumps({"step": "rendering", "fraction": 0.5, "sensor": "front"}))
        print("@@ rendered 4 s, 30 °")
        print("@@ result " + json.dumps({"frames": 30}))
        print("@@done 0")
        """,
        monkeypatch,
    )
    progress, lines = [], []
    assert run("x:y", on_progress=progress.append, on_line=lines.append) == {"frames": 30}
    assert progress[1] == process.Progress("rendering", 0.5, {"sensor": "front"})
    assert lines == ["rendered 4 s, 30 °"]


def test_a_target_that_raises_fails_saying_why_and_where_its_log_is():
    with pytest.raises(process.Failed, match="Expecting value") as failed:
        run("json:loads", "not json")
    assert "ovrtx's whole log" in str(failed.value)
    assert any("JSONDecodeError" in e for e in failed.value.errors)
    assert failed.value.log is not None and failed.value.log.is_file()


def test_a_process_that_stops_without_saying_so_failed(tmp_path, monkeypatch):
    _stand_in(tmp_path, "raise SystemExit(0)\n", monkeypatch)
    with pytest.raises(process.Failed, match="stopped without finishing"):
        run("x:y")


def test_a_process_whose_gpu_runs_out_of_memory_is_killed_and_failed(tmp_path, monkeypatch):
    _stand_in(
        tmp_path,
        """
        import time
        print("2026 [5ms] [Error] [omni.rtx] Out of GPU memory allocating resource 'x'", flush=True)
        time.sleep(30)  # it would go on rendering black: killed instead
        print("@@done 0")
        """,
        monkeypatch,
    )
    began = time.monotonic()
    with pytest.raises(process.Failed, match="ran out of memory"):
        run("x:y")
    assert time.monotonic() - began < 10


def test_env_is_set_in_the_process_over_the_callers(monkeypatch):
    monkeypatch.setenv("DARKROOM_TEST_SWITCH", "off")
    assert run("os:getenv", "DARKROOM_TEST_SWITCH", env={"DARKROOM_TEST_SWITCH": "on"}) == "on"


def test_a_long_call_is_handed_over_whole():
    # In a file, not the environment: Windows caps a variable at 32,767 characters.
    text = "x" * 100_000
    assert run("builtins:len", text) == 100_000


def test_one_renderer_at_a_time_holds_the_machines_lock_while_it_runs():
    from filelock import FileLock, Timeout

    with process.one_at_a_time(on_line=None), pytest.raises(Timeout):  # another would wait
        FileLock(process.LOGS / process.LOCK).acquire(blocking=False)
    other = FileLock(process.LOGS / process.LOCK)  # and goes once it is done
    other.acquire(blocking=False)
    other.release()
