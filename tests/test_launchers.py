"""ovrtx's process: a function run in it, what it returns handed back, its progress as it
comes, its failing and the GPU running out said."""

import sys

import pytest

from darkroom import process
from darkroom.renderers import launchers


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    """A launcher whose process is this Python, its logs and the machine's one-renderer lock
    in the test's own folder (a renderer running on the machine does not hold the test up)."""
    monkeypatch.setenv(launchers.PYTHON, sys.executable)
    monkeypatch.setattr(process, "LOGS", tmp_path)
    return launchers.OvrtxLauncher()


def said(text):  # run in the process: says how far it is, then what it was told
    process.progress("half way", 0.5, sensor="front")
    print("a line of its own")
    return {"said": text}


def telemetry():
    import os

    return os.environ.get("NVIDIA_OMNIVERSE_PRODUCT_TELEMETRY_DISABLED")


def fails():
    raise RuntimeError("no such camera")


def runs_out():
    print("[Error] [rtx] ERROR_OUT_OF_DEVICE_MEMORY", flush=True)
    import time

    time.sleep(60)  # goes on, as ovrtx does: the launcher stops it


def _target(name):
    return f"test_launchers:{name}"


@pytest.fixture(autouse=True)
def _importable(monkeypatch):
    """This module importable by the process: its folder handed on with the workspace's."""
    from pathlib import Path

    sources = process.source_paths()
    monkeypatch.setattr(process, "source_paths", lambda: [str(Path(__file__).parent), *sources])


def test_a_function_runs_in_the_process_and_its_progress_and_result_come_back(launcher, tmp_path):
    heard, started = [], []
    result = launcher.call(
        _target("said"), "hello", on_progress=heard.append, on_start=started.append, on_line=None
    )
    assert result == {"said": "hello"}
    assert heard[0].step == "starting ovrtx" and heard[0].extra["log"].startswith(str(tmp_path))
    assert (heard[1].step, heard[1].fraction, heard[1].extra) == (
        "half way",
        0.5,
        {"sensor": "front"},
    )
    assert len(started) == 1 and hasattr(started[0], "terminate")
    log = next(tmp_path.glob("*.log")).read_text(encoding="utf-8")
    assert "a line of its own" in log and "The same call again" in log
    assert "darkroom.process.child" in log


def test_ovrtxs_process_is_started_told_to_send_nvidia_no_report(launcher, monkeypatch):
    monkeypatch.delenv("NVIDIA_OMNIVERSE_PRODUCT_TELEMETRY_DISABLED", raising=False)
    assert launcher.call(_target("telemetry"), on_line=None) == "1"


def test_what_fails_in_the_process_is_raised_with_why(launcher):
    with pytest.raises(process.Failed, match="no such camera") as raised:
        launcher.call(_target("fails"), on_line=None)
    assert any("RuntimeError: no such camera" in e for e in raised.value.errors)


def test_the_gpu_running_out_stops_the_process_and_says_so(launcher):
    with pytest.raises(process.Failed, match="ran out of memory") as raised:
        launcher.call(_target("runs_out"), on_line=None)
    assert any("ERROR_OUT_OF_DEVICE_MEMORY" in e for e in raised.value.errors)


def test_no_python_with_ovrtx_says_how_to_get_one(monkeypatch):
    monkeypatch.delenv(launchers.PYTHON, raising=False)
    monkeypatch.setattr(launchers.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="uv sync"):
        launchers.OvrtxLauncher().python()
