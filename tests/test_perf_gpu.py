"""The GPU as NVML says, the one with the least free on a machine of several; a capture's
timings in one line; a sensor's mosaic."""

import ctypes
import json

from PIL import Image

from darkroom import gpu, perf
from darkroom.mosaic import mosaic


class _Nvml:
    """NVML stood in for: two GPUs, the second with less free."""

    def __init__(self):
        self.gpus = [(16, 4, 50), (16, 12, 90)]  # GB in all, in use; busy percent

    def nvmlInit_v2(self):
        return 0

    def nvmlDeviceGetCount_v2(self, count):
        count._obj.value = len(self.gpus)
        return 0

    def nvmlDeviceGetHandleByIndex_v2(self, index, handle):
        handle._obj.value = index + 1
        return 0

    def nvmlDeviceGetMemoryInfo_v2(self, handle, info):
        total, used, _ = self.gpus[handle.value - 1]
        info._obj.total, info._obj.used = total * 2**30, used * 2**30
        info._obj.free = (total - used) * 2**30
        return 0

    def nvmlDeviceGetUtilizationRates(self, handle, rates):
        rates._obj.gpu = self.gpus[handle.value - 1][2]
        return 0


def test_on_several_gpus_the_one_with_least_free_is_told_of(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _Nvml())
    monkeypatch.setattr(gpu, "_nvml", False)
    assert gpu.memory() == (12.0, 16.0) and gpu.busy() == 90.0


def test_a_captures_timings_are_one_line_as_strider_reads_them():
    took = {
        "seconds": 61.2,
        "parts": [{"name": "stage", "seconds": 1.1}, {"name": "cameras", "seconds": 60.1}],
    }
    assert perf.line(took) == "took 61 s: stage 1 s, cameras 60 s"  # no GPU measured
    measured = {**took, "gpu_memory_most_mb": 9523.2, "gpu_busy_average_percent": 32}
    assert perf.line(measured) == (
        "took 61 s: stage 1 s, cameras 60 s; GPU memory 9.3 GB at most, GPU 32% busy on average"
    )


def test_a_runs_parts_are_timed_one_after_another():
    watch = perf.Watch()
    assert watch.finish()["parts"] == []  # nothing begun: nothing to say, and no failure
    watch.part("stage")
    watch.part("cameras")
    took = watch.finish()
    assert [p["name"] for p in took["parts"]] == ["stage", "cameras"]
    assert took["seconds"] >= sum(p["seconds"] for p in took["parts"]) - 0.02


def test_a_capture_carried_on_adds_its_timings_to_those_it_had(tmp_path):
    # As darkroom 0.1.2 left one on Windows: a figure it had no number for.
    had = {
        "seconds": 10.0,
        "parts": [{"name": "cameras", "seconds": 10.0}],
        "memory_most_mb": None,
        "gpu_memory_most_mb": 5000.0,
        "ovrtx": {"camera": {"gpu_ms": {}}},
    }
    (tmp_path / perf.FILE).write_text(json.dumps(had), encoding="utf-8")
    now = {
        "seconds": 5.0,
        "parts": [{"name": "lidars", "seconds": 5.0}],
        "gpu_memory_most_mb": 3000.0,
        "ovrtx": {"lidar": {"gpu_ms": {}}},
    }
    both = perf.add(tmp_path, now)
    assert both["seconds"] == 15.0 and [p["name"] for p in both["parts"]] == ["cameras", "lidars"]
    assert both["gpu_memory_most_mb"] == 5000.0 and set(both["ovrtx"]) == {"camera", "lidar"}
    assert json.loads((tmp_path / perf.FILE).read_text(encoding="utf-8")) == both
    assert perf.line(both).startswith("took 15 s: cameras 10 s, lidars 5 s; GPU memory 4.9 GB")


def test_a_mosaic_is_a_dozen_of_the_frames_each_shown_small(tmp_path):
    frames = tmp_path / "front" / "frames"
    frames.mkdir(parents=True)
    for i in range(30):
        Image.new("RGB", (1600, 1200), (i * 8, 0, 0)).save(frames / f"{i:06d}.jpg")
    made = mosaic(tmp_path / "front", tmp_path / "front_mosaic.jpg", width=200)
    assert made is not None
    with Image.open(made) as sheet:  # 4 a row, 3 rows, 200 wide and 150 high each
        assert sheet.size == (4 * 208 - 8, 3 * (150 + 28 + 8) - 8)
    assert mosaic(tmp_path / "nothing", tmp_path / "x.jpg") is None
