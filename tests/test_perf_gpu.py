"""The GPU as NVML says, the one with the least free on a machine of several; a capture's
timings in one line; a sensor's mosaic."""

import ctypes

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
        "memory_most_mb": 13312,
        "gpu_memory_before_mb": 4198.4,
        "gpu_memory_most_mb": 9523.2,
        "gpu_busy_average_percent": 32,
    }
    assert perf.line(took) == (
        "took 61 s: stage 1 s, cameras 60 s; GPU memory 9.3 GB at most (4.1 GB before), "
        "GPU 32% busy on average; 13.0 GB of memory"
    )


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
