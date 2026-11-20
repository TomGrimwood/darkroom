"""ovrtx's own measures of a capture, in short (`darkroom.ovrtx.profile`)."""

from darkroom.ovrtx import profile


def _metric(uid, name, group, units, values, parent=0):
    samples = [
        {"timestamp_ns": n, "value": v, "parent_uid": parent, "execution_id": 0}
        for n, v in enumerate(values)
    ]
    return {
        "uid": uid,
        "parent_uid": parent,
        "name": name,
        "group_name": group,
        "units": units,
        "values": samples,
    }


METRICS = [
    _metric(1, "RTX Rendering [1]", profile.PASSES, "ms", [20.0, 40.0]),
    _metric(2, "Viewport Texture 0 (1280x960)", profile.PASSES, "ms", [12.0, 14.0], parent=1),
    _metric(3, "Path Tracing", profile.PASSES, "ms", [7.0, 8.0], parent=2),
    _metric(4, "Viewport Texture 1 (960x540)", profile.PASSES, "ms", [8.0], parent=1),
    _metric(5, "Path Tracing", profile.PASSES, "ms", [4.5], parent=4),
    _metric(6, "GPU Memory v2: GPU0", profile.MEMORY, "MiB", [2400.2, 2792.4]),
    _metric(7, "BVH", profile.MEMORY, "MiB", [480.0, 764.2], parent=6),
    _metric(8, profile.GPU_MEMORY, "VRAM", "MB", [2900.0, 3430.5]),
    _metric(9, profile.GPU_BUSY, "", "%", [60.0, 100.0]),
    _metric(10, "Render Thread", "", "ms", []),
]


def test_a_frames_parts_are_said_by_their_path_each_cameras_its_own():
    said = profile.summary(METRICS)
    tracing = "RTX Rendering [1] / Viewport Texture 0 (1280x960) / Path Tracing"
    assert said["gpu_ms"][tracing] == {"average": 7.5, "most": 8.0, "times": 2}
    other = "RTX Rendering [1] / Viewport Texture 1 (960x540) / Path Tracing"
    assert said["gpu_ms"][other] == {"average": 4.5, "most": 4.5, "times": 1}
    assert said["memory_mib"] == {"GPU Memory v2: GPU0": 2792, "GPU Memory v2: GPU0 / BVH": 764}
    assert said["gpu_memory_most_mb"] == 3430.5 and said["gpu_memory_average_mb"] == 3165
    assert said["gpu_busy_average_percent"] == 80 and said["gpu_busy_most_percent"] == 100.0


def test_a_captures_gpu_numbers_are_over_every_kind_ovrtx_rendered():
    cameras = profile.summary(METRICS)
    lidars = {**cameras, "gpu_memory_most_mb": 5000.0, "gpu_busy_average_percent": 40}
    said = profile.stats({"camera": cameras, "lidar": lidars})
    assert said["ovrtx"] == {"camera": cameras, "lidar": lidars}
    assert said["gpu_memory_most_mb"] == 5000.0 and said["gpu_busy_average_percent"] == 60
    assert said["gpu_memory_before_mb"] == 0.0  # its own memory, not the whole card's
    assert profile.stats({}) == {"ovrtx": {}}
