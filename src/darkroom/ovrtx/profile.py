"""What ovrtx measured of itself as it rendered (its profiling API): how long the GPU took
over each part of a frame, and the GPU's memory by what holds it. A capture's `perf.json`
has it per kind rendered, under "ovrtx" (`summary`), and its line's GPU numbers
(`darkroom.perf.line`) are these: this process's own, not the whole card's.

    measured = Profile()      # once an engine is made: profiling is the process's
    ...
    said = measured.finish()  # before the engine goes
"""

from __future__ import annotations

from typing import Any

#: The group of ovrtx's metrics that are a frame's parts on the GPU, and that of its memory.
PASSES, MEMORY = "RTX Render Pass", "RTX Memory"
#: ovrtx's metrics a capture's line says: the GPU's memory in use (MB), how busy it is (%).
GPU_MEMORY, GPU_BUSY = "GPU Memory", "GPU Utilization"


class Profile:
    """ovrtx's profiling on from when it is made until `finish`."""

    def __init__(self) -> None:
        import ovrtx

        self._data = ovrtx.ovrtx_performance_data_update_t()
        ovrtx.start_profiling()

    def finish(self) -> dict[str, Any]:
        """Profiling stopped: what ovrtx measured since (`summary`)."""
        import ovrtx

        try:
            _, metrics = ovrtx.get_performance_data_update(self._data)
        finally:
            try:
                ovrtx.release_performance_data_update(self._data)
            finally:
                ovrtx.stop_profiling()
        return summary(metrics)


def summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """ovrtx's `metrics` (as `ovrtx.get_performance_data_update` returns them) in short:
    "gpu_ms", each part of a frame the GPU worked on by its path ("RTX Rendering [1] /
    Viewport Texture 0 (1280x960) / Path Tracing": milliseconds on average and at most, and
    how many times); "memory_mib", the GPU's memory at its most by what held it; and the
    GPU's memory in use and how busy it was, at most and on average."""
    by_uid = {m.get("uid"): m for m in metrics}

    def path(metric: dict[str, Any]) -> str:
        names, seen = [metric["name"]], {metric.get("uid")}
        parent = by_uid.get(metric.get("parent_uid"))
        while parent is not None and parent.get("uid") not in seen:
            seen.add(parent.get("uid"))
            names.append(parent["name"])
            parent = by_uid.get(parent.get("parent_uid"))
        return " / ".join(reversed(names))

    said: dict[str, Any] = {"gpu_ms": {}, "memory_mib": {}}
    for metric in metrics:
        values = [v["value"] for v in metric["values"]]
        if not values:
            continue
        if metric["group_name"] == PASSES and metric["units"] == "ms":
            said["gpu_ms"][path(metric)] = {
                "average": round(sum(values) / len(values), 3),
                "most": round(max(values), 3),
                "times": len(values),
            }
        elif metric["group_name"] == MEMORY:
            said["memory_mib"][path(metric)] = round(max(values))
        elif metric["name"] == GPU_MEMORY:
            said["gpu_memory_most_mb"] = round(max(values), 1)
            said["gpu_memory_average_mb"] = round(sum(values) / len(values))
        elif metric["name"] == GPU_BUSY:
            said["gpu_busy_average_percent"] = round(sum(values) / len(values))
            said["gpu_busy_most_percent"] = round(max(values), 1)
    return said


#: Of a capture's kinds' summaries, what is one number for them all, and how.
OVER_ALL = {
    "gpu_memory_most_mb": max,
    "gpu_memory_average_mb": lambda values: round(sum(values) / len(values)),
    "gpu_busy_average_percent": lambda values: round(sum(values) / len(values)),
    "gpu_busy_most_percent": max,
}


def stats(kinds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """What a capture's `perf.json` has of ovrtx's own measures: each kind's `summary` under
    "ovrtx", and the GPU's memory and how busy it was over them all."""
    said: dict[str, Any] = {"ovrtx": kinds}
    for key, over in OVER_ALL.items():
        values = [kind[key] for kind in kinds.values() if key in kind]
        if values:
            said[key] = over(values)
    return said
