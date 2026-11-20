"""How long a capture's renderer took over each part, and the GPU while it went (`gpu`):
written beside the capture as `FILE`, and said in one line (`line`).

    watch = perf.Watch()          # the GPU read once a second from now
    watch.part("stage")
    ...
    watch.part("cameras")
    ...
    took = watch.finish()         # how long each part took, the GPU at its most and on average
    print(perf.line(took))        # "took 61 s: stage 1 s, cameras 60 s; GPU memory ..."

GPU memory is the whole card's, so `gpu_memory_before_mb` says what was in use before.
Plain Python: the renderer's side uses it.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import gpu

#: The file a capture's timings are kept in.
FILE = "perf.json"


def _read() -> tuple[float, float] | None:
    """The GPU's memory in use, MB, and how busy it is, percent; None unread."""
    used, busy = gpu.memory(), gpu.busy()
    if used is None or busy is None:
        return None
    return used[0] * 1024, busy


class Watch:
    """The parts of a run timed, the GPU read every `every` seconds until `finish`."""

    def __init__(
        self, every: float = 1.0, read: Callable[[], tuple[float, float] | None] = _read
    ) -> None:
        self._began = time.perf_counter()
        self._parts: list[dict[str, Any]] = []
        self._samples: list[tuple[float, float]] = []
        self._read = read
        first = read()
        self._before = first[0] if first else None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, args=(every,), daemon=True)
        self._thread.start()

    def _watch(self, every: float) -> None:
        while not self._stop.wait(every):
            if seen := self._read():
                self._samples.append(seen)

    def part(self, name: str) -> None:
        """A part of the run called `name` begun; the one before it ended."""
        now = time.perf_counter()
        if self._parts:
            self._parts[-1]["seconds"] = now - self._parts[-1].pop("began")
        self._parts.append({"name": name, "began": now})

    def finish(self) -> dict[str, Any]:
        """Stopped: how long the run and each part took, the GPU's memory at its most and on
        average and how busy it was, and the most memory this process used."""
        now = time.perf_counter()
        self._stop.set()
        self._thread.join()
        if self._parts and "began" in self._parts[-1]:
            self._parts[-1]["seconds"] = now - self._parts[-1].pop("began")
        for part in self._parts:
            part["seconds"] = round(part["seconds"], 2)
        stats: dict[str, Any] = {
            "seconds": round(now - self._began, 2),
            "parts": self._parts,
            "memory_most_mb": _memory_most_mb(),
        }
        if self._samples:
            used = [sample[0] for sample in self._samples]
            busy = [sample[1] for sample in self._samples]
            stats.update(
                gpu_memory_before_mb=self._before,
                gpu_memory_most_mb=max(used),
                gpu_memory_average_mb=round(sum(used) / len(used)),
                gpu_busy_average_percent=round(sum(busy) / len(busy)),
                gpu_busy_most_percent=max(busy),
            )
        return stats


def line(stats: dict[str, Any]) -> str:
    """What `finish` returned, as one line: "took 61 s: stage 1 s, cameras 60 s; GPU memory
    9.3 GB at most (4.1 GB before), GPU 32% busy on average; 13.0 GB of memory"."""
    parts = ", ".join(f"{p['name']} {p['seconds']:.0f} s" for p in stats["parts"])
    said = f"took {stats['seconds']:.0f} s: {parts}"
    if "gpu_memory_most_mb" in stats:
        before = stats["gpu_memory_before_mb"] or 0.0
        said += (
            f"; GPU memory {stats['gpu_memory_most_mb'] / 1024:.1f} GB at most "
            f"({before / 1024:.1f} GB before), "
            f"GPU {stats['gpu_busy_average_percent']}% busy on average"
        )
    return said + f"; {stats['memory_most_mb'] / 1024:.1f} GB of memory"


def add(out: Path, stats: dict[str, Any]) -> dict[str, Any]:
    """`stats` added to those in `out`'s `FILE` (another process's, for the same capture),
    written there. Returns them all."""
    file = out / FILE
    if file.is_file():
        had = json.loads(file.read_text(encoding="utf-8"))
        both = {**had, **stats}
        both["seconds"] = round(had["seconds"] + stats["seconds"], 2)
        both["parts"] = had["parts"] + stats["parts"]
        for key in ("memory_most_mb", "gpu_memory_most_mb", "gpu_busy_most_percent"):
            if key in had and key in stats:
                both[key] = max(had[key], stats[key])
        stats = both
    file.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def _memory_most_mb() -> int | None:
    """The most memory this process has held, MB; None where the system does not say (Windows)."""
    try:
        import resource
    except ImportError:
        return None
    # ru_maxrss is in kilobytes on Linux, bytes on macOS.
    most = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(most / (1024 * 1024 if sys.platform == "darwin" else 1024))
