"""How long a capture's renderer took over each part: kept beside the capture as `FILE`, with
what ovrtx measured of itself (`ovrtx.profile`), and said in one line (`line`).

    watch = perf.Watch()
    watch.part("stage")
    ...
    watch.part("cameras")
    ...
    took = watch.finish()         # how long it took, and each part
    print(perf.line(took))        # "took 61 s: stage 1 s, cameras 60 s"

Plain Python: the renderer's side uses it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

#: The file a capture's timings are kept in.
FILE = "perf.json"
#: What of two processes' timings is, for the capture, the greater of the two.
MOST = ("gpu_memory_most_mb", "gpu_busy_most_percent")


class Watch:
    """The parts of a run timed, from when it is made until `finish`."""

    def __init__(self) -> None:
        self._began = time.perf_counter()
        self._parts: list[tuple[str, float]] = []

    def part(self, name: str) -> None:
        """A part of the run called `name` begun; the one before it ended."""
        self._parts.append((name, time.perf_counter()))

    def finish(self) -> dict[str, Any]:
        """Stopped: how long the run took ("seconds") and each of its parts ("parts": a name
        and its seconds each, in order)."""
        now = time.perf_counter()
        begun = [began for _, began in self._parts] + [now]  # each ends as the next begins
        parts = [
            {"name": name, "seconds": round(end - began, 2)}
            for (name, began), end in zip(self._parts, begun[1:], strict=True)
        ]
        return {"seconds": round(now - self._began, 2), "parts": parts}


def line(stats: dict[str, Any]) -> str:
    """A process's timings (`Watch.finish`'s, with ovrtx's own of the GPU if it has them) as
    one line: "took 61 s: stage 1 s, cameras 60 s; GPU memory 9.3 GB at most, GPU 32% busy on
    average"."""
    parts = ", ".join(f"{p['name']} {p['seconds']:.0f} s" for p in stats["parts"])
    said = f"took {stats['seconds']:.0f} s: {parts}"
    if "gpu_memory_most_mb" in stats:
        said += f"; GPU memory {stats['gpu_memory_most_mb'] / 1024:.1f} GB at most"
    if "gpu_busy_average_percent" in stats:
        said += f", GPU {stats['gpu_busy_average_percent']}% busy on average"
    return said


def add(out: Path, stats: dict[str, Any]) -> dict[str, Any]:
    """`stats` added to those capture `out` has already (`FILE`: an earlier process's, of a
    capture carried on) and written there: the seconds summed, the parts one after another,
    each kind's ovrtx measures kept, the GPU's most the greater. Returns them all."""
    file = out / FILE
    if file.is_file():
        had = json.loads(file.read_text(encoding="utf-8"))
        both = {**had, **stats}
        both["seconds"] = round(had["seconds"] + stats["seconds"], 2)
        both["parts"] = had["parts"] + stats["parts"]
        if "ovrtx" in had and "ovrtx" in stats:
            both["ovrtx"] = {**had["ovrtx"], **stats["ovrtx"]}
        for key in MOST:
            if key in had and key in stats:
                both[key] = max(had[key], stats[key])
        stats = both
    file.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats
