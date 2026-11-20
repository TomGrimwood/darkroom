"""What a sensor made, in words, whichever renderer made it: a `Result` (how many frames or
scans, at what rate), worded with the time it took by `said` alone, and how its rendering
ended (`Outcome`: that sentence, or why it failed). Plain Python: loads no USD.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Result:
    """What a sensor made: how many (`unit`: "frames" or "scans") at what rate, what else is
    said of it (a camera's "4096x3000", a lidar's "1.42 M points"), and notes after ("3 rendered
    off their poses, not kept"). `timed`: its time said after the detail (a lidar that saw
    nothing says none)."""

    count: int
    unit: str
    rate: float
    detail: str = ""
    notes: tuple[str, ...] = ()
    timed: bool = True


def said(result: Result, seconds: float) -> str:
    """`result` in words, with the `seconds` it took: "30 frames at 15 Hz, 1280x960, in 2 s"."""
    words = f"{result.count} {result.unit} at {result.rate:g} Hz"
    if result.detail:
        words += f", {result.detail}"
    if result.timed:
        words += f", in {seconds:.0f} s"
    return words + "".join(f" ({note})" for note in result.notes)


@dataclass(frozen=True)
class Outcome:
    """How a sensor's rendering ended: what it made, in words (`said`), or why it
    failed; how many frames or scans."""

    name: str
    said: str
    failed: bool = False
    count: int | None = None

    @property
    def line(self) -> str:
        """ "front: 30 frames at 15 Hz, ...", or "front: FAILED: why"."""
        return f"{self.name}: FAILED: {self.said}" if self.failed else f"{self.name}: {self.said}"


def made_before(out: Path, kind: str, sensors: list[Any]) -> tuple[list[Outcome], list[Any]]:
    """Those of `sensors` made before in capture `out` (a capture carried on: `format.was_made`)
    kept as they are, said, not rendered again: their outcomes, and the sensors left to render."""
    from darkroom import events, format, process

    kept: list[Outcome] = []
    left: list[Any] = []
    for sensor in sensors:
        before = format.was_made(out, sensor.name)
        if before is None:
            left.append(sensor)
            continue
        kept.append(Outcome(sensor.name, f"{before.said} (made before)", count=before.count))
        made = events.Made(kind, sensor.name, before.said, rate=sensor.rate, count=before.count)
        process.progress(f"{sensor.name} made before", None, **events.extra(made))
    return kept, left


def finished(
    kind: str,
    sensor: Any,
    running: Any,
    seconds: Callable[[Any], Any],
    span: float,
    took: float,
    out: Path,
    step: str,
    keep_from: float = -math.inf,
) -> Outcome:
    """`sensor`, rendered, finished (its `running.finish`): its mosaic made, marked made with what it
    made, and said done (`events.Made`, its progress under `step`); one failing is said so and
    leaves the others."""
    from darkroom import events, format, process
    from darkroom.mosaic import mosaic

    try:
        result = running.finish(seconds, span, keep_from)
        made_of = said(result, took)
        process.progress(f"{step}: mosaic of {sensor.name}")
        mosaic(out / sensor.name, out / f"{sensor.name}_mosaic.jpg")
        format.made(out, sensor.name, made_of, result.count)
        outcome = Outcome(sensor.name, made_of, count=result.count)
        made = events.Made(kind, sensor.name, made_of, seconds=took, count=result.count)
    except Exception as exc:  # one sensor failing leaves the others
        outcome = Outcome(sensor.name, str(exc), failed=True)
        made = events.Made(kind, sensor.name, str(exc), failed=True, seconds=took)
    process.progress(f"{step}: {sensor.name} done", None, **events.extra(made))
    return outcome
