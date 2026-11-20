"""Cameras rendered as the timeline plays the window once, as many on together as the GPU's
memory holds: ovrtx renders each camera only when it is due (every camera has its own rate,
and a step renders those whose period it ends), so the window is played once for the cameras
that fit together, again for those left.

- Who is on (`board`): the biggest first; another while what it would need, by what the last
  put on took of the GPU's memory a pixel, fits in what is free less the packing's headroom
  (`turns.PACKING`; "one by one": each alone). Each is rendered a couple of frames as it is
  put on, the timeline held: that is when it takes its memory.
- The timeline then starts `lead` seconds before the window (frames not kept: a render's
  first are black, its denoiser has no history), where the window's start is a whole number
  of every camera's periods of the renderer's clock on: each renders on its own poses.
- A frame is named by the time it shows (`frame_store.FrameStore`): the renderer's clock at
  it, less what that clock is ahead of the timeline, the same all the pass.
- Out of memory all the same, a capture is stepped back (`RenderJob.most`: no more pixels
  on at once than those on then, less the smallest's).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import gpu, turns
from .base_renderer import BaseRenderer
from .outfit import PROGRESS, TRAILING, covered, outfit
from .simulation_context import SimulationContext


def board(
    waiting: list[Any],
    pixels: Callable[[Any], int],
    put_on: Callable[[Any], None],
    most: int | None,
    headroom: float | None,
    read: Callable[[], tuple[float, float] | None] = gpu.memory,
) -> list[Any]:
    """The cameras of `waiting` (biggest first) put on for a pass (`put_on`: on the renderer,
    rendered until it has taken its memory), as many as the GPU's memory has room for: the
    first always; another while what it would need, by what the last put on took a pixel
    (`read`: GB in use, of how many), fits in what is free less `headroom` of the whole
    (None: each alone), and they have no more than `most` pixels between them. Returns those
    on, taken out of `waiting`."""
    on: list[Any] = []
    a_pixel = 0.0  # bytes a pixel, as the last put on took
    for sensor in list(waiting):
        size = pixels(sensor)
        if on:
            if headroom is None:
                break
            if most is not None and sum(pixels(s) for s in on) + size > most:
                continue
            used = read()
            if used is not None and size * a_pixel > turns.room(*used, headroom):
                continue
        before = read()
        put_on(sensor)
        after = read()
        if before is not None and after is not None and size:
            a_pixel = max(after[0] - before[0], 0.0) * 2**30 / size
        waiting.remove(sensor)
        on.append(sensor)
    return on


def play(
    renderer: BaseRenderer,
    sim: SimulationContext,
    sensors: list[Any],
    *,
    first: float,
    last: float,
    lead: float,
    rate: float,
) -> list[Any]:
    """The cameras `sensors` (`sensorkit.Sensor`s) rendered by `renderer` as the timeline
    plays the window of `sim.job` once (`first` to `last`, from `lead` before, `rate` updates
    a second), all on together, into its capture folder (`outfit`), said on a board.
    Returns an outcome each (`darkroom.outcome.Outcome`)."""
    from darkroom import events, process
    from darkroom.events import Planned, Rendering, Riding, Writing

    job, out, kind = sim.job, Path(sim.job.out), "camera"
    made = outfit(renderer, sim, [sensors])
    said = f"{len(made.order)} cameras, as many on at once as the GPU has room for ({job.packing}"
    said += ", stepped back after running out of memory)" if job.most is not None else ")"
    process.progress(f"cameras: planned, {said}", None, **events.extra(Planned(kind, made.planned)))
    span = last - first
    waiting = list(made.order)
    outcomes: list[Any] = []
    while waiting:
        running: dict[str, Any] = {}

        def update(running: dict[str, Any] = running) -> None:
            """An update: the simulation a step on, the scene as it is then shown to the
            renderer, each camera's pose with it, every camera on rendered and read."""
            sim.step()
            renderer.update_transforms()
            renderer.update_geometries()
            for one in running.values():
                one.update(sim.time)
            renderer.render_batch([one.data for one in running.values()])
            for one in running.values():
                one.read()

        def put_on(sensor: Any, running: dict[str, Any] = running) -> None:
            """`sensor` on the renderer, rendered two of its frames with the timeline held
            before the window: what it takes of the GPU's memory is taken by then."""
            running[sensor.name] = made.camera(sensor, out / sensor.name)
            for _ in range(2 * round(rate / sensor.rate)):
                sim.seek(first - lead - 1 / rate)
                update()

        try:
            sim.span(first - lead - 2 / rate, last + (TRAILING + 2) / rate)
            sim.play()
            on = board(
                waiting,
                lambda s: math.prod(made.size(s)),
                put_on,
                job.most,
                turns.PACKING[job.packing],
            )
            names = tuple(s.name for s in on)
            riding = f"cameras: on, {', '.join(names)}"
            process.progress(riding, None, **events.extra(Riding(kind, names)))
            # The window's start a whole number of every camera's periods of the renderer's
            # clock on, the lead a whole number of them too.
            beat = math.lcm(*(round(rate / s.rate) for s in on))
            before = math.ceil(round(lead * rate) / beat) * beat / rate
            sim.seek(first - before - (-round(sim.clock * rate) % beat) / rate)
            ahead = sim.clock - sim.time  # the renderer's clock over the timeline's

            def seconds(stamp_ns: Any, ahead: float = ahead) -> Any:
                """Seconds after the window's start the timeline was at, at `stamp_ns` of
                the renderer's clock."""
                return float(stamp_ns) / 1e9 - ahead - first

            began = told = time.perf_counter()
            while sim.time < last + TRAILING / rate:
                update()
                now = time.perf_counter()
                if now - told >= PROGRESS:
                    told = now
                    for s in on:  # its frames named as they come: kept should the renderer stop
                        have = running[s.name].keep(seconds, span, 0.0)
                        begin, share = covered(have, made.frames(s))
                        turn = made.turn_of[s.name]
                        told_of = Rendering(kind, turn, share, now - began, begin)
                        process.progress("rendering cameras", share, **events.extra(told_of))
        finally:
            for one in running.values():  # each render product off the stage, its memory back
                one.stop()
        took = time.perf_counter() - began
        process.progress("cameras: on, none", None, **events.extra(Riding(kind, ())))
        for s in on:
            off = Writing(kind, made.turn_of[s.name])
            process.progress(f"cameras: {s.name} off", None, **events.extra(off))
            outcomes.append(made.off(s, running[s.name], 0.0, took, seconds))
    sim.stop()
    return outcomes
