"""A capture, start to end: the truth from USD, then ovrtx in a process of its own for the
cameras, the lidars and the GNSS antennas' skies (its launcher, `darkroom.process.call`), their progress
handed to `on_progress` as it comes, then how much of each antenna's view was sky (`skies`).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from darkroom import process, sensorkit
from darkroom.process import OUT_OF_GPU

from . import events, format, looks, renderers, renderjob, skies, truth, turns
from .options import Job, Options
from .renderers import renderer_cfg
from .renderjob import RenderJob


@dataclass
class _Ready:
    """A job its truth written, waiting for its renderer: where it is in what `captures`
    returns, what is rendered of it, and what the renderer is handed (`job`)."""

    at: int
    skies: bool
    rendered: set[str]
    options: Options
    job: RenderJob


def _out_of_memory(failed: process.Failed) -> bool:
    """Whether the renderer's process failed for want of the GPU's memory."""
    return any(o in e for e in failed.errors for o in OUT_OF_GPU)


def _step_back(
    one: _Ready,
    planned: dict[str, events.Plan],
    riding: list[str],
    left: list[str],
    runtime: str,
    on_line: Callable[[str], object] | None,
    on_progress: Callable[[process.Progress], object] | None,
) -> bool:
    """`one`'s cameras rendered again in a new process of their renderer's (`runtime`) after
    one ran out of the GPU's memory, what it made kept. The cameras on
    the loop then (`riding`, sized as `planned`) were too many at once: no more pixels on at
    once than theirs less the smallest's (`RenderJob.most`). One camera on alone goes again as
    it was if any camera of `left` (those unmade as it started) was made first: a new process
    starts with the GPU's memory free, where RTX kept what earlier cameras took. False when a
    camera alone ran out with nothing made: nothing smaller to try."""
    unmade = _unmade(one)
    on = [planned[n] for n in riding if n in planned and planned[n].size]
    if not on:
        return False
    names = ", ".join(p.name for p in on)
    most = turns.step_back([p.size for p in on if p.size])
    if most is not None:
        was = one.job.most
        most = min(most, was) if was else most
        one.job = replace(one.job, most=most)
        said = (
            f"cameras: the GPU ran out of memory rendering {names} at once; stepping back, "
            f"at most {most / 1e6:.1f} Mpx at once, in a new {runtime}"
        )
    elif len(unmade) < len(left):
        said = (
            f"cameras: the GPU ran out of memory rendering {names} after other cameras; "
            f"carrying on in a new {runtime}, what was made kept"
        )
    else:
        return False
    if on_line:
        on_line(said)
    if on_progress:
        on_progress(process.Progress(said, None, {}))
    return True


def _unmade(one: _Ready) -> list[str]:
    """The cameras of `one`'s capture its renderer has not made all of (`format.was_made`)."""
    out = Path(one.job.out)
    meta = format.read_meta(out)
    return [
        n for n, s in meta.sensors.items() if s.kind == "camera" and not format.was_made(out, n)
    ]


def _whole(out: Path, name: str, kind: str) -> bool:
    """Whether sensor `name` of the capture in `out` was made whole (`format.was_made`), a
    lidar with its scans' LAZ there: one an older darkroom left with its scans not yet in a
    LAZ is rendered again."""
    if format.was_made(out, name) is None:
        return False
    return kind != "lidar" or (out / name / format.PACKED).is_file()


def _carry_on(out: Path, stage_file: Path, on_line: Callable[[str], object] | None) -> list[str]:
    """A capture of `stage_file` in `out` that did not finish, made ready to carry on: the
    sensors its renderer made all of kept (`format.was_made`), the rest cleared to render again
    (the truth is written again as before). Returns those kept."""
    before = format.read_meta(out)
    if Path(before.stage).resolve() != stage_file:
        raise RuntimeError(
            f"{out} has an unfinished capture of {before.stage}, not {stage_file}: "
            "capture into a new folder."
        )
    kept = [name for name in before.sensors if _whole(out, name, before.sensors[name].kind)]
    part = 0
    for name in set(before.sensors) - set(kept):
        (out / f"{name}_mosaic.jpg").unlink(missing_ok=True)
        # (a segmented camera's labels are numbered as a run finds them: all again)
        if before.sensors[name].kind != "camera" or (out / name / format.SEGMENTS).is_dir():
            shutil.rmtree(out / name, ignore_errors=True)
            continue
        # A camera part made keeps its whole frames, named by their time (`renderers.frame_store`): it
        # renders only those it has not. What else the renderer left goes.
        whole = format.frames_kept(out, name)
        part += bool(whole)
        frames = out / name / format.FRAMES
        for file in frames.iterdir() if frames.is_dir() else ():
            if file.is_dir():
                shutil.rmtree(file)
            elif not (file.suffix == ".jpg" and file.stem.isdigit() and int(file.stem) in whole):
                file.unlink()
    if on_line:
        said = f"carrying on {out}: {len(kept)} of {len(before.sensors)} sensors made before, kept"
        on_line(said + (f"; {part} cameras part made, their frames kept" if part else ""))
    return kept


#: Told each line a capture says (what it made, what it took).
Said = Callable[[str], object]


def capture(
    stage_file: str | Path,
    out: str | Path,
    options: Options = Options(),
    *,
    on_progress: Callable[[process.Progress], object] | None = None,
    on_start: Callable[[Any], object] | None = None,
    on_line: Said | None = print,
) -> Path:
    """Capture every sensor on the stage (or of `options.kinds`) over `options.window` into
    the folder `out`, made as `options` say: a new or empty folder, or one whose capture did
    not finish, carried on (only the sensors not made all of before rendered). Returns it.

    `on_progress` is handed the render's progress as it comes (its `extra` a `darkroom.events`
    event for a board of the sensors, `board.Board.take`); `on_start` the renderer's process
    started (a `process.Stopper`, to stop it); `on_line` each line said.
    Raises `darkroom.Failed` when the renderer's side fails, `RuntimeError` for a stage that cannot
    be captured as it is (no sensors, no time range).
    """
    (made,) = captures(
        [Job(stage_file, out, options)],
        on_progress=on_progress,
        on_start=on_start,
        on_line=on_line,
    )
    if isinstance(made, Exception):
        raise made
    return made


#: The most stages one renderer process renders: GPU memory a stage took is not all given back
#: before the next, so more start another.
STAGES_AT_ONCE = 4


def captures(
    jobs: Sequence[Job],
    *,
    on_progress: Callable[[process.Progress], object] | None = None,
    on_start: Callable[[Any], object] | None = None,
    on_line: Said | None = print,
) -> list[Path | Exception]:
    """Several captures (`jobs`) in as few renderer processes as render them: one for the jobs
    rendering the same kinds of sensor, another every `STAGES_AT_ONCE` stages.
    Returns each capture's folder, or what went wrong with it, in the order of `jobs`: one
    failing leaves the others, but the GPU running out of memory stops them all
    (`darkroom.Failed`, raised)."""
    made: list[Path | Exception] = [RuntimeError("not prepared")] * len(jobs)
    ready: list[_Ready] = []
    for at, job in enumerate(jobs):
        try:
            one = _prepare(at, job, on_progress, on_line)
        except Exception as exc:
            made[at] = exc
            continue
        made[at] = Path(job.out)
        ready.append(one)
    for sim in _group(ready):
        alive = [one for one in sim.jobs if not isinstance(made[one.at], Exception)]
        # At most `STAGES_AT_ONCE` stages a process: it does not give back all the GPU memory
        # one stage took before the next.
        for start in range(0, len(alive), STAGES_AT_ONCE):
            batch = alive[start : start + STAGES_AT_ONCE]
            _render(sim, batch, made, on_progress, on_start, on_line)
    for one in ready:
        out = made[one.at]
        if not isinstance(out, Exception):
            _finish(one, out, on_progress, on_line)
    return made


def _prepare(
    at: int,
    job: Job,
    on_progress: Callable[[process.Progress], object] | None,
    on_line: Said | None,
) -> _Ready:
    """Job `at` of `captures` made ready for its renderer: an unfinished capture in its folder
    carried on, its truth written, its fisheye lenses and antennas' skies worked out. Raises
    what is wrong with it."""
    o = job.options
    look, warmup = looks.resolve(o.look)
    spans = tuple((float(d.start), float(d.end), looks.resolve(d.look)[0]) for d in o.during)
    out = Path(job.out)
    if format.finished(out):
        raise RuntimeError(f"{out} has a capture in it already: capture into a new folder.")
    stage_file = Path(job.stage_file).resolve()
    kept = _carry_on(out, stage_file, on_line) if (out / format.META).exists() else []
    meta = truth.write(
        stage_file, out, o.window, o.kinds, on_line=on_line, cell=o.scenery_cell, bodies=o.bodies
    )
    for name in kept:  # a board shows them done from the start
        sensor, was = meta.sensors.get(name), format.was_made(out, name)
        if on_progress and sensor is not None and was is not None:
            s = sensor.settings
            size = (int(s["width"]), int(s["height"])) if "width" in s else None
            before = events.Made(
                sensor.kind, name, was.said, rate=sensor.rate, size=size, count=was.count
            )
            on_progress(process.Progress(f"{name} made before", None, events.extra(before)))
    # Each fisheye camera rendered as ovrtx's own OpenCV fisheye, said in its settings
    # ("rendered"): it fills its frame, corners and all, where OpenCV's model puts every point
    # (a tenth of a pixel, to 164 degrees off its axis).
    lensed = _fisheyes(meta)
    for sensor in meta.sensors.values():
        if sensor.prim in lensed:
            sensor.settings["rendered"] = lensed[sensor.prim]["model"]
    for sensor in meta.sensors.values():  # and the quality its frames are written at
        if sensor.kind == "camera":
            sensor.settings["jpeg_quality"] = o.jpeg_quality
    meta.extra[format.RENDERER] = renderers.resolve(o.renderer_cfg).renderer_type
    format.write_meta(out, meta)
    antennas = skies.wanted(meta) if o.skies else []
    rendered: set[str] = {s.kind for s in meta.sensors.values() if s.kind in sensorkit.RENDERED}
    if antennas:
        rendered.add("gnss")
    return _Ready(
        at=at,
        skies=bool(antennas),
        rendered=rendered,
        options=o,
        job=RenderJob(
            stage_file=str(stage_file),
            out=str(out.resolve()),
            window=(meta.start, meta.end),
            look=look,
            warmup=warmup,
            spans=spans,
            lenses=lensed,
            blind=o.blind,
            hits=o.lidar_hits,
            packing=o.packing,
            quality=o.jpeg_quality,
            renderer_cfg=renderer_cfg.plain(renderers.resolve(o.renderer_cfg)),
        ),
    )


def _fisheyes(meta: format.Meta) -> dict[str, dict[str, Any]]:
    """How each fisheye camera of a capture is rendered, by its prim path: as ovrtx's own
    OpenCV fisheye (`eyes.lenses`)."""
    return {
        s.prim: {"model": sensorkit.camera.FISHEYE}
        for s in meta.sensors.values()
        if s.kind == "camera" and s.settings.get("model") == sensorkit.camera.FISHEYE
    }


@dataclass
class _Sim:
    """A renderer process's worth of ready jobs: the kinds it renders, and the jobs."""

    kinds: list[str]
    jobs: list[_Ready]


#: The kinds of sensor the renderer renders, in the order it renders them.
RENDERED = ("camera", "lidar", "gnss")


def _group(ready: list[_Ready]) -> list[_Sim]:
    """The processes `ready` render in: one for the jobs that render the same kinds."""
    sims: dict[tuple[str, ...], _Sim] = {}
    for one in ready:
        kinds = tuple(kind for kind in RENDERED if kind in one.rendered)
        if kinds:
            sims.setdefault(kinds, _Sim(list(kinds), [])).jobs.append(one)
    return list(sims.values())


def _render(
    sim: _Sim,
    batch: list[_Ready],
    made: list[Path | Exception],
    on_progress: Callable[[process.Progress], object] | None,
    on_start: Callable[[Any], object] | None,
    on_line: Said | None,
) -> None:
    """`batch` rendered in a process of the renderer's own (its launcher's), what went wrong
    with each put in `made`: after the GPU ran out of memory, its cameras again in a new one
    with fewer on at once (`_step_back`); again for what a process left unmade without
    failing, each sensor made kept (`format.MADE`)."""
    planned: dict[str, events.Plan] = {}  # each camera's size, as planned
    riding: list[str] = []  # the cameras on the loop, as last said

    def heard(progress: process.Progress) -> None:
        said = events.read(progress.extra)
        if isinstance(said, events.Planned) and said.kind == "camera":
            planned.update((p.name, p) for p in said.sensors)
        if isinstance(said, events.Riding) and said.kind == "camera":
            riding[:] = said.sensors
        if on_progress:
            on_progress(progress)

    cfg = renderers.resolve(batch[0].options.renderer_cfg)
    if cfg.launcher_type is None:
        raise ValueError(f"{type(cfg).__name__} names no launcher to render in.")
    launcher = renderer_cfg.string_to_callable(cfg.launcher_type)()
    while batch:
        left = {one.at: _unmade(one) for one in batch}
        planned.clear()
        riding.clear()
        try:
            said = launcher.call(
                launcher.target,
                [renderjob.plain(one.job) for one in batch],
                kinds=sim.kinds,
                on_progress=heard,
                on_start=on_start,
                on_line=on_line,
            )
            for one, got in zip(batch, said["jobs"], strict=True):
                if got.get("error"):
                    made[one.at] = RuntimeError(got["error"])
                elif on_line:  # each sensor: how many frames, at what rate, how long
                    for line in [*got.get("sensors", []), got.get("timings", "")]:
                        if line:
                            on_line(line)
        except process.Failed as exc:
            # The process itself failed (out of the GPU's memory, say): each of its stages
            # with it, what it made kept to carry on from. The next process starts with the
            # GPU's memory free (the lidars'), so goes on.
            if "camera" in sim.kinds and _out_of_memory(exc):
                # A step back, not a failure: what is left again in a new process, the
                # cameras that ran out without the smallest (`turns.step_back`).
                smaller = []
                for one in batch:
                    if _step_back(
                        one, planned, riding, left[one.at], launcher.name, on_line, on_progress
                    ):
                        smaller.append(one)
                    else:
                        made[one.at] = exc  # a camera alone ran out: no further back
                if smaller:
                    batch = smaller
                    continue
            for one in batch:
                made[one.at] = exc
        except Exception as exc:
            for one in batch:
                made[one.at] = exc
        again = []
        for one in batch:
            if isinstance(made[one.at], Exception) or "camera" not in sim.kinds:
                continue
            unmade = _unmade(one)
            if unmade and len(unmade) == len(left[one.at]):  # none made: never would be
                made[one.at] = RuntimeError(f"{launcher.name} made none of {', '.join(unmade)}.")
            elif unmade:
                again.append(one)
        batch = again


def _finish(
    one: _Ready,
    out: Path,
    on_progress: Callable[[process.Progress], object] | None,
    on_line: Said | None,
) -> None:
    """`one`'s capture in `out` finished once all is rendered: how much of each antenna's view
    was sky measured (`skies`), and it marked done."""
    if one.skies and on_progress:
        on_progress(process.Progress("measuring how much sky each antenna saw", None, {}))
    for line in skies.write(one.job.stage_file, out) if one.skies else []:
        if on_line:
            on_line(line)
    format.done(out)
