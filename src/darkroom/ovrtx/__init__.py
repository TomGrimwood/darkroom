"""ovrtx's side of a capture: a stage's cameras, its lidars and the sky above each GNSS
antenna, rendered by NVIDIA's ovrtx (RTX without Isaac Sim), in the process
`renderers.launchers.OvrtxLauncher` started for it. A capture whose `renderer_cfg` is an
`OVRTXRendererCfg` starts no Isaac Sim.

Run there through `OvrtxLauncher.call("darkroom.ovrtx:render_all", ...)` (`run.captures`
does it): several stages in one process, as Isaac Sim's side (`darkroom.kit`). A kind at a
time, each with an engine of its own (`render`): RTX's motion BVH is an engine's from when it
is made, and only lidars and rolling shutters need it.

- The cameras (`_cameras`) are rendered as the timeline plays the window once, as many on
  together as the GPU's memory holds (`renderers.once`): ovrtx renders each only when it is
  due, so they need no loop of laps (`renderers.loop`, Isaac Sim's). Their pictures, and
  where the stage asks, each pixel's segment, normal and distance.
- The lidars (`_lidars`) are measured as the timeline plays the window once: each a copy of
  the lidar with a `PointCloud` render product, its scans handed over as they end
  (`ovrtx_lidar`) and written into its LAZ on a thread of its own as the render goes on
  (`lidarscans.ScanWriter`).
- The skies (`_skies`): a fisheye camera beside each antenna (`darkroom.eyes.sky`) rendering
  each pixel's semantic class, a view a fix (`darkroom.skies.view`).

Time is a plain `SimulationContext`: nothing plays by itself; it and ovrtx's clock are
stepped together, an update at a time. What ovrtx could not make it says itself: an output's
`status` and `error_message` (`ovrtx_renderer._raise_failed`), or the step raising.

What was measured with ovrtx 0.7.0 and ovstage 0.4.0 (an RTX 5080, 16 GB), kept here
(0.6.0's where it differs):

- pxr (usd-core) and ovrtx run in one process; ovstage plays a stage's USD time itself,
  skinned characters and all (`update_from_usd_time`, in seconds).
- Two clocks: USD time, set before each step, and ovrtx's own, which each step advances and
  which schedules its sensors (`omni:sensor:frameRate`). A camera at 15 Hz stepped at 60 has
  a frame in the step that ends on each of its periods, showing the stage at the USD time
  set before that step: a bar crossing at 10 m/s was within half a pixel of where the stage
  has it in every frame (a frame's travel: 21 pixels).
- A rolling shutter (`OmniRtxRollingShutterAPI`; `ovrtx_usd.rolling_shutter` has how a
  camera's readout and shutter time become its line offset, line exposure and start time;
  `sensorkit.camera` the convention): ovrtx samples a line once, at the end of the interval
  it is given, and blurs nothing by it. The bar's rows were within 1.3 pixels of where it
  was as each was read. `omni:sensor:Core:partialOutputs` has to be written false through
  ovstage, or a part-read frame comes back each step.
- Rows are traced from where things are in the step they fall in, a thing turned between a
  step's two poses the short way: a blade at 40 turns a second read over 100 ms comes out as
  the stage says at 240 and at 960 steps a second (60 and 15 degrees a step: 0.87 and 0.88
  of its outline shared with what was expected), and wrong at the stage's 60 (240 degrees a
  step: 0.17). `OVRTXRendererCfg.step_rate` says how often to step. Isaac Sim's rolling
  shutter has one update a frame: 0.20.
- RTX's motion BVH on, 0.7.0 path traces a camera 2.4 times slower (a 1280x960 frame of a
  street 27 ms for 7 ms; it turns shader execution reordering off) and `MotionBvh.AUTO` is
  no faster (23 ms). Pictures are the same with and without it (2.5 of 255 apart, as two
  runs are). So it is on only for the lidars and for cameras with a rolling shutter.
- A render product and its camera can be referenced into the open stage and taken out again
  (`add_usd_reference_from_string`, `remove_usd`). The product's `camera` relationship has
  to be written through ovstage: a reference drops a target outside it.
- A camera's eye needs its lens's API schema authored on it: pxr's `GetAppliedSchemas`
  leaves out NVIDIA's lens schemas where they are not registered (anywhere but Isaac Sim),
  and without it a fisheye renders as a pinhole (`darkroom.eyes.eye`).
- A stage loads in 2 s once ovrtx's caches are warm; a stage's first render ever compiles
  shaders (85 s once for a street, 230 s the first time ever), a camera's first render
  after that takes 4 to 9 s; then a step with no camera due takes 3 ms and one with every
  camera's frame as long as the GPU takes over them (three cameras 50 ms: 44 ms of GPU, by
  ovrtx's own profile, kept in a capture's `perf.json` under "ovrtx"). A 12 Mpx camera
  takes 3 GB of the GPU's memory: fourteen were rendered five, five and four at once.
- Out of the GPU's memory ovrtx says so two ways: a camera too big by itself, "Out of GPU
  memory" on its output (the process is stopped on it, `darkroom.process.call`); many at once, the
  step fails with nothing said of memory (DLSS cannot be made: "NGX CreateFeature failed"),
  which `OVRTXRenderer._step` says as out of memory when the GPU is full. Either way the
  capture steps back. No output's `status` was seen to say it.
- The same stage, window and look by both renderers (a walk past a curb, three cameras, 16
  frames each): each of ovrtx's frames nearest Isaac Sim's of the same pose, not its
  neighbours'; mean absolute difference 10.0, 7.7 and 1.3 of 255 (two runs of one renderer:
  2.5, 2.3, 0.0; 0.6.0: 10.3, 7.3, 1.3), half that at a quarter of the size: Isaac Sim's
  pictures are softer, and ovrtx's shadows and grass a little lighter.
- Segments (ovrtx's `SemanticSegmentation`, named by its `SemanticIdMap`): 99.4 to 99.99 %
  of pixels labelled as Isaac Sim labels them. Distances (`DistanceToCameraSD`, metres from
  the camera) are exact on a wall 5 m ahead, plain camera or OmniLens, still or moving (Isaac
  Sim's an OmniLens camera's near clip short, made up by `darkroom.kit.camera.near_of`: the
  1.3 to 1.5 % they were apart on the curb stage). There each renderer's agree with its own
  lidars' points to half a millimetre (the median). Normals (`NormalSD`, the world frame, as
  Isaac Sim's) are the surface's own, where Isaac Sim's have its material's normal maps in
  them (13 degrees apart, the median; a plain wall's within 0.3, alike).
- A sky view: ovrtx renders a fisheye past 90 degrees off its axis, Isaac Sim nothing
  there (left as nothing hit, `_Sky`). Within, 97.5 % of pixels classed alike, the open sky
  a fix 0.6 % apart on average and 2.7 % at most (0.918 and 0.921 of it open on a walk);
  ovrtx's view of a fix is nearer Isaac Sim's of the fix before (98.3 %), so one of them is
  under a fix late; ovrtx's cameras were checked against the stage (the bar), Isaac Sim's
  skies were not.
- ovrtx's lidar is Isaac Sim's (the generic lidar model, the same attributes but for a few
  names: `ovrtx_usd.LIDAR_RENAMED`), and takes a capture's lidar prims as they are but for
  how they lie: ovrtx's sensor axes are a camera's (ahead down the prim's -z), Isaac Sim's the
  prim's own, so the lidar that measures is turned (`ovrtx_usd.LIDAR_TURN`). Unturned, every
  point is in other axes and nothing says so.
- A scan comes whole in the step that ends it (`partialOutputs` false), stamped with the
  time of ovrtx's clock it began at, each point with when it fired after that; its channel
  is the lidar's `channelId`, one more than Isaac Sim's ring for the same emitter.
- The same lidars over the same two seconds of a walk, by both (four lidars, 20 scans each):
  points a scan within 2 to 6 %; ranges', materials' and hit prims' shares alike to 97 to
  99 %; the same channel fired at the same instant in the same direction. Intensities are
  no longer alike for the rotating lidars (their distributions share 69 to 79 %, 98 % with
  0.6.0; a level one's median twice Isaac Sim's, one tilted 30 degrees down 1.35 times): 0.7.0
  changed its Gaussian beam at grazing angles (its changelog: NVBug 6897288), Isaac Sim 6.2
  has not. The solid-state lidar's are alike (93 %).
- Motion within a scan: ovrtx traces a firing from where the lidar was in the step it fell
  in. Stepped at the stage's frames (six steps a scan), a lidar swaying as it goes has its
  points 3.4 mm off a room's walls (the median), as one moving steadily has; stepped once a
  scan, as Isaac Sim's lidars are, 98 mm, the same as Isaac Sim's 98 mm: both then trace
  between a scan's first and last pose. So ovrtx's lidars step at the stage's frames, which
  costs time: four lidars over 18 s took 33 s (23 ms of GPU a step), Isaac Sim's 11 s.
- What a lidar's points hit: with no ID map on the stage every point's object is 0; with
  one on the lidar's own product (`StableIdMap` or `StableIdMapDeltas`) a step took 340 to
  460 ms for 5 to 19; with one on a product of its own, rendered once before the lidars
  are put on (`OVRTXRenderer.object_table`, 8 s for a street's 45 MB), their steps cost
  nothing more. 20 scans with hits: 20 s (58 s the map on the lidar; 0.6.0 66 s; Isaac Sim
  45 s). Two seconds of four lidars without: 9 s (0.6.0: 24 s; Isaac Sim 13 s).
- A camera's `omni:rtx:post:motionblur:*` go on its render product and ovrtx blurs by them
  as Isaac Sim does, with its rolling shutter (a bar crossing at 640 pixels a second, an
  exposureFraction of 1.19: a streak of 18.5 pixels for Isaac Sim's 16.5; 10.7 expected).
  It is RTX's post-process, not an exposure: one edge smeared, the other sharp. Its streak
  is `exposureFraction` of the camera's frame interval, whatever the steps, as Isaac Sim's
  is whatever its updates: cameras at 30 down to 7.5 Hz, each with exposureFraction for one
  1/60 s shutter, streaks of 18.5 to 19.5 pixels (Isaac Sim 17 to 20), 2 to 8 steps a frame;
  30 Hz at 60, 120 and 240 steps a second, 19.5. Past 8 steps between a camera's frames
  (ovrtx keeps a thing's last 8 transforms: "MaxHistoryTransformCount 8") it breaks, a
  doubled streak at the 40 pixels the measure stops at, at any step rate (12 Hz at 120, 30
  Hz at 480, a 5 Hz camera at the stage's 60): such a camera is refused (`BLUR_STEPS`).
  With a rolling shutter it comes out shorter the finer the steps (30 Hz: 18.8 pixels at 60
  and 120 steps a second, 11 at 240).
  `omni:rtx:rollingShutter:lineExposure` blurs nothing (it moves when a row is read, as
  ovrtx's docs say). A true exposure is the mean of renders whose readout begins ever
  earlier across the shutter's time (12 gave a box streak of 10.5 pixels): valid, K
  renders a frame, and done by hand, not offered here.
- AV1 off the GPU (`ovrtx:compression:type = "av1"` on a RenderVar, its QPs set: its
  `targetQuality` changed nothing): a 1280x960 frame in 90 kB at 28.9 dB of the picture
  rendered, where the JPEG a capture writes, encoded to AV1 afterwards as sensors2rrd does,
  is 90 kB at 27.9 dB; the step no slower. Measured only: a capture keeps JPEGs.
- ovrtx sends NVIDIA a report as it starts and uploads crash dumps unless told not to: a
  capture's process is told (`launchers.OVRTX_ENVIRONMENT`, `OVRTXBackend`).

This module imports neither `ovrtx` nor `pxr` at the top: a cfg is read (`ovrtx_renderer_cfg`)
where neither is installed.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from darkroom.outcome import Outcome
    from darkroom.renderers.simulation_context import SimulationContext
    from darkroom.renderjob import RenderJob


def render_all(jobs: list[dict[str, Any]], kinds: list[str] | None = None) -> dict:
    """Each of `jobs` (`darkroom.renderjob.RenderJob`s as plain data) rendered in turn in this
    one process, its cameras, lidars and skies (those of `kinds`). Returns {"jobs": [{"sensors",
    "timings"} or {"error"}, ...]}: one failing leaves the rest."""
    from darkroom import renderjob

    done = []
    for job in jobs:
        try:
            done.append(render(renderjob.read(job), kinds=kinds))
        except Exception as exc:
            done.append({"error": f"{type(exc).__name__}: {exc}"})
    return {"jobs": done}


def render(job: RenderJob, kinds: list[str] | None = None) -> dict:
    """Render `job`'s stage's cameras, its lidars, then its skies (those of `kinds`), into its capture
    folder with ovrtx. Returns a line about each sensor, and the timings; raises when any
    sensor failed, or for a kind that is not ovrtx's to render."""
    from pxr import Usd

    from darkroom import perf, process, renderers, sensorkit, skies
    from darkroom.renderers.simulation_context import SimulationContext

    from . import profile

    other = sorted(set(kinds or ()) - set(KINDS))
    if other:
        raise RuntimeError(f"ovrtx renders no {', '.join(other)}: those are Isaac Sim's.")
    kinds = [kind for kind in KINDS if kinds is None or kind in kinds]
    # The parts timed; the GPU's numbers are ovrtx's own (`profile`), not read off the card.
    watch = perf.Watch(read=lambda: None)
    cfg: Any = renderers.of(job.renderer_cfg)
    outcomes: list[Outcome] = []
    measured: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        watch.part("stage")
        process.progress("opening the stage")
        stage = Usd.Stage.Open(str(Path(job.stage_file).resolve()))
        if not sensorkit.sensors(stage, [kind]):
            continue
        if kind == "gnss" and (over := skies.layer(stage)):
            # The skies are rendered without what the stage's GNSS layer takes out.
            stage.GetSessionLayer().subLayerPaths.append(over)
        origin = stage.GetStartTimeCode() / stage.GetTimeCodesPerSecond()
        sim = SimulationContext(stage, job=job, origin=origin)
        try:
            # An engine a kind: RTX's motion BVH is an engine's from when it is made, and
            # cameras with no rolling shutter render 2.4 times faster without it.
            moving = _moving(stage, kind) if cfg.motion_bvh is None else cfg.motion_bvh
            renderer = sim.get_or_create_backend(dataclasses.replace(cfg, motion_bvh=moving))
            renderer.initialize()
            process.progress("loading the stage into ovrtx")
            renderer.prepare_stage(stage, 1)
            watch.part({"camera": "cameras", "lidar": "lidars", "gnss": "skies"}[kind])
            profiling = profile.Profile()
            try:
                outcomes += KINDS[kind](sim, renderer)
            finally:
                measured[kind] = profiling.finish()  # before the engine goes
        finally:
            sim.clear_instance()
    took = {**watch.finish(), **profile.stats(measured)}
    perf.add(Path(job.out), took)  # the capture's all, in perf.json
    timings = perf.line(took)  # this process's own
    failed = [o.line for o in outcomes if o.failed]
    if failed:
        raise RuntimeError("; ".join(failed))
    return {"sensors": [o.line for o in outcomes], "timings": timings}


def _moving(stage: Any, kind: str) -> bool:
    """Whether rendering `stage`'s sensors of `kind` needs RTX's motion BVH: its lidars do (a
    scan's firings each at their own time), its cameras when one has a rolling shutter, its
    skies never."""
    from darkroom import sensorkit

    if kind != "camera":
        return kind == "lidar"
    cameras = sensorkit.sensors(stage, [kind])
    return any(sensorkit.camera.lens(stage.GetPrimAtPath(c.path)).readout > 0 for c in cameras)


def _cameras(sim: SimulationContext, renderer: Any) -> list[Outcome]:
    """Every camera of the simulation's stage rendered by `renderer` as the timeline plays
    its job's window once (`renderers.once`), all on together, from `job.warmup` camera
    periods before it (frames not kept). An outcome each; those made before (a capture
    carried on) kept as they are."""
    from darkroom import sensorkit
    from darkroom.outcome import made_before
    from darkroom.rates import update_rate
    from darkroom.renderers import once

    stage, job = sim.stage, sim.job
    kept, sensors = made_before(Path(job.out), "camera", sensorkit.sensors(stage, ["camera"]))
    if not sensors:
        return kept
    start, end = job.window
    # At least as often as the stage's own frames (or as the renderer's cfg says: its
    # `step_rate`), and a whole number of updates between each camera's frames: ovrtx renders
    # a camera in the update that ends on its period.
    at_least = max(stage.GetTimeCodesPerSecond(), getattr(renderer.cfg, "step_rate", None) or 0)
    rate = update_rate([s.rate for s in sensors], at_least)
    refuse_blur(stage, sensors, rate)
    sim.dt = 1 / rate
    lead = max(1 / s.rate for s in sensors) * max(1, job.warmup)
    return kept + once.play(
        renderer,
        sim,
        sensors,
        first=sim.origin + start,
        last=sim.origin + end,
        lead=lead,
        rate=rate,
    )


#: The most steps between a camera's frames ovrtx blurs its motion right at (its RTX
#: post-process motion blur, `omni:rtx:post:motionblur:*`): more, and the streak doubles.
BLUR_STEPS = 8


def refuse_blur(stage: Any, sensors: list[Any], rate: float) -> None:
    """Raises, saying what to do, when ovrtx stepping `rate` times a second would put more
    than `BLUR_STEPS` steps between the frames of a camera of `sensors` that blurs motion."""
    from darkroom import sensorkit

    for sensor in sensors:
        blur = stage.GetPrimAtPath(sensor.path).GetAttribute(sensorkit.camera.MOTION_BLUR)
        steps = round(rate / sensor.rate)
        if blur and blur.Get() and steps > BLUR_STEPS:
            raise RuntimeError(
                f"{sensor.name} blurs motion, which ovrtx gets right with at most {BLUR_STEPS} "
                f"steps between its frames; {rate:g} steps a second put {steps} between its "
                f"{sensor.rate:g} Hz frames. Step ovrtx at most {BLUR_STEPS * sensor.rate:g} "
                "times a second (its cfg's step_rate), or turn its blur off."
            )


def _lidars(sim: SimulationContext, renderer: Any) -> list[Outcome]:
    """Every lidar of the simulation's stage measured by `renderer` (ovrtx's) as the timeline
    plays its job's window once, from a scan before it (not kept), all in the same updates.
    An outcome each; those made before (a capture carried on) kept as they are.

    The updates are the stage's own frames (60 a second) or a whole number to each lidar's
    scan (`darkroom.rates`), not one a scan: ovrtx traces a firing from where the lidar was
    in the update it fell in, so a scan gathered over six has six of the lidar's poses in it."""
    import math
    import time

    from darkroom import events, format, lidarscans, process, sensorkit
    from darkroom.events import Plan, Planned, Rendering, Writing
    from darkroom.outcome import finished, made_before
    from darkroom.rates import update_rate
    from darkroom.renderers import outfit
    from darkroom.sensorkit import semantics

    from .ovrtx_lidar import Names

    stage, job = sim.stage, sim.job
    out, kind = Path(job.out), "lidar"
    kept, sensors = made_before(out, kind, sensorkit.sensors(stage, [kind]))
    if not sensors:
        return kept
    start, end = job.window
    first, last = sim.origin + start, sim.origin + end
    rate = update_rate([s.rate for s in sensors], stage.GetTimeCodesPerSecond())
    sim.dt = 1 / rate
    planned = [
        Plan(s.name, kind, s.rate, None, 0, 1, int((end - start) * s.rate + 1e-9)) for s in sensors
    ]
    process.progress("lidars: planned", None, **events.extra(Planned(kind, tuple(planned))))
    # With hits, what every object in the scene is called, read once, before the lidars are
    # on: only with that on the stage do their points say what they hit.
    table = renderer.object_table() if job.hits else None
    names = Names(table) if table is not None else None
    # A segmented stage's labels name what each point hit, as its cameras' segments do.
    taxonomy = stage.GetRootLayer().customLayerData.get(format.SEGMENT_TAXONOMY)
    labels = semantics.labeller(stage, taxonomy) if taxonomy and names else None
    span = end - start
    measuring = {
        s.name: renderer.create_lidar_data(
            str(s.path),
            lidarscans.ScanWriter(out, s.name, s.rate, span, names, job.blind, labels),
            job.hits,
        )
        for s in sensors
    }
    # ovrtx starts a lidar's scans on its own clock's whole periods, and that clock ran on
    # through the cameras: the timeline is started early by the updates it takes to come to
    # a whole number of every lidar's periods (`beat`), and a beat more, so each lidar's
    # scans start with the window, those before it not kept.
    beat = math.lcm(*(round(rate / s.rate) for s in sensors))
    early = -round(sim.clock * rate) % beat
    sim.seek(first - (beat + early) / rate)
    ahead = sim.clock - sim.time  # ovrtx's clock over the timeline: the same all the way

    def seconds(stamp_ns: Any) -> Any:
        """Seconds after the window's start the stage was at when ovrtx's clock said `stamp_ns`."""
        return stamp_ns / 1e9 - ahead - first

    for one in measuring.values():
        one.scans.start(seconds)
    process.progress("rendering lidars", 0.0, **events.extra(Rendering(kind, 0, 0.0)))
    began = told = time.perf_counter()
    try:
        # Past the window's end by the updates a scan takes to be handed over, as every pass.
        while sim.time < last + outfit.TRAILING / rate:
            sim.step()
            renderer.update_transforms()
            renderer.scan_batch(list(measuring.values()))
            now = time.perf_counter()
            if now - told >= outfit.PROGRESS:
                told = now
                fraction = min(max(sim.time - first, 0.0), end - start) / (end - start)
                said = Rendering(kind, 0, fraction, now - began)
                process.progress("rendering lidars", fraction, **events.extra(said))
    except BaseException:
        for one in measuring.values():  # nothing half-written left in the capture
            one.scans.abort()
        raise
    finally:
        for one in measuring.values():  # each render product off the stage
            renderer.cleanup(one)
    took = time.perf_counter() - began
    process.progress("lidars: writing", None, **events.extra(Writing(kind, 0)))
    return kept + [
        finished(kind, s, _Scans(measuring[s.name]), seconds, span, took, out, "lidars")
        for s in sensors
    ]


class _Scans:
    """A lidar measured, for `outcome.finished`: its scans' LAZ finished."""

    def __init__(self, measured: Any) -> None:
        self.measured = measured

    def finish(self, seconds: Any, span: float, keep_from: float = 0.0) -> Any:
        """Its writer finished (`lidarscans.ScanWriter.finish`, which kept the whole scans
        within the window). Raises when ovrtx gave its points another way than asked."""
        measured = self.measured
        if measured.wrong is not None:
            measured.scans.abort()
            raise RuntimeError(f"ovrtx gave points as {measured.wrong}, not x, y, z in its axes.")
        return measured.scans.finish()


def _skies(sim: SimulationContext, renderer: Any) -> list[Outcome]:
    """The sky above every GNSS antenna of the simulation's stage, rendered by `renderer`
    (ovrtx's) as the timeline plays its job's window once, a view a fix from one before it
    (not kept): a camera beside each antenna (`darkroom.eyes.sky`) rendering each pixel's
    semantic class, classed open, leaves or blocked (`darkroom.skies.view`). An outcome each;
    those made before (a capture carried on) kept as they are."""
    import math
    import time

    from darkroom import events, eyes, process, sensorkit
    from darkroom.events import Plan, Planned, Rendering, Writing
    from darkroom.outcome import finished, made_before
    from darkroom.rates import update_rate
    from darkroom.renderers import CameraCfg, CameraData, CameraRenderSpec, outfit

    stage, job = sim.stage, sim.job
    out, kind = Path(job.out), "gnss"
    kept, sensors = made_before(out, kind, sensorkit.sensors(stage, [kind]))
    if not sensors:
        return kept
    start, end = job.window
    first, last = sim.origin + start, sim.origin + end
    # A step a fix (a whole number to each antenna's): a sky view needs none between.
    rate = update_rate([s.rate for s in sensors], max(s.rate for s in sensors))
    sim.dt = 1 / rate
    size = sensorkit.sky.SIZE
    planned = [
        Plan(s.name, kind, s.rate, (size, size), 0, 1, int((end - start) * s.rate + 1e-9) + 1)
        for s in sensors
    ]
    process.progress("skies: planned", None, **events.extra(Planned(kind, tuple(planned))))
    watching: dict[str, _Sky] = {}
    began = told = time.perf_counter()
    try:
        for s in sensors:
            camera = eyes.sky(stage, stage.GetPrimAtPath(s.path))
            cfg = CameraCfg(
                prim_path=str(camera.GetPath()),
                width=size,
                height=size,
                update_period=1 / s.rate,
                data_types=("semantic_segmentation",),
            )
            data = CameraData()
            data.allocate(cfg.data_types, size, size, 1, renderer.supported_output_types())
            spec = CameraRenderSpec(cfg, sim.device, 1, (cfg.prim_path,), 1)
            made = renderer.create_render_data(spec)
            renderer.set_outputs(made, data.output)
            watching[s.name] = _Sky(s, made, data, out)
        # The window's start a whole number of every antenna's periods of ovrtx's clock on,
        # two views before it not kept: a sky camera's first view shows the stage where it
        # was loaded, not where it is (measured: an antenna moving out from under a roof,
        # its first view at the stage's start, every later one exactly at its fix).
        beat = math.lcm(*(round(rate / s.rate) for s in sensors))
        sim.seek(first - (2 * beat + (-round(sim.clock * rate) % beat)) / rate)
        ahead = sim.clock - sim.time  # ovrtx's clock over the timeline: the same all the way
        process.progress("rendering skies", 0.0, **events.extra(Rendering(kind, 0, 0.0)))
        while sim.time < last + outfit.TRAILING / rate:
            sim.step()
            renderer.update_transforms()
            renderer.render_batch([one.made for one in watching.values()])
            for one in watching.values():
                one.read(renderer)
            now = time.perf_counter()
            if now - told >= outfit.PROGRESS:
                told = now
                fraction = min(max(sim.time - first, 0.0), end - start) / (end - start)
                said = Rendering(kind, 0, fraction, now - began)
                process.progress("rendering skies", fraction, **events.extra(said))
    finally:
        for one in watching.values():  # each render product off the stage
            renderer.cleanup(one.made)
    took = time.perf_counter() - began
    process.progress("skies: writing", None, **events.extra(Writing(kind, 0)))

    def seconds(stamp_ns: Any) -> Any:
        """Seconds after the window's start the stage was at when ovrtx's clock said `stamp_ns`."""
        return float(stamp_ns) / 1e9 - ahead - first

    return kept + [
        finished(kind, s, watching[s.name], seconds, end - start, took, out, "skies")
        for s in sensors
    ]


class _Sky:
    """An antenna's sky camera on ovrtx (`made`, rendering into `data`), for
    `outcome.finished`: its views as they come, kept by the fix they belong to."""

    def __init__(self, sensor: Any, made: Any, data: Any, out: Path) -> None:
        from darkroom import sensorkit

        self.sensor, self.made, self.data, self.out = sensor, made, data, out
        #: Each view rendered: the time of ovrtx's clock it shows, seconds, and the view.
        self.views: list[tuple[float, Any]] = []
        #: The pixels more than 90 degrees off the view's axis: ovrtx renders its fisheye
        #: there too (the ground under a level antenna), Isaac Sim nothing. Left as nothing
        #: hit, so a view is the same either way; no signal is counted from there.
        self.under = sensorkit.sky.directions()[..., 2] <= 0

    def read(self, renderer: Any) -> None:
        """What ovrtx rendered since the last read, classed (`darkroom.skies.view`)."""
        from darkroom import sensorkit, skies

        renderer.read_output(self.made, self.data)
        info = self.data.info
        for capture in (info.get("rgba") or {}).get("capture", ()):
            named = (info.get("semantic_segmentation") or {}).get("idToLabels") or {}
            ids = self.data.output["semantic_segmentation"][0, ..., 0]
            seen = skies.view(ids, named)
            seen[self.under] = sensorkit.sky.OPEN
            self.views.append((capture["time"], seen))

    def finish(self, seconds: Any, span: float, keep_from: float = 0.0) -> Any:
        """The views within the window written, one a fix, named by the fix's number
        (`format.write_sky`). Raises with none in the window."""
        import numpy as np

        from darkroom import format, sensorkit
        from darkroom.outcome import Result

        sensor, kept = self.sensor, set()
        for stamp, seen in self.views:
            second = float(seconds(stamp * 10**9))
            number = int(np.floor(second * sensor.rate + 1e-3))
            if -1e-6 <= second <= span + 1e-6 and number not in kept:
                format.write_sky(self.out, sensor.name, number, seen)
                kept.add(number)
        if not kept:
            raise RuntimeError(f"no sky views within the window's {span:g} s.")
        size = sensorkit.sky.SIZE
        return Result(len(kept), "frames", sensor.rate, f"{size}x{size}")


#: What renders each kind ovrtx renders, in the order they are rendered.
KINDS = {"camera": _cameras, "lidar": _lidars, "gnss": _skies}
