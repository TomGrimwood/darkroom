"""A render's sensors as a board, from what it reports (its `Progress`, carrying `events`): a row a sensor with its
kind, size, rate, how far it has got, what it made and the rigs it is for. Plain data, for a
window or a terminal to draw: `Board.take` each report, then `rows()`, `row_view` and `headline`.
A caller's own work on what was rendered (a recording made of it, say) is a row too, under the
sensors: sent as a `Step` (`extra(Step(...))`, a progress's `extra`), waiting from the start."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Protocol

from darkroom.process import Progress

from . import events, gpu
from .events import Step as Step  # a caller's own rows, sent as `extra(Step(...))`
from .events import extra as extra

#: A row's states, in the order a sensor goes through them.
WAITING, RENDERING, WRITING, DONE, FAILED = "waiting", "rendering", "writing", "done", "failed"
#: A lidar rendered, its scans being packed in one LAZ (`events.Packing`): not done till packed.
PACKING = "packing"
#: A caller's step (`Step`) under way: work on what was rendered, after it.
MAKING = "making"
#: A row left when its job was cancelled.
STOPPED = "stopped"
#: Kinds worked out from the walk, not rendered (a GNSS antenna also has its sky rendered).
WORKED_OUT = frozenset({"imu", "gnss"})


class Shown(Protocol):
    """What a board is drawn from: a render's kind, its rows, how many of its sensors are
    shared, its step and its state ("running", "done", "failed", "cancelled": a job's status,
    say)."""

    @property
    def kind(self) -> str: ...
    @property
    def sensors(self) -> tuple[SensorRow, ...]: ...
    @property
    def sharing(self) -> tuple[int, int] | None: ...
    @property
    def step(self) -> str: ...
    @property
    def state(self) -> str: ...


@dataclass(frozen=True)
class SensorRow:
    name: str
    kind: str
    rate: float
    #: A camera's picture, (width, height) in pixels; None for other kinds.
    size: tuple[int, int] | None
    #: Its turn of `turns`, from 0: each camera its own (it rides the loop on a render product
    #: of its own, `renderers.once`); a kind's other sensors all in one.
    turn: int
    turns: int
    #: The frames (or scans) it should make over the window.
    frames: int
    state: str = WAITING
    #: How far through the window it has rendered, 0 to 1, from `begin` of the way (a camera
    #: that got on the loop part way: what it rendered runs on round from there, `covered`).
    fraction: float = 0.0
    begin: float = 0.0
    #: How far it was as it started rendering (a camera carried on: what it had), for its eta.
    started: float | None = None
    #: What it made, as its renderer said ("12 frames at 30 Hz, ...").
    said: str = ""
    #: The rigs it is for: more than one when the same sensor is in each (rendered once).
    rigs: tuple[str, ...] = ()
    #: How many it made, once done (its renderer's count, not the `frames` planned).
    made: int | None = None
    #: Its turn's time so far, seconds (wall-clock): the sensors of a turn render together.
    seconds: float = 0.0
    #: A caller's step after the render (`Step`: a recording made of it), not a sensor.
    after: bool = False


def covered(begin: float, fraction: float) -> tuple[tuple[float, float], ...]:
    """The stretches of a bar (0 to 1) a row has rendered: `fraction` of it from `begin`, on
    round past the end to the start (a camera that got on the loop half way: 0.5 to 1, then 0
    up to where it is)."""
    begin, fraction = begin % 1.0, min(max(fraction, 0.0), 1.0)
    end = begin + fraction
    if end <= 1.0:
        return ((begin, end),)
    return ((begin, 1.0), (0.0, end - 1.0))


class Board:
    """The rows, as reports come in (`take`)."""

    def __init__(self) -> None:
        self._rows: dict[str, SensorRow] = {}
        self._uses: dict[str, tuple[str, ...]] = {}
        #: How many sensors the rigs have between them, and how many are rendered (or worked
        #: out) once for them: fewer when rigs share a sensor. None until said.
        self.sharing: tuple[int, int] | None = None

    def take(self, progress: Progress) -> bool:
        """A render's progress taken in (its event, `events.read`): True when it changed a row."""
        match events.read(progress.extra):
            case events.Uses() as uses:
                # Every sensor the capture makes, with the rigs it is for: those the renderer
                # renders are planned again by it; the rest (an IMU, a GNSS's fixes) are worked
                # out from the walk before it starts, done when it does.
                self.sharing = (uses.in_rigs, uses.made)
                for name, use in uses.sensors.items():
                    self._uses[name] = use.rigs
                    row = self._rows.get(name) or SensorRow(name, use.kind, use.rate, None, 0, 1, 0)
                    self._rows[name] = replace(row, rigs=use.rigs)
                return True
            case events.Planned() as planned:
                for (
                    name,
                    row,
                ) in self._rows.items():  # the renderer started: the walk's truth is out
                    if row.kind in WORKED_OUT and row.state == WAITING and row.frames == 0:
                        self._rows[name] = replace(
                            row, state=DONE, fraction=1.0, said="worked out from the walk"
                        )
                for p in planned.sensors:
                    self._rows[p.name] = SensorRow(
                        p.name,
                        p.kind,
                        p.rate,
                        p.size,
                        p.turn,
                        p.turns,
                        p.frames,
                        rigs=self._uses.get(p.name, ()),
                    )
                return True
            case events.Made() as made:  # one made before (a capture carried on) too: done
                row = self._rows.get(made.sensor) or SensorRow(
                    made.sensor, made.kind, made.rate or 0.0, made.size, 0, 1, made.count or 0
                )
                self._rows[row.name] = replace(
                    row,
                    size=row.size or made.size,
                    state=FAILED if made.failed else DONE,
                    fraction=1.0,
                    said=made.said,
                    made=made.count,
                    seconds=row.seconds if made.seconds is None else made.seconds,
                )
                return True
            case events.Packing(sensor, done):
                row = self._rows.get(sensor)
                if row is None or row.state == FAILED:
                    return False
                packed = done >= 1.0
                new = replace(
                    row, state=DONE if packed else PACKING, fraction=1.0 if packed else done
                )
                changed = new != row
                self._rows[sensor] = new
                return changed
            case events.Step() as step:
                return self._step(step)
            case events.Rendering(kind, turn, done, seconds, begin):
                return self._turn(kind, turn, RENDERING, done, seconds, begin)
            case events.Writing(kind, turn):
                return self._turn(kind, turn, WRITING, 1.0)
        return False

    def _turn(
        self,
        kind: str,
        turn: int,
        state: str,
        fraction: float,
        seconds: float | None = None,
        begin: float | None = None,
    ) -> bool:
        """The rows of `kind`'s turn `turn` not yet done, now `state`, `fraction` of the way
        (from `begin`, when said); each with the turn's `seconds` so far, when said."""
        changed = False
        rows = [r for r in self._rows.values() if r.kind == kind and r.turn == turn]
        for row in rows:
            if row.state not in (DONE, FAILED):
                took = row.seconds if seconds is None else seconds
                at = row.begin if begin is None else begin
                started = row.started
                if started is None and state == RENDERING and seconds is not None:
                    started = fraction
                new = replace(
                    row, state=state, fraction=fraction, seconds=took, begin=at, started=started
                )
                if new != row:
                    self._rows[row.name], changed = new, True
        return changed

    def _step(self, step: Step) -> bool:
        """A caller's step's row, as `step` says it is now: waiting, under way (what it is
        doing for what it says), done with what it made, or failed with why."""
        row = self._rows.get(step.name) or SensorRow(
            step.name, step.kind, 0.0, None, 0, 1, 0, after=True
        )
        if step.failed:
            new = replace(row, state=FAILED, said=step.said)
        elif step.finished:
            new = replace(row, state=DONE, fraction=1.0, said=step.said)
        elif step.done is None:
            new = replace(row, state=WAITING, fraction=0.0, said="")
        else:
            # its seconds count from its start: its eta from nothing done
            new = replace(row, state=MAKING, fraction=step.done, said=step.doing, started=0.0)
        new = replace(new, rigs=step.rigs or row.rigs, seconds=step.seconds or row.seconds)
        self._rows[step.name] = new
        return new != row

    def ended(self, state: str) -> tuple[SensorRow, ...]:
        """The rows as a job that ended `state` ("failed", "cancelled") leaves them: those not
        done, failed (or stopped) where they were. Returns them."""
        left = FAILED if state == "failed" else STOPPED
        for name, row in self._rows.items():
            if row.state not in (DONE, FAILED):
                self._rows[name] = replace(row, state=left)
        return self.rows()

    def rows(self) -> tuple[SensorRow, ...]:
        """Each sensor's row: the cameras by turn, the skies, the lidars, then those worked out
        from the walk (an IMU, a GNSS's fixes); else in the order they came. Then the caller's
        steps after the render (`Step`), in the order they were first said."""
        order = {"camera": 0, "gnss": 1, "lidar": 2}

        def place(row: SensorRow) -> tuple[int, int]:
            if row.after:
                return (4, 0)
            worked_out = row.kind in WORKED_OUT and row.frames == 0
            return (3 if worked_out else order.get(row.kind, 3), row.turn)

        return tuple(sorted(self._rows.values(), key=place))


@dataclass(frozen=True)
class RowView:
    """A row as both boards show it (the terminal's, the Run tab's): its state, its cells in
    `COLUMNS`' order, its bar's fraction, and what it made (for a tooltip or the last column)."""

    state: str
    cells: tuple[str, ...]
    fraction: float
    said: str
    #: What its bar says: its state, or "starting" while nothing of it has rendered yet.
    label: str = ""
    #: Where its bar starts filling, 0 to 1 (`covered`).
    begin: float = 0.0


#: The cells' headings, after the state's mark and before the bar; then those after it.
COLUMNS = ("sensor", "size", "rate")
AFTER_BAR = ("frames", "elapsed", "eta", "fps", "for")


def frames_made(r: SensorRow) -> int:
    """The frames (or scans) row `r` has made so far: its renderer's count once done, else
    as far as it has got."""
    if r.made is not None:
        return r.made
    if r.state == RENDERING:
        return round(r.frames * r.fraction)
    return r.frames if r.state in (WRITING, PACKING, DONE) else 0


#: Seconds the pixels a second are worked out over (`Meter`).
SMOOTH = 10.0
#: A chart's marks within a row, low to high (eighths of it).
SPARK = " ▁▂▃▄▅▆▇█"


@dataclass
class Meter:
    """The GPU's busy share and memory in use, and the pixels the cameras render a second,
    read at most every `every` seconds (`read`), every reading of the render kept: for a
    window or a terminal to chart (as nvtop does, from NVML: `gpu`)."""

    every: float = 1.0
    busy: list[float] = field(default_factory=list)  # percent
    used: list[float] = field(default_factory=list)  # GB
    pixels: list[float] = field(default_factory=list)  # a second, over the last `SMOOTH` s
    total: float = 0.0  # the GPU's memory, GB
    _at: float = 0.0
    _rendered: list[tuple[float, float]] = field(default_factory=list)  # (when, pixels made)

    def read(self, rows: tuple[SensorRow, ...], now: float | None = None) -> bool:
        """A reading taken, if `every` seconds have passed: True when one was."""
        now = time.perf_counter() if now is None else now
        if self._at and now - self._at < self.every:
            return False
        rendered = sum(frames_made(r) * r.size[0] * r.size[1] for r in rows if r.size)
        # Over the last `SMOOTH` seconds: frames are counted as they are kept, in bursts, so
        # from one reading to the next the rate jumps about.
        self._rendered = [(t, n) for t, n in self._rendered if now - t <= SMOOTH]
        then, before = self._rendered[0] if self._rendered else (now, rendered)
        rate = (rendered - before) / (now - then) if now > then else 0.0
        self._rendered.append((now, rendered))
        memory, busy = gpu.memory(), gpu.busy()
        self.busy.append(busy or 0.0)
        self.used.append(memory[0] if memory else 0.0)
        self.pixels.append(max(rate, 0.0))
        self.total = memory[1] if memory else self.total
        self._at = now
        return True


def window(values: list[float], columns: int, per: int = 1) -> list[float | None]:
    """The last `columns` x `per` of `values` (a reading a second) as exactly `columns`
    columns of `per` readings each, the newest at the right, each the most in its readings (a
    peak kept); None where there are none yet. A column holds the same readings until it
    scrolls off the left: the chart moves at one column every `per` readings, never rescaled."""
    whole = len(values) // per  # columns filled; the newest, filling, after them
    filling = len(values) % per
    out: list[float | None] = []
    for k in range(columns):
        column = whole - (columns - 1 - k) - (0 if filling else 1)  # the newest at the right
        if column < 0:
            out.append(None)
            continue
        part = values[column * per : (column + 1) * per]
        out.append(max(part) if part else None)
    return out


def chart(values: list[float], top: float, width: int, rows: int = 4, per: int = 5) -> list[str]:
    """`values` as a chart `rows` high and `width` wide, 0 to `top`, scrolling at `per`
    readings a column (`window`): lines, top first, each column filled to its value in eighths
    of a row (`SPARK`), blank before the first reading."""
    columns = window(values, width, per)
    eighths = [
        None if v is None else round(min(max(v / top, 0.0), 1.0) * rows * 8) if top > 0 else 0
        for v in columns
    ]
    return [
        "".join(" " if e is None else SPARK[min(max(e - row * 8, 0), 8)] for e in eighths)
        for row in reversed(range(rows))
    ]


def meter_charts(m: Meter, width: int, rows: int = 4, per: int = 5) -> list[tuple[str, list[str]]]:
    """The meter's three charts, `width` wide, `per` readings (seconds) a column: each its
    title (what it is, the latest, the span shown) and its lines; none read yet, none."""
    if not m.busy:
        return []
    span = f"last {width * per / 60:.0f} min"
    return [
        (f"GPU {m.busy[-1]:.0f}%  {span}", chart(m.busy, 100.0, width, rows, per)),
        (f"VRAM {m.used[-1]:.1f} / {m.total:.0f} GB", chart(m.used, m.total, width, rows, per)),
        (
            f"{m.pixels[-1] / 1e6:.0f} Mpx/s",
            chart(m.pixels, max(m.pixels, default=0.0), width, rows, per),
        ),
    ]


def duration(seconds: float) -> str:
    """A time as a board says it: "40 s", "12 min", "1 h 05"."""
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} min"
    hours, minutes = divmod(round(seconds / 60), 60)
    return f"{hours} h {minutes:02d}"


def eta(r: SensorRow) -> float | None:
    """Seconds `r` has left to render (or make, a step after the render), at the pace it has
    gone since it started (from what it had, when carried on); None when not under way or too
    soon to say."""
    if r.state not in (RENDERING, MAKING) or not r.seconds or r.started is None:
        return None
    gone = r.fraction - r.started
    return r.seconds * (1.0 - r.fraction) / gone if gone > 0.005 else None


def row_view(r: SensorRow) -> RowView:
    """`r`'s cells: its name, size (or kind), rate, frames made, its time so far (its turn's:
    sensors in a turn render together), what it has left at that pace (`eta`), the frames it
    made a second, and the rigs it is for."""
    made = frames_made(r)
    left = eta(r)
    cells = (
        r.name,
        f"{r.size[0]}x{r.size[1]}" if r.size else r.kind,
        f"{r.rate:g} Hz" if r.rate else "",
        f"{made}/{r.frames}" if r.frames else "",
        duration(r.seconds) if r.seconds else "",
        duration(left) if left is not None else "",
        f"{made / r.seconds:.1f}" if r.seconds and made else "",
        (" + ".join(r.rigs) + " (shared)") if len(r.rigs) > 1 else "".join(r.rigs),
    )
    label = "starting" if r.state == RENDERING and r.fraction <= 0 else r.state
    if r.after and r.state in (MAKING, DONE):  # what a step is doing now, or what it made
        label = r.said or label
    return RowView(r.state, cells, r.fraction, r.said, label, r.begin)


def headline(
    kind: str,
    rows: tuple[SensorRow, ...],
    sharing: tuple[int, int] | None,
    took: float | None = None,
    step: str = "",
    state: str = "running",
) -> str:
    """The board's first line: the job, whether it is still running or how it ended, the
    sensors done, how many the rigs share, each kind of step after the render done (`Step`),
    its time and, while running, its step ("Capture running: 19 of 19 sensors, 23 in the rigs
    as 19 (4 shared), rrd 4 of 4, mcap 2 of 4, 18 min, ..."; "Capture done: ..., 21 min")."""
    sensors = [r for r in rows if not r.after]
    done = sum(r.state in (DONE, FAILED) for r in sensors)
    said = f"{kind.capitalize()} {state}: {done} of {len(sensors)} sensors"
    if sharing and sharing[0] > sharing[1]:
        said += f", {sharing[0]} in the rigs as {sharing[1]} ({sharing[0] - sharing[1]} shared)"
    kinds: dict[str, list[SensorRow]] = {}
    for r in rows:
        if r.after:
            kinds.setdefault(r.kind, []).append(r)
    for after, of in kinds.items():
        said += f", {after} {sum(r.state in (DONE, FAILED) for r in of)} of {len(of)}"
    if took is not None:
        said += f", {duration(took)}"
    running = state == "running" and took is not None
    return said + (f", {step}" if step and running else "")
