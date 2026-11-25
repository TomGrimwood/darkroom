"""What a render tells of its sensors as it goes, for a board of them (`board.Board.take`).

Each event is sent with the step it belongs to as a `process.Progress`'s `extra` (`extra(event)`,
plain data: it crosses from the renderer's process as JSON) and read back from it (`read`):

- `Uses`: every sensor the capture makes, with the rigs each is for (before the renderer starts);
- `Planned`: the sensors of a kind the renderer renders, their sizes, turns and frames;
- `Rendering`, `Writing`: a turn of a kind playing the window, then writing what it rendered;
- `Riding`: the cameras on now (`renderers.once`), said as one gets on or off;
- `Made`: a sensor done, with what it made or why it failed;
- `Step`: the caller's own work on what was rendered, after it (a recording made of the
  capture, say): a row of its own, said from the start, waiting till its turn.

Plain Python.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

#: The key in a progress's `extra` naming its event.
KEY = "darkroom"


@dataclass(frozen=True)
class Use:
    """A sensor a capture makes: its kind, rate and the rigs it is for (more than one when
    each has the same sensor: made once for them all)."""

    kind: str
    rate: float
    rigs: tuple[str, ...]


@dataclass(frozen=True)
class Uses:
    """Every sensor a capture makes, by its name in the capture; how many sensors the rigs
    have between them (`in_rigs`), and how many are made once for them (`made`)."""

    sensors: Mapping[str, Use]
    in_rigs: int
    made: int


@dataclass(frozen=True)
class Plan:
    """A sensor the renderer renders: a camera's picture size (else None), its turn of `turns`
    (each camera its own, on as the GPU has room: `renderers.once`; a kind's other sensors render together
    in one), the frames (or a lidar's scans) it should make."""

    name: str
    kind: str
    rate: float
    size: tuple[int, int] | None
    turn: int
    turns: int
    frames: int


@dataclass(frozen=True)
class Planned:
    kind: str
    sensors: tuple[Plan, ...]


@dataclass(frozen=True)
class Rendering:
    """Turn `turn` of `kind` playing the window, `done` of it (0 to 1) rendered, in `seconds`
    so far, from `begin` of the way through it (a camera that got on the loop part way: its
    rendered stretch runs on round from there)."""

    kind: str
    turn: int
    done: float
    seconds: float = 0.0
    begin: float = 0.0


@dataclass(frozen=True)
class Riding:
    """The cameras of `kind` rendering now (`renderers.once`): got on, none off yet."""

    kind: str
    sensors: tuple[str, ...]


@dataclass(frozen=True)
class Writing:
    """Turn `turn` of `kind` played, what it rendered being written."""

    kind: str
    turn: int


@dataclass(frozen=True)
class Made:
    """Sensor `sensor` done: what it made ("12 frames at 30 Hz, ..."), or why it failed; its
    turn's time, in seconds (the sensors of a turn render together)."""

    kind: str
    sensor: str
    said: str
    failed: bool = False
    seconds: float | None = None
    #: Its rate, a second, and a camera's picture size (width, height): for a board with no
    #: row for it yet (made in a run before this one, of a capture carried on).
    rate: float | None = None
    size: tuple[int, int] | None = None
    #: How many frames (or scans) it made; None when failed or not known.
    count: int | None = None


@dataclass(frozen=True)
class Step:
    """Work on what the render made, after it (a recording or a bag made of a capture): a row
    `name` of `kind` ("rrd", "mcap"), for `rigs`. `done` of it so far, 0 to 1 (None: not
    started, waiting its turn), what it is doing now, its `seconds` so far; `finished` with what
    it made (`said`), or `failed` with why."""

    name: str
    kind: str
    rigs: tuple[str, ...] = ()
    done: float | None = None
    doing: str = ""
    seconds: float = 0.0
    said: str = ""
    finished: bool = False
    failed: bool = False


Event = Uses | Planned | Rendering | Riding | Writing | Made | Step
_TYPES: dict[str, type] = {
    t.__name__.lower(): t for t in (Uses, Planned, Rendering, Riding, Writing, Made, Step)
}


def extra(event: Event) -> dict[str, Any]:
    """`event` as a progress's `extra`: plain data, its type under `KEY`."""
    return {KEY: type(event).__name__.lower(), **asdict(event)}


def read(extra: Mapping[str, Any] | None) -> Event | None:
    """The event in a progress's `extra`; None when it has none."""
    if not extra or extra.get(KEY) not in _TYPES:
        return None
    fields = {k: v for k, v in extra.items() if k != KEY}
    match extra[KEY]:
        case "uses":
            sensors = {
                n: Use(u["kind"], float(u["rate"]), tuple(u["rigs"]))
                for n, u in fields["sensors"].items()
            }
            return Uses(sensors, int(fields["in_rigs"]), int(fields["made"]))
        case "planned":
            return Planned(fields["kind"], tuple(_plan(p) for p in fields["sensors"]))
        case "riding":
            return Riding(fields["kind"], tuple(fields["sensors"]))
        case "made":
            size = fields.get("size")
            return Made(
                str(fields["kind"]),
                str(fields["sensor"]),
                str(fields["said"]),
                bool(fields.get("failed", False)),
                fields.get("seconds"),
                fields.get("rate"),
                (int(size[0]), int(size[1])) if size else None,
                fields.get("count"),
            )
        case "step":
            done = fields.get("done")
            return Step(
                str(fields["name"]),
                str(fields["kind"]),
                tuple(fields.get("rigs", ())),
                None if done is None else float(done),
                str(fields.get("doing", "")),
                float(fields.get("seconds", 0.0)),
                str(fields.get("said", "")),
                bool(fields.get("finished", False)),
                bool(fields.get("failed", False)),
            )
        case name:
            return _TYPES[name](**fields)


def _plan(p: Mapping[str, Any]) -> Plan:
    size = p.get("size")
    return Plan(
        str(p["name"]),
        str(p["kind"]),
        float(p["rate"]),
        (int(size[0]), int(size[1])) if size else None,
        int(p["turn"]),
        int(p["turns"]),
        int(p["frames"]),
    )
