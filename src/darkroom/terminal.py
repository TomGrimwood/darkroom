"""A render's sensors live in a terminal, as `docker compose pull` shows its layers: a row a
sensor (its bar, kind, size, rate, frames and what it made) under a line saying the render, its
step, how long it has taken and the GPU's memory, redrawn as it goes. Lines said meanwhile go
above it. Drawn with `rich` from any status with a board's rows (`board.Shown`); when the render
ends, the board is left as it ended (written whole where the output is a file or a pipe)."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .board import Shown

#: Each state's colour and mark.
LOOK = {
    "waiting": ("grey50", "·"),
    "rendering": ("cyan", "▶"),
    "writing": ("yellow", "✎"),
    "making": ("cyan", "▶"),
    "done": ("green", "✓"),
    "failed": ("red", "✗"),
    "stopped": ("grey50", "■"),
}
#: Redraws a second, at most.
RATE = 4
#: How wide a board written to a file or a pipe is (no terminal to say).
WIDE = 200
#: The meter's charts: how many rows high, and the gap between them, characters.
CHART_ROWS, GAP = 5, 3


#: A row's bar, characters.
BAR = 24


def bar(begin: float, fraction: float, width: int, colour: str) -> Any:
    """A row's bar: `width` cells on a quiet track, those it has rendered (`board.covered`:
    from `begin`, on round past the end) in `colour`."""
    from rich.text import Text

    from .board import covered

    drawn = Text("━" * width, style="grey23")
    for a, b in covered(begin, fraction):
        drawn.stylize(colour, round(a * width), round(b * width))
    return drawn


class TerminalBoard:
    """The live board while a job with sensors runs (`show`), left as it ended (`close`)."""

    def __init__(self) -> None:
        self._live: Any = None
        self._seen = False  # a board shown for the job running
        self._drawn = 0.0
        self._meter: Any = None  # `board.Meter`: the GPU and the pixels a second, charted
        self._charted: tuple[int, list[str]] = (0, [])  # its charts as last drawn, and how wide

    def show(self, name: str, status: Shown, started: float) -> None:
        """The board of `status`'s sensors, for job `name` running since `started`."""
        if not status.sensors:
            return
        self._seen = True
        if not sys.stdout.isatty():  # a file or a pipe: the board once, as it ends (`close`)
            return
        now = time.perf_counter()
        if self._live is None:
            from rich.live import Live

            from .board import Meter

            self._meter, self._charted = Meter(), (0, [])
            self._live = Live(
                self._table(name, status, started),
                refresh_per_second=RATE,
                redirect_stdout=True,
                redirect_stderr=True,
                transient=False,
            )
            self._live.start()
        elif now - self._drawn < 1 / RATE:
            return
        self._drawn = now
        self._live.update(self._table(name, status, started))

    def close(self, name: str, status: Shown, started: float) -> None:
        """The board drawn once more as the job ended, and left in the terminal (written whole,
        at full width, where the output is a file or a pipe)."""
        if self._live is not None:
            self._live.update(self._table(name, status, started), refresh=True)
            self._live.stop()
            self._live = None
        elif self._seen and status.sensors:
            from rich.console import Console

            Console(file=sys.stdout, width=WIDE, force_terminal=False).print(
                self._table(name, status, started)
            )
        self._seen = False

    def _width(self) -> int:
        """The terminal's width, characters (`WIDE` written to a file)."""
        import shutil

        return shutil.get_terminal_size((WIDE, 24)).columns

    def _table(self, name: str, status: Shown, started: float) -> Any:
        """The board drawn: its headline and rows (`board.headline`, `board.row_view`), in
        colour, with bars."""
        from rich.console import Group
        from rich.table import Table
        from rich.text import Text

        from .board import AFTER_BAR, COLUMNS, headline, meter_charts, row_view

        took = time.perf_counter() - started
        head = Text(
            headline(status.kind, status.sensors, status.sharing, took, status.step, status.state),
            style="bold" if status.state == "running" else LOOK.get(status.state, ("", ""))[0],
        )
        if self._meter is not None:  # read once a second at most: one call into the driver
            width = max(20, (self._width() - 3 * GAP) // 3)
            if self._meter.read(status.sensors) or width != self._charted[0]:  # drawn again
                drawn = meter_charts(self._meter, width, CHART_ROWS)  # only with a new reading
                self._charted = (width, [f"{t.ljust(width)}\n" + "\n".join(c) for t, c in drawn])
        charts = [Text(c, style="magenta") for c in self._charted[1]]
        table = Table(box=None, pad_edge=False, show_header=True, header_style="grey50")
        for column in ("", *COLUMNS, "", *AFTER_BAR, ""):
            table.add_column(column, no_wrap=True)
        for r in map(row_view, status.sensors):
            colour, mark = LOOK.get(r.state, ("white", "?"))
            name, *middle = r.cells[: len(COLUMNS)]
            table.add_row(
                Text(mark, style=colour),
                Text(name, style="bold" if r.state in ("rendering", "making") else ""),
                *middle,
                bar(r.begin, r.fraction, BAR, colour),
                *r.cells[len(COLUMNS) :],
                Text(r.said if r.state == "failed" else r.label, style=colour),  # why, if it failed
            )
        if charts:  # side by side, under the headline
            from rich.columns import Columns

            return Group(head, Columns(charts, padding=(0, GAP)), table)
        return Group(head, table)
