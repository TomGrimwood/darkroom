"""What a renderer is bound to: Isaac Lab's `sim` (`SimulationContext`), as much of it as its
renderers use. There it is a physics simulation that publishes where things are; here it is a
stage played from USD time: nothing is simulated, `time` says how far along its timeline the
stage is, and a renderer shows the stage as it is then (`BaseRenderer.update_transforms`).

Two clocks, as RTX has them: `time`, the stage's timeline, which a capture takes back to the
window's start each lap (`seek`); and `clock`, the renderer's own, which only runs on, and
which its sensors tick by and stamp their pictures with.

It also owns what renderers share (`get_or_create_backend`: one resource for equal cfgs, as
Isaac Lab's registry), closed with it.
"""

from __future__ import annotations

import json
from typing import Any

from .renderer_cfg import BackendCfg, RendererCfg, instantiate, plain


class SimulationContext:
    """A stage (`Usd.Stage`) at a time of its timeline, stepped `dt` seconds at a time; `job`,
    the capture being made of it (`darkroom.renderjob.RenderJob`), for renderers to read."""

    def __init__(
        self,
        stage: Any,
        *,
        device: str = "cuda:0",
        dt: float = 1 / 60,
        job: Any = None,
        origin: float = 0.0,
    ) -> None:
        self.stage = stage
        self.device = device
        self.dt = dt
        self.job = job
        #: The timeline's time at the stage's first time code, seconds: a job's window and
        #: spans count from it.
        self.origin = origin
        self._time = 0.0
        self._clock = 0.0
        self._backends: dict[str, Any] = {}

    @property
    def time(self) -> float:
        """Seconds of the stage's timeline it is at (its time code over its time codes a second)."""
        return self._time

    @property
    def clock(self) -> float:
        """Seconds the renderer's own clock is at: it runs on over every `seek`."""
        return self._clock

    def span(self, start: float, end: float) -> None:
        """The stretch of the timeline about to be played, seconds: for one that has to be told."""

    def play(self) -> None:
        """Playing begun: for a timeline that has to be started."""

    def seek(self, time: float) -> None:
        """The timeline taken to `time` seconds; the renderer's clock stays where it is."""
        self._time = time

    def step(self, dt: float | None = None) -> None:
        """Both clocks advanced by `dt` seconds (its own by default)."""
        dt = self.dt if dt is None else dt
        self._time += dt
        self._clock += dt

    def stop(self) -> None:
        """Playing stopped."""

    def get_or_create_backend(self, cfg: BackendCfg) -> Any:
        """The resource `cfg` describes (a renderer, bound to this; ovrtx's native engine):
        made the first time, the same one for an equal cfg after."""
        key = json.dumps(plain(cfg), sort_keys=True, default=str)
        if key not in self._backends:
            bound = {"sim": self} if isinstance(cfg, RendererCfg) else {}
            self._backends[key] = instantiate(cfg, **bound)
        return self._backends[key]

    def clear_instance(self) -> None:
        """Every resource it made closed, last made first (Isaac Lab's teardown)."""
        while self._backends:
            _, backend = self._backends.popitem()
            backend.close()
