"""What starts the renderer's process (`RendererCfg.launcher_type`): `OvrtxLauncher`, a Python
with NVIDIA's ovrtx. A capture renders in a process of its own (a renderer that runs out of the
GPU's memory has to be killed), so the launcher runs a function there and hands back what it
returned (`call`), through `darkroom.process.call`: one renderer on the GPU at a time, killed
when the GPU's memory runs out, what it says read as it comes, its whole output kept in a log.
Loads no USD or ovrtx.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping
from typing import Any

from darkroom import process

#: The environment variable naming the Python that has ovrtx, when the caller's own has not.
PYTHON = "DARKROOM_OVRTX_PYTHON"
#: What ovrtx's process is started with in its environment: no report to NVIDIA as ovrtx
#: starts (0.7.0 sends one a process unless this is set before it does).
OVRTX_ENVIRONMENT = {"NVIDIA_OMNIVERSE_PRODUCT_TELEMETRY_DISABLED": "1"}


class OvrtxLauncher:
    """A Python with ovrtx, in a process of its own: the caller's own Python when it has
    ovrtx, else the one `$DARKROOM_OVRTX_PYTHON` names (it needs
    darkroom's other dependencies too: USD, numpy, PIL). ovrtx takes nothing as it starts."""

    name = "ovrtx"
    target = "darkroom.ovrtx:render_all"
    kinds = ("camera", "lidar", "gnss")

    def __init__(self, launcher_args: Mapping[str, Any] | None = None) -> None:
        pass

    def python(self) -> str:
        """The Python to run. Raises RuntimeError, saying how to get one, when none has ovrtx."""
        if os.environ.get(PYTHON):
            return os.environ[PYTHON]
        if importlib.util.find_spec("ovrtx") is not None:
            return sys.executable
        raise RuntimeError(
            "ovrtx is not installed: `uv sync` in darkroom, or point "
            f"${PYTHON} at a Python that has it."
        )

    def call(self, target: str, *args: Any, **kwargs: Any) -> Any:
        """`target` ("module:function") run with `args` and `kwargs` in a Python with ovrtx
        (`darkroom.process`). Returns what it returned; raises `process.Failed` when it does
        not finish, or when RTX says the GPU's memory ran out."""
        return process.call(
            target, *args, python=self.python(), name=self.name, env=OVRTX_ENVIRONMENT, **kwargs
        )
