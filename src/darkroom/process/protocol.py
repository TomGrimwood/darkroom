"""The lines the renderer's process prints for its caller to read.

A line meant to be read starts with `@@ `, and the exit code is said as `@@done N`. Within
what is said, two kinds of line are data, not text:

    @@ progress {"step": "rendering cameras", "fraction": 0.4}
    @@ result {"frames": 30}
"""

from __future__ import annotations

import json
from typing import Any

SAID = "@@ "
DONE = "@@done "
PROGRESS = "progress "
RESULT = "result "


def say(text: str) -> None:
    for line in str(text).splitlines() or [""]:
        print(SAID + line, flush=True)


def done(code: int) -> None:
    print(f"{DONE}{int(code)}", flush=True)


def say_progress(step: str, fraction: float | None, extra: dict[str, Any]) -> None:
    say(PROGRESS + json.dumps({"step": step, "fraction": fraction, **extra}))


def say_result(value: Any) -> None:
    say(RESULT + json.dumps(value))
