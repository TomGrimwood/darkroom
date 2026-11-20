"""The renderer's process's side: reading the call, running the target, reporting.

`python -c "import darkroom.process.child as c; c.main()" CALL.json` is what `call` runs:
it reads the call from the file named, imports and calls the target, says its result and
its exit code (0, 130 for Ctrl-C, 1 for anything raised, its traceback said), and ends.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from . import protocol


def progress(step: str, fraction: float | None = None, **extra: Any) -> None:
    """Report how far the work has got: what it is doing, and how much of it is done (0 to 1)."""
    protocol.say_progress(step, fraction, extra)


def main() -> None:
    """Run the call in `sys.argv[1]`. It ends without the interpreter's own teardown: a
    renderer's, at exit, can hang."""
    said = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    module, _, name = said["target"].partition(":")
    code = 1
    try:
        function = getattr(importlib.import_module(module), name)
        protocol.say_result(function(*said["args"], **said["kwargs"]))
        code = 0
    except KeyboardInterrupt:
        code = 130
    except Exception as exc:
        protocol.say(f"FAILED: {exc}")
        for line in traceback.format_exc().splitlines():
            print(line, flush=True)
    protocol.done(code)
    sys.stdout.flush()
    os._exit(code)
