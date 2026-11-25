"""A function run in a Python process of its own, and what it did handed back as data.

A capture renders in a process of its own: a renderer that runs out of the GPU's memory goes
on rendering black and has to be killed, and a driver crash must not take its caller with it.

    result = process.call(
        "darkroom.ovrtx:render_all", job, out,
        python=sys.executable, name="ovrtx",
        on_progress=lambda p: print(p.step, p.fraction),
    )

`call` starts `python` running `child.main`, which imports the target, calls it with the
arguments given (as JSON) and says its result (as JSON). Inside it, `progress(step, fraction)`
reports how far it has got. A failure raises `Failed`, with the error lines and the whole
log's path. One renderer runs on the machine at a time (a lock file); the process is killed
when the GPU's memory runs out.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from . import protocol
from .child import progress

__all__ = ["OUT_OF_GPU", "Failed", "Progress", "call", "progress"]

#: How many error lines a failure keeps.
ERRORS = 30
#: Where each call's whole log is kept, with the command to run it again at its top.
LOGS = Path(tempfile.gettempdir()) / "darkroom"
#: How many calls' logs are kept there, the newest: the rest go as a call starts.
KEPT = 10
#: The lock every call takes first, machine-wide: one renderer at a time (two share one GPU's
#: memory and run out).
LOCK = "renderer.lock"
#: What RTX says when the GPU's memory runs out: after it, renders are black or missing.
OUT_OF_GPU = ("Out of GPU memory", "ERROR_OUT_OF_DEVICE_MEMORY")
#: A renderer's log prefix (time and milliseconds), taken off error lines.
_PREFIX = re.compile(r"^\S+ \[[\d,]+ms\] ")
#: How the process is started: the call's file as its one argument.
ENTRY = "import darkroom.process.child as c; c.main()"


@dataclass(frozen=True)
class Progress:
    """How far a call has got: what it is doing, how much of it is done (0 to 1, or None)."""

    step: str
    fraction: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Failed(RuntimeError):
    """A call that did not finish: what went wrong, said plainly, and where its whole log is."""

    def __init__(self, message: str, errors: list[str], log: Path | None):
        super().__init__(message)
        self.errors = errors
        self.log = log


class Stopper:
    """What `on_start` is handed to stop the process with (`terminate`, as a process would
    be): it kills it, as a renderer does not stop when asked to."""

    def __init__(self, process: subprocess.Popen):
        self._process = process

    def terminate(self) -> None:
        self._process.kill()

    kill = terminate


def source_paths() -> list[str]:
    """The source folders of the packages installed editable here (each one's `src`), handed
    on to the process so it imports the same code."""
    return list(dict.fromkeys(p for p in sys.path if Path(p).name == "src" and Path(p).is_dir()))


def _errors(lines: list[str]) -> list[str]:
    """The renderer's `[Error]` lines and any Python traceback, without repeats."""
    found: list[str] = []
    inside = False
    for line in lines:
        if "[Error]" in line:
            text = _PREFIX.sub("", line)
        elif line.startswith("Traceback (most recent call last)"):
            inside, text = True, line
        elif inside:
            text, inside = line, line.startswith(" ")
        elif line.startswith(protocol.SAID + "FAILED"):
            text = line[len(protocol.SAID) :]
        else:
            continue
        if text not in found:
            found.append(text)
    return found[:ERRORS]


@contextlib.contextmanager
def one_at_a_time(on_line: Callable[[str], object] | None = print) -> Generator[None]:
    """Hold the machine's renderer lock (`LOCK`, in `LOGS`) while inside: wait, saying so,
    while another process's renderer holds it."""
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = FileLock(LOGS / LOCK)
    try:
        lock.acquire(blocking=False)
    except Timeout:
        if on_line:
            on_line("waiting for another renderer to finish (one at a time: they share the GPU)")
        lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _forget(keep: int) -> None:
    """All but the newest `keep` calls' logs in `LOGS` gone, each with its call's file. One
    that will not go (another process's, open still, on Windows) is left for the next call."""
    logs = sorted(LOGS.glob("*.log"), key=lambda log: log.stat().st_mtime)
    for log in logs[: max(0, len(logs) - keep)]:
        for file in (log, log.with_suffix(".call.json")):
            with contextlib.suppress(OSError):
                file.unlink()


def _shell_line(env: Mapping[str, str], command: list[str]) -> str:
    """The command as a person would type it again, in this system's shell."""
    if sys.platform == "win32":
        sets = "".join(f'set "{k}={v}" && ' for k, v in env.items())
        return sets + subprocess.list2cmdline(command)
    sets = "".join(f"{k}={shlex.quote(v)} " for k, v in env.items())
    return sets + shlex.join(command)


def call(
    target: str,
    *args: Any,
    python: str,
    name: str,
    env: Mapping[str, str] | None = None,
    on_progress: Callable[[Progress], object] | None = None,
    on_line: Callable[[str], object] | None = print,
    on_start: Callable[[Stopper], object] | None = None,
    **kwargs: Any,
) -> Any:
    """Run `target` ("module:function") with `args` and `kwargs` (JSON values) in `python`, a
    process of its own called `name` in what is said of it ("starting ovrtx", "ovrtx's whole
    log"). Returns what it returned.

    What it says (`progress`, and any other line it means to be read) goes to `on_progress`
    and `on_line` as it comes; the rest of its output only to a log in `LOGS` (the newest
    `KEPT` calls' are kept), which starts with the command that makes the same call again. The
    first progress is "starting <name>", its `extra` the target and the log's path. `on_start`
    is handed a `Stopper` as it starts. `env` is set in the process's environment over the
    caller's own.
    Raises `Failed` when it does not finish, or when it says the GPU's memory ran out
    (`OUT_OF_GPU`): it would go on rendering black, so it is killed then.
    """
    LOGS.mkdir(parents=True, exist_ok=True)
    _forget(KEPT - 1)
    handle, made = tempfile.mkstemp(prefix=f"{target.replace(':', '_')}_", suffix=".log", dir=LOGS)
    os.close(handle)
    log = Path(made)
    call_file = log.with_suffix(".call.json")
    call_file.write_text(
        json.dumps({"target": target, "args": list(args), "kwargs": kwargs}), encoding="utf-8"
    )
    added = {**(env or {}), "PYTHONPATH": os.pathsep.join(source_paths())}
    environment = {**os.environ, **added, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    command = [python, "-c", ENTRY, str(call_file)]
    lines: list[str] = []
    code: int | None = None
    result: Any = None
    out_of_gpu: str | None = None

    with (
        one_at_a_time(on_line),
        log.open("w", encoding="utf-8") as written,
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
        ) as process,
    ):
        assert process.stdout is not None
        written.write(f"# The same call again:\n# {_shell_line(added, command)}\n\n")
        written.flush()
        if on_progress:
            on_progress(Progress(f"starting {name}", None, {"target": target, "log": str(log)}))
        if on_start:
            on_start(Stopper(process))
        for raw in process.stdout:
            written.write(raw)
            line = raw.rstrip("\n")
            lines.append(line)
            if out_of_gpu is None and any(o in line for o in OUT_OF_GPU):
                out_of_gpu = line.strip()
                process.kill()  # it goes on rendering black, holding the GPU
            if line.startswith(protocol.DONE):
                code = int(line[len(protocol.DONE) :] or 1)
            elif line.startswith(protocol.SAID):
                text = line[len(protocol.SAID) :]
                if text.startswith(protocol.PROGRESS):
                    data = json.loads(text[len(protocol.PROGRESS) :])
                    if on_progress:
                        step, fraction = data.pop("step"), data.pop("fraction", None)
                        on_progress(Progress(step, fraction, data))
                elif text.startswith(protocol.RESULT):
                    result = json.loads(text[len(protocol.RESULT) :])
                elif on_line:
                    on_line(text)
    if out_of_gpu is not None:
        raise Failed(
            f"{target} failed: the GPU ran out of memory (is another renderer running?): "
            f"{out_of_gpu}. {name}'s whole log: {log}",
            [out_of_gpu],
            log,
        )
    if code == 0:
        return result
    errors = _errors(lines)
    why = f"exit code {code}" if code is not None else f"{name} stopped without finishing"
    first = next((e for e in errors if not e.startswith("Traceback")), why)
    raise Failed(f"{target} failed ({why}): {first}. {name}'s whole log: {log}", errors, log)
