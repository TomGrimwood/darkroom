"""How cameras share the GPU in one renderer process (the scene stays loaded): as many on
together as the GPU has room for (`renderers.once`), each render product destroyed once off.

What a camera needs is measured as each gets on, not guessed; `PACKING` says how much of the
GPU's memory is kept free beside them (textures streamed in as they see more, RTX's caches
growing, the driver's own), or that each renders alone ("one by one"). Out of memory all the
same, a new process takes no more pixels at once than the cameras on then, less the
smallest's (`step_back`).
"""

from __future__ import annotations

#: The share of the GPU's memory kept free beside the cameras, by `Options.packing`; None,
#: each camera alone.
PACKING: dict[str, float | None] = {
    "one by one": None,
    "conservative": 0.15,
    "normal": 0.08,
    "aggressive": 0.04,
}


def step_back(sizes: list[tuple[int, int]]) -> int | None:
    """The most pixels the cameras on at once may have between them after cameras of `sizes`
    ran out of the GPU's memory together: theirs less the smallest's, so that one waits. None
    for one camera: alone already, it can step back no further."""
    if len(sizes) < 2:
        return None
    pixels = [w * h for w, h in sizes]
    return sum(pixels) - min(pixels)


def room(used: float, total: float, headroom: float = 0.08) -> float:
    """The GPU memory cameras may yet take, bytes: what is free (GB `used` of `total`, as
    `gpu.memory` reads them), less `headroom` of the whole."""
    return (total * (1 - headroom) - used) * 2**30
