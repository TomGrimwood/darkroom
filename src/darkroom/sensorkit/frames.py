"""Frames: where rigs of sensors go.

A frame is any `Xform` whose parent is a prim named `Frames`: a pole's
`/World/Trajectories/walk/Props/Pole/Frames/SensorMount`, `/Drone/Frames/Gimbal`. Whatever declares one (a pole, a drone, a character's hand), a rig
attached under it (`rigs.attach`) moves with it by USD's own transform inheritance. Nothing
here knows what declared a frame or how it moves.

Whatever carries sensors declares `SENSOR_MOUNT`: the origin its rigs are placed about (a
pole's, the antenna's top).
"""

from __future__ import annotations

import os
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom

#: The name of the scope frames are declared in.
FRAMES = "Frames"
#: The frame a carrier's rigs go on: their sensors are placed about it.
SENSOR_MOUNT = "SensorMount"
#: What a carrier (a pole, a drone) has beside its `Frames`: its shape. A prim with both carries
#: the sensors under it (`carrier`).
GEOMETRY = "Geometry"
#: The frame at a carrier's point (a pole's tip), kept as a body of its own.
TIP = "Tip"


def frames(stage: Usd.Stage) -> list[Sdf.Path]:
    """Every frame on `stage`, in the order the stage has them."""
    return [
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Xformable) and prim.GetParent().GetName() == FRAMES
    ]


def carrier(stage: Usd.Stage, path: str | Sdf.Path) -> Usd.Prim | None:
    """The prim carrying the sensor at `path`: its nearest ancestor with both `Frames` and
    `Geometry` children; None when nothing does."""
    prim = stage.GetPrimAtPath(str(path)).GetParent()
    while prim and not prim.IsPseudoRoot():
        if prim.GetChild(FRAMES) and prim.GetChild(GEOMETRY):
            return prim
        prim = prim.GetParent()
    return None


def identifier(text: str) -> str:
    """`text` as a prim name: letters, digits and underscores, not starting with a digit."""
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in text)
    return name if name and not name[0].isdigit() else f"_{name}"


def asset_path(path: str | Path, layer: Sdf.Layer) -> str:
    """`path` as written in `layer`: relative (`./…`) when the layer is a file on the same
    drive, so the two can move together; absolute otherwise. With forward slashes, as USD
    writes them on every system."""
    path = Path(path).resolve()
    if not layer.realPath:
        return path.as_posix()
    here = Path(layer.realPath).resolve().parent
    if path.anchor != here.anchor or _device(path) != _device(here):
        return path.as_posix()
    relative = Path(os.path.relpath(path, here)).as_posix()
    return relative if relative.startswith(".") else f"./{relative}"


def _device(path: Path) -> int:
    """The device `path` is on, or will be on: that of its nearest ancestor that exists."""
    while not path.exists():
        path = path.parent
    return path.stat().st_dev
