"""How much of each GNSS antenna's view was open sky, a share a fix (`gnss.npz`'s `sky`, 0 to
1), from the sky views rendered of it (ovrtx's, `darkroom.ovrtx`): the whole 180° the
antenna sees, a tilted antenna's view under the horizon not counted (`sensorkit.sky.sky_share`).
Leaves are not sky. A fix without a sky view of its own (the window's last, at times) takes the
nearest one's.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Usd

from darkroom import sensorkit
from darkroom.sensorkit import motion

from . import format, truth


def layer(stage: Usd.Stage) -> str | None:
    """The stage's GNSS layer (`darkroom.sensorkit.sky.LAYER` in a layer's custom data,
    strongest layer first): what the sky views are rendered without. None when no layer names
    one."""
    for one in stage.GetLayerStack(includeSessionLayers=False):
        found = one.customLayerData.get(sensorkit.sky.LAYER)
        if found:
            if not Path(found).is_file():
                raise FileNotFoundError(f"the stage's GNSS layer {found} is not there.")
            return str(found)
    return None


def view(ids: np.ndarray, named: Mapping[str, Any]) -> np.ndarray:
    """A sky view from a semantic segmentation: each pixel's class (`ids`, rows x columns)
    as open, leaves or blocked (`sensorkit.sky.classify`) by what its class is called
    (`named`: Replicator's `idToLabels`, {id: {"class": labels}}); a class not named blocks."""
    seen = np.full(ids.shape, sensorkit.sky.BLOCKED, np.uint8)
    for key, labels in named.items():
        said = labels.get("class", "") if isinstance(labels, dict) else str(labels)
        seen[ids == int(key)] = sensorkit.sky.classify(said)
    return seen


def wanted(meta: format.Meta) -> list[str]:
    """The GNSS antennas in `meta`: each has its sky views rendered."""
    return [name for name, s in meta.sensors.items() if s.kind == "gnss"]


def write(stage_file: str | Path, out: str | Path) -> list[str]:
    """Each antenna's share of open sky a fix added to its `gnss.npz` in the capture `out`. A
    line each."""
    meta = format.read_meta(out)
    names = wanted(meta)
    if not names:
        return []
    stage = Usd.Stage.Open(str(stage_file))
    span = truth.Window(stage, (meta.start, meta.end))
    where = sensorkit.site.find(stage)
    lines = []
    for name in names:
        fixes = format.read_arrays(out, name, format.GNSS)
        seconds = fixes["seconds"]
        views = _views(Path(out), name, len(seconds))
        matrices = truth.poses(stage, meta.sensors[name].prim, span.code(span.start + seconds))
        if where is not None:
            matrices = matrices @ where.to_site
        _, quaternions = motion.from_usd_matrices(matrices)
        rotations = motion.rotation_matrices(quaternions)
        sky = np.array(
            [
                sensorkit.sky.sky_share(v, r, through=0.0)
                for v, r in zip(views, rotations, strict=True)
            ]
        )
        format.write_arrays(out, name, format.GNSS, **{**fixes, "sky": sky})
        lines.append(
            f"{name}: {sky.mean():.0%} of its view open sky on average, {sky.min():.0%} at least"
        )
    return lines


def _views(out: Path, name: str, count: int) -> list[np.ndarray]:
    """A sky view for each of `count` fixes: its own, else the nearest one there is."""
    there = [i for i in range(count) if format.sky_path(out, name, i).is_file()]
    if not there:
        raise RuntimeError(f"{name}: no sky views of it were rendered.")
    read = {i: format.read_sky(out, name, i) for i in there}
    return [read[min(there, key=lambda j: abs(j - i))] for i in range(count)]
