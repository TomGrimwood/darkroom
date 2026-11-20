"""The sensors on a stage: prims marked by usdsensors' schemas (`SensorCameraAPI`, ...).

- `camera`: a `Camera`; its rate is OmniSensorAPI's `omni:sensor:tickRate` (else 30).
- `lidar`: an `OmniLidar`; its rate is its `omni:sensor:tickRate`, which has to be its scan rate.
- `imu`, `gnss`: an `Xform`; its rate is `sensor:imu:rateHz`, `sensor:gnss:rateHz` (else 200, 10).

A sensor is off when its prim is inactive; nothing else switches it off (a camera asset is
often authored invisible, so its body does not show, and still renders).
Its name is its prim's name in lower case; where two share one, each is named with its kit in
front (`oak4d_front_camera`): the prim under the frame it is attached to.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal, cast

import usdsensors
from pxr import Sdf, Usd

TICK_RATE = usdsensors.TICK_RATE
Kind = Literal["camera", "lidar", "imu", "gnss"]
KINDS: tuple[Kind, ...] = ("camera", "lidar", "imu", "gnss")
#: The kinds a renderer renders; the rest are worked out from poses alone.
RENDERED = frozenset({"camera", "lidar"})
#: A sensor added to look into a capture, not part of what it hands on (a GNSS antenna added only
#: to see the sky): a bool on its prim, said in its settings (`"debug": true`).
DEBUG = "sensorkit:debug"
FALLBACK_RATES = {"camera": 30.0, "imu": 200.0, "gnss": 10.0}


@dataclass(frozen=True)
class Sensor:
    path: Sdf.Path
    kind: Kind
    name: str
    rate: float


def kind_of(prim: Usd.Prim) -> Kind | None:
    """What sensor `prim` is (its usdsensors schema applied), else None."""
    return cast(Kind | None, usdsensors.kind_of(prim))


def rate_of(prim: Usd.Prim, kind: Kind) -> float:
    name = TICK_RATE if kind in RENDERED else usdsensors.RATE[kind]
    value = prim.GetAttribute(name).Get()
    if value is None:
        if kind not in FALLBACK_RATES:
            raise RuntimeError(f"{prim.GetPath()} ({kind}) has no {name}.")
        return FALLBACK_RATES[kind]
    return float(value)


def sensors(stage: Usd.Stage, kinds: Collection[str] | None = None) -> list[Sensor]:
    """The stage's sensors that are on, of `kinds` (all by default), in stage order."""
    found: list[tuple[Usd.Prim, Kind]] = []
    for prim in stage.Traverse():
        kind = kind_of(prim)
        if kind is None:
            continue
        if kinds is not None and kind not in kinds:
            continue
        found.append((prim, kind))
    plain = [prim.GetName().lower() for prim, _ in found]
    names: list[str] = []
    for (prim, _), name in zip(found, plain, strict=True):
        if plain.count(name) > 1:
            name = f"{_kit(prim).lower()}_{name}"
        while name in names:
            name += "_"
        names.append(name)
    return [
        Sensor(prim.GetPath(), kind, name, rate_of(prim, kind))
        for (prim, kind), name in zip(found, names, strict=True)
    ]


def _kit(prim: Usd.Prim) -> str:
    """The name of the kit `prim` is in: its ancestor just under a frame, else its parent."""
    here = prim
    while here.GetParent() and here.GetParent().GetParent():
        if here.GetParent().GetParent().GetName() == "Frames":
            return here.GetName()
        here = here.GetParent()
    return prim.GetParent().GetName()


def mark(layer: Sdf.Layer, path: str | Sdf.Path, kind: Kind, rate: float | None = None) -> None:
    """Say in `layer` that the prim at `path` is a sensor of `kind` (its usdsensors schema
    applied), at `rate` when given."""
    spec = layer.GetPrimAtPath(path) or Sdf.CreatePrimInLayer(layer, path)
    listed = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else Sdf.TokenListOp()
    schema = usdsensors.SCHEMAS[kind]
    rest = [s for s in listed.prependedItems if s != schema]
    spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=[schema, *rest]))
    if rate is not None:
        name = TICK_RATE if kind in RENDERED else usdsensors.RATE[kind]
        kind_of_value = Sdf.ValueTypeNames.Float if kind in RENDERED else Sdf.ValueTypeNames.Double
        attribute = spec.attributes.get(name) or Sdf.AttributeSpec(spec, name, kind_of_value)
        attribute.default = float(rate)
