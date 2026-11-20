"""A capture's RTX settings as ovrtx takes them. Isaac Sim takes settings by path
(`/rtx/post/histogram/enabled`: a capture's look, a stage's own `renderSettings`); ovrtx takes
them as attributes, some on a camera (`omni:rtx:autoExposure:enabled`, `exposure:iso`), most on
its render product (`omni:rtx:rtpt:maxBounces`). ovrtx's own schemas say which attribute stands
for which path (each attribute's doc: "deprecated carb setting path: '...'") and which kind of
prim each schema is for (their `plugInfo.json`), so the table from one to the other is read
from the ovrtx that is installed (`read`), and a look means the same to both renderers
(`translate`). Needs USD only: no ovrtx (its renderer says where its schema files are).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: How an attribute's doc names the Kit setting it stands for.
CARB = re.compile(r"deprecated carb setting path: '([^']+)'")
#: The USD types a setting's value is taken as it is for.
SCALARS = {"bool": bool, "int": int, "uint": int, "float": float, "double": float}
TEXT = ("token", "string")
#: The kinds of prim a schema's attributes are authored on.
CAMERA, PRODUCT = "Camera", "RenderProduct"

#: A value for USD, with its type's name: `CameraCfg.render_settings`' values.
Typed = tuple[str, Any]


@dataclass(frozen=True)
class Schemas:
    """What ovrtx's schemas say of its settings (`read`)."""

    #: Each Kit setting (its path) with the attributes that stand for it: (API schema,
    #: attribute, USD type) each.
    by_setting: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)
    #: Each attribute's API schema and USD type.
    by_attribute: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: The kind of prim each API schema is for ("Camera", "RenderProduct", "RenderSettings").
    on: dict[str, str] = field(default_factory=dict)
    #: Each attribute's value where none is authored (its schema's).
    defaults: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Translated:
    """Settings as attributes, each `(USD type, value)` by its name: those for the camera,
    those for its render product, and the settings left out (no attribute stands for them,
    their value is not of the attribute's type, or they are render-wide)."""

    camera: dict[str, Typed] = field(default_factory=dict)
    product: dict[str, Typed] = field(default_factory=dict)
    left: tuple[str, ...] = ()


def read(files: Iterable[str | Path]) -> Schemas:
    """What the schemas in `files` (`generatedSchema.usda` each, its `plugInfo.json` beside it)
    say: the attribute for each Kit setting, each attribute's schema, each schema's prim."""
    from pxr import Sdf

    found = Schemas()
    for file in files:
        layer = Sdf.Layer.FindOrOpen(str(file))
        for schema in layer.rootPrims if layer else ():
            for attribute in schema.attributes:
                kind = str(attribute.typeName)
                found.by_attribute.setdefault(attribute.name, (schema.name, kind))
                if attribute.default is not None:
                    found.defaults.setdefault(attribute.name, attribute.default)
                for path in CARB.findall(attribute.documentation or ""):
                    found.by_setting.setdefault(path, []).append(
                        (schema.name, attribute.name, kind)
                    )
        found.on.update(_applies_to(Path(file).with_name("plugInfo.json")))
    return found


def _applies_to(plug_info: Path) -> dict[str, str]:
    """Each API schema a `plugInfo.json` describes with the kind of prim it is for (the first
    it can only apply to, else the first it is applied to by itself)."""
    if not plug_info.is_file():
        return {}
    lines = plug_info.read_text(
        encoding="utf-8"
    ).splitlines()  # USD's plugInfo may start with comments
    data = json.loads("\n".join(line for line in lines if not line.lstrip().startswith("#")))
    on: dict[str, str] = {}
    for plugin in data.get("Plugins", []):
        info = plugin.get("Info", {})
        for name, said in info.get("AutoApplyAPISchemas", {}).items():
            if said.get("apiSchemaAutoApplyTo"):
                on[name] = said["apiSchemaAutoApplyTo"][0]
        for name, said in info.get("Types", {}).items():
            if said.get("apiSchemaCanOnlyApplyTo"):
                schema = said.get("alias", {}).get("UsdSchemaBase", name)
                on[schema] = said["apiSchemaCanOnlyApplyTo"][0]
    return on


def _typed(kind: str, value: Any) -> Typed | None:
    """`value` as an attribute of USD type `kind` takes it; None when it is not of that type
    (Kit keeps some of what ovrtx names as a number: a tone mapper by its index)."""
    if kind in SCALARS and isinstance(value, bool | int | float):
        return kind, SCALARS[kind](value)
    if kind in TEXT and isinstance(value, str):
        return kind, value
    if isinstance(value, list | tuple) and all(isinstance(v, int | float) for v in value):
        from pxr import Sdf

        made = Sdf.ValueTypeNames.Find(kind).type.pythonClass
        try:  # a colour, a pair: as the type's own class
            return kind, made(*value)
        except Exception:
            return None
    return None


def translate(settings: Mapping[str, Any], schemas: Schemas) -> Translated:
    """Kit `settings` (path: value) as ovrtx's attributes, by `schemas` (`read`): on the
    camera or on its render product, as each one's schema is for."""
    camera: dict[str, Typed] = {}
    product: dict[str, Typed] = {}
    left: list[str] = []
    for path, value in settings.items():
        stood = schemas.by_setting.get("/" + path.lstrip("/"))
        taken = False
        for schema, name, kind in stood or ():
            typed = _typed(kind, value)
            where = {CAMERA: camera, PRODUCT: product}.get(schemas.on.get(schema, PRODUCT))
            if typed is not None and where is not None:
                where[name] = typed
                taken = True
        if not taken:
            left.append(path)
    return Translated(camera, product, tuple(left))


def split(attributes: Mapping[str, Typed], schemas: Schemas) -> Translated:
    """Attributes authored on a camera prim itself (`ovrtx_usd.camera_settings`: its
    `omni:rtx:*`) as ovrtx takes them: those its schemas say are a camera's stay on the camera,
    the rest go on its render product (as Isaac Sim takes them all)."""
    camera: dict[str, Typed] = {}
    product: dict[str, Typed] = {}
    for name, typed in attributes.items():
        schema = schemas.by_attribute.get(name, ("", ""))[0]
        (camera if schemas.on.get(schema) == CAMERA else product)[name] = typed
    return Translated(camera, product)


#: A span's settings as attributes: from `start` to `end`, seconds of USD time.
Timed = tuple[float, float, Translated]


def samples(
    spans: Sequence[Timed],
    side: str,
    base: Mapping[str, Typed],
    schemas: Schemas,
    per_second: float,
    step: float,
) -> dict[str, tuple[str, list[tuple[float, Any]]]]:
    """The attributes `spans` change on a camera (`side` "camera") or its render product
    ("product"), each as time samples (time code, value): its own value (`base`, else its
    schema's) up to a span, the span's while it lasts, its own again after. Each change is
    half a `step` (seconds) before the span's start and end, so the steps either side of them
    read the right value whatever the rounding of their times, and within a thousandth of a
    time code, so a number is not ramped between."""
    names = {name for _, _, said in spans for name in getattr(said, side)}
    eps, half = 1e-3, step * per_second / 2
    timed: dict[str, tuple[str, list[tuple[float, Any]]]] = {}
    for name in sorted(names):
        kind, own = base.get(name) or (
            schemas.by_attribute.get(name, ("", ""))[1],
            schemas.defaults.get(name),
        )
        if own is None:
            continue  # nothing to go back to: left as the span has it from its start
        keys: list[tuple[float, Any]] = []
        for start, end, said in sorted(spans, key=lambda s: s[0]):
            value = getattr(said, side).get(name, (kind, own))[1]
            a, b = start * per_second - half, end * per_second - half
            keys += [(a - eps, own), (a, value), (b - eps, value), (b, own)]
        timed[name] = (kind, keys)
    return timed
