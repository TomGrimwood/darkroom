"""What a stage's semantic labels call a prim: its own and its ancestors' in one taxonomy,
joined as Replicator joins them for a camera's segments ("car, vehicle"), so what a lidar's
point hit is named as a camera's pixel is.

    label = semantics.labeller(stage, "class")
    label("/World/Cars/Wagon/Body/mesh")        # "car, vehicle"; "" for nothing labelled
"""

from __future__ import annotations

from collections.abc import Callable

from pxr import Usd, UsdSemantics


def labeller(stage: Usd.Stage, taxonomy: str) -> Callable[[str], str]:
    """A prim's labels in `taxonomy` by its path, its ancestors' with them. Within an
    instance's prototype (`/__Prototype_*`, a tree's branches) up to the prototype, then on
    from one of its instances: what it is is said where it is instanced."""
    said: dict[str, str] = {}

    def own(prim: Usd.Prim) -> list[str]:
        if not prim.HasAPI(UsdSemantics.LabelsAPI, taxonomy):
            return []
        return [str(x) for x in UsdSemantics.LabelsAPI(prim, taxonomy).GetLabelsAttr().Get() or []]

    def label(path: str) -> str:
        if path not in said:
            found: set[str] = set()
            prim = stage.GetPrimAtPath(path)
            while prim and not prim.IsPseudoRoot():
                found.update(own(prim))
                if prim.IsPrototype():
                    instances = prim.GetInstances()
                    if not instances:
                        break
                    prim = instances[0]
                    continue
                prim = prim.GetParent()
            said[path] = ", ".join(sorted(found))
        return said[path]

    return label
