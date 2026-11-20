"""ovrtx's side of `darkroom.stills`: each shot a camera and a render product of its own on a
root layer over the stage, rendered by ovrtx a few steps (RTX's lighting settles over the first
few), the last step's picture written as the PNG. Run in ovrtx's process
(`renderers.launchers.OvrtxLauncher.call`). The stage's own lights, or darkroom's sun and sky
(`darkroom.sun.own_light`, as darkroom's stills have always had); a still's settings (Kit's names)
as ovrtx takes them (`settings.translate`), on every shot's product and camera.
"""

from __future__ import annotations

import time
from pathlib import Path

#: An orthographic camera's apertures are in tenths of a stage unit .
TENTHS = 10.0
#: Steps of a still: none of the stage's time passes between them.
STEP = 1.0 / 60


def stills(
    stage_file: str,
    shots: list[dict],
    settle: list[int] | None = None,
    subframes: int = 4,
    render_settings: dict | None = None,
    sun: list[float] | None = None,
) -> dict:
    """Each shot (as `darkroom.stills` hands it over) rendered into its PNG. Returns
    {"seconds": [each shot's]}. `subframes`: renders a frame accumulates:
    ovrtx's real-time path tracer accumulates over its steps, which `settle` counts."""
    import numpy as np
    import ovrtx
    import ovstage
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom

    from darkroom import process
    from darkroom.sun import own_light

    from . import ovrtx_usd, settings
    from .ovrtx_renderer import schema_files

    process.progress("opening the stage")
    # Everything a still adds in the session layer, which ovrtx is handed over the stage.
    over = Usd.Stage.Open(str(Path(stage_file).resolve()))
    over.SetEditTarget(over.GetSessionLayer())
    if sun is not None:
        own_light(over, *sun)
    translated = settings.translate(dict(render_settings or {}), settings.read(schema_files()))
    for key in translated.left:
        process.progress(f"stills: ovrtx has no setting for {key}, left out")
    products = []
    for index, shot in enumerate(shots):
        said = shot["camera"]
        width, height = said["resolution"]
        path = f"/Stills/Shot_{index}"
        camera = UsdGeom.Camera.Define(over, f"{path}/Camera")
        if said["projection"] == "orthographic":
            camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
            across = said["width"] * TENTHS
            camera.CreateHorizontalApertureAttr(across)
            camera.CreateVerticalApertureAttr(across * height / width)
        else:
            camera.CreateFocalLengthAttr(said["focal_length"])
            camera.CreateHorizontalApertureAttr(said["horizontal_aperture"])
            camera.CreateVerticalApertureAttr(said["horizontal_aperture"] * height / width)
        camera.CreateClippingRangeAttr(Gf.Vec2f(said["near"], said["far"]))
        x, y, z = said["position"]
        target = said["look_at"] or (x, y, z - 1.0)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(x, y, z), Gf.Vec3d(*target), Gf.Vec3d(*said["up"]))
        camera.AddTransformOp().Set(view.GetInverse())
        for name, (kind, value) in translated.camera.items():
            camera.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Find(kind)).Set(value)
        product = over.DefinePrim(f"{path}/Product", "RenderProduct")
        product.CreateRelationship("camera").SetTargets([camera.GetPath()])
        product.CreateAttribute(
            "resolution", Sdf.ValueTypeNames.Int2, variability=Sdf.VariabilityUniform
        ).Set(Gf.Vec2i(width, height))
        product.CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set(
            "RealTimePathTracing"
        )
        for name, (kind, value) in translated.product.items():
            product.CreateAttribute(name, Sdf.ValueTypeNames.Find(kind)).Set(value)
        var = over.DefinePrim(f"{path}/Product/LdrColor", "RenderVar")
        var.CreateAttribute(
            "sourceName", Sdf.ValueTypeNames.String, variability=Sdf.VariabilityUniform
        ).Set("LdrColor")
        product.CreateRelationship("orderedVars").SetTargets([var.GetPath()])
        products.append((str(product.GetPath()), str(var.GetPath()), shot["png"], (width, height)))

    renderer = ovrtx.Renderer()
    scene = ovstage.Stage("darkroom.stills")
    renderer.attach_ovstage(scene)
    took: list[float] = []
    try:
        usd = ovrtx_usd.build_stage_as_string(over)
        ovstage.population.open_usd_from_string(scene, usd, ordinal=1)
        scene.advance_write_floor(1, ovstage.Scope.ALL).wait()
        first, again = settle or (8, 4)
        for index, (product, var, png, (width, height)) in enumerate(products):
            began = time.perf_counter()
            process.progress(f"rendering {Path(png).name}", index / len(products))
            picture = None
            for _ in range(max(1, (first if index == 0 else again)) * max(1, subframes)):
                result = renderer.step(render_products={product}, delta_time=STEP, ordinal=1)
                for _, made in result.items():
                    for frame in made.frames:
                        with frame.render_vars[var].map(device=ovrtx.Device.CPU) as mapped:
                            picture = np.from_dlpack(mapped).copy()
                del result
            if picture is None or picture.shape[:2] != (height, width):
                shape = None if picture is None else picture.shape
                raise RuntimeError(f"ovrtx gave a {shape} picture, not {height} x {width}.")
            Image.fromarray(picture[..., :3].astype(np.uint8)).save(png, compress_level=1)
            took.append(round(time.perf_counter() - began, 2))
    finally:
        renderer.detach_ovstage()
        scene.destroy()
        renderer.destroy()
    return {"seconds": took}
