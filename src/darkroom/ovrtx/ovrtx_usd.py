# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD authored for OVRTX: the stage as ovrtx opens it, each camera's eye, and a camera's
render product. Needs USD only: no ovrtx.

The render product and its render vars are Isaac Lab's (`isaaclab_ov/renderers/ovrtx_usd.py`:
`get_render_var_configs`, `build_render_product_as_string`, the outputs' names), so an output
means the same RenderVar to both. What is darkroom's own:

- `sensor_eye`: the camera that renders for a camera (`darkroom.eyes`), scheduled by ovrtx at
  the camera's rate (`OmniSensorAPI`'s `omni:sensor:frameRate`), and with a rolling shutter
  where the camera has a readout (`sensor:camera:readoutSeconds`): `OmniRtxRollingShutterAPI`,
  its rows read top to bottom over the readout, the last at the frame's time, each row at
  the middle of its exposure (`rolling_shutter`: how a readout and a shutter time become
  ovrtx's line timings), as Isaac Sim's rows' times end on the frame.
- `sensor_lidar`: the lidar that measures for a lidar: a copy of it beside it, its attributes
  as ovrtx's lidar schema has them (Isaac Sim's and ovrtx's differ in a few names), giving
  whole scans in its own axes; `build_lidar_product_as_string`: its render product, a
  `PointCloud` RenderVar of the channels a capture keeps.
- `build_sensor_as_string`: an eye or a lidar as a layer of its own, referenced into the stage
  ovrtx has open where it is (beside its sensor), and taken out again with it.
- `build_stage_as_string`: the root layer ovrtx opens, the stage's own file under it.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from pxr import Gf, Sdf, Usd

from darkroom import sensorkit
from darkroom.eyes import eye

if TYPE_CHECKING:
    from darkroom.renderers.camera_render_spec import CameraRenderSpec

    from .ovrtx_renderer import OVRTXCameraRenderData

# Maps camera data types to (prim name, source name). Shared sources use one config.
_RENDER_VAR_BY_DATA_TYPE: dict[str, tuple[str, str]] = {
    "rgb": ("LdrColor", "LdrColor"),
    "rgba": ("LdrColor", "LdrColor"),
    # Simple shading uses LdrColor in per-product RTX Minimal mode.
    "simple_shading_constant_diffuse": ("LdrColor", "LdrColor"),
    "simple_shading_diffuse_mdl": ("LdrColor", "LdrColor"),
    "simple_shading_full_mdl": ("LdrColor", "LdrColor"),
    "rgb_hdr": ("HdrColor", "HdrColor"),
    "albedo": ("albedo", "DiffuseAlbedoSD"),
    "depth": ("depth", "DistanceToImagePlaneSD"),
    "distance_to_image_plane": ("depth", "DistanceToImagePlaneSD"),
    # This source requires a distinct render-var prim.
    "distance_to_camera": ("DistanceToCameraSD", "DistanceToCameraSD"),
    "normals": ("NormalSD", "NormalSD"),
    "motion_vectors": ("TargetMotionSD", "TargetMotionSD"),
    "semantic_segmentation": ("semantic", "SemanticSegmentation"),
    "instance_segmentation": (
        "NonStableInstanceSegmentation",
        "NonStableInstanceSegmentation",
    ),
}

# Maps simple shading data types to the render product's ``omni:rtx:minimal:mode``.
_RTX_MINIMAL_MODES = {
    "simple_shading_constant_diffuse": 1,
    "simple_shading_diffuse_mdl": 2,
    "simple_shading_full_mdl": 3,
}

_COLOR_DATA_TYPES = frozenset({"rgb", "rgba"})

# Segmentation ID-map vars are authored alongside the pixel AOVs, not as camera data types.
_SEGMENTATION_MAP_RENDER_VARS: tuple[tuple[str, str], ...] = (
    ("StableIdSemanticIdMap", "StableIdSemanticIdMap"),
    ("StableIdMap", "StableIdMap"),
    ("SemanticIdMap", "SemanticIdMap"),
)

_RENDER_VAR_PRIM_NAME_BY_SOURCE: Mapping[str, str] = MappingProxyType(
    {
        source: name
        for name, source in (*_RENDER_VAR_BY_DATA_TYPE.values(), *_SEGMENTATION_MAP_RENDER_VARS)
    }
)

#: A rolling shutter's texture of row times, Isaac Sim's way: not an eye's, for ovrtx.
SHUTTER_TEXTURE = sensorkit.camera.SHUTTER_TEXTURE
#: The stage's own metadata the root layer ovrtx opens repeats (a sublayer's is not the stage's).
METADATA = (
    "defaultPrim",
    "startTimeCode",
    "endTimeCode",
    "timeCodesPerSecond",
    "framesPerSecond",
    "upAxis",
    "metersPerUnit",
    "customLayerData",
)


def get_render_var_configs(
    data_types: list[str], render_scope_name: str
) -> list[tuple[str, str, str]]:
    """Return render-var configs for the requested camera outputs.

    Shared sources are de-duplicated. Segmentation requests also add their ID-map vars.

    Args:
        data_types: Requested camera data types.
        render_scope_name: Root scope containing the render product and its render vars.

    Returns:
        Render-var configs as (absolute prim path, prim name, source name) tuples.

    Raises:
        ValueError: If no outputs are requested, an output is unsupported, or outputs are incompatible.
    """
    if not data_types:
        raise ValueError("OVRTX render products require at least one output.")
    unsupported = set(data_types) - _RENDER_VAR_BY_DATA_TYPE.keys()
    if unsupported:
        raise ValueError(f"Unsupported OVRTX output types: {sorted(unsupported)}.")
    simple_shading = list(
        dict.fromkeys(data_type for data_type in data_types if data_type in _RTX_MINIMAL_MODES)
    )
    color = list(
        dict.fromkeys(data_type for data_type in data_types if data_type in _COLOR_DATA_TYPES)
    )
    if simple_shading and color:
        raise ValueError(
            f"OVRTX cannot render simple shading {simple_shading} together with {color} on one render product:"
            " both read the 'LdrColor' render var, and simple shading additionally requires RTX Minimal mode."
            " Request them from separate cameras."
        )
    if len(simple_shading) > 1:
        raise ValueError(
            f"OVRTX supports at most one simple shading data type per render product, got {simple_shading}."
            " RTX Minimal mode is a per-render-product setting. Request them from separate cameras."
        )
    render_vars = list(
        dict.fromkeys(_RENDER_VAR_BY_DATA_TYPE[data_type] for data_type in data_types)
    )
    if "instance_segmentation" in data_types:
        render_vars.append(_SEGMENTATION_MAP_RENDER_VARS[0])
        render_vars.append(_SEGMENTATION_MAP_RENDER_VARS[1])
    if "semantic_segmentation" in data_types or "instance_segmentation" in data_types:
        render_vars.append(_SEGMENTATION_MAP_RENDER_VARS[2])
    return [(f"/{render_scope_name}/Vars/{name}", name, source) for name, source in render_vars]


def render_var_prim_names_by_source() -> Mapping[str, str]:
    """Return the scope-independent RenderVar prim name of every OVRTX render-var source."""
    return _RENDER_VAR_PRIM_NAME_BY_SOURCE


def build_render_product_as_string(
    spec: CameraRenderSpec,
    render_data: OVRTXCameraRenderData,
    *,
    device_id: int,
    enable_shadows: bool = False,
) -> str:
    """Build a complete render product USD layer as a string.

    The layer sets the camera scope as its default prim for referencing into OVRTX: a
    RenderProduct at the camera's resolution, pointed at the camera's eye, with a RenderVar
    for each output asked for, in real-time path tracing (RTX Minimal for a simple shading
    output), and `spec.cfg.render_settings` authored on it (name: (USD type, value)).

    Args:
        spec: Camera configuration and camera paths.
        render_data: Camera's render scope and product identity.
        device_id: CUDA device index the render product is pinned to.
        enable_shadows: Whether lights cast shadows. Only honored for the
            ``simple_shading_*`` data types, which are the ones that select RTX Minimal mode.

    Returns:
        Render product USD layer, including the USDA header and default prim metadata.
    """
    data_types = list(spec.cfg.data_types)
    tiled_width, tiled_height = (
        render_data.num_cols * render_data.width,
        render_data.num_rows * render_data.height,
    )
    render_var_configs = get_render_var_configs(data_types, render_data.render_scope_name)
    minimal_mode = next(
        (
            _RTX_MINIMAL_MODES[data_type]
            for data_type in data_types
            if data_type in _RTX_MINIMAL_MODES
        ),
        None,
    )
    background_color = spec.cfg.background_color
    if background_color is None:
        bg_type_line = 'token omni:rtx:background:source:type = "domeLight"'
    else:
        r, g, b = background_color
        bg_type_line = (
            f'token omni:rtx:background:source:type = "color"\n'
            f"        color3f omni:rtx:background:source:color = ({r}, {g}, {b})"
        )
    if minimal_mode is None:
        render_mode_lines = ['token omni:rtx:rendermode = "RealTimePathTracing"']
    else:
        render_mode_lines = [
            'token omni:rtx:rendermode = "Minimal"',
            f"int omni:rtx:minimal:mode = {minimal_mode}",
            f"bool omni:rtx:minimal:castShadows = {'true' if enable_shadows else 'false'}",
        ]
    render_mode_block = "\n        ".join(render_mode_lines)
    ordered_vars = ", ".join(f"<{path}>" for path, _, _ in render_var_configs)
    render_var_defs = "\n".join(
        f'''        def RenderVar "{name}"
        {{
            uniform string sourceName = "{source}"
        }}'''
        for _, name, source in render_var_configs
    )
    layer = f'''#usda 1.0
(defaultPrim = "{render_data.render_scope_name}")

def Scope "{render_data.render_scope_name}"
{{
    def RenderProduct "RenderProduct" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    ) {{
        rel camera = [<{render_data.camera_paths[0]}>]
        uint[] deviceIds = [{device_id}]
        {bg_type_line}
        {render_mode_block}
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = [{ordered_vars}]
        uniform int2 resolution = ({tiled_width}, {tiled_height})
    }}

    def "Vars"
    {{
{render_var_defs}
    }}
}}
'''
    if not spec.cfg.render_settings:
        return layer
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().ImportFromString(layer)
    product = stage.GetPrimAtPath(render_data.render_product_path)
    for name, (type_name, value) in spec.cfg.render_settings.items():
        product.CreateAttribute(name, _type(type_name)).Set(value)
    return stage.GetRootLayer().ExportToString()


def with_samples(
    layer: str,
    path: str,
    timed: Mapping[str, tuple[str, list[tuple[float, Any]]]],
    per_second: float,
) -> str:
    """Layer text `layer` (a render product's, `build_render_product_as_string`) with `timed`
    authored on its prim at `path` as time samples (`author_samples`), and the stage's
    `per_second` time codes a second said in it, so they are not retimed where it goes."""
    if not timed:
        return layer
    stage = Usd.Stage.CreateInMemory()
    stage.GetRootLayer().ImportFromString(layer)
    stage.GetRootLayer().timeCodesPerSecond = per_second
    author_samples(stage.GetPrimAtPath(path), timed)
    return stage.GetRootLayer().ExportToString()


def author_samples(
    prim: Usd.Prim, timed: Mapping[str, tuple[str, list[tuple[float, Any]]]]
) -> None:
    """`timed` (name: (USD type, [(time code, value)])) authored on `prim` as time samples, in
    its stage's edit target's layer."""
    for name, (type_name, keys) in timed.items():
        attribute = prim.GetAttribute(name) or prim.CreateAttribute(name, _type(type_name))
        for code, value in keys:
            attribute.Set(value, Usd.TimeCode(code))


def _type(name: str) -> Sdf.ValueTypeName:
    """The USD type called `name`: as Isaac Lab's settings say it ("Token") or as USD's
    schemas do ("token")."""
    return getattr(Sdf.ValueTypeNames, name, None) or Sdf.ValueTypeNames.Find(name)


def rational(value: Fraction) -> Gf.Vec2d:
    """`value` as ovrtx takes times and rates: (numerator, denominator)."""
    return Gf.Vec2d(value.numerator, value.denominator)


def rolling_shutter(
    rate: Fraction, rows: int, readout: Fraction, exposure: Fraction
) -> tuple[Fraction, Fraction, Fraction]:
    """How ovrtx is told a rolling shutter that reads `rows` rows over `readout` seconds, its
    shutter open `exposure` seconds, at `rate` frames a second: its `lineOffset`, its
    `lineExposure` and its `omni:sensor:startTime`, seconds.

    A capture's frame is at the middle of its last row's exposure, and row i `readout *
    (1 - i / (rows - 1))` before it (`darkroom.sensorkit.camera`). ovrtx takes line i to be
    exposed from `startTime + i * lineOffset` for `lineExposure`, and samples it once, at the
    END of that interval; it gathers no light over it. So:

    - `lineOffset = readout / (rows - 1)`: a row after a row.
    - `lineExposure = exposure / 2`: the first half of the row's exposure, whose end, where
      ovrtx samples, is the exposure's middle.
    - `startTime = -(readout + exposure / 2)`, a whole number of frames on (ovrtx takes none
      before 0): the first row's exposure opens then, and the last row is sampled on a frame's
      time, which is when ovrtx says the frame ended (`FrameOutput.end_time`).

    Raises ValueError when the first row opens more than two frames before the last is
    sampled: ovrtx keeps two exposures at once, and a third's rows come back black."""
    span = readout + exposure / 2
    if span > 2 / rate:
        raise ValueError(
            f"a readout of {float(readout):g} s with a shutter open {float(exposure):g} s spans "
            f"{float(span):g} s: more than two frames at {float(rate):g} Hz, which ovrtx cannot keep."
        )
    period = 1 / rate
    return readout / max(rows - 1, 1), exposure / 2, -span % period


def sensor_eye(stage: Usd.Stage, prim: Usd.Prim, rate: float, rows: int) -> Usd.Prim:
    """The camera that renders for camera `prim` in ovrtx: its eye (`darkroom.eyes.eye`, in
    the session layer), scheduled at `rate` frames a second, and where `prim` has a readout
    (`sensorkit.camera.READOUT`), a rolling shutter of its `rows` rows, top to bottom, timed
    as `rolling_shutter` says. Raises ValueError for a readout ovrtx cannot keep."""
    made = eye(stage, prim, leave=(SHUTTER_TEXTURE,))
    every = Fraction(rate).limit_denominator(1000)
    lens = sensorkit.camera.lens(prim)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        made.AddAppliedSchema("OmniSensorAPI")
        made.CreateAttribute("omni:sensor:frameRate", Sdf.ValueTypeNames.Double2).Set(
            rational(every)
        )
        if lens.readout > 0:
            try:
                offset, exposed, start = rolling_shutter(
                    every,
                    rows,
                    Fraction(lens.readout).limit_denominator(1_000_000),
                    Fraction(max(lens.exposure, 0.0)).limit_denominator(1_000_000),
                )
            except ValueError as exc:
                raise ValueError(f"{prim.GetPath()}: {exc}") from None
            made.AddAppliedSchema("OmniRtxRollingShutterAPI")
            made.CreateAttribute("omni:rtx:rollingShutter:scanOrder", Sdf.ValueTypeNames.Token).Set(
                "topToBottom"
            )
            for name, value in (
                ("omni:rtx:rollingShutter:lineOffset", offset),
                ("omni:rtx:rollingShutter:lineExposure", exposed),
                ("omni:sensor:startTime", start),
            ):
                made.CreateAttribute(name, Sdf.ValueTypeNames.Double2).Set(rational(value))
    return made


def author(prim: Usd.Prim, attributes: Mapping[str, tuple[str, Any]]) -> None:
    """`attributes` (name: (USD type, value)) authored on `prim`, in the session layer."""
    with Usd.EditContext(prim.GetStage(), prim.GetStage().GetSessionLayer()):
        for name, (type_name, value) in attributes.items():
            prim.CreateAttribute(name, _type(type_name)).Set(value)


def build_sensor_as_string(made: Usd.Prim) -> str:
    """What renders for a sensor (`sensor_eye`, `sensor_lidar`: `made`, in its stage's session
    layer) as a layer of its own, its default prim: what ovrtx is handed to put it on the
    stage it has open. It is taken out of the session layer: the stage is as it was."""
    session = made.GetStage().GetSessionLayer()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    # The stage's time codes a second: a sensor's time samples (a span's look) not retimed.
    layer.timeCodesPerSecond = made.GetStage().GetTimeCodesPerSecond()
    Sdf.CreatePrimInLayer(layer, "/Sensor")
    Sdf.CopySpec(session, made.GetPath(), layer, Sdf.Path("/Sensor"))
    layer.defaultPrim = "Sensor"
    with Usd.EditContext(made.GetStage(), session):
        made.GetStage().RemovePrim(made.GetPath())
    return layer.ExportToString()


#: What a lidar that measures is named: its lidar's name and this (Isaac Sim's alike).
LIDAR = sensorkit.lidar.CAPTURE
#: Isaac Sim's lidar attributes ovrtx's schema has not, by what stands for each (None: nothing
#: does; the scan rate is the sensor's frame rate, whole scans are asked for by
#: `partialOutputs`).
LIDAR_RENAMED: dict[str, str | None] = {
    "omni:sensor:Core:skipDroppingInvalidPoints": "omni:sensor:Core:includeInvalidPoints",
    "omni:sensor:Core:accumulateOutputs": None,
    "omni:sensor:Core:scanRateBaseHz": None,
    "omni:sensor:Core:reportRateBaseHz": None,
    "omni:sensor:Core:patternFiringRateHz": None,
    "omni:sensor:tickRate": None,
    "omni:sensor:frameRate": None,
    # A maker's own version ("0.1.0") is refused: ovrtx takes a whole number or "latest", its own.
    "omni:sensor:modelVersion": None,
}
#: How often a spinning lidar fires its pattern: Isaac Sim's attribute (firings a second) and
#: ovrtx's (degrees turned between firings).
LIDAR_FIRING_RATE = "omni:sensor:Core:patternFiringRateHz"
LIDAR_FIRING_ANGLE = "omni:sensor:Core:rotPatternFiringResDeg"
#: How a lidar that measures gives its points, whatever its lidar says: in its own axes, as
#: fired (not motion compensated), as x, y, z, a whole scan at a time (Isaac Sim's the same:
#: `kit.lidar.OVERRIDES`).
LIDAR_OVERRIDES: dict[str, tuple[str, Any]] = {
    "omni:sensor:Core:outputFrameOfReference": ("token", "SENSOR"),
    "omni:sensor:Core:outputMotionCompensationState": ("token", "NONCOMPENSATED"),
    "omni:sensor:Core:elementsCoordsType": ("token", "CARTESIAN"),
    "omni:sensor:Core:partialOutputs": ("bool", False),
}
#: How a lidar that measures is turned in its lidar's frame, degrees about x, y, z. ovrtx
#: takes a lidar prim as a camera lies (ahead down its -z, its +y up) and gives points in the
#: sensor's axes (+x ahead, +z up); a capture's lidar prim is those axes itself, as Isaac Sim
#: takes it. Turned so (NVIDIA's own lidar example's turn), ovrtx's sensor axes are the
#: lidar prim's.
LIDAR_TURN = (90.0, 0.0, -90.0)
#: The channels of a lidar's `PointCloud` a capture keeps (`Counts` and `Flags` come anyway),
#: and those kept besides when each point's object is (`RenderJob.hits`).
LIDAR_CHANNELS = ("Coordinates", "Intensity", "TimeOffsetNs", "EchoId", "ChannelId", "MaterialId")
LIDAR_HIT_CHANNELS = ("ObjectId",)


def sensor_lidar(stage: Usd.Stage, prim: Usd.Prim, rate: float) -> Usd.Prim:
    """The lidar that measures for lidar `prim` in ovrtx: a copy of it beside it (same
    parent), in the session layer, scanning `rate` times a second, with every sensor attribute
    authored on it under the name ovrtx's schema has for it (`LIDAR_RENAMED`), and
    `LIDAR_OVERRIDES`."""
    path = prim.GetParent().GetPath().AppendChild(f"{prim.GetName()}{LIDAR}")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        made = stage.DefinePrim(path, prim.GetTypeName())
        for api in prim.GetMetadata("apiSchemas").GetAppliedItems():
            if api.startswith("OmniSensor"):
                made.AddAppliedSchema(api)
        for attribute in prim.GetAttributes():
            name = attribute.GetName()
            if not name.startswith("omni:sensor:") or not attribute.HasAuthoredValue():
                continue
            name = LIDAR_RENAMED.get(name, name)
            if name is not None:
                made.CreateAttribute(name, attribute.GetTypeName()).Set(attribute.Get())
        made.CreateAttribute("omni:sensor:frameRate", Sdf.ValueTypeNames.Double2).Set(
            rational(Fraction(rate).limit_denominator(1000))
        )
        # A lidar that spins fires its pattern so many times a second (Isaac Sim's
        # attribute); ovrtx has it as the angle it turns between firings.
        fired = prim.GetAttribute(LIDAR_FIRING_RATE).Get()
        if fired and prim.GetAttribute(sensorkit.lidar.SCAN_TYPE).Get() == "ROTARY":
            made.CreateAttribute(LIDAR_FIRING_ANGLE, Sdf.ValueTypeNames.Float).Set(
                360.0 * rate / fired
            )
        made.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3).Set(LIDAR_TURN)
        made.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray, False).Set(
            ["xformOp:rotateXYZ"]
        )
    author(made, LIDAR_OVERRIDES)
    return made


def build_lidar_product_as_string(render_scope_name: str, lidar_path: str, hits: bool) -> str:
    """A lidar's render product as a layer: a RenderProduct pointed at the lidar at
    `lidar_path`, with one `PointCloud` RenderVar of the channels a capture keeps (with
    `hits`, each point's object ID too), under the scope `render_scope_name` (the layer's
    default prim, for referencing into OVRTX). What the object IDs are called is another
    product's to say (`build_names_product_as_string`), and has to be on the stage first."""
    channels = LIDAR_CHANNELS + (LIDAR_HIT_CHANNELS if hits else ())
    listed = ", ".join(f'"{channel}"' for channel in channels)
    return f'''#usda 1.0
(defaultPrim = "{render_scope_name}")

def Scope "{render_scope_name}"
{{
    def RenderProduct "RenderProduct"
    {{
        rel camera = [<{lidar_path}>]
        rel orderedVars = [</{render_scope_name}/Vars/PointCloud>]
    }}

    def "Vars"
    {{
        def RenderVar "PointCloud"
        {{
            uniform string sourceName = "PointCloud"
            token[] channels = [{listed}]
        }}
    }}
}}
'''


def build_names_product_as_string(render_scope_name: str) -> str:
    """What names a scene's objects, as a layer: a camera of a few pixels and its render
    product, whose one RenderVar is `StableIdMap`: the prim path of every object ID in the
    scene (what a lidar's points' `ObjectId`s are). Rendered once, before the lidars are put
    on, and left on the stage: with no ID map on the stage ovrtx 0.7.0 gives every point
    object 0, and with one on a lidar's own product each of its steps took 340 ms for 5 (four
    lidars on a street, the map 45 MB)."""
    return f'''#usda 1.0
(defaultPrim = "{render_scope_name}")

def Scope "{render_scope_name}"
{{
    def Camera "Camera"
    {{
    }}

    def RenderProduct "RenderProduct"
    {{
        rel camera = [</{render_scope_name}/Camera>]
        rel orderedVars = [</{render_scope_name}/Vars/StableIdMap>]
        uniform int2 resolution = (16, 16)
    }}

    def "Vars"
    {{
        def RenderVar "StableIdMap"
        {{
            uniform string sourceName = "StableIdMap"
        }}
    }}
}}
'''


def camera_settings(prim: Usd.Prim) -> dict[str, tuple[str, Any]]:
    """The RTX settings authored on camera `prim` itself (`omni:rtx:*`: its motion blur, its
    exposure), for its render product: name to (USD type, value)."""
    found: dict[str, tuple[str, Any]] = {}
    for attribute in prim.GetAttributes():
        name = attribute.GetName()
        if name.startswith("omni:rtx:") and attribute.HasAuthoredValue():
            found[name] = (str(attribute.GetTypeName()), attribute.Get())
    return found


def build_stage_as_string(stage: Usd.Stage) -> str:
    """The root layer ovrtx opens for `stage`: what its session layer holds (the eyes made
    for its cameras), over the stage's own root layer as a sublayer, with the stage's
    metadata (its time codes, its axes and units, its custom data)."""
    source = stage.GetRootLayer()
    root = Sdf.Layer.CreateAnonymous(".usda")
    root.TransferContent(stage.GetSessionLayer())
    for key in METADATA:
        if source.pseudoRoot.HasInfo(key):
            root.pseudoRoot.SetInfo(key, source.pseudoRoot.GetInfo(key))
    root.subLayerPaths.append(source.realPath or source.identifier)
    return root.ExportToString()
