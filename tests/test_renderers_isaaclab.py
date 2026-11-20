"""darkroom's renderers held against Isaac Lab's, whose design they copy: the same classes, the
same methods with the same parameters, the same cfg fields with the same defaults, the same
names for a camera's outputs. Isaac Lab is read as source (`$ISAACLAB_SOURCE`: a clone's
`source` folder), never imported: it needs torch and Warp. Skipped where it is not.

What differs is listed here, each with why (`DIFFERENT`, `ONLY_OURS`, `ONLY_THEIRS`): anything
else differing fails, so a change in either is seen. `NEEDS` lists what each of Isaac Lab's own
renderers asks of its simulation that darkroom's has not: what porting one would take.
"""

import ast
import os
from pathlib import Path

import pytest

LAB = Path(os.environ.get("ISAACLAB_SOURCE") or "no-isaaclab-source")
OURS = Path(__file__).parents[1] / "src" / "darkroom"

pytestmark = pytest.mark.skipif(
    not (LAB / "isaaclab" / "isaaclab" / "renderers" / "base_renderer.py").is_file(),
    reason=f"Isaac Lab's source is not at {LAB}",
)

BASE = "isaaclab/isaaclab/renderers/base_renderer.py"
OV_CFG = "isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer_cfg.py"
OV = "isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer.py"
OV_USD = "isaaclab_ov/isaaclab_ov/renderers/ovrtx_usd.py"
NEWTON = "isaaclab_newton/isaaclab_newton/renderers/newton_warp_renderer.py"

#: Each class of the interface: Isaac Lab's file, darkroom's.
CLASSES = {
    "BaseRenderer": (BASE, "renderers/base_renderer.py"),
    "VisualMaterialBatch": (BASE, "renderers/base_renderer.py"),
    "RendererCfg": ("isaaclab/isaaclab/renderers/renderer_cfg.py", "renderers/renderer_cfg.py"),
    "BackendCfg": ("isaaclab/isaaclab/sim/simulation_cfg.py", "renderers/renderer_cfg.py"),
    "CameraRenderSpec": (
        "isaaclab/isaaclab/renderers/camera_render_spec.py",
        "renderers/camera_render_spec.py",
    ),
    "RenderBufferKind": (
        "isaaclab/isaaclab/renderers/output_contract.py",
        "renderers/output_contract.py",
    ),
    "RenderBufferSpec": (
        "isaaclab/isaaclab/renderers/output_contract.py",
        "renderers/output_contract.py",
    ),
    "OVRTXRendererCfg": (OV_CFG, "ovrtx/ovrtx_renderer_cfg.py"),
    "OVRTXBackendCfg": (OV_CFG, "ovrtx/ovrtx_renderer_cfg.py"),
    "OvstageBackendCfg": ("isaaclab_ov/isaaclab_ov/stage.py", "ovrtx/stage.py"),
}
#: Each renderer, and what it owns: Isaac Lab's file, darkroom's.
RENDERERS = {
    "OVRTXRenderer": (OV, "ovrtx/ovrtx_renderer.py"),
    "OVRTXBackend": (OV, "ovrtx/ovrtx_renderer.py"),
    "OVRTXCameraRenderData": (OV, "ovrtx/ovrtx_renderer.py"),
    "OvstageBackend": ("isaaclab_ov/isaaclab_ov/stage.py", "ovrtx/stage.py"),
}

A_MODULE = "where the class is: darkroom's own module, said whole (no configclass's {DIR})"
FROZEN = "a frozen dataclass: a default made by a factory, not shared"
NO_MISSING = "no MISSING in a dataclass: the value a capture has"
INDICES = "a capture keeps each pixel's label index, never a colour"
#: Fields whose defaults differ from Isaac Lab's, and why.
DIFFERENT = {
    "BackendCfg": {"class_type": "None until a concrete cfg sets it: no MISSING in a dataclass"},
    "OVRTXRendererCfg": {
        "class_type": A_MODULE,
        "launcher_type": "darkroom's launcher starts a process; Isaac Lab's registers schemas",
        "cloning_contexts": "a capture has one stage: nothing is cloned",
        "colorize_semantic_segmentation": INDICES,
        "colorize_instance_segmentation": INDICES,
        "colorize_instance_id_segmentation": INDICES,
    },
    "OVRTXBackendCfg": {
        "class_type": A_MODULE,
        "use_ovstage": NO_MISSING + ": through an OVStage, which plays the stage's USD time",
        "read_gpu_transforms": NO_MISSING + ": ovstage's own transforms",
        "device": NO_MISSING,
    },
    "OvstageBackendCfg": {"class_type": A_MODULE},
}
#: Fields and public methods darkroom has that Isaac Lab has not, and why.
ONLY_OURS = {
    "OVRTXBackendCfg": {
        "motion_bvh": "RTX's motion BVH, which a rolling shutter needs",
        "cache_root_path": "where OVRTX keeps its compiled shaders",
    },
    "OVRTXRendererCfg": {
        "step_rate": "steps a second: fast rotation under a rolling shutter",
        "motion_bvh": "RTX's motion BVH only for what needs it: 2.4 times slower with it",
        "cache_root_path": "where OVRTX keeps its compiled shaders",
    },
    "OVRTXBackend": {
        "attach": "the engine put on its stage (Isaac Lab's renderer does it)",
    },
    "OvstageBackend": {"write": "an attribute written at the write ordinal, in one call"},
    "OVRTXRenderer": {
        "create_lidar_data": "a capture's lidars: Isaac Lab's renderers render cameras only",
        "scan_batch": "a step for the lidars, as `render_batch` is for the cameras",
        "object_table": "what each object a lidar hit is called, read once",
    },
}
#: Fields and public methods Isaac Lab has that darkroom has not, and why.
ONLY_THEIRS = {
    "OvstageBackendCfg": {
        "population_domains": "a capture populates rendering only",
        "populate_physics": "no physics",
        "defer_render_copies": "nothing is cloned",
    },
    "OvstageBackend": {
        "render_copies_pending": "nothing is cloned",
        "complete_render_copies": "nothing is cloned",
        "reset": "a capture's stage is made once",
    },
    "OVRTXRenderer": {
        "drain_pending_renders": "async rendering only: a capture renders synchronously",
        "log_mesh": "runtime meshes (markers): a capture's stage is all there is",
    },
}
#: What each of Isaac Lab's renderers asks of its simulation (`sim.<name>`) that darkroom's
#: `SimulationContext` has not: what running that renderer on a capture would take.
NEEDS = {
    "OVRTXRenderer": {
        "get_scene_data_provider": "where things are (`PROVIDER`): ovstage plays the stage",
        "render_context": "what drives its renderers: darkroom's loop",
    },
    "NewtonWarpRenderer": {
        "get_scene_data_provider": "where things are (`PROVIDER`): the port's main work",
        "cfg": "the physics cfg its Newton model is built with",
        "get_clone_plan": "its environments' copies: none in a capture",
        "requires_newton_model": "a Newton model built from the stage (newton's USD importer)",
        "requires_usd_stage": "a flag: darkroom's simulation always has its stage",
    },
}
#: What Isaac Lab's renderers ask of where things are (`sim.get_scene_data_provider()`), none
#: of which darkroom's simulation gives: its renderers follow the stage's USD time
#: themselves. A Newton Warp renderer needs them all: every body's world transform and every
#: deformed mesh's points at the simulation's time, worked out from the stage (USD's xform
#: cache, UsdSkel's skinning) and handed over as arrays, with a version to tell a change by.
PROVIDER = {"get_transforms", "get_geometry_points", "create_mapping", "backend"}


def _class(file: Path, name: str) -> ast.ClassDef:
    for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{file} has no class {name}")


def _methods(cls: ast.ClassDef) -> dict[str, tuple[list[str], list[str]]]:
    """Each public method's parameters: (positional, keyword-only)."""
    found = {}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and (
            not node.name.startswith("_") or node.name == "__init__"
        ):
            args = node.args
            found[node.name] = (
                [a.arg for a in args.posonlyargs + args.args],
                [a.arg for a in args.kwonlyargs],
            )
    return found


def _fields(cls: ast.ClassDef) -> dict[str, str | None]:
    """Each annotated field's default, as source; None when it has none."""
    return {
        node.target.id: ast.unparse(node.value) if node.value is not None else None
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }


def _abstract(cls: ast.ClassDef) -> set[str]:
    return {
        node.name
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and any(ast.unparse(d) == "abstractmethod" for d in node.decorator_list)
    }


def _pair(name: str, files: dict[str, tuple[str, str]]) -> tuple[ast.ClassDef, ast.ClassDef]:
    theirs, ours = files[name]
    return _class(LAB / theirs, name), _class(OURS / ours, name)


def test_base_renderer_has_isaac_labs_methods_with_its_parameters():
    theirs, ours = _pair("BaseRenderer", CLASSES)
    assert _methods(ours) == _methods(theirs)
    assert _abstract(ours) == _abstract(theirs)
    # In the order Isaac Lab declares them: the order they are used in.
    assert list(_methods(ours)) == list(_methods(theirs))


@pytest.mark.parametrize("name", CLASSES)
def test_a_class_has_isaac_labs_fields_and_defaults_but_for_those_listed(name):
    theirs, ours = _pair(name, CLASSES)
    fields_theirs, fields_ours = _fields(theirs), _fields(ours)
    different, only_ours = DIFFERENT.get(name, {}), ONLY_OURS.get(name, {})
    only_theirs = ONLY_THEIRS.get(name, {})
    assert set(fields_ours) - set(fields_theirs) == set(only_ours) & set(fields_ours)
    assert set(fields_theirs) - set(fields_ours) == set(only_theirs) & set(fields_theirs)
    differing = {
        field
        for field in set(fields_ours) & set(fields_theirs)
        if fields_ours[field] != fields_theirs[field]
    }
    assert differing == set(different), (name, sorted(differing ^ set(different)))


@pytest.mark.parametrize("name", [*CLASSES, *RENDERERS])
def test_a_class_has_isaac_labs_methods_but_for_those_listed(name):
    theirs, ours = _pair(name, {**CLASSES, **RENDERERS})
    methods_theirs, methods_ours = _methods(theirs), _methods(ours)
    base = _methods(_class(OURS / "renderers/base_renderer.py", "BaseRenderer"))
    # A method both have takes the same parameters.
    for method in set(methods_ours) & set(methods_theirs):
        assert methods_ours[method] == methods_theirs[method], (name, method)
    listed = set(ONLY_OURS.get(name, {})) - set(_fields(ours))
    assert set(methods_ours) - set(methods_theirs) - set(base) == listed
    # Isaac Lab's public methods darkroom's class has not: the base's own, or listed.
    lacking = set(methods_theirs) - set(methods_ours)
    if name in RENDERERS and name.endswith("Renderer"):
        lacking -= set(base)
    assert lacking == set(ONLY_THEIRS.get(name, {})) - set(_fields(theirs)), (name, lacking)


@pytest.mark.parametrize("name", ["OVRTXCameraRenderData"])
def test_render_data_fields_differ_only_as_listed(name):
    theirs, ours = _pair(name, RENDERERS)
    fields_theirs, fields_ours = _fields(theirs), _fields(ours)
    assert set(fields_ours) - set(fields_theirs) == set(ONLY_OURS.get(name, {}))
    assert set(fields_theirs) - set(fields_ours) == set(ONLY_THEIRS.get(name, {}))


@pytest.mark.parametrize("name", ["OVRTXRenderer"])
def test_a_renderer_implements_every_abstract_method(name):
    _, ours = _pair(name, RENDERERS)
    abstract = _abstract(_class(OURS / "renderers/base_renderer.py", "BaseRenderer"))
    assert abstract <= set(_methods(ours))


def test_renderers_are_named_as_isaac_labs():
    from darkroom import renderers

    for cfg_name, kind, cls, module in (
        ("OVRTXRendererCfg", "ovrtx", "OVRTXRenderer", "ovrtx_renderer"),
    ):
        cfg = getattr(renderers, cfg_name)()
        assert (
            cfg.renderer_type
            == kind
            == ast.literal_eval(
                _fields(_class(LAB / CLASSES[cfg_name][0], cfg_name))["renderer_type"] or ""
            )
        )
        # "module:Class", the module and the class named as Isaac Lab's.
        assert cfg.class_type.rpartition(".")[2] == f"{module}:{cls}"
        theirs = ast.literal_eval(
            _fields(_class(LAB / CLASSES[cfg_name][0], cfg_name))["class_type"] or ""
        )
        assert theirs == f"{{DIR}}.{module}:{cls}"
    assert renderers.RendererCfg().renderer_type == "default"
    assert type(renderers.get_default_renderer_cfg()).__name__ == "OVRTXRendererCfg"


def _source(node: ast.AST | None) -> str:
    return ast.unparse(node) if node is not None else ""


def _outputs(file: Path, name: str) -> dict[str, tuple[int, str] | None]:
    """The outputs a cfg's `supported_output_types` returns, read from its source: each
    kind's (channels, dtype's name); None for one laid out by a call (a segmentation's)."""
    cls = _class(file, name)
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "supported_output_types"
    )
    table = next(  # the first table of outputs in it: returned as it is, or added to after
        n
        for n in ast.walk(method)
        if isinstance(n, ast.Dict) and n.keys and _source(n.keys[0]).startswith("RenderBuffer")
    )
    found: dict[str, tuple[int, str] | None] = {}
    for key, value in zip(table.keys, table.values, strict=True):
        assert isinstance(key, ast.Attribute) and isinstance(value, ast.Call)
        if ast.unparse(value.func) == "RenderBufferSpec":
            channels, dtype = value.args
            assert isinstance(channels, ast.Constant) and isinstance(dtype, ast.Attribute)
            found[key.attr] = (int(str(channels.value)), dtype.attr)
        else:
            found[key.attr] = None
    return found


@pytest.mark.parametrize("name", ["OVRTXRendererCfg"])
def test_a_renderers_outputs_are_among_isaac_labs_laid_out_alike(name):
    theirs = _outputs(LAB / CLASSES[name][0], name)
    ours = _outputs(OURS / CLASSES[name][1], name)
    assert set(ours) <= set(theirs)  # by the same names; fewer: what a capture keeps
    for kind, layout in ours.items():
        # Isaac Lab lays a segmentation out by whether it is coloured; a capture's never is:
        # one int32 a pixel, as Isaac Lab's when not coloured.
        assert layout == (theirs[kind] or (1, "int32")), kind


def test_output_names_are_isaac_labs():
    theirs, ours = _pair("RenderBufferKind", CLASSES)

    def members(cls):
        return {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in cls.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        }

    assert members(ours) == members(theirs)


def _assigned(file: Path, name: str) -> str:
    """What module `file` assigns `name` at its top, as source."""
    for node in ast.parse(file.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign):
            target, value = node.targets[0], node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name and value is not None:
            return ast.unparse(value)
    raise AssertionError(f"{file} assigns no {name}")


def _functions(file: Path) -> dict[str, tuple[list[str], list[str]]]:
    return {
        n.name: ([a.arg for a in n.args.args], [a.arg for a in n.args.kwonlyargs])
        for n in ast.parse(file.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef)
    }


def test_ovrtx_render_vars_and_their_functions_are_isaac_labs():
    theirs, ours = LAB / OV_USD, OURS / "ovrtx/ovrtx_usd.py"
    for table in (
        "_RENDER_VAR_BY_DATA_TYPE",
        "_RTX_MINIMAL_MODES",
        "_COLOR_DATA_TYPES",
        "_SEGMENTATION_MAP_RENDER_VARS",
    ):
        assert ast.literal_eval(
            _assigned(ours, table).replace("frozenset(", "(")
        ) == ast.literal_eval(_assigned(theirs, table).replace("frozenset(", "(")), table
    for function in (
        "get_render_var_configs",
        "render_var_prim_names_by_source",
        "build_render_product_as_string",
    ):
        assert _functions(ours)[function] == _functions(theirs)[function], function


def _asked_of_sim(file: Path, name: str) -> tuple[set[str], set[str]]:
    """What a renderer in Isaac Lab asks of its simulation (`self._sim.x`, `sim.x`), and of
    what says where things are (`provider.x`, whatever it calls it)."""
    sim: set[str] = set()
    provider: set[str] = set()
    for node in ast.walk(_class(file, name)):
        if not isinstance(node, ast.Attribute):
            continue
        owner = ast.unparse(node.value)
        if owner in ("self._sim", "sim"):
            sim.add(node.attr)
        if owner in ("self._sdp", "self._scene_data_provider", "provider"):
            provider.add(node.attr)
    return sim, provider


@pytest.mark.parametrize("name, file", [("OVRTXRenderer", OV), ("NewtonWarpRenderer", NEWTON)])
def test_isaac_labs_renderers_would_fit_darkrooms_interface(name, file):
    """Each of Isaac Lab's renderers, Newton Warp's too, overrides the base's methods with
    the parameters darkroom's base has, and asks of its simulation only what darkroom's has
    or `NEEDS` lists."""
    from darkroom.renderers import SimulationContext

    base = _methods(_class(OURS / "renderers/base_renderer.py", "BaseRenderer"))
    for method, parameters in _methods(_class(LAB / file, name)).items():
        if method in base:
            assert parameters == base[method], (name, method)
    sim, provider = _asked_of_sim(LAB / file, name)
    lacking = {x for x in sim if not hasattr(SimulationContext, x) and x not in ("stage", "device")}
    assert lacking == set(NEEDS[name]), (name, sorted(lacking ^ set(NEEDS[name])))
    assert provider <= PROVIDER, (name, sorted(provider - PROVIDER))
