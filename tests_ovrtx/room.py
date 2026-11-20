"""A box room a lidar moves through, and how far its points are off the walls: what
`test_ovrtx_lidar` holds ovrtx's lidars to (an XT32 moving 3 m/s and turning 180 degrees a
second; swaying; lidars of several rates)."""

import os
from pathlib import Path

import numpy as np
import usdsensors as us
from pxr import Gf, Sdf, Usd, UsdGeom

from darkroom import format, truth

HALF, HIGH = 10.0, 6.0
#: NVIDIA's Isaac Sim assets, which the room's lidars are made over (`$ISAAC_ASSETS`, NVIDIA's
#: own download): the tests that need them are skipped without.
ASSETS = Path(os.environ["ISAAC_ASSETS"]) if os.environ.get("ISAAC_ASSETS") else None
#: Each lidar: its file in the assets and its prim there (None: its default prim).
XT32 = ("Sensors/HESAI/XT32_SD10/Hesai_XT32_SD10.usda", "/XT_32_10hz")
#: A solid-state lidar firing all its rays at once: it can scan at any rate.
SOLID = ("Sensors/NVIDIA/Simple_Example_Solid_State.usda", None)


def have_assets() -> bool:
    return ASSETS is not None and (ASSETS / XT32[0]).is_file()


def _sensor(folder: Path, asset: tuple[str, str | None]) -> Path:
    """The lidar `asset` as a sensor file in `folder` (made the first time)."""
    assert ASSETS is not None
    file = folder / Path(asset[0]).name
    if not file.is_file():
        us.make_sensor(file, "lidar", ASSETS / asset[0], asset[1])
    return file


def _room(folder: Path) -> Path:
    """A 20 x 20 x 6 m room; an XT32 1.5 m up, along x at 3 m/s and turning 180°/s, for 3 s
    (the motion in a sublayer, as a trajectory's is)."""
    layer = Sdf.Layer.CreateNew(str(folder / "room.usda"))
    layer.timeCodesPerSecond = 60
    layer.startTimeCode, layer.endTimeCode = 0, 180
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    slabs = {
        "Floor": ((0, 0, -0.5), (24, 24, 1)),
        "Ceiling": ((0, 0, HIGH + 0.5), (24, 24, 1)),
        "East": ((HALF + 0.5, 0, 3), (1, 24, 8)),
        "West": ((-HALF - 0.5, 0, 3), (1, 24, 8)),
        "North": ((0, HALF + 0.5, 3), (24, 1, 8)),
        "South": ((0, -HALF - 0.5, 3), (24, 1, 8)),
    }
    for name, (centre, size) in slabs.items():
        slab = UsdGeom.Xformable(UsdGeom.Cube.Define(stage, f"/Room/{name}"))
        slab.AddTranslateOp().Set(Gf.Vec3d(*centre))
        slab.AddScaleOp().Set(Gf.Vec3f(*size) / 2)
    moving = Sdf.Layer.CreateNew(str(folder / "motion.usda"))
    moving.timeCodesPerSecond = 60
    on = Usd.Stage.Open(moving)
    UsdGeom.Scope.Define(on, "/Thing/Frames")
    mount = UsdGeom.Xform.Define(on, "/Thing/Frames/Mount")
    slide, turn = mount.AddTranslateOp(), mount.AddRotateZOp()
    for code in range(181):
        slide.Set(Gf.Vec3d(-4.5 + 3.0 * code / 60, 0.0, 1.5), code)
        turn.Set(180.0 * code / 60, code)
    moving.Save()
    layer.subLayerPaths.append("./motion.usda")
    rig = us.new_rig(folder / "XT32.usda")
    us.add(rig, "lidar", _sensor(folder, XT32), us.Pose())
    us.attach(layer, rig, "/Thing/Frames/Mount", "XT32")
    layer.Save()
    return folder / "room.usda"


def _off_the_walls(out: Path, late: float = 0.0, name: str = "lidar") -> np.ndarray:
    """How far each point of lidar `name` on the east and west walls is off them, put in the
    world by the lidar's pose `late` seconds after it was fired."""
    meta = format.read_meta(out)
    sensor = meta.sensors[name]
    stage = Usd.Stage.Open(meta.stage)
    span = truth.Window(stage, (meta.start, meta.end))
    off = []
    for number in format.scan_numbers(out, name)[2:-2]:
        scan = format.read_scan(out, name, number)
        fired = number / sensor.rate + scan.offset.astype(float) + late
        poses = truth.poses(stage, sensor.prim, span.code(span.start + fired))
        ones = np.c_[scan.points, np.ones(len(scan.points))]
        p = np.einsum("ni,nij->nj", ones, poses)[:, :3]
        wall = (
            (np.abs(np.abs(p[:, 0]) - HALF) < 1.5)
            & (np.abs(p[:, 1]) < HALF - 1.5)
            & (p[:, 2] > 0.5)
            & (p[:, 2] < HIGH - 0.5)
        )
        off.append(np.abs(np.abs(p[wall, 0]) - HALF))
    return np.concatenate(off)


def _sway(stage_file: Path) -> Path:
    """The room's lidar swaying as it goes, as one carried does: half a metre to and fro and
    40 degrees either way, twice a second, over its steady 3 m/s and 180 degrees a second."""
    motion = Usd.Stage.Open(str(stage_file.with_name("motion.usda")))
    mount = motion.GetPrimAtPath("/Thing/Frames/Mount")
    slide, turn = mount.GetAttribute("xformOp:translate"), mount.GetAttribute("xformOp:rotateZ")
    for code in range(181):
        sway = np.sin(2 * np.pi * 2.0 * code / 60)
        slide.Set(Gf.Vec3d(-4.5 + 3.0 * code / 60 + 0.5 * sway, 0.0, 1.5), code)
        turn.Set(180.0 * code / 60 + 40.0 * sway, code)
    motion.GetRootLayer().Save()
    return stage_file


def _rates(stage_file: Path, rates: tuple[float, ...] = (30.0,)) -> Path:
    """The room's XT32 (10 Hz) with a solid-state lidar beside it at each of `rates` (`st<rate>`:
    NVIDIA's simple example, made for 30 Hz, re-rated, its report and pattern rates with it,
    and its 12 emitters fired at once, as a flash lidar's are: spread over 1/30 s as NVIDIA has
    them, a scan at 60 Hz would end before its last firing, and ovrtx makes no points)."""
    rig = stage_file.with_name("XT32.usda")
    for rate in rates:
        name = f"st{rate:g}"
        us.add(
            rig, name, _sensor(stage_file.parent, SOLID), us.Pose((0.0, 0.0, 0.2), (0.0, 0.0, 0.0))
        )
        for attribute in ("tickRate", "Core:scanRateBaseHz", "Core:reportRateBaseHz",
                          "Core:patternFiringRateHz"):  # fmt: skip
            us.change(rig, name, f"omni:sensor:{attribute}", rate)
        us.change(rig, name, "omni:sensor:Core:emitterState:s001:fireTimeNs", [0] * 12)
    return stage_file


def check_rates(out: Path, rates: tuple[float, ...]) -> None:
    """`_rates`' room captured over 3 s: every lidar with each of the window's whole scans,
    once; the XT32's first as full as the rest and each of its scans begun on its pose (its
    points on the walls)."""
    meta = format.read_meta(out)
    for name, rate in (("lidar", 10.0), *((f"st{r:g}", r) for r in rates)):
        numbers = format.scan_numbers(out, name)
        assert meta.sensors[name].rate == rate and numbers == list(range(round(3 * rate))), (
            name,
            numbers,
        )
    counts = [len(format.read_scan(out, "lidar", n).points) for n in (0, 1, 2, 15)]
    assert counts[0] > 0.98 * np.median(counts), counts
    off = _off_the_walls(out)
    assert np.median(off) < 0.006, np.median(off)
