"""A lidar measured by ovrtx for real (the GPU, seconds): a room (`room`: an XT32 moving 3 m/s
and turning 180 degrees a second between walls): every point, put in the world by where the lidar was when it
fired it, lies on the walls. Any error in the scans' timing, in the axes ovrtx gives points
in, or in how motion through a scan is rendered moves them off.

    uv run pytest tests_ovrtx
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ovrtx")

from darkroom import Options, capture, format
from darkroom.renderers import OVRTXRendererCfg


@pytest.fixture(scope="module")
def room():
    """The room and its measure of how far points are off the walls."""
    import room as module

    if not module.have_assets():
        pytest.skip("no NVIDIA sensor assets: set ISAAC_ASSETS to NVIDIA's Isaac assets folder")
    return module


@pytest.fixture(scope="module")
def captured(room, tmp_path_factory):
    folder = tmp_path_factory.mktemp("room")
    options = Options(kinds=["lidar"], renderer_cfg=OVRTXRendererCfg(), lidar_hits=True)
    return capture(room._room(folder), folder / "capture", options, on_line=None)


def test_a_moving_lidars_points_lie_on_the_walls_where_it_fired_them(room, captured):
    off = room._off_the_walls(captured)
    assert len(off) > 10_000
    # How far a still lidar's points are off the walls, at most.
    assert np.median(off) < 0.006 and np.percentile(off, 90) < 0.015, (
        np.median(off),
        np.percentile(off, 90),
    )
    # A thirtieth of a second out, they are well off: the test sees timing that small.
    assert np.median(room._off_the_walls(captured, late=1 / 30)) > 0.2


def test_every_whole_scan_in_the_window_is_kept_with_what_each_point_hit(captured):
    meta = format.read_meta(captured)
    name = next(n for n, s in meta.sensors.items() if s.kind == "lidar")
    assert format.scan_numbers(captured, name) == list(range(30))  # 3 s at 10 Hz
    scan = format.read_scan(captured, name, 10)
    hits = format.read_hits(captured, name)
    assert set(hits) <= {
        f"/Room/{wall}" for wall in ("Floor", "Ceiling", "East", "West", "North", "South")
    }
    assert len(hits) >= 4 and scan.hit.max() < len(hits)  # every point's wall named
    east = scan.points[np.array(hits)[scan.hit] == "/Room/East"]
    assert len(east) > 100


def test_lidars_at_10_15_and_60_hz_keep_every_scan_each_on_its_pose(room, tmp_path_factory):
    """Lidars at several rates (`room.check_rates`), which ovrtx renders losing no
    scan of a lidar off its model's own rate: 15 Hz (its periods off the XT32's but every
    fifth of a second) and 60 Hz (as fast as the stage's frames)."""
    folder = tmp_path_factory.mktemp("rates")
    options = Options(kinds=["lidar"], renderer_cfg=OVRTXRendererCfg())
    rates = (15.0, 30.0, 60.0)
    room.check_rates(
        capture(room._rates(room._room(folder), rates), folder / "capture", options), rates
    )


def _sway(stage_file: Path) -> Path:
    """The room's lidar swaying as it goes, as one carried does: half a metre to and fro and
    40 degrees either way, twice a second, over its steady 3 m/s and 180 degrees a second."""
    from pxr import Gf, Usd

    motion = Usd.Stage.Open(str(stage_file.with_name("motion.usda")))
    mount = motion.GetPrimAtPath("/Thing/Frames/Mount")
    slide, turn = mount.GetAttribute("xformOp:translate"), mount.GetAttribute("xformOp:rotateZ")
    for code in range(181):
        sway = np.sin(2 * np.pi * 2.0 * code / 60)
        slide.Set(Gf.Vec3d(-4.5 + 3.0 * code / 60 + 0.5 * sway, 0.0, 1.5), code)
        turn.Set(180.0 * code / 60 + 40.0 * sway, code)
    motion.GetRootLayer().Save()
    return stage_file


def test_a_swaying_lidars_points_lie_on_the_walls_too(room, tmp_path_factory):
    """A scan is gathered over the stage's own frames, six to a scan, each firing traced from
    where the lidar was in its frame: a sway within a scan is in its points (3.4 mm off the
    walls, the median). Traced between the scan's first and last pose alone (an update a
    scan, as Isaac Sim's lidars are measured), they were 98 mm off."""
    folder = tmp_path_factory.mktemp("sway")
    options = Options(kinds=["lidar"], renderer_cfg=OVRTXRendererCfg())
    out = capture(_sway(room._room(folder)), folder / "capture", options, on_line=None)
    off = room._off_the_walls(out)
    assert len(off) > 10_000
    assert np.median(off) < 0.006 and np.percentile(off, 90) < 0.02, (
        np.median(off),
        np.percentile(off, 90),
    )


def test_object_ids_are_named_from_ovrtxs_table_none_for_one_it_has_not():
    from darkroom.ovrtx import ovrtx_lidar

    paths = ["/World/Wall", "/World/Ground/dirt6", "/World/Tree", "/World/Rock"]
    ids = np.array([[7, 0, 0, 0], [1, 2, 3, 0], [9, 9, 9, 9], [1, 2, 4, 0]], np.uint32)
    ends = np.cumsum([0, *(len(p) for p in paths)]).astype(np.uint64)
    table = (ids, ends, np.frombuffer("".join(paths).encode(), np.uint8))
    asked = np.array([[9, 9, 9, 9], [1, 2, 4, 0], [1, 2, 3, 0], [5, 5, 5, 5]], np.uint32)
    names = ovrtx_lidar.Names(table)  # two IDs alike in their low half: told apart
    assert names(asked.view(np.uint8)) == [
        "/World/Tree",
        "/World/Rock",
        "/World/Ground/dirt6",
        None,
    ]
    empty = (np.zeros((0, 4), np.uint32), np.zeros(1, np.uint64), np.zeros(0, np.uint8))
    assert ovrtx_lidar.Names(empty)(asked.view(np.uint8)) == [None] * 4


def test_an_output_ovrtx_could_not_make_is_said_with_its_reason():
    from types import SimpleNamespace

    import ovrtx

    from darkroom.ovrtx import ovrtx_renderer

    made = SimpleNamespace(status=ovrtx.EventStatus.COMPLETED, error_message="")
    failed = SimpleNamespace(status=ovrtx.EventStatus.FAILURE, error_message="out of device memory")
    ovrtx_renderer._raise_failed(
        "/Rig/front", SimpleNamespace(render_vars={"/R/Vars/LdrColor": made})
    )
    frame = SimpleNamespace(render_vars={"/R/Vars/LdrColor": failed, "/R/Vars/NormalSD": made})
    with pytest.raises(
        RuntimeError, match="/Rig/front: ovrtx could not make LdrColor: out of device memory"
    ):
        ovrtx_renderer._raise_failed("/Rig/front", frame)


def test_ovrtx_is_started_sending_nvidia_neither_reports_nor_crash_dumps(monkeypatch):
    import ovrtx

    from darkroom.ovrtx import ovrtx_renderer
    from darkroom.ovrtx.ovrtx_renderer_cfg import OVRTXBackendCfg

    said = []
    monkeypatch.setattr(ovrtx, "library_telemetry_opt_out", lambda: said.append("no reports"))
    monkeypatch.setattr(ovrtx, "Renderer", lambda config: said.append(config))
    ovrtx_renderer.OVRTXBackend(OVRTXBackendCfg())
    assert said[0] == "no reports"  # before the engine is made
    assert said[1].enable_crash_reporter is False
    assert said[1].cache_root_path is None  # its caches where it keeps them itself
    ovrtx_renderer.OVRTXBackend(OVRTXBackendCfg(cache_root_path="/var/cache/ovrtx"))
    assert said[3].cache_root_path == "/var/cache/ovrtx"


def test_ovrtxs_semantic_classes_are_named_as_replicators_are():
    from darkroom.ovrtx import ovrtx_renderer

    said = ["class: BACKGROUND;", "class: UNLABELLED;", "class: tree; other: oak;", ""]
    ends = np.cumsum([0, *(len(s) for s in said)]).astype(np.uint64)
    ids = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [5, 0, 0, 0], [6, 0, 0, 0]], np.uint32)
    named = ovrtx_renderer.semantic_labels(
        ids, ends, np.frombuffer("".join(said).encode(), np.uint8)
    )
    assert named["5"] == {"class": "tree", "other": "oak"}
    assert named["0"] == {"class": "BACKGROUND"} and named["6"] == {}
