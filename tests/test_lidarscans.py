"""A lidar's scans as Isaac Sim writes them (a file a field, appended) packed into its one LAZ:
each scan read back as it went in, its points' times from its pose, its hits named."""

import json
from pathlib import Path

import numpy as np
import pytest

from darkroom import format, lidarscans

RATE = 10.0


def _scan(rng, n, k):
    return {
        "points": rng.uniform(-30, 30, (n, 3)).astype(np.float32),
        "intensity": rng.uniform(0, 1.3, n).astype(np.float32),
        "offsets": np.sort(rng.integers(0, 99_000_000, n)).astype(np.int64),
        "ret": rng.integers(0, 2, n).astype(np.uint8),  # first and second returns
        "ring": rng.integers(0, 128, n).astype(np.uint32),
        "material": rng.integers(0, 40, n).astype(np.uint32),
        "objects": np.repeat(
            np.array([[k % 3 + 1] + [0] * 15], np.uint8), n, axis=0
        ),  # IDs 1, 2 or 3
    }


@pytest.mark.parametrize("hits", [False, True])
def test_raw_scans_pack_into_one_laz_each_as_it_went_in(tmp_path, hits, monkeypatch):
    rng = np.random.default_rng(3)
    count = 6
    format.write_poses(
        tmp_path,
        "lidar",
        format.Poses(
            np.arange(count) / RATE, np.zeros((count, 3)), np.tile([1.0, 0, 0, 0], (count, 1))
        ),
    )
    raw = tmp_path / "lidar" / lidarscans.RAW
    writer = lidarscans.RawWriter(raw, with_hits=hits)
    went = {}
    stamp0 = 7_000_000_000
    for k in range(count):
        scan = _scan(rng, 0 if k == 2 else 500 + 100 * k, k)  # scan 2 saw nothing
        writer.add(stamp0 + k * 100_000_000, **scan)
        went[k] = scan
    writer.close()
    # Kept: every scan but the first, which began a little late (2 ms) and is numbered so.
    kept = [(k, k, k / RATE + 0.002) for k in range(1, count)]
    names = {1: "/World/Wall", 2: "/World/Car"}  # ID 3 not named
    lidarscans.write_index(raw, writer.stamps, writer.counts, kept, RATE, names if hits else None)
    told = []
    rmtree = lidarscans.shutil.rmtree

    def deleting(folder):  # as Windows would: no raw file may still be mapped
        import gc

        gc.collect()
        mapped = [m for m in gc.get_objects() if isinstance(m, np.memmap) and m.filename]
        held = [m.filename for m in mapped if Path(m.filename).is_relative_to(folder)]
        assert not held, f"still mapped as {folder} goes: {held}"
        rmtree(folder)

    monkeypatch.setattr(lidarscans.shutil, "rmtree", deleting)
    packed, before, after = lidarscans.pack_all(tmp_path, lambda s, d: told.append((s, d)))
    assert packed == count - 1 and not raw.exists() and after < before
    assert told[0] == ("lidar", 0.0) and told[-1] == ("lidar", 1.0)
    assert format.scan_numbers(tmp_path, "lidar") == list(range(1, count))
    paths = format.read_hits(tmp_path, "lidar")
    assert paths == (["/World/Car", "/World/Wall"] if hits else [])
    for k in range(1, count):
        back, scan = format.read_scan(tmp_path, "lidar", k), went[k]
        assert np.abs(back.points - scan["points"]).max(initial=0) <= 0.6e-4
        assert np.array_equal(back.intensity, scan["intensity"])
        assert np.array_equal(back.ring, scan["ring"]) and np.array_equal(back.ret, scan["ret"])
        offset = (scan["offsets"] / 1e9 + 0.002).astype(np.float32)  # from its pose
        assert np.array_equal(back.offset, offset)
        if len(scan["ring"]):
            if hits:
                want = {1: paths.index("/World/Wall"), 2: paths.index("/World/Car")}
                assert np.all(back.hit == want.get(k % 3 + 1, format.NO_HIT))
            else:
                assert np.all(back.hit == format.NO_HIT)
            assert np.array_equal(back.material, scan["material"])
    import laspy

    las = laspy.read(str(tmp_path / "lidar" / format.PACKED))
    for k in range(1, count):
        mine = np.asarray(las.scan) == k
        if mine.any():  # each scan's own greatest return, its time from its pose
            assert set(np.asarray(las.number_of_returns)[mine]) == {went[k]["ret"].max() + 1}
            assert np.allclose(
                las.gps_time[mine], k / RATE + (went[k]["offsets"] / 1e9 + 0.002), atol=1e-6
            )


def test_scans_out_to_a_long_range_lidars_reach_pack(tmp_path):
    rng = np.random.default_rng(5)
    format.write_poses(
        tmp_path, "lidar", format.Poses(np.zeros(1), np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]]))
    )
    raw = tmp_path / "lidar" / lidarscans.RAW
    writer = lidarscans.RawWriter(raw)
    scan = _scan(rng, 20_000, 0)
    scan["points"] = rng.uniform(-350, 350, (20_000, 3)).astype(np.float32)  # an OS2's
    del scan["objects"]
    writer.add(7_000_000_000, **scan)
    writer.close()
    lidarscans.write_index(raw, writer.stamps, writer.counts, [(0, 0, 0.0)], RATE, None)
    assert lidarscans.pack_all(tmp_path)[0] == 1
    back = format.read_scan(tmp_path, "lidar", 0)
    assert np.abs(back.points - scan["points"]).max() <= 0.6e-4 + 350 * np.finfo(np.float32).eps


def test_several_lidars_each_pack_as_it_went_in(tmp_path):
    rng = np.random.default_rng(7)
    went = {}
    for sensor in ("left", "right", "middle"):
        pose = format.Poses(np.zeros(1), np.zeros((1, 3)), np.array([[1.0, 0, 0, 0]]))
        format.write_poses(tmp_path, sensor, pose)
        raw = tmp_path / sensor / lidarscans.RAW
        writer = lidarscans.RawWriter(raw)
        scan = _scan(rng, 3_000, 0)
        del scan["objects"]
        writer.add(7_000_000_000, **scan)
        writer.close()
        lidarscans.write_index(raw, writer.stamps, writer.counts, [(0, 0, 0.0)], RATE, None)
        went[sensor] = scan
    told = []
    assert lidarscans.pack_all(tmp_path, lambda s, d: told.append((s, d)))[0] == 3
    assert {s for s, d in told if d == 1.0} == set(went)
    for sensor, scan in went.items():
        back = format.read_scan(tmp_path, sensor, 0)
        assert np.abs(back.points - scan["points"]).max() <= 0.6e-4
        assert np.array_equal(back.ring, scan["ring"])


def test_a_lidar_left_raw_is_found_to_pack_and_its_index_says_what_was_kept(tmp_path):
    raw = tmp_path / "lidar" / lidarscans.RAW
    writer = lidarscans.RawWriter(raw)
    writer.add(5, **_scan(np.random.default_rng(1), 3, 0))
    writer.close()
    assert lidarscans.raw_lidars(tmp_path) == []  # not indexed: not finished rendering
    lidarscans.write_index(raw, writer.stamps, writer.counts, [(0, 0, 0.0)], RATE)
    assert lidarscans.raw_lidars(tmp_path) == ["lidar"]
    index = json.loads((raw / lidarscans.INDEX).read_text(encoding="utf-8"))
    assert index["counts"] == [3] and index["kept"] == [[0, 0, 0.0]]


def test_a_renderers_raw_scans_are_indexed_whichever_renderer_wrote_them(tmp_path):
    """Scans stamped by a renderer's clock, 5 s ahead of the window: the whole scans within
    the window kept, numbered by when each began; one that began before it, and one that
    would end after it, left out."""
    rng = np.random.default_rng(5)
    writer = lidarscans.RawWriter(tmp_path / "lidar" / lidarscans.RAW)
    for k in range(-1, 4):  # began at -0.1, 0, 0.1, 0.2, 0.3 s of the window
        writer.add(round((5.0 + k / RATE) * 1e9), **_scan(rng, 50, k))
    result = lidarscans.finish(writer, RATE, lambda ns: ns / 1e9 - 5.0, span=0.3)
    assert (result.count, result.unit, result.detail) == (3, "scans", "0.00 M points")
    index = json.loads((writer.raw / lidarscans.INDEX).read_text(encoding="utf-8"))
    assert [(place, number) for place, number, _ in index["kept"]] == [(1, 0), (2, 1), (3, 2)]
    assert [began for _, _, began in index["kept"]] == pytest.approx([0.0, 0.1, 0.2])
    assert index["counts"] == [50] * 5 and index["rate"] == RATE


def test_scans_that_hit_nothing_fail_unless_the_lidar_may_be_blind(tmp_path):
    def blank():
        writer = lidarscans.RawWriter(tmp_path / "lidar" / lidarscans.RAW)
        writer.add(0, **_scan(np.random.default_rng(1), 0, 0))
        return writer

    with pytest.raises(RuntimeError, match="hit anything"):
        lidarscans.finish(blank(), RATE, lambda ns: ns / 1e9, span=0.1)
    result = lidarscans.finish(blank(), RATE, lambda ns: ns / 1e9, span=0.1, blind=True)
    assert result.detail == "no points: nothing in reach" and not result.timed
    with pytest.raises(RuntimeError, match="no whole scans"):
        lidarscans.finish(blank(), RATE, lambda ns: ns / 1e9 - 9.0, span=0.1)


def test_a_window_with_a_scan_lost_or_scans_begun_off_their_poses_is_refused(tmp_path):
    """The window's whole scans, each once, each begun on its pose: one lost at either end,
    or all begun an update late (their points would be placed by the wrong poses), refused."""

    def given(began):
        writer = lidarscans.RawWriter(tmp_path / "lidar" / lidarscans.RAW)
        rng = np.random.default_rng(2)
        for k, second in enumerate(began):
            writer.add(round((5.0 + second) * 1e9), **_scan(rng, 20, k))
        return writer

    def finish(writer):
        return lidarscans.finish(writer, RATE, lambda ns: ns / 1e9 - 5.0, span=0.3)

    assert finish(given([0.0, 0.1, 0.2])).count == 3
    with pytest.raises(RuntimeError, match="2 of the window's 3 whole scans: none for scan 0"):
        finish(given([0.1, 0.2]))
    with pytest.raises(RuntimeError, match="2 of the window's 3 whole scans: none for scan 2"):
        finish(given([0.0, 0.1]))
    with pytest.raises(RuntimeError, match=r"16\.7 ms off their poses"):
        finish(given([1 / 60, 0.1 + 1 / 60]))
