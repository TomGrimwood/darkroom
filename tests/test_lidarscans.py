"""A lidar's scans written into its one LAZ as its renderer hands them over (`ScanWriter`):
each read back as it went in, its index filled in place, the window's whole scans kept and
numbered, what each point hit named as it first comes back."""

import numpy as np
import pytest

from darkroom import format, lidarscans

RATE = 10.0
#: A renderer's clock, 5 s ahead of the window.
AHEAD = 5.0


def _seconds(ns):
    return ns / 1e9 - AHEAD


def _stamp(second):
    return round((AHEAD + second) * 1e9)


def _scan(rng, n, ids):
    """A scan of `n` points, each hitting one of object `ids` (runs of them, as a lidar's do)."""
    which = np.repeat(rng.integers(0, len(ids), n // 7 + 1), 7)[:n]
    objects = np.zeros((n, 16), np.uint8)
    objects[:, 0] = np.asarray(ids, np.uint8)[which]
    objects[:, 9] = 3  # the high half too
    return {
        "points": rng.uniform(-300, 300, (n, 3)).astype(np.float32),
        "intensity": rng.uniform(0, 1.3, n).astype(np.float32),
        "offsets": np.sort(rng.integers(0, 99_000_000, n)).astype(np.int64),
        "ret": rng.integers(0, 2, n).astype(np.uint8),
        "ring": rng.integers(0, 128, n).astype(np.uint32),
        "material": rng.integers(0, 40, n).astype(np.uint32),
        "objects": objects,
    }


def _named(ids):
    """Object `ids` as a renderer's table names them: 1 a wall, 2 a car, the rest unnamed."""
    called = {1: "/World/Wall", 2: "/World/Car"}
    return [called.get(int(row[0])) for row in np.asarray(ids).reshape(-1, 16)]


def _poses(folder, sensor, count):
    seconds = np.arange(count) / RATE
    rotation = np.tile([1.0, 0, 0, 0], (count, 1))
    format.write_poses(folder, sensor, format.Poses(seconds, np.zeros((count, 3)), rotation))


def _given(rng, began, n=400):
    """Scans begun at `began` (seconds of the window), stamped by the renderer's clock."""
    return [(_stamp(s), _scan(rng, 0 if k == 2 else n + 50 * k, [1, 2, 3])) for k, s in enumerate(began)]  # fmt: skip


def _streamed(folder, given, span, names=_named, blind=False):
    writer = lidarscans.ScanWriter(folder, "lidar", RATE, span, names, blind)
    writer.start(_seconds)
    for stamp, scan in given:
        writer.add(stamp, **scan)
    return writer.finish()


def test_scans_written_as_they_come_read_back_as_they_went_in(tmp_path):
    """One before the window and one ending after it left out; scan 2 saw nothing. Points out
    to a long-range lidar's reach, where float32 rounds coarser than the LAZ's grid."""
    rng = np.random.default_rng(3)
    given = _given(rng, [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    _poses(tmp_path, "lidar", 5)
    result = _streamed(tmp_path, given, span=0.5)
    assert (result.count, result.unit) == (5, "scans")
    assert format.scan_numbers(tmp_path, "lidar") == [0, 1, 2, 3, 4]
    paths = format.read_hits(tmp_path, "lidar")
    assert sorted(paths) == ["/World/Car", "/World/Wall"]
    for number, (_, went) in enumerate(given[1:6]):
        back = format.read_scan(tmp_path, "lidar", number)
        near = format.LAZ_SCALE / 2 + np.abs(went["points"]) * np.finfo(np.float32).eps
        assert np.all(np.abs(back.points - went["points"]) <= near), number
        assert np.array_equal(back.intensity, went["intensity"]), number
        assert np.allclose(back.offset, went["offsets"] / 1e9, atol=1e-6), number
        for field in ("ring", "ret", "material"):
            assert np.array_equal(getattr(back, field), went[field]), (number, field)
        # The prim each point hit, as the renderer's table names its object.
        assert list(_paths(paths, back.hit)) == _named(went["objects"]), number
    import laspy

    las = laspy.read(str(tmp_path / "lidar" / format.PACKED))  # each point's time from its pose
    third = np.asarray(las.scan) == 2
    assert third.sum() == len(given[3][1]["ring"]) > 0
    assert np.allclose(las.gps_time[third], 0.2 + given[3][1]["offsets"] / 1e9, atol=1e-6)
    assert not (tmp_path / "lidar" / (format.PACKED + ".part")).exists()


def test_several_lidars_are_each_written_as_they_went_in(tmp_path):
    rng = np.random.default_rng(7)
    went, writers = {}, {}
    for sensor in ("left", "right", "middle"):
        _poses(tmp_path, sensor, 2)
        went[sensor] = _given(rng, [0.0, 0.1], n=3_000)
        writers[sensor] = lidarscans.ScanWriter(tmp_path, sensor, RATE, 0.2)
        writers[sensor].start(_seconds)
    for k in range(2):  # a step of the renderer: each lidar's scan that ended in it
        for sensor, writer in writers.items():
            stamp, scan = went[sensor][k]
            writer.add(stamp, **scan)
    for sensor, writer in writers.items():
        assert writer.finish().count == 2
        for k in range(2):
            back = format.read_scan(tmp_path, sensor, k)
            assert np.array_equal(back.ring, went[sensor][k][1]["ring"]), (sensor, k)
        assert format.read_hits(tmp_path, sensor) == []  # no table to name them by


def _paths(paths, hit):
    named = np.array([*paths, None], object)
    return named[np.minimum(hit, len(paths))]


def test_an_index_of_many_records_is_filled_in_place(tmp_path, monkeypatch):
    monkeypatch.setattr(format, "_INDEX_PAIRS", 2)  # 5 scans: records of 2, 2 and 1
    rng = np.random.default_rng(4)
    _poses(tmp_path, "lidar", 5)
    _streamed(tmp_path, _given(rng, [0.0, 0.1, 0.2, 0.3, 0.4], n=30), span=0.5, names=None)
    assert format.scan_numbers(tmp_path, "lidar") == [0, 1, 2, 3, 4]
    assert [len(s.ring) for _, s in format.iter_scans(tmp_path, "lidar")] == [30, 80, 0, 180, 230]
    assert format.read_hits(tmp_path, "lidar") == []
    assert np.all(format.read_scan(tmp_path, "lidar", 1).hit == format.NO_HIT)


def test_a_window_with_a_scan_lost_or_begun_off_its_pose_is_refused_and_nothing_left(tmp_path):
    rng = np.random.default_rng(2)
    _poses(tmp_path, "lidar", 3)
    laz = tmp_path / "lidar" / format.PACKED
    assert _streamed(tmp_path, _given(rng, [0.0, 0.1, 0.2]), span=0.3).count == 3
    laz.unlink()
    with pytest.raises(RuntimeError, match="2 of the window's 3 whole scans: none for scan 0"):
        _streamed(tmp_path, _given(rng, [0.1, 0.2]), span=0.3)
    with pytest.raises(RuntimeError, match="2 of the window's 3 whole scans: none for scan 2"):
        _streamed(tmp_path, _given(rng, [0.0, 0.1]), span=0.3)
    with pytest.raises(RuntimeError, match=r"16\.7 ms off their poses"):
        _streamed(tmp_path, _given(rng, [1 / 60, 0.1 + 1 / 60, 0.2 + 1 / 60]), span=0.3)
    with pytest.raises(RuntimeError, match="no whole scans"):
        _streamed(tmp_path, _given(rng, [-9.0]), span=0.3)
    with pytest.raises(RuntimeError, match="writing its scans failed"):  # out of order
        _streamed(tmp_path, _given(rng, [0.1, 0.0, 0.2]), span=0.3)
    assert not laz.exists() and not laz.with_name(laz.name + ".part").exists()


def test_scans_that_hit_nothing_fail_unless_the_lidar_may_be_blind(tmp_path):
    rng = np.random.default_rng(1)
    _poses(tmp_path, "lidar", 1)
    blank = [(_stamp(0.0), _scan(rng, 0, [1]))]
    with pytest.raises(RuntimeError, match="hit anything"):
        _streamed(tmp_path, blank, span=0.1)
    result = _streamed(tmp_path, blank, span=0.1, blind=True)
    assert result.detail == "no points: nothing in reach" and not result.timed
    assert format.scan_numbers(tmp_path, "lidar") == [0]
