import numpy as np

from darkroom import format as fmt


def test_a_capture_folder_round_trips(tmp_path):
    meta = fmt.Meta(
        stage="/x/run.usda",
        start=10.0,
        end=12.0,
        sensors={
            "front": fmt.Sensor(
                "camera", "/World/Trajectories/w/Frames/Mount/Rig/Front", 10.0, {"width": 640}
            ),
            "lidar": fmt.Sensor("lidar", "/World/Trajectories/w/Frames/Mount/Rig/Lidar", 10.0),
        },
        site=fmt.Site(-36.85, 174.76, 40.0),
    )
    fmt.write_meta(tmp_path, meta)
    assert fmt.read_meta(tmp_path) == meta

    poses = fmt.Poses(np.linspace(0, 2, 21), np.zeros((21, 3)), np.tile([1.0, 0, 0, 0], (21, 1)))
    fmt.write_poses(tmp_path, "front", poses)
    again = fmt.read_poses(tmp_path, "front")
    assert np.array_equal(again.seconds, poses.seconds) and again.rotation.shape == (21, 4)

    scan = fmt.Scan(
        np.ones((5, 3)), np.ones(5), np.zeros(5), np.arange(5), np.zeros(5, dtype=np.uint8)
    )
    fmt.write_scans(tmp_path, "lidar", [(3, scan)])
    assert np.array_equal(fmt.read_scan(tmp_path, "lidar", 3).ring, np.arange(5))

    assert not fmt.finished(tmp_path)
    fmt.done(tmp_path)
    assert fmt.finished(tmp_path)


def test_a_captures_lanes_of_what_was_going_on_round_trip_and_may_be_missing(tmp_path):
    from darkroom import format

    assert format.read_states(tmp_path) == {}
    format.write_states(tmp_path, {"location": [(0, 12.5, "field")], "action": []})
    assert format.read_states(tmp_path) == {"location": [(0.0, 12.5, "field")]}


def test_scans_in_their_laz_keep_every_field_and_each_points_time(tmp_path):
    rng = np.random.default_rng(1)
    n = 2000
    scan = fmt.Scan(
        rng.uniform(-80, 80, (n, 3)).astype(np.float32),
        rng.uniform(0.005, 1.4, n).astype(np.float32),  # intensities over 1 too
        np.sort(rng.uniform(0, 0.1, n)).astype(np.float32),  # fired across a 0.1 s sweep
        rng.integers(0, 108_000, n).astype(np.uint32),  # an MLX-S's rings run past 65535
        rng.integers(0, 2, n).astype(np.uint8),  # first and second returns
        np.where(rng.random(n) < 0.1, fmt.NO_HIT, rng.integers(0, 40, n)).astype(np.uint32),
        rng.integers(0, 48, n).astype(np.uint32),  # RTX's lidar material IDs
    )
    fmt.write_poses(
        tmp_path,
        "mlxs",
        fmt.Poses(np.array([3.0, 3.1, 3.2]), np.zeros((3, 3)), np.tile([1.0, 0, 0, 0], (3, 1))),
    )
    empty = fmt.Scan(
        np.zeros((0, 3), np.float32),
        *(np.zeros(0, t) for t in (np.float32, np.float32, np.uint32, np.uint8)),
    )
    fmt.write_scans(tmp_path, "mlxs", [(1, empty), (2, scan)])
    assert fmt.scan_numbers(tmp_path, "mlxs") == [1, 2]
    assert fmt.has_scan(tmp_path, "mlxs", 2) and not fmt.has_scan(tmp_path, "mlxs", 0)
    back = fmt.read_scan(tmp_path, "mlxs", 2)
    assert (
        np.abs(back.points - scan.points).max() <= 0.6e-4
    )  # on a 0.1 mm grid (and float32 at 80 m)
    for field in ("intensity", "offset", "ring", "ret", "hit", "material"):
        assert np.array_equal(getattr(back, field), getattr(scan, field)), field
        assert getattr(back, field).dtype == getattr(scan, field).dtype, field
    # Each point's own time, in the capture's seconds, and its scan, for any LAZ reader.
    import laspy

    las = laspy.read(str(tmp_path / "mlxs" / fmt.PACKED))
    mine = np.asarray(las.scan) == 2
    assert mine.sum() == n and set(np.unique(las.scan)) == {2}  # scan 1 has no points
    assert np.allclose(las.gps_time[mine], 3.2 + scan.offset.astype(np.float64), atol=1e-9)
    empty = fmt.read_scan(tmp_path, "mlxs", 1)
    assert len(empty.points) == 0 and len(empty.hit) == 0


def test_many_scans_in_one_file_each_read_alone_and_read_through_a_batch_at_a_time(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(2)
    count = 9000  # more than one index record holds
    fmt.write_poses(
        tmp_path,
        "lidar",
        fmt.Poses(
            np.arange(count) * 0.1, np.zeros((count, 3)), np.tile([1.0, 0, 0, 0], (count, 1))
        ),
    )
    sizes = rng.integers(0, 40, count)
    scans = [
        (
            k,
            fmt.Scan(
                rng.uniform(-5, 5, (size, 3)).astype(np.float32),
                np.full(size, 0.5, np.float32),
                np.zeros(size, np.float32),
                np.full(size, k, np.uint32),
                np.zeros(size, np.uint8),
            ),
        )
        for k, size in enumerate(sizes)
    ]
    fmt.write_scans(tmp_path, "lidar", scans)
    for k in (0, 1, 4321, 8000, count - 1):
        scan = fmt.read_scan(tmp_path, "lidar", k)
        assert len(scan.points) == sizes[k] and np.all(scan.ring == k), k
    monkeypatch.setattr(fmt, "READ_AHEAD", 500)  # many reads, a few dozen scans each
    every = list(fmt.iter_scans(tmp_path, "lidar"))  # read through once, in order
    assert [k for k, _ in every] == list(range(count))
    assert all(len(s.points) == sizes[k] and np.all(s.ring == k) for k, s in every)


def test_an_imus_readings_with_noise_sit_beside_its_truth_by_the_noise_models_name(tmp_path):
    t = np.arange(4) / 100
    fmt.write_arrays(
        tmp_path, "imu", fmt.IMU, seconds=t, accel=np.zeros((4, 3)), gyro=np.zeros((4, 3))
    )
    assert fmt.imu_errors(tmp_path, "imu") == []
    for noise in ("hpc1", "cheap"):
        fmt.write_arrays(
            tmp_path,
            "imu",
            fmt.imu_with(noise),
            seconds=t,
            accel=np.ones((4, 3)),
            gyro=np.ones((4, 3)),
        )
    assert fmt.imu_errors(tmp_path, "imu") == ["cheap", "hpc1"]
    assert fmt.read_arrays(tmp_path, "imu", fmt.imu_with("hpc1"))["accel"].sum() == 12
