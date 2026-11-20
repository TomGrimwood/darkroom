import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom

from darkroom.sensorkit import gnss, imu, kinds, motion, site
from darkroom.sensorkit.camera import lens


def _yaw(angles):
    half = np.asarray(angles) / 2
    return np.stack([np.cos(half), 0 * half, 0 * half, np.sin(half)], 1)


def test_still_imu_feels_gravity_up():
    t = np.linspace(0, 2, 241)
    moving = motion.motion(t, np.zeros((len(t), 3)), np.tile([1.0, 0, 0, 0], (len(t), 1)))
    got = imu.samples(moving, 200)
    assert np.allclose(got.accel, (0, 0, 9.80665)) and np.allclose(got.gyro, 0, atol=1e-9)
    assert len(got.seconds) == 401


def test_turntable_imu_reads_its_rate():
    t = np.linspace(0, 2, 481)
    rate = 0.5  # rad/s about +Z
    moving = motion.motion(t, np.zeros((len(t), 3)), _yaw(rate * t))
    got = imu.samples(moving, 100)
    assert np.allclose(got.gyro[5:-5], (0, 0, rate), atol=1e-4)


def test_accelerating_imu_reads_the_push_in_its_own_axes():
    t = np.linspace(0, 2, 481)
    # Accelerating 1 m/s² along the world's +X, the IMU turned 90° about Z: its -Y.
    moving = motion.motion(
        t, np.stack([0.5 * t**2, 0 * t, 0 * t], 1), _yaw(np.full_like(t, np.pi / 2))
    )
    got = imu.samples(moving, 100)
    assert np.allclose(got.accel[5:-5], (0, -1, 9.80665), atol=1e-6)


def test_quaternions_round_trip_through_usd_matrices():
    q = motion.continuous(np.random.default_rng(1).normal(size=(50, 4)))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.sign(q[:, :1])
    usd = np.zeros((50, 4, 4))
    usd[:, :3, :3] = np.swapaxes(motion.rotation_matrices(q), 1, 2)
    usd[:, 3, :3] = np.arange(150).reshape(50, 3)
    usd[:, 3, 3] = 1
    positions, back = motion.from_usd_matrices(usd)
    back *= np.sign(back[:, :1])
    assert np.allclose(back, q) and np.allclose(positions, usd[:, 3, :3])


def test_gnss_at_the_origin_is_the_site():
    where = site.Site(-36.85, 174.76, 40.0, np.eye(4), geoid=34.0)
    t = np.linspace(0, 1, 11)
    still = motion.motion(t, np.zeros((11, 3)), np.tile([1.0, 0, 0, 0], (11, 1)))
    got = gnss.fixes(still, where, 10)
    assert np.allclose(got.latitude, -36.85) and np.allclose(got.longitude, 174.76)
    assert got.altitude is not None and np.allclose(got.altitude, 6.0)


def test_sensors_are_marked_prims_that_are_active():
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    UsdGeom.Camera.Define(stage, "/Kit/Front")
    UsdGeom.Camera.Define(stage, "/Kit/Hidden")
    UsdGeom.Xform.Define(stage, "/Kit/Imu")
    UsdGeom.Camera.Define(stage, "/Scene/Overview")  # a scene's camera: not a sensor
    kinds.mark(layer, "/Kit/Front", "camera", 15)
    kinds.mark(layer, "/Kit/Hidden", "camera")
    kinds.mark(layer, "/Kit/Imu", "imu")
    stage.GetPrimAtPath("/Kit/Hidden").SetActive(False)
    found = kinds.sensors(stage)
    assert [(s.name, s.kind, s.rate) for s in found] == [
        ("front", "camera", 15.0),
        ("imu", "imu", 200.0),
    ]
    assert [s.name for s in kinds.sensors(stage, ["imu"])] == ["imu"]


def test_a_plain_camera_is_the_pinhole_its_apertures_say():
    stage = Usd.Stage.CreateInMemory()
    camera = UsdGeom.Camera.Define(stage, "/C")
    camera.GetFocalLengthAttr().Set(24.0)
    camera.GetHorizontalApertureAttr().Set(36.0)
    camera.GetVerticalApertureAttr().Set(20.25)
    found = lens(camera.GetPrim())
    assert (found.width, found.height) == (960, 540)
    assert found.fx == pytest.approx(640.0) and found.cx == 480


def test_sensors_sharing_a_name_take_their_kit_s():
    stage = Usd.Stage.CreateInMemory()
    for kit in ("Pinhole", "OAK4D_Front"):
        UsdGeom.Camera.Define(stage, f"/Thing/Frames/Mount/{kit}/Camera")
        kinds.mark(stage.GetRootLayer(), f"/Thing/Frames/Mount/{kit}/Camera", "camera")
    assert [s.name for s in kinds.sensors(stage)] == ["pinhole_camera", "oak4d_front_camera"]
