"""A camera's frames as its renderer publishes them, whichever renderer: named by the pose each
shows, those rendered before the camera is to keep any (or off its poses) let go; a camera on
a renderer read into its store; and where a camera is, said as a renderer is told it."""

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom

from darkroom import format
from darkroom.renderers import (
    CameraCfg,
    CameraRenderSpec,
    RenderBufferKind,
    RenderBufferSpec,
    outfit,
)
from darkroom.renderers.base_renderer import BaseRenderer
from darkroom.renderers.frame_store import Camera, FrameStore

RATE = 10.0


def _seconds(stamp_ns):
    """The renderer's clock is 5 s ahead of the window's start."""
    return float(stamp_ns) / 1e9 - 5.0


def _picture(value):
    return np.full((6, 8, 4), value, np.uint8)


def test_frames_handed_over_as_pixels_are_written_and_named_by_their_poses(tmp_path):
    store = FrameStore(tmp_path, "front", RATE, (8, 6))
    # Before the window, then poses 0 and 1, one half way to pose 3, pose 1 again, pose 2.
    for value, clock in enumerate((4.9, 5.0, 5.1, 5.25, 5.1, 5.2)):
        store.write(clock, _picture(40 * value))
    result = store.finish(_seconds, span=0.3)
    assert store.have == {0, 1, 2}
    assert (result.count, result.unit, result.rate, result.detail) == (3, "frames", RATE, "8x6")
    assert result.notes == ("1 rendered off their poses, not kept",)
    kept = sorted(p.name for p in (tmp_path / "front" / format.FRAMES).iterdir())
    assert kept == ["000000.jpg", "000001.jpg", "000002.jpg"]  # the rest let go
    assert format.frames_kept(tmp_path, "front") == {0, 1, 2}


def test_frames_rendered_before_a_camera_is_to_keep_any_are_let_go(tmp_path):
    store = FrameStore(tmp_path, "front", RATE, (8, 6))
    for clock in (5.0, 5.1, 5.2, 5.3):
        store.write(clock, _picture(100))
    store.wait()
    # It got on part way: black until 0.2 s into the window, so it keeps from there.
    assert store.keep(_seconds, span=0.3, keep_from=0.2) == {2, 3}
    with pytest.raises(RuntimeError, match="no frames within"):
        FrameStore(tmp_path, "back", RATE, (8, 6)).finish(_seconds, span=0.3)


def test_a_capture_carried_on_keeps_the_frames_it_had(tmp_path):
    first = FrameStore(tmp_path, "front", RATE, (8, 6))
    first.write(5.0, _picture(10))
    first.finish(_seconds, span=0.1)
    again = FrameStore(tmp_path, "front", RATE, (8, 6))
    assert again.had == {0}
    again.write(5.0, _picture(200))  # pose 0 again: it has it
    again.write(5.1, _picture(200))
    assert again.finish(_seconds, span=0.1).count == 2 and again.kept == {1}


def test_a_frame_its_renderer_wrote_itself_is_named_where_it_lies(tmp_path):
    store = FrameStore(tmp_path, "front", RATE, (8, 6))
    (store.dir / "render_000000.jpg").write_bytes(b"jpeg")
    store.took(5.1, "render_000000.jpg")
    assert store.finish(_seconds, span=0.2).count == 1
    assert format.frame_path(tmp_path, "front", 1).read_bytes() == b"jpeg"


class _Renderer(BaseRenderer):
    """A renderer stood in for: a frame a render, the simulation's time in its first pixel."""

    def __init__(self):
        self.clock, self.cleaned = 0.0, []

    def supported_output_types(self):
        return {
            RenderBufferKind.RGBA: RenderBufferSpec(4, np.uint8),
            RenderBufferKind.RGB: RenderBufferSpec(3, np.uint8),
        }

    def prepare_stage(self, stage, num_envs): ...

    def create_render_data(self, spec):
        return {"spec": spec, "captures": [], "poses": []}

    def set_outputs(self, render_data, output_data):
        render_data["rgba"] = output_data["rgba"]

    def update_transforms(self): ...

    def update_geometries(self): ...

    def update_camera(self, render_data, positions, orientations, intrinsics):
        render_data["poses"].append(positions)

    def render(self, render_data):
        self.clock += 0.1
        render_data["rgba"][:] = round(self.clock * 10)
        render_data["captures"].append({"time": self.clock})

    def read_output(self, render_data, camera_data):
        camera_data.info["rgba"] = (
            {"capture": render_data["captures"]} if render_data["captures"] else None
        )
        render_data["captures"] = []

    def cleanup(self, render_data):
        self.cleaned.append(render_data)


def test_a_camera_on_a_renderer_is_told_its_pose_and_read_into_its_store(tmp_path):
    renderer = _Renderer()
    cfg = CameraCfg("/Rig/front", 8, 6, update_period=1 / RATE)
    spec = CameraRenderSpec(cfg, "cuda:0", 1, ("/Rig/front",), 1)
    store = FrameStore(tmp_path, "front", RATE, (8, 6))
    where = (
        np.ones((1, 3), np.float32),
        np.zeros((1, 4), np.float32),
        np.eye(3, dtype=np.float32)[None],
    )
    camera = Camera(renderer, spec, store, lambda time: where)
    for _ in range(3):
        camera.update(0.0)
        renderer.render(camera.data)
        camera.read()
        camera.read()  # nothing new: nothing written twice
    data = camera.data
    camera.stop()
    assert renderer.cleaned == [data] and camera.data is None and len(data["poses"]) == 3
    result = camera.finish(lambda stamp_ns: float(stamp_ns) / 1e9 - 0.1, span=0.2)
    assert result.count == 3 and camera.have == {0, 1, 2}
    # Each frame its own picture: copied as it was published, not the array's last.
    from PIL import Image

    means = [
        np.asarray(Image.open(format.frame_path(tmp_path, "front", n))).mean() for n in range(3)
    ]
    assert means == pytest.approx([1, 2, 3], abs=0.6)


def test_where_a_camera_is_is_said_in_isaac_labs_world_convention():
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(60)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    rig = UsdGeom.Xform.Define(stage, "/Rig")
    move = rig.AddTranslateOp()
    move.Set(Gf.Vec3d(0, 0, 2), 0)
    move.Set(Gf.Vec3d(6, 0, 2), 60)
    UsdGeom.Camera.Define(stage, "/Rig/front")  # as USD has a camera: looking down -z, +y up
    lens = type("Lens", (), {"fx": 640.0, "fy": 650.0, "cx": 480.0, "cy": 270.0})()
    pose = outfit.poser(stage, "/Rig/front", lens)
    positions, orientations, intrinsics = pose(0.5)
    assert positions.shape == (1, 3) and orientations.shape == (1, 4)
    assert positions[0] == pytest.approx((3, 0, 2))  # half way along, at half a second
    assert intrinsics[0].tolist() == [[640, 0, 480], [0, 650, 270], [0, 0, 1]]
    # A quaternion (x, y, z, w) turning Isaac Lab's camera axes (ahead +x, up +z) into the
    # world's: this camera looks down the world's -z with the world's +y up.
    x, y, z, w = (float(v) for v in orientations[0])
    turn = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    assert tuple(turn.TransformDir(Gf.Vec3d(1, 0, 0))) == pytest.approx((0, 0, -1), abs=1e-6)
    assert tuple(turn.TransformDir(Gf.Vec3d(0, 0, 1))) == pytest.approx((0, 1, 0), abs=1e-6)


def test_a_frame_handed_over_with_its_segments_normals_and_distances_has_each_beside_it(tmp_path):
    from PIL import Image

    store = FrameStore(tmp_path, "front", 10.0, (6, 4), taxonomy="class", geometry=True)
    rgb = np.full((4, 6, 4), 200, np.uint8)
    segments = np.zeros((4, 6, 1), np.int32)
    segments[1:, :] = 3  # ovrtx's class 3 is a tree; 0 and 1 are nothing labelled
    named = {"0": {"class": "BACKGROUND"}, "1": {"class": "UNLABELLED"}, "3": {"class": "tree"}}
    normals = np.zeros((4, 6, 3), np.float32)
    normals[..., 2] = 1.0
    store.write(0.0, rgb, segments, named, normals, np.full((4, 6, 1), 2.5, np.float32))
    store.write(0.1, rgb)  # a frame with nothing beside it is a picture all the same
    store.wait()
    assert store.labels == ["", "tree"]
    frames = tmp_path / "front" / format.FRAMES
    kept = np.asarray(Image.open(frames / format.SEGMENTS / "render_000000.png"))
    assert kept[0, 0] == 0 and kept[3, 5] == 1
    assert (frames / format.NORMALS / "render_000000.png").is_file()
    assert (frames / format.DISTANCES / "render_000000.png").is_file()
    assert not (frames / format.SEGMENTS / "render_000001.png").exists()
    assert (frames / "render_000001.jpg").is_file()
