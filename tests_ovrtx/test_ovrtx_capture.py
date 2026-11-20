"""A capture rendered by ovrtx for real (the GPU, about half a minute): a bar crossing under
two cameras at 10 m/s, one with a global shutter, one read out over 30 ms. Each frame must show
the bar where the stage has it at the frame's time, and the rolling shutter's rows where it
was as each was read: its last row at the frame's time, its first a readout before.

    uv run --extra ovrtx pytest tests_ovrtx

Not among `pytest`'s own tests (`tests/`): those need no GPU.
"""

import numpy as np
import pytest
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

pytest.importorskip("ovrtx")

import darkroom
from darkroom import Options, format, sensorkit
from darkroom.renderers import OVRTXRendererCfg

RATE, READOUT, SPEED = 30.0, 0.03, 10.0  # frames a second; seconds; metres a second
HEIGHT = 10.0  # the cameras above the bar, metres
WINDOW = (1.0, 2.0)


def bar_at(seconds):
    """Where the bar is along x, metres, `seconds` into the stage (a number, or an array of
    them): under the cameras at 1.5 s."""
    return SPEED * (seconds - 1.5)


def _stage(folder):
    layer = Sdf.Layer.CreateNew(str(folder / "bar.usda"))
    layer.timeCodesPerSecond = 60
    layer.startTimeCode, layer.endTimeCode = 0, 180
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    # Lit by the sky alone: a sun overhead glints off the bar into the cameras.
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1000)
    bar = UsdGeom.Cube.Define(stage, "/World/Bar")
    bar.CreateSizeAttr(1.0)
    bar.CreateDisplayColorAttr([Gf.Vec3f(1, 0, 0)])
    move = bar.AddTranslateOp()
    move.Set(Gf.Vec3d(bar_at(0.0), 0, 0), 0)
    move.Set(Gf.Vec3d(bar_at(3.0), 0, 0), 180)
    bar.AddScaleOp().Set(Gf.Vec3f(0.2, 20.0, 0.02))  # thin, flat, across every row
    rig = UsdGeom.Xform.Define(stage, "/World/Rig")
    rig.AddTranslateOp().Set(Gf.Vec3d(0, 0, HEIGHT))
    for name, readout in (("still", None), ("rolling", READOUT)):
        camera = UsdGeom.Camera.Define(stage, f"/World/Rig/{name}")  # looks down, +x right
        camera.CreateFocalLengthAttr(24.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
        sensorkit.mark(layer, f"/World/Rig/{name}", "camera", RATE)
        if readout:
            camera.GetPrim().CreateAttribute(
                "sensor:camera:readoutSeconds", Sdf.ValueTypeNames.Double
            ).Set(readout)
    layer.Save()
    return folder / "bar.usda"


def _bar(file) -> np.ndarray:
    """The column the bar's middle is in, in each row of picture `file`; NaN where no bar."""
    image = np.asarray(Image.open(file).convert("RGB"), np.int16)
    red = (image[..., 0] - image[..., 1]) > 60
    columns = np.arange(red.shape[1])[None, :]
    count = red.sum(axis=1)
    middle = (red * columns).sum(axis=1) / np.maximum(count, 1) + 0.5
    return np.where(count > 3, middle, np.nan)


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ovrtx")
    out = darkroom.capture(
        _stage(folder),
        folder / "capture",
        Options(window=WINDOW, kinds=["camera"], renderer_cfg=OVRTXRendererCfg()),
        on_line=None,
    )
    return out


def test_every_frame_of_both_cameras_is_made_and_the_capture_finished(capture):
    meta = format.read_meta(capture)
    assert set(meta.sensors) == {"still", "rolling"} and format.finished(capture)
    assert meta.extra[format.RENDERER] == "ovrtx"
    for name in meta.sensors:
        assert sorted(format.frames_kept(capture, name)) == list(range(31))
        assert (capture / f"{name}_mosaic.jpg").is_file()


def test_a_frame_shows_the_stage_at_its_own_time(capture):
    width = 960
    per_metre = (24.0 * width / 36.0) / HEIGHT  # pixels a metre, at the bar
    for number in range(31):
        seconds = WINDOW[0] + number / RATE
        expected = width / 2 + bar_at(seconds) * per_metre
        if not 20 < expected < width - 20:
            continue
        seen = _bar(format.frame_path(capture, "still", number))
        assert np.nanmax(seen) - np.nanmin(seen) < 1.5, number  # upright: one moment
        # Half a frame's travel is 10 pixels: a frame late or early would be 21 off.
        assert abs(np.nanmedian(seen) - expected) < 1.5, (number, np.nanmedian(seen), expected)


def test_a_rolling_shutters_rows_show_the_stage_as_each_was_read(capture):
    width, height = 960, 540
    per_metre = (24.0 * width / 36.0) / HEIGHT
    checked = 0
    for number in range(31):
        seconds = WINDOW[0] + number / RATE
        # Row 0 is read a readout before the frame's time, the last row at it.
        read = seconds - READOUT * (1 - np.arange(height) / (height - 1))
        expected = width / 2 + bar_at(read) * per_metre
        if expected.min() < 20 or expected.max() > width - 20:
            continue
        seen = _bar(format.frame_path(capture, "rolling", number))
        assert np.isfinite(seen).mean() > 0.95, number
        # The bar leans by the readout's travel (19 pixels), each row where it was then.
        assert np.nanmax(np.abs(seen - expected)) < 2.0, (
            number,
            np.nanmax(np.abs(seen - expected)),
        )
        lean = np.nanmean(seen[-20:]) - np.nanmean(seen[:20])
        assert lean == pytest.approx(READOUT * SPEED * per_metre, abs=2.0)
        checked += 1
    assert checked > 20


def _blurred(folder):
    """The bar's stage with a third camera beside the still one, blurring motion (RTX's
    post-process: exposureFraction 1.19, a 1/60 s shutter by the rtx-camera-shutters skill)."""
    stage_file = _stage(folder)
    stage = Usd.Stage.Open(str(stage_file))
    camera = UsdGeom.Camera.Define(stage, "/World/Rig/blur")
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100.0))
    for name, kind, value in (
        ("enabled", Sdf.ValueTypeNames.Bool, True),
        ("exposureFraction", Sdf.ValueTypeNames.Float, RATE / 60 / 0.42),
        ("maxBlurDiameterFraction", Sdf.ValueTypeNames.Float, 0.1),
        ("numSamples", Sdf.ValueTypeNames.Int, 32),
    ):
        camera.GetPrim().CreateAttribute(f"omni:rtx:post:motionblur:{name}", kind).Set(value)
    sensorkit.mark(stage.GetRootLayer(), "/World/Rig/blur", "camera", RATE)
    stage.GetRootLayer().Save()
    return stage_file


def _wide(file) -> float:
    """How wide the bar is across the picture's middle rows, pixels: red at all."""
    image = np.asarray(Image.open(file).convert("RGB"), np.int16)
    return float(np.median(((image[200:340, :, 0] - image[200:340, :, 1]) > 25).sum(axis=1)))


def test_motion_blur_streaks_alike_however_ovrtx_steps_and_more_steps_than_it_blurs_are_refused(
    tmp_path,
):
    """RTX's blur is `exposureFraction` of the camera's frame interval, at 2 and 4 steps a
    frame alike; past `BLUR_STEPS` it doubles, so such a capture is refused."""
    from darkroom.ovrtx import BLUR_STEPS

    stage_file = _blurred(tmp_path)
    widened = {}
    for steps in (None, 120.0):
        out = darkroom.capture(
            stage_file,
            tmp_path / f"steps_{steps}",
            Options(
                window=WINDOW, kinds=["camera"], renderer_cfg=OVRTXRendererCfg(step_rate=steps)
            ),
            on_line=None,
        )
        frame = 15  # the bar under the cameras
        widened[steps] = _wide(format.frame_path(out, "blur", frame)) - _wide(
            format.frame_path(out, "still", frame)
        )
    assert widened[None] > 5 and abs(widened[120.0] - widened[None]) <= 2, widened
    with pytest.raises(RuntimeError, match=f"at most {BLUR_STEPS} steps between its frames"):
        darkroom.capture(
            stage_file,
            tmp_path / "steps_480",
            Options(
                window=WINDOW, kinds=["camera"], renderer_cfg=OVRTXRendererCfg(step_rate=480.0)
            ),
            on_line=None,
        )


def test_distances_are_from_the_camera_whatever_its_lens_and_near_clip(tmp_path):
    """A wall 5 m ahead of a plain camera, an OmniLens pinhole and fisheye (`walls`)."""
    import walls

    options = Options(window=(0.1, 0.3), kinds=["camera"], renderer_cfg=OVRTXRendererCfg())
    walls.check(darkroom.capture(walls.wall(tmp_path), tmp_path / "capture", options, on_line=None))
