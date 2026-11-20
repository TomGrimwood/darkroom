"""A camera's frame as the writer threads write it (`darkroom.frames`): the JPEG PIL would make,
whole or not there; its segments through the labels looked up on the render thread."""

import io

import numpy as np
from PIL import Image

from darkroom import format
from darkroom.frames import QUALITY, labels_of, write_frame


def _picture():
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (48, 64, 3), np.uint8)


def test_a_frames_jpeg_is_pils_own_and_rgba_is_written_as_its_rgb(tmp_path):
    rgb = _picture()
    expected = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(expected, format="jpeg", quality=QUALITY)
    write_frame(str(tmp_path), "a.jpg", rgb)
    rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
    write_frame(str(tmp_path), "b.jpg", rgba)
    assert (tmp_path / "a.jpg").read_bytes() == expected.getvalue()
    assert (tmp_path / "b.jpg").read_bytes() == expected.getvalue()
    assert not list(tmp_path.glob("*.part"))  # renamed once whole


def test_a_frames_segments_are_indexed_through_the_labels_and_written_before_it(tmp_path):
    labels = [""]
    named = {"0": {"class": "BACKGROUND"}, "3": {"class": "tree"}, "7": {"class": "car"}}
    table = labels_of(named, "class", labels)
    assert labels == ["", "tree", "car"] and list(table[[0, 3, 7]]) == [0, 1, 2]
    ids = np.array([[0, 3], [7, 3]], np.uint32)
    write_frame(str(tmp_path), "f.jpg", _picture()[:2, :2], segments=ids, table=table)
    written = np.asarray(Image.open(tmp_path / format.SEGMENTS / "f.png"))
    assert written.tolist() == [[0, 1], [2, 1]] and (tmp_path / "f.jpg").is_file()


def test_a_frame_is_written_at_the_quality_asked_and_smaller_for_it(tmp_path):
    rgb = np.random.default_rng(0).integers(0, 255, (64, 96, 3), np.uint8)
    sizes = {}
    for quality in (92, 80, 75):
        write_frame(str(tmp_path), f"q{quality}.jpg", rgb, quality=quality)
        expected = tmp_path / "expected.jpg"
        Image.fromarray(rgb, "RGB").save(expected, format="jpeg", quality=quality)
        made = tmp_path / f"q{quality}.jpg"
        assert made.read_bytes() == expected.read_bytes()
        sizes[quality] = made.stat().st_size
    assert sizes[92] > sizes[80] > sizes[75]
