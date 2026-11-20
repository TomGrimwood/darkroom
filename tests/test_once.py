"""Which cameras are on together when the window is played once (`renderers.once`)."""

from darkroom.renderers import once

BIG, MID, SMALL = 1920 * 1080, 1280 * 960, 960 * 540
TOTAL = 16.0  # the GPU's memory, GB


class _Gpu:
    """A GPU whose memory in use grows by `a_pixel` bytes for every pixel put on."""

    def __init__(self, used: float, a_pixel: float) -> None:
        self.used, self.a_pixel, self.on = used, a_pixel, []

    def put_on(self, pixels: int) -> None:
        self.on.append(pixels)
        self.used += pixels * self.a_pixel / 2**30

    def read(self) -> tuple[float, float]:
        return self.used, TOTAL


def _board(waiting, gpu, most=None, headroom: float | None = 0.08):
    return once.board(waiting, lambda s: s, gpu.put_on, most, headroom, gpu.read)


def test_every_camera_is_on_at_once_while_the_gpu_has_room():
    waiting = [BIG, MID, SMALL]
    assert _board(waiting, _Gpu(2.0, 240.0)) == [BIG, MID, SMALL] and waiting == []


def test_a_camera_waits_when_what_it_would_need_does_not_fit_and_a_smaller_one_gets_on():
    # 13 GB in use: with 8% kept free, 1.25 GB to spare after the biggest is on (0.47 GB).
    gpu = _Gpu(13.0, 240.0)
    waiting = [BIG, 3 * BIG, SMALL]  # the second would need 1.4 GB, the third 0.12 GB
    assert _board(waiting, gpu) == [BIG, SMALL]
    assert waiting == [3 * BIG]  # for the next pass, first on whatever the GPU says


def test_the_first_camera_is_on_whatever_the_gpu_says():
    assert _board([BIG], _Gpu(15.9, 240.0)) == [BIG]


def test_a_capture_stepped_back_has_no_more_pixels_on_than_it_was_told():
    waiting = [BIG, MID, SMALL]
    assert _board(waiting, _Gpu(2.0, 240.0), most=BIG + SMALL) == [BIG, SMALL]
    assert waiting == [MID]


def test_one_by_one_is_a_pass_each():
    waiting = [BIG, MID, SMALL]
    assert _board(waiting, _Gpu(2.0, 240.0), headroom=None) == [BIG]
    assert waiting == [MID, SMALL]


def test_with_no_reading_of_the_gpu_every_camera_is_on():
    put = []
    assert once.board([BIG, MID], lambda s: s, put.append, None, 0.08, lambda: None) == [BIG, MID]
