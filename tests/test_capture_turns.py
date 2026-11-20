"""The GPU memory cameras may take, and a step back after running out."""

from darkroom.turns import PACKING, room, step_back

BIG, SMALL, DRONE = (4096, 3000), (1280, 960), (960, 540)
GB = 2**30


def test_packing_says_the_memory_kept_free_or_each_camera_alone():
    assert PACKING == {"one by one": None, "conservative": 0.15, "normal": 0.08, "aggressive": 0.04}


def test_the_room_is_what_is_free_less_a_share_of_the_whole():
    assert room(4.0, 16.0) == (16.0 * 0.92 - 4.0) * GB
    assert room(4.0, 16.0, 0.04) == (16.0 * 0.96 - 4.0) * GB


def test_out_of_memory_the_cameras_on_lose_their_smallest_and_one_alone_goes_no_further():
    huge, tiny = (10000, 10000), (100, 100)
    assert step_back([huge] + [tiny] * 10) == 10000 * 10000 + 9 * 100 * 100
    assert step_back([BIG, BIG, SMALL]) == 2 * 4096 * 3000  # less the small one's
    assert step_back([huge]) is None
