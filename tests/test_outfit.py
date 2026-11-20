from darkroom.renderers import outfit


def test_a_bar_fills_from_the_end_of_the_stretch_missing():
    assert outfit.covered(set(range(50, 100)), 100) == (0.5, 0.5)
    assert outfit.covered(set(range(70, 100)) | set(range(10)), 100) == (0.7, 0.4)
    assert outfit.covered(set(), 100) == (0.0, 0.0)
    assert outfit.lacking_from(set(range(50)), 100) == 50
    assert outfit.lacking_from(set(range(20, 30)) | set(range(60, 100)), 100) == 30  # 30-59
    assert outfit.lacking_from(set(), 100) == 0
