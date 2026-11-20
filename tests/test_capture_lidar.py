"""A lidar scan's object IDs: which of them each point hit, as numpy's own unique says."""

import numpy as np

from darkroom.lidarscans import distinct as _distinct


def test_distinct_object_ids_are_numpys_unique_rows_faster():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 2**63, (40, 2), dtype=np.uint64)
    objects = ids[rng.integers(0, 40, 5000)].view(np.uint8).reshape(-1, 16)
    unique, back = _distinct(objects)
    expected, expected_back = np.unique(objects, axis=0, return_inverse=True)
    assert np.array_equal(unique[back], objects)  # each point's ID found again
    assert {r.tobytes() for r in unique} == {r.tobytes() for r in expected}
    assert len(unique) == len(expected) and len(back) == len(expected_back.reshape(-1))
    empty, none = _distinct(np.zeros((0, 16), np.uint8))
    assert empty.shape == (0, 16) and none.shape == (0,)
