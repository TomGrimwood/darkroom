"""What a capture hands Isaac Sim, the same on both sides."""

import json

import pytest

from darkroom.renderjob import RenderJob, plain, read


def test_a_job_crosses_to_isaac_sim_as_it_was():
    job = RenderJob(
        "/runs/a/run.usda",
        "/out/a",
        (10.0, 12.5),
        look={"/rtx/rendermode": "RaytracedLighting"},
        warmup=3,
        spans=((11.0, 12.0, {"/rtx/pathtracing/spp": 64}),),
        lenses={"/Rig/fisheye": {"model": "lut", "name": "f.exr"}},
        blind=True,
        hits=True,
        packing="conservative",
        quality=75,
        most=4096 * 3000,
    )
    assert read(json.loads(json.dumps(plain(job)))) == job


def test_a_key_one_side_does_not_know_is_refused():
    data = plain(RenderJob("run.usda", "out", (0.0, 1.0)))
    with pytest.raises(ValueError, match="no shutter"):
        read({**data, "shutter": 1})
