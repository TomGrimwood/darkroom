import pytest

from darkroom import looks


def test_a_preset_by_name() -> None:
    settings, warmup = looks.resolve("path_traced")
    assert settings["/rtx/rendermode"] == "PathTracing"
    assert warmup > 1


def test_no_look_is_the_default_preset_and_an_empty_one_the_scene_as_it_is() -> None:
    assert looks.resolve(None) == looks.resolve(looks.DEFAULT) != ({}, 1)
    assert looks.resolve({}) == ({}, 1)


def test_a_look_is_a_preset_file_as_isaac_sim_writes_one(tmp_path) -> None:
    from pxr import Sdf

    layer = Sdf.Layer.CreateNew(str(tmp_path / "mine.settings.usda"))
    layer.customLayerData = {"renderSettings": {"rtx:post:aa:op": 4}, "darkroom:warmup": 3}
    layer.Save()
    assert looks.resolve(tmp_path / "mine.settings.usda") == ({"/rtx/post/aa/op": 4}, 3)
    assert looks.resolve(["real_time", str(tmp_path / "mine.settings.usda")])[1] == 15


def test_there_are_real_time_and_path_traced_and_a_quicker_path_traced() -> None:
    assert list(looks.LOOKS) == ["real_time", "path_traced", "path_traced_draft"]


def test_later_parts_win() -> None:
    settings, warmup = looks.resolve(["path_traced", {"/rtx/pathtracing/spp": 128}])
    assert settings["/rtx/pathtracing/spp"] == 128
    assert settings["/rtx/rendermode"] == "PathTracing"
    assert warmup == looks.LOOKS["path_traced"].warmup


def test_every_preset_says_what_it_is_for() -> None:
    assert all(look.about.endswith(".") for look in looks.LOOKS.values())


def test_an_unknown_name_is_a_sentence() -> None:
    with pytest.raises(ValueError, match="no render look 'glossy'"):
        looks.resolve("glossy")


def test_the_real_time_look_is_isaac_sims_own_light_and_no_look_has_render_wide_auto_exposure():
    from darkroom import looks

    settings, warmup = looks.resolve("real_time")
    for name in looks.LOOKS:  # auto exposure is each camera's own
        assert looks.resolve(name)[0]["/rtx/post/histogram/enabled"] is False
    assert settings["/rtx/raytracing/globalVolumetricEffects/enabled"] is False
    assert settings["/rtx/sceneDb/ambientLightIntensity"] == 0.0
    assert warmup >= 8
