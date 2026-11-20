"""darkroom's front door: all of its interface, and loading no USD until asked."""

import subprocess
import sys

import darkroom
from darkroom import surroundings


def test_importing_darkroom_loads_no_usd():
    code = "import sys, darkroom; darkroom.Options(); print('pxr' in sys.modules)"
    said = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert said.stdout.strip() == "False"


def test_everything_it_says_it_has_is_there():
    for name in darkroom.__all__:
        assert getattr(darkroom, name) is not None


def test_options_merge_the_scenery_as_finely_as_surroundings_does():
    assert darkroom.Options().scenery_cell == surroundings.CELL


def test_what_the_renderers_process_failing_raises_is_at_the_front_door():
    from darkroom import process

    assert darkroom.Failed is process.Failed and darkroom.Progress is process.Progress
