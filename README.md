# darkroom

A stage's sensors rendered into capture folders. Cameras, lidars and the antennas' skies are
rendered by NVIDIA's ovrtx in a process of its own (`darkroom.process`); poses, IMUs and GNSS
fixes are worked out from the stage with USD.

```python
import darkroom
from darkroom import Options

folder = darkroom.capture("run.usda", "out/run", Options(window=(10, 12)))
darkroom.format.read_meta(folder).sensors  # what was made: plain files

rigs = [darkroom.Rig("A", "/World/Walk/Frames/SensorMount/A"), darkroom.Rig("B", "...")]
board = darkroom.board.Board()
darkroom.capture_rigs("run.usda", "out/run", rigs, on_progress=board.take)  # a capture each
board.rows()  # each sensor: its size, rate, how far it has got, the rigs it is for
```

## The renderer

NVIDIA's ovrtx 0.7.0 (`darkroom/ovrtx/`), in a process of its own (`renderers.launchers`;
`$DARKROOM_OVRTX_PYTHON` names another Python with it). Its settings are a capture's
`Options.renderer_cfg`; left out, its own.

```python
from darkroom.renderers import OVRTXRendererCfg

darkroom.capture("run.usda", "out/run", Options(renderer_cfg=OVRTXRendererCfg(step_rate=960)))
```

The interface is Isaac Lab's (`isaaclab.renderers` and its ovrtx backend), copied, not
imported: a `RendererCfg` names its renderer's class by where it is (`class_type`,
"module:Class") and says what it makes without loading it; the renderer is a `BaseRenderer`;
what renders the cameras (`renderers/once.py`) talks to that alone.
`tests/test_renderers_isaaclab.py` reads Isaac Lab's source (`$ISAACLAB_SOURCE`, when set)
and fails when the two differ other than as it lists, with why.

ovrtx renders all a capture has, a kind at a time in one process: its cameras (their
pictures, and where the stage asks their segments, normals and distances), its lidars and the
sky above each antenna; spans in another look, several rigs at once, a capture carried on and
each camera's own auto exposure among them. What ovrtx measures about itself is in
`darkroom/ovrtx/__init__.py`; in short:

- The cameras play the window once, as many on together as the GPU's memory has room for
  (measured as each is put on); ovrtx renders each when it is due. RTX's motion BVH is on
  only for lidars and rolling shutters: with it a camera renders 2.4 times slower.
- A camera's `sensor:camera:readoutSeconds` is ovrtx's own rolling shutter
  (`OmniRtxRollingShutterAPI`). A frame's time is the middle of its last row's exposure
  (`darkroom/sensorkit/camera.py` has the convention). Things that turn fast under it need
  more steps a second than the stage has frames: `OVRTXRendererCfg(step_rate=960)`.
- The lidars are the stage's own, unchanged (ovrtx's lidar is RTX's, the same model:
  `OmniLidar`, `OmniSensorGenericLidarCoreAPI`), each point in the lidar's own frame as a
  driver gives it, with its ring, material and (with `Options.lidar_hits`) prim, and on a
  segmented stage that prim's label as the cameras' segments name it; a scan is gathered over
  the stage's frames, each firing from where the lidar was in its frame (`darkroom/rates.py`
  has every kind's update rate), and written into the lidar's LAZ as the render goes on; a
  lidar missing any scan is refused.
- What carries the sensors (a prim with `Frames` and `Geometry`) and any prims
  `Options.bodies` names (`"/World/Machine/*"`) keep their poses as bodies: a sensor under one
  rides on it.
- A camera's distances are from its centre along each pixel's ray. Fisheyes are ovrtx's own
  OpenCV fisheye, corners and all. RTX's motion blur is right up to 8 steps between a camera's frames, and refused
  past (`darkroom.ovrtx.BLUR_STEPS`). `darkroom/format.py` says every output's frame, units
  and time.
- A look's RTX settings (by their setting paths) are turned into the attributes ovrtx's schemas say stand for them,
  those it has none for said and left out.
- What ovrtx could not make it says (an output's status, a step failing); out of the GPU's
  memory, the capture steps back, fewer cameras on at once. Its timings in `perf.json` are ovrtx's own
  profile. It is started sending NVIDIA neither its start-up report nor crash dumps.

## The front door

`darkroom/__init__.py`'s `__all__` is the whole interface; importing `darkroom` loads no USD.

| Name | Is |
|---|---|
| `capture`, `captures`, `Job` | a stage's sensors into a capture folder; several, in as few renderer processes as render them |
| `capture_rigs`, `Rig`, `count_shared` | rigs on one stage, a sensor the same in two rendered once, a capture each |
| `Options`, `Span` | how a capture is made: window, kinds, look, spans in another look, packing, the renderer's settings |
| `LOOKS` | render presets by name, RTX settings as Isaac Sim saves them (`presets/*.settings.usda`) |
| `stills`, `Camera`, `Shot` | pictures from any cameras at chosen times |
| `Progress` | what `on_progress` is handed, carrying a render's events |
| `Failed` | raised when the renderer's side fails, the GPU out of memory say |
| `board`, `terminal` | a render's progress as rows of sensors; drawn live in a terminal (`rich`) |
| `format` | the capture folder as plain files: `meta.json`, poses, frames, scans, IMU, GNSS |
| `sensorkit` | what the sensors on a stage are; GNSS and IMU maths |

## Inside

| Module | Does |
|---|---|
| `run` | the truth, then the renderer's process, into each capture folder |
| `process` | a function run in a Python process of its own: progress, result and errors as data, killed when the GPU runs out, one renderer on the machine at a time |
| `share` | sensors keyed by everything that decides what they record; the duplicates off; a folder per rig, hard links, the render then only how it was made |
| `options`, `events` | plain data: a capture's options; what a render reports of its sensors, typed |
| `ovrtx` | the ovrtx side (run only through its launcher): cameras, lidars, skies; its renderer, "ovrtx" |
| `renderers` | Isaac Lab's renderer interface, the launchers that start each renderer's process, a camera's frames named by their poses |
| `turns` | the GPU memory kept free beside the cameras, a step back after running out |
| `renderers/once.py`, `renderers/outfit.py` | cameras as the window plays once, as many on as the GPU has room for; each camera made ready, where it is at a time |
| `looks`, `presets` | how cameras are rendered: RTX presets |
| `truth`, `skies`, `surroundings` | poses, IMU, true GNSS fixes (antenna and tip), how much sky each antenna saw, and the scenery near the way |

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
uv run lint-imports
uv run pytest                                # no rendering: seconds
uv run pytest tests_ovrtx                    # rendered by ovrtx: a minute or two, the GPU
                                             # (ISAAC_ASSETS: NVIDIA's Isaac assets, for its lidars)
```

The import rules (`pyproject.toml`): `sensorkit` knows nothing of rendering; the front door's
own modules load no USD; a board knows only the events it is told; `ovrtx` is run only
through its launcher, and nothing but its renderer, its lidars and its stage loads ovrtx.
