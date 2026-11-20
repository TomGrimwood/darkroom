# darkroom

A stage's sensors rendered into capture folders. Knows nothing of environments, characters,
trajectories or any app: whoever uses it hands it a stage (strider composes one per run).

- Its interface is `darkroom/__init__.py`'s `__all__`, and nothing else: users import from
  `darkroom` (and `darkroom.format`, `darkroom.sensorkit`, `darkroom.board`,
  `darkroom.terminal`), never its other modules. Importing `darkroom` loads no USD.
- A capture's options are one frozen `Options`, its progress typed `events` (sent from the
  renderer's process as plain data, read back by `board`). A new option or event goes there, not in kwargs or
  string keys.
- Before committing, the README's checks (ruff, ty, the import rules in `pyproject.toml`,
  pytest); `tests_ovrtx` too when the rendering side changed.

- `truth.py` works out poses, IMU and GNSS with USD alone; `ovrtx/` renders cameras, lidars
  and skies, in a process of its own (`renderers.launchers`).
- The renderer's interface is Isaac Lab's (`renderers/`, copied; README "The renderer"); the
  cameras play the window once (`renderers/once.py`), talking only to `BaseRenderer`. ovrtx
  code stays in `ovrtx/`; what is plain beside it (`eyes`, `frames`, `outcome`, `looks`,
  `skies`). Change the interface only with `tests/test_renderers_isaaclab.py` saying why it
  differs.
- What was measured with ovrtx is kept in `ovrtx/__init__.py` (read its docstring first):
  one kind at a time (cameras without RTX's motion BVH, 2.4 times faster; lidars and rolling
  shutters with it); the stage's frames stepped, or `step_rate`; at most 8 steps between a
  blurred camera's frames.
- The render copies (a camera's `_CaptureEye`, a lidar's `_CaptureLidar`) are siblings of the
  sensor in the session layer: under an invisible camera asset a camera renders black, and a
  copied `xformOpOrder` loses the camera's own rotation.
- `darkroom.format` is the only thing the exporters may import: change it with a `VERSION` bump.
