"""A camera's frames as its renderer publishes them, named by the pose each belongs to, and
the camera itself as the loop holds it (`Camera`): a renderer's render data, the arrays it
renders into, and its frames' store. The same for every renderer: none of this knows which.

A renderer publishes a frame with the time of its own clock it shows, either as pixels in the
camera's output array (written here, off the thread that renders, as a JPEG) or as a file it
wrote itself. Either way it lands in the
camera's frames folder under the name it was rendered by (`render_000012.jpg`), and is given
its pose's name once its time is known to be a pose's (`keep`): frame i is at i / rate seconds
into the window. So what a renderer stopped part way made is kept, to carry on from.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from darkroom import format
from darkroom.frames import QUALITY, labels_of, write_frame
from darkroom.outcome import Result

from .base_renderer import BaseRenderer
from .camera_cfg import CameraData
from .camera_render_spec import CameraRenderSpec

#: How far off its pose a frame may be rendered and kept, of a period.
OFF_POSE = 0.1
#: Threads writing the frames a renderer hands over as pixels (PIL encodes without Python's
#: lock: 85 frames a second with 4, of 12 MP).
WRITERS = 4


class FrameStore:
    """A camera's frames in its capture folder: as rendered (`took`, `write`), then named by
    their time in the window as they come (`keep`: a frame a pose, the first rendered of
    it); once the camera is off, the rest (`finish`). Those already named (`had`: from a
    capture carried on) are not kept again."""

    def __init__(
        self,
        capture: Path,
        name: str,
        rate: float,
        size: tuple[int, int],
        *,
        quality: int = QUALITY,
        taxonomy: str | None = None,
        geometry: bool = False,
    ) -> None:
        self.rate, self.size, self.quality = rate, size, quality
        self.taxonomy, self.geometry = taxonomy, geometry
        self.folder = capture / name
        self.dir = self.folder / format.FRAMES
        self.dir.mkdir(parents=True, exist_ok=True)
        #: Each frame rendered: the time of the renderer's clock it shows (seconds), its file.
        self.frames: list[tuple[Any, str]] = []
        #: The labels a segmented camera's pixels index, as its renderer says them.
        self.labels: list[str] = [""]
        #: The frames' numbers made: before this run, and since.
        self.had = format.frames_kept(capture, name)
        self.kept: set[int] = set()
        self.seen = 0  # of `frames`, those looked at
        self.keeping = False  # since the first frame at `keep_from` (`keep`)
        #: Frames rendered off their poses (more than `OFF_POSE` of a period): not kept; and the
        #: furthest off of those kept, seconds.
        self.off_pose = 0
        self.worst = 0.0
        self._pool: ThreadPoolExecutor | None = None
        self._writing: list[Future] = []

    @property
    def have(self) -> set[int]:
        return self.had | self.kept

    def took(self, stamp: Any, name: str) -> None:
        """A frame its renderer wrote itself as `name` in the frames folder, showing `stamp`."""
        self.frames.append((stamp, name))

    def write(
        self,
        stamp: Any,
        rgb: np.ndarray,
        segments: np.ndarray | None = None,
        named: Mapping[str, Any] | None = None,
        normals: np.ndarray | None = None,
        far: np.ndarray | None = None,
    ) -> None:
        """A frame handed over as pixels (height x width x 3 or 4, uint8), showing `stamp`:
        copied, and written as a JPEG by a writer thread. With it, where the capture keeps
        them: each pixel's segment (`segments`, ids `named` as Replicator's `idToLabels`),
        and its normal and distance (`normals`, `far`), written beside it."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(WRITERS, thread_name_prefix="frames")
        name = f"render_{len(self.frames):06d}.jpg"
        beside: dict[str, np.ndarray] = {}
        if self.taxonomy and segments is not None and named is not None:
            beside["segments"] = np.array(segments, copy=True)
            beside["table"] = labels_of(named, self.taxonomy, self.labels)
            if self.geometry and normals is not None and far is not None:
                beside["normals"] = np.array(normals, copy=True)
                beside["far"] = np.array(far, copy=True)
        self._writing.append(
            self._pool.submit(
                write_frame,
                folder=str(self.dir),
                name=name,
                rgb=np.array(rgb, copy=True),
                quality=self.quality,
                **beside,
            )
        )
        self.frames.append((stamp, name))

    def wait(self) -> None:
        """Every frame handed over as pixels written; raises what a write raised."""
        writing, self._writing = self._writing, []
        for write in writing:
            write.result()
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def keep(
        self,
        seconds: Callable[[Any], Any],
        span: float,
        keep_from: float = -math.inf,
        every: bool = False,
    ) -> set[int]:
        """The frames rendered since, in order, named by their time, up to the first not yet
        written (its JPEG not there yet: `frames.write_frame` writes it last, whole or not at
        all); with `every` (all handed over, `wait`), one not written raises. Let
        go: those before the first at `keep_from` seconds of the window or after (rendered as
        the camera got on: black, its denoiser without history), outside the window, of a time
        it has, or off their pose. Returns the numbers it has."""
        rate = self.rate
        for k in range(self.seen, len(self.frames)):
            stamp, name = self.frames[k]
            if not (self.dir / name).is_file():
                if every:
                    raise RuntimeError(f"frame {name} was rendered but never written.")
                break
            self.seen = k + 1
            second = float(seconds(stamp * 10**9))
            self.keeping = self.keeping or second >= keep_from - 1e-6
            # The period it falls in: frames come a period apart, on the grid or not, and one
            # on the grid can be stamped a millionth of a period early.
            number = int(np.floor(second * rate + 1e-3))
            off = abs(second - number / rate)
            wanted = self.keeping and -1e-6 <= second <= span + 1e-6 and number not in self.have
            if wanted and off > OFF_POSE / rate:  # the scene at another time than its pose's
                self.off_pose += 1
                wanted = False
            if wanted:
                self._name(name, number)
                self.kept.add(number)
                self.worst = max(self.worst, off)
            else:
                self._drop(name)
        return self.have

    def _name(self, name: str, number: int) -> None:
        """Frame `name` (and its segments, normals, distances) named as pose `number`."""
        capture, sensor = self.folder.parent, self.folder.name
        (self.dir / name).replace(format.frame_path(capture, sensor, number))
        for kind, path in self._beside():
            made = self.dir / kind / f"{name[:-4]}.png"
            if made.is_file():
                to = path(capture, sensor, number)
                to.parent.mkdir(parents=True, exist_ok=True)
                made.replace(to)

    def _drop(self, name: str) -> None:
        (self.dir / name).unlink(missing_ok=True)
        for kind, _ in self._beside():
            (self.dir / kind / f"{name[:-4]}.png").unlink(missing_ok=True)

    def _beside(self) -> list[tuple[str, Callable[[Any, str, int], Path]]]:
        """What is written beside each frame: its segments, and its normals and distances."""
        beside: list[tuple[str, Callable[[Any, str, int], Path]]] = []
        if self.taxonomy:
            beside.append((format.SEGMENTS, format.segment_path))
        if self.geometry:
            beside += [
                (format.NORMALS, format.normals_path),
                (format.DISTANCES, format.distances_path),
            ]
        return beside

    def finish(
        self, seconds: Callable[[Any], Any], span: float, keep_from: float = -math.inf
    ) -> Result:
        """Every frame named (`keep`); fails with none in the window."""
        self.wait()
        have = self.keep(seconds, span, keep_from, every=True)
        for kind, _ in self._beside():  # what Replicator wrote them into, emptied
            made = self.dir / kind
            if made.is_dir() and not any(made.iterdir()):
                made.rmdir()
        if self.taxonomy:
            labels = format.segment_path(self.folder.parent, self.folder.name, 0).parent
            labels.mkdir(parents=True, exist_ok=True)
            (labels / format.SEGMENT_LABELS).write_text(json.dumps(self.labels), encoding="utf-8")
        if not have:
            raise RuntimeError(f"no frames within the window's {span:g} s.")
        notes = []
        if self.off_pose:
            notes.append(f"{self.off_pose} rendered off their poses, not kept")
        if self.worst > 1e-4:
            notes.append(f"up to {self.worst * 1000:.1f} ms off their poses")
        size = f"{self.size[0]}x{self.size[1]}"
        return Result(len(have), "frames", self.rate, size, tuple(notes))


class Camera:
    """A camera on a renderer, as the loop holds it: made (`create_render_data`,
    `set_outputs`), told where it is before each render (`update`), read after each
    (`read`: what the renderer published goes to its `store`), taken off (`stop`: `cleanup`).
    `data` is the renderer's own render data for it."""

    def __init__(
        self,
        renderer: BaseRenderer,
        spec: CameraRenderSpec,
        store: FrameStore,
        pose: Callable[[float], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.renderer, self.spec, self.store, self.pose = renderer, spec, store, pose
        self.camera_data = CameraData()
        self.camera_data.allocate(
            spec.cfg.data_types,
            spec.cfg.height,
            spec.cfg.width,
            spec.num_instances,
            renderer.supported_output_types(),
        )
        self.data: Any = renderer.create_render_data(spec)
        renderer.set_outputs(self.data, self.camera_data.output)

    @property
    def have(self) -> set[int]:
        """The poses it has a frame of."""
        return self.store.have

    def update(self, time: float) -> None:
        """The renderer told where the camera is at `time` seconds of the timeline, and its
        intrinsics (`pose`): a renderer that follows the stage itself takes no notice."""
        if self.pose is not None and self.data is not None:
            positions, orientations, intrinsics = self.pose(time)
            self.camera_data.pos_w, self.camera_data.quat_w_world = positions, orientations
            self.camera_data.intrinsic_matrices = intrinsics
            self.renderer.update_camera(self.data, positions, orientations, intrinsics)

    def read(self) -> None:
        """What the renderer published since the last read, into the store: each picture
        with its time (`CameraData.info`'s "capture"), a segmentation's labels."""
        if self.data is None:
            return
        self.renderer.read_output(self.data, self.camera_data)
        info = self.camera_data.info
        for capture in (info.get("rgb") or info.get("rgba") or {}).get("capture", ()):
            if "file" in capture:
                self.store.took(capture["time"], capture["file"])
            else:
                made = self.camera_data.output
                segmented = info.get("semantic_segmentation") or {}
                self.store.write(
                    capture["time"],
                    made["rgba"][0],
                    segments=made["semantic_segmentation"][0] if segmented else None,
                    named=segmented.get("idToLabels"),
                    normals=made["normals"][0] if "normals" in made else None,
                    far=made["distance_to_camera"][0] if "distance_to_camera" in made else None,
                )
        labels = (info.get("semantic_segmentation") or {}).get("labels")
        if labels is not None:
            self.store.labels = labels

    def stop(self) -> None:
        """Rendering stopped: what it published last read, its render data cleaned up (its
        render product destroyed, its GPU memory given back)."""
        if self.data is not None:
            self.read()
            self.renderer.cleanup(self.data)
            self.data = None

    def keep(
        self, seconds: Callable[[Any], Any], span: float, keep_from: float = -math.inf
    ) -> set[int]:
        """The poses it has a frame of so far, those rendered since named (`FrameStore.keep`)."""
        return self.store.keep(seconds, span, keep_from)

    def finish(
        self, seconds: Callable[[Any], Any], span: float, keep_from: float = -math.inf
    ) -> Result:
        """What it made of the window's `span` seconds (`FrameStore.finish`)."""
        return self.store.finish(seconds, span, keep_from)
