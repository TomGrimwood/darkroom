"""A camera's frame written to its capture folder by ovrtx's side as it reads each frame
back. Everything for the frame in one job, its JPEG last, so a frame is
written once its JPEG is there (`renderers.frame_store.FrameStore.keep`).

On the render thread only copies are made (a contiguous copy of the picture as the annotator
hands it, four channels: 3.9 ms for a 12 MP frame, where taking three channels out took 68 ms)
and the segment labels looked up. In the job, the picture is encoded by PIL straight into a file
(85 frames a second with 4 threads, where Replicator's own JPEG writer, encoding into memory
holding Python's lock, made 30), the same bytes as before. A file is written as `<name>.part`
and renamed: there whole, or not at all. Plain Python.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from darkroom import format

#: The frames' JPEG quality when a capture does not say (`Options.jpeg_quality`): on 12 MP
#: frames 62% of the bytes 92 makes (75: 56%, 85: 82%), told apart from 92 by eye at 200% by
#: no one who looked; the frames are most of a capture, and of its MCAP.
QUALITY = 80


def labels_of(named: Mapping[str, Any], taxonomy: str, labels: list[str]) -> np.ndarray:
    """Replicator's segment ids (`idToLabels`) as a table from id to index into `labels`
    (grown with new ones here, on the render thread alone; 0 is nothing labelled)."""
    ids = [int(k) for k in named]
    table = np.zeros(max(ids, default=0) + 1, np.uint8)
    for key, said in named.items():
        label = said.get(taxonomy, "") if isinstance(said, dict) else str(said)
        if not label or label.upper() in ("BACKGROUND", "UNLABELLED"):
            continue
        if label not in labels:
            labels.append(label)
        table[int(key)] = labels.index(label)
    return table


def write_frame(
    folder: str,
    name: str,
    rgb: np.ndarray,
    segments: np.ndarray | None = None,
    table: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    far: np.ndarray | None = None,
    quality: int = QUALITY,
) -> None:
    """Frame `name` written into `folder` (a camera's frames folder): its segments (`segments`
    ids indexed through `table`, `labels_of`), normals and distances first, each beside it in
    its own folder, then its JPEG, from `rgb` as the annotator gave it (RGB, or RGBA whose
    fourth channel is left out)."""
    into, stem = Path(folder), name[:-4]
    if segments is not None and table is not None:
        ids = segments.reshape(segments.shape[:2])
        indexed = table[np.clip(ids, 0, len(table) - 1)]
        _png(into / format.SEGMENTS / f"{stem}.png", indexed.astype(np.uint8))
    if normals is not None:
        _png(into / format.NORMALS / f"{stem}.png", format.encode_normals(normals[..., :3]))
    if far is not None:
        _png(into / format.DISTANCES / f"{stem}.png", format.encode_distances(far))
    jpeg(into / name, rgb, quality)


def jpeg(file: Path, rgb: np.ndarray, quality: int = QUALITY) -> None:
    """`rgb` (height x width x 3 or 4, uint8; a fourth channel left out) as a JPEG in `file`."""
    from PIL import Image

    height, width = rgb.shape[:2]
    mode = "RGBX" if rgb.shape[2] == 4 else "RGB"
    image = Image.frombuffer(mode, (width, height), np.ascontiguousarray(rgb), "raw", mode, 0, 1)
    part = file.with_name(file.name + ".part")
    with part.open("wb") as out:
        image.save(out, format="jpeg", quality=quality)
    part.replace(file)


def _png(file: Path, data: np.ndarray) -> None:
    from PIL import Image

    file.parent.mkdir(parents=True, exist_ok=True)
    part = file.with_name(file.name + ".part")
    with part.open("wb") as out:
        Image.fromarray(data).save(out, format="png", compress_level=3)
    part.replace(file)
