"""A sheet of a sensor's pictures: a dozen of them spread over its frames, each named, written
beside its folder (`<sensor>_mosaic.jpg`). Each frame is decoded at the size it is shown
(PIL's `draft`: a 12 MP JPEG at an eighth of its size), as it is made while the other
cameras wait. Plain Python, PIL only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

#: The pictures a mosaic is made of.
SUFFIXES = (".jpg", ".jpeg", ".png")
#: Pixels between tiles, and the strip above each for its name.
GAP, STRIP = 8, 28


def _spread(names: list[str], count: int) -> list[str]:
    """`count` of `names` spread evenly from first to last; all of them with fewer."""
    if count <= 0 or count >= len(names):
        return names
    return [names[i] for i in np.linspace(0, len(names) - 1, count).round().astype(int)]


def mosaic(
    folder: Path, out: Path, frames: int = 12, columns: int = 4, width: int = 480
) -> Path | None:
    """`frames` of the pictures under `folder`, spread evenly, in one picture `out`:
    `columns` to a row, each `width` pixels wide, its shape kept, its name above it. Returns
    `out`, or None when `folder` has no pictures."""
    found = {
        str(p.relative_to(folder)): p
        for p in sorted(folder.rglob("*"))
        if p.suffix.lower() in SUFFIXES
    }
    names = _spread(sorted(found), frames)
    if not names:
        return None
    tiles = []
    for name in names:
        with Image.open(found[name]) as image:
            image.draft("RGB", (width, round(width * image.height / image.width)))
            shown = image.convert("RGB")
        tiles.append((name, shown.resize((width, round(width * shown.height / shown.width)))))
    height = max(tile.height for _, tile in tiles)
    rows = -(-len(tiles) // columns)
    sheet = Image.new(
        "RGB",
        (min(columns, len(tiles)) * (width + GAP) - GAP, rows * (height + STRIP + GAP) - GAP),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (name, tile) in enumerate(tiles):
        x = (index % columns) * (width + GAP)
        y = (index // columns) * (height + STRIP + GAP)
        draw.text((x + 6, y + 7), name, fill="black")
        sheet.paste(tile, (x, y + STRIP))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=90)
    return out
