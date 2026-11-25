"""A lidar's scans as ovrtx hands them over: its `PointCloud` (the lidar model's output:
points, intensities, firing times, echoes, channels, materials, objects) read scan by scan
and handed to the lidar's `darkroom.lidarscans.ScanWriter`, which writes them on a thread of
its own; and what each object ID is called, from ovrtx's `StableIdMap` (`Names`).

A scan is one frame of the lidar's render product: its `timestampNs` is the time of ovrtx's
clock it began at, each point's `TimeOffsetNs` when it was fired after that. With
`partialOutputs` false a scan comes whole, in the step that ends it, gathered over the steps
it took, each firing from where the lidar was in its step. Its points are in the lidar's own
frame, not compensated for its motion (`CARTESIAN`, `SENSOR`), as a real lidar's driver gives
them.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import ovrtx

from darkroom import lidarscans

#: The bit of a point's `Flags` that says it is a real return.
VALID = 0x40
#: ovrtx's words for how a scan's points are given (`coordsType`, `frameOfReference`), as it
#: says them of a scan asked for as x, y, z in the lidar's own axes.
CARTESIAN, SENSOR = 0, 0

#: What ovrtx names of object IDs (`StableIdMap`, an `IdentifierMap`): each ID's four words,
#: where each one's prim path ends in the bytes after, and those bytes.
Table = tuple[np.ndarray, np.ndarray, np.ndarray]


class OVRTXLidarRenderData:
    """Owns one lidar's native resources (its copy and its render product on the stage) and
    the writer its scans are handed to."""

    def __init__(
        self, prim_path: str, render_scope_name: str, scans: lidarscans.ScanWriter, hits: bool
    ) -> None:
        """Create render data for the lidar at `prim_path` in its assigned render scope.

        Args:
            prim_path: The lidar's prim, as the stage has it.
            render_scope_name: Root scope containing this lidar's render product and RenderVars.
            scans: What its scans are handed to.
            hits: Whether each point's object is kept (and named: `Names`).
        """
        self.prim_path = prim_path
        self.render_scope_name = render_scope_name
        self.render_product_path = f"/{render_scope_name}/RenderProduct"
        self.point_cloud_key = f"/{render_scope_name}/Vars/PointCloud"
        self.scans, self.hits = scans, hits
        self.resources = contextlib.ExitStack()
        #: The stamps of the scans handed over: a scan is kept once.
        self.stamps: set[int] = set()
        #: What a scan's points were given in, when not x, y, z in the lidar's axes.
        self.wrong: dict[str, int] | None = None

    def read(self, frame: Any) -> None:
        """A frame of the lidar's render product read: its scan, if it has not been handed
        over, copied out (its real returns that hit something) and handed to `scans`. Only
        the copying is done here, on the render thread: the rest is the writer's."""
        cloud = dict(frame.render_vars.items()).get(self.point_cloud_key)
        if cloud is None:
            return
        with cloud.map(device=ovrtx.Device.CPU) as mapped:
            said = {name: _param(mapped, name) for name in ("coordsType", "frameOfReference")}
            stamp = _param(mapped, "timestampNs")
            if stamp in self.stamps:
                return
            if said != {"coordsType": CARTESIAN, "frameOfReference": SENSOR}:
                self.wrong = said
            count = int(np.from_dlpack(mapped["Counts"]).ravel()[0])
            points = np.from_dlpack(mapped["Coordinates"])[:, :count].T
            flags = np.asarray(np.from_dlpack(mapped["Flags"])[:count], np.uint8)
            # A real return, and one that hit something (none lies at the lidar itself).
            real = (np.bitwise_and(flags, VALID) > 0) & points.any(axis=1)

            def channel(name: str) -> np.ndarray:
                return np.from_dlpack(mapped[name])[:count][real]  # a copy: the map goes

            objects = None
            if self.hits:
                objects = np.ascontiguousarray(channel("ObjectId")).view(np.uint8).reshape(-1, 16)
            self.scans.add(
                stamp,
                points=points[real],
                intensity=channel("Intensity"),
                offsets=channel("TimeOffsetNs").astype(np.int64),
                ret=channel("EchoId"),
                # A channel numbered from 0, as the capture keeps it: ovrtx says the lidar's own
                # `channelId`, from 1 (the same emitter's is one more: measured, point by point).
                ring=np.maximum(channel("ChannelId"), 1) - 1,
                material=channel("MaterialId"),
                objects=objects,
            )
            self.stamps.add(stamp)

    def cleanup(self) -> None:
        """Release this lidar's native resources. Safe to call repeatedly."""
        self.resources.close()


def _param(mapped: Any, name: str) -> int:
    """One of a mapped `PointCloud`'s params, a whole number."""
    return int(np.from_dlpack(mapped.params[name]).ravel()[0])


class Names:
    """What object IDs are called, from ovrtx's `table`, for a lidar's writer to ask of each ID
    as it first comes back (`lidarscans.Names`): only those asked for are looked up, the table's
    IDs sorted once by their low half (a street's table names 700,000 prims)."""

    def __init__(self, table: Table) -> None:
        words, self._ends, self._paths = table
        keys = np.ascontiguousarray(words, "<u4").view(np.uint64).reshape(-1, 2)
        self._order = np.argsort(keys[:, 0], kind="stable")
        self._low, self._high = keys[self._order, 0], keys[self._order, 1]

    def __call__(self, ids: np.ndarray) -> list[str | None]:
        """Each of `ids`' (k x 16 bytes) prim path, or None for one the table has not."""
        pairs = np.ascontiguousarray(ids, np.uint8).view(np.uint64).reshape(-1, 2)
        first = np.searchsorted(self._low, pairs[:, 0], "left")
        last = np.searchsorted(self._low, pairs[:, 0], "right")
        found: list[str | None] = []
        for (_, high), a, b in zip(pairs, first, last, strict=True):
            rows = [int(self._order[r]) for r in range(int(a), int(b)) if self._high[r] == high]
            found.append(self._path(rows[0]) if rows else None)
        return found

    def _path(self, row: int) -> str:
        path = self._paths[int(self._ends[row]) : int(self._ends[row + 1])].tobytes()
        return path.decode("utf-8").rstrip("\x00")
