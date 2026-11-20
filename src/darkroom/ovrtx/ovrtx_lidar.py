"""A lidar's scans as ovrtx hands them over: its `PointCloud` (the lidar model's output:
points, intensities, firing times, echoes, channels, materials, objects) read scan by scan
into the lidar's raw files (`darkroom.lidarscans.RawWriter`), the files a lidar
writes; and what each object ID is called, from ovrtx's `StableIdMap` (`names_of`).

A scan is one frame of the lidar's render product: its `timestampNs` is the time of ovrtx's
clock it began at, each point's `TimeOffsetNs` when it was fired after that. With
`partialOutputs` false a scan comes whole, in the step that ends it, gathered over the steps
it took, each firing from where the lidar was in its step.
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
#: A 128-bit object ID as two numbers, to tell IDs apart whole arrays at a time.
ID = np.dtype([("low", "<u8"), ("high", "<u8")])
#: What ovrtx names of object IDs (`StableIdMap`, an `IdentifierMap`): each ID's four words,
#: where each one's prim path ends in the bytes after, and those bytes.
Table = tuple[np.ndarray, np.ndarray, np.ndarray]


class OVRTXLidarRenderData:
    """Owns one lidar's native resources (its copy and its render product on the stage) and
    the raw files its scans are written into."""

    def __init__(
        self, prim_path: str, render_scope_name: str, raws: lidarscans.RawWriter, hits: bool
    ) -> None:
        """Create render data for the lidar at `prim_path` in its assigned render scope.

        Args:
            prim_path: The lidar's prim, as the stage has it.
            render_scope_name: Root scope containing this lidar's render product and RenderVars.
            raws: The raw files its scans are appended to.
            hits: Whether each point's object is kept (and named: `names_of`).
        """
        self.prim_path = prim_path
        self.render_scope_name = render_scope_name
        self.render_product_path = f"/{render_scope_name}/RenderProduct"
        self.point_cloud_key = f"/{render_scope_name}/Vars/PointCloud"
        self.raws, self.hits = raws, hits
        self.resources = contextlib.ExitStack()
        #: The stamps of the scans written: a scan is kept once.
        self.stamps: set[int] = set()
        #: What a scan's points were given in, when not x, y, z in the lidar's axes.
        self.wrong: dict[str, int] | None = None
        #: The object IDs its scans' points came back from (with `hits`).
        self.seen = np.zeros(0, ID)

    def read(self, frame: Any) -> None:
        """A frame of the lidar's render product read: its scan, if it has not been written,
        appended to the raw files (its real returns that hit something)."""
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
                return np.from_dlpack(mapped[name])[:count][real]

            objects = np.zeros((0, 16), np.uint8)
            if self.hits:
                objects = np.ascontiguousarray(channel("ObjectId")).view(np.uint8).reshape(-1, 16)
                self.seen = np.union1d(self.seen, objects.view(ID).reshape(-1))
            self.raws.add(
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


def names_of(table: Table, ids: np.ndarray) -> dict[int, str]:
    """The prim paths `table` has for object `ids` (an `ID` array), by each ID as one number.
    Only those are looked up: a scene's table names every prim it has (700,000 on a street)."""
    words, ends, paths = table
    keys = np.ascontiguousarray(words, "<u4").view(ID).reshape(-1)
    found: dict[int, str] = {}
    for row in np.flatnonzero(np.isin(keys, ids)):
        path = paths[int(ends[row]) : int(ends[row + 1])].tobytes()
        found[int.from_bytes(np.ascontiguousarray(words[row], "<u4").tobytes(), "little")] = (
            path.decode("utf-8").rstrip("\x00")
        )
    return found
