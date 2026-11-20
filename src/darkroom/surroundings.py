"""The place a capture is of, and what carries its sensors, as plain grey meshes: something to read a
lidar's map and the trails against, not a render.

- `nearby(stage, way, within)`: the stage's still meshes within `within` metres (across the ground)
  of `way`, as one mesh in the stage's frame, in metres. Thinned by merging points `cell` apart; a
  group of prims whose bounds are all far is never opened; it stops after `budget` seconds (what it
  has by then is kept, and it says so).
- `model(stage, prop, code)`: the meshes under a prop (a pole), in the prop's own frame.

Skinned meshes and anything whose transform moves (who walks, what is carried, onlookers) are left
out of the place: they would be wherever they were at the default time.
"""

from __future__ import annotations

import time

import numpy as np
from pxr import Usd, UsdGeom, UsdSkel

from .format import Mesh

#: How far from the way, across the ground, the place is kept (m); how fine the ground grid that
#: decides it is (m); and points merged closer than `CELL` (m).
WITHIN, GRID, CELL = 25.0, 1.0, 0.2
#: Seconds `nearby` may take before it stops adding meshes.
BUDGET = 20.0


def _triangles(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """A mesh's faces as triangles (N x 3): each face fanned from its first corner."""
    if not len(counts) or counts.max() < 3:
        return np.zeros((0, 3), np.int64)
    first = np.cumsum(counts) - counts
    fans = []
    for corner in range(1, int(counts.max()) - 1):
        wide = first[counts > corner + 1]
        fans.append(
            np.stack([indices[wide], indices[wide + corner], indices[wide + corner + 1]], 1)
        )
    return np.concatenate(fans)


#: A unit cube's corners and its twelve triangles, facing out.
CUBE_POINTS = np.array([(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
CUBE_FACES = np.array(
    [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
     (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
)  # fmt: skip


def _faces(prim: Usd.Prim, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """A mesh's (or a cube's) points taken by `matrix` (USD's row-vector convention) and its
    triangles."""
    if prim.IsA(UsdGeom.Cube):
        size = UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0
        return CUBE_POINTS * size @ matrix[:3, :3] + matrix[3, :3], CUBE_FACES.copy()
    mesh = UsdGeom.Mesh(prim)
    own = mesh.GetPointsAttr().Get()
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], np.int64)
    if not own or not len(counts):
        return None
    points = np.asarray(own, np.float64) @ matrix[:3, :3] + matrix[3, :3]
    faces = _triangles(counts, np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64))
    if not len(faces):
        return None
    if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
        faces = faces[:, ::-1]
    return points, faces


def thin(points: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Points merged on a `cell` m grid (each cell's to their mean), triangles that collapse dropped."""
    if cell <= 0 or not len(points):
        return points, faces
    keys = np.floor(points / cell).astype(np.int64)
    _, which, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    which = which.reshape(-1)
    merged = np.zeros((len(counts), 3))
    np.add.at(merged, which, points)
    merged /= counts[:, None]
    faces = which[faces]
    keep = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[keep]
    if len(faces):  # duplicates found by their corners whichever way round; each keeps its winding
        _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        faces = faces[np.sort(first)]
    return merged, faces


def _normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (points[faces[:, k]] for k in range(3))
    facing = np.cross(b - a, c - a)
    normal = np.zeros_like(points)
    for k in range(3):
        np.add.at(normal, faces[:, k], facing)
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)


def _as_one(found: list[tuple[np.ndarray, np.ndarray]], cell: float) -> Mesh | None:
    found = [f for f in found if len(f[1])]
    if not found:
        return None
    first = np.cumsum([0] + [len(p) for p, _ in found[:-1]])
    points = np.concatenate([p for p, _ in found])
    faces = np.concatenate([f + k for (_, f), k in zip(found, first, strict=True)])
    points, faces = thin(points, faces, cell)
    if not len(faces):
        return None
    return Mesh(
        points.astype(np.float32),
        faces.astype(np.uint32),
        _normals(points, faces).astype(np.float32),
    )


class _Ground:
    """Which ground is near the way: a grid of `GRID` m cells, widened by `within`; and heights within
    `within` of the way's (a sky's dome over it is not the place)."""

    def __init__(self, way: np.ndarray, within: float):
        lo, hi = way[:, :2].min(0) - within - GRID, way[:, :2].max(0) + within + GRID
        self.lo, self.hi = lo, hi
        self.z = (way[:, 2].min() - within, way[:, 2].max() + within)
        shape = np.ceil((hi - lo) / GRID).astype(int) + 1
        near = np.zeros(shape, bool)
        r = int(np.ceil(within / GRID))
        dx, dy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        disc = np.stack([dx[dx**2 + dy**2 <= r * r], dy[dx**2 + dy**2 <= r * r]], 1)
        cells = np.unique(np.floor((way[:, :2] - lo) / GRID).astype(int), axis=0)
        for ox, oy in disc:  # the way's cells, widened (a few hundred shifts of a small set)
            at = cells + np.array([ox, oy])
            ok = (at >= 0).all(1) & (at < shape).all(1)
            near[at[ok, 0], at[ok, 1]] = True
        self.near = near

    def of(self, points: np.ndarray) -> np.ndarray:
        """Which of `points` (N x 3, metres) are near the way."""
        at = np.floor((points[:, :2] - self.lo) / GRID).astype(int)
        ok = (at >= 0).all(1) & (at[:, 0] < self.near.shape[0]) & (at[:, 1] < self.near.shape[1])
        ok &= (points[:, 2] >= self.z[0]) & (points[:, 2] <= self.z[1])
        out = np.zeros(len(points), bool)
        out[ok] = self.near[at[ok, 0], at[ok, 1]]
        return out

    def box_far(self, box) -> bool:
        """Whether a world box (in metres) misses the widened way's bounds."""
        mn, mx = box.GetMin(), box.GetMax()
        return (
            mx[0] < self.lo[0] or mx[1] < self.lo[1] or mn[0] > self.hi[0] or mn[1] > self.hi[1]
            or mx[2] < self.z[0] or mn[2] > self.z[1]
        )  # fmt: skip


def _moving(prim: Usd.Prim) -> bool:
    return prim.IsA(UsdSkel.Root) or (
        prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).TransformMightBeTimeVarying()
    )


def nearby(
    stage: Usd.Stage,
    way: np.ndarray,
    within: float = WITHIN,
    cell: float = CELL,
    budget: float = BUDGET,
    said: list[str] | None = None,
) -> Mesh | None:
    """The stage's still meshes and cubes within `within` m (across the ground) of `way` (N x 3,
    metres, the stage's frame), as one grey mesh, merged on a `cell` m grid. Stops after `budget` s, keeping what
    it has (a line in `said`). None when nothing is near."""
    began = time.monotonic()
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    ground = _Ground(np.asarray(way, float), within)
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    placed = UsdGeom.XformCache()
    found: list[tuple[np.ndarray, np.ndarray]] = []
    prims = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for prim in prims:
        if time.monotonic() - began > budget:
            if said is not None:
                said.append(f"scenery: stopped after {budget:g} s; what was found by then is kept")
            break
        if _moving(prim) or not prim.IsActive():
            prims.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Imageable):
            continue
        # Its own visibility: an invisible parent's children are not visited.
        if UsdGeom.Imageable(prim).GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible:
            prims.PruneChildren()
            continue
        box = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            continue
        if ground.box_far(box.__class__(box.GetMin() * metres, box.GetMax() * metres)):
            prims.PruneChildren()
            continue
        if prim.GetTypeName() not in ("Mesh", "Cube"):
            continue
        got = _faces(prim, np.asarray(placed.GetLocalToWorldTransform(prim)))
        if got is None:
            continue
        points, faces = got[0] * metres, got[1]
        near = ground.of(points)  # a triangle is near by a corner, or its middle (a big floor's)
        faces = faces[near[faces].any(1) | ground.of(points[faces].mean(1))]
        if not len(faces):
            continue
        used, faces = np.unique(faces, return_inverse=True)
        found.append(thin(points[used], faces.reshape(-1, 3), cell))
    return _as_one(found, cell)


def model(stage: Usd.Stage, prop: str, code: float, cell: float = 0.005) -> Mesh | None:
    """The meshes under `prop` (a carried pole), in the prop's own frame at time code `code`, metres."""
    root = stage.GetPrimAtPath(prop)
    if not root:
        return None
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    to_prop = np.linalg.inv(np.asarray(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(code)))
    found = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() != "Mesh":
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility(code) == UsdGeom.Tokens.invisible:
            continue
        world = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(code))
        got = _faces(prim, world @ to_prop)
        if got is not None:
            found.append((got[0] * metres, got[1]))
    return _as_one(found, cell)
