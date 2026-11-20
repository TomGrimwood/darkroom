# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The stage ovrtx renders, as ovstage holds it: Isaac Lab's `OvstageBackend` and its cfg
(`isaaclab_ov/stage.py`), as much of them as a capture uses. Isaac Lab's also clones its
environments and populates physics; a capture has one stage and neither.

Every write to the stage carries an ordinal; a renderer sees a write once its ordinal is sealed
(`commit`), and is stepped at the ordinal it is to see.
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from typing import Any

import ovstage

from darkroom.renderers.renderer_cfg import BackendCfg


def _handle_raw_id(handle: object) -> int:
    """Return a numeric ID from an OVStage integer or typed token/path handle."""
    return getattr(handle, "raw_id", handle)  # ty: ignore[invalid-return-type]


@dataclass(frozen=True)
class OvstageBackendCfg(BackendCfg):
    """Configuration of a simulation-owned stage."""

    class_type: type[OvstageBackend] | str = "darkroom.ovrtx.stage:OvstageBackend"
    scene_key: BackendCfg | int | None = None
    """None selects the simulation scene; isolated consumers use their configuration."""


class OvstageBackend:
    """Own a detached stage and its paths until the simulation has closed its consumers."""

    def __init__(self, cfg: OvstageBackendCfg) -> None:
        self.cfg = cfg
        self.ordinal = 0
        """The ordinal writes are made at; 0 before the stage is populated."""
        self.next_camera_id = 0
        self.resources = contextlib.ExitStack()
        """Release borrowers before paths and stage, in reverse registration order."""
        self.stage: Any = self.resources.enter_context(ovstage.Stage("darkroom.scene"))
        self.paths: Any = self.resources.enter_context(ovstage.PathDictionary(self.stage))

    def populate(self, usda: str) -> None:
        """The stage filled from `usda` (a root layer's text), once: ordinal 1, sealed; what
        comes after (render products, times) is written from ordinal 2."""
        if self.ordinal:
            return
        ovstage.population.open_usd_from_string(self.stage, usda, ordinal=1)
        self.ordinal = 1
        self.commit()

    @contextlib.contextmanager
    def query(self, prim_paths: Sequence[str]) -> Generator[Any]:
        """Retain a query and its path list together until the caller leaves the context.

        Args:
            prim_paths: Prim paths in the query's binding order.

        Yields:
            The query, released before its caller-owned path list.
        """
        paths = self.paths.create_path_list_from_strings(list(prim_paths))
        try:
            with self.stage.query_from_path_list(paths) as query:
                yield query
        finally:
            self.paths.destroy_path_list(paths)

    def write(self, prim_paths: Sequence[str], attribute: str, tensors: Any, **said: Any) -> None:
        """`attribute` written on `prim_paths` (a value each in `tensors`) at the ordinal
        writes are made at; `said` as `ovstage.Stage.write_attribute` takes it."""
        with self.query(prim_paths) as query:
            self.stage.write_attribute(
                query, attribute, ordinal=self.ordinal, tensors=tensors, **said
            ).wait()

    def commit(self) -> int:
        """Seal pending writes and return their ordinal; all consumers share the next write ordinal."""
        ordinal = self.ordinal
        self.stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
        self.ordinal += 1
        return ordinal

    def close(self) -> None:
        """Release borrowers, then the path dictionary and stage."""
        self.resources.close()
        self.stage = self.paths = None
