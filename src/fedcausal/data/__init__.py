"""The splits that simulate sites.

The datasets themselves and their encoding are causalbase's
(:mod:`causalbase.data`); the old names here still work and warn.
"""

from __future__ import annotations

import importlib
from typing import Any

from fedcausal._aliases import warn_moved as _warn_moved
from fedcausal.data.partition import contiguous_split, dirichlet_split

__all__ = ["contiguous_split", "dirichlet_split"]

_MOVED = {
    "ContinuousDataset",
    "DiscreteDataset",
    "as_continuous",
    "as_continuous_sites",
    "as_discrete",
    "as_discrete_sites",
}


def __getattr__(name: str) -> Any:
    if name in _MOVED:
        _warn_moved(f"fedcausal.data.{name}", f"causalbase.data.{name}")
        return getattr(importlib.import_module("causalbase.data"), name)
    raise AttributeError(f"module 'fedcausal.data' has no attribute {name!r}")
