"""Old import paths of the modules that moved to causalbase.

fedcausal 0.x held the core of structure learning itself. That core is now the
causalbase package, and ``fedcausal.graph``, ``fedcausal.scores``, ... (with
every submodule) are aliases of ``causalbase.graph``, ``causalbase.scores``, ...:
importing an old path returns the causalbase module itself, so classes are the
same objects under both names, and warns once with the path to use instead.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from types import ModuleType
from typing import Any

__all__ = ["MOVED", "install"]

# Old top-level subpackages of fedcausal that are now causalbase's, whole.
MOVED = ("graph", "scores", "search", "metrics", "io", "datasets")
# Submodules of packages that stayed, whose contents moved.
MOVED_MODULES = {
    "fedcausal.data.dataset": "causalbase.data.dataset",
    "fedcausal.data.continuous": "causalbase.data.continuous",
}


def new_name(old: str) -> str | None:
    """Return the causalbase module that `old` stands for, or None if it did not move."""
    if old in MOVED_MODULES:
        return MOVED_MODULES[old]
    parts = old.split(".")
    if len(parts) >= 2 and parts[0] == "fedcausal" and parts[1] in MOVED:
        return ".".join(["causalbase", *parts[1:]])
    return None


def warn_moved(old: str, new: str, stacklevel: int = 3) -> None:
    """Warn that `old` is deprecated in favour of `new`."""
    warnings.warn(
        f"{old} moved to {new}; import it from there (the old path goes in fedcausal 2.0)",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, new: str) -> None:
        self.new = new

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        warn_moved(spec.name, self.new, stacklevel=6)
        return importlib.import_module(self.new)

    def exec_module(self, module: ModuleType) -> None:
        # The module is causalbase's, already executed; the import system has
        # registered it under the old name too.
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        new = new_name(fullname)
        if new is None:
            return None
        target_spec = importlib.util.find_spec(new)
        if target_spec is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            _AliasLoader(new),
            is_package=target_spec.submodule_search_locations is not None,
        )


def install() -> None:
    """Put the alias finder in front of the import system, once."""
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
