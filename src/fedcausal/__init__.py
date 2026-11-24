"""Federated causal structure learning.

Four estimators cover the common cases:

* :class:`GES` learns a causal structure from discrete data in one place, and
  :class:`XGES` does so with a search that removes edges as it goes.
* :class:`FedGES` learns one from discrete data held by several sites, which
  exchange only structures and scores.
* :class:`FedRCD` learns a linear non-Gaussian model from continuous data held
  by several sites, which exchange only moment sums.

Each follows scikit-learn's conventions: hyperparameters go to the
constructor, ``fit`` learns, and results end in an underscore.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fedcausal.fedges import FedGES
    from fedcausal.graph import DAG, PDAG
    from fedcausal.lingam import FedRCD
    from fedcausal.search import GES, XGES

__version__ = "1.0.0"

__all__ = ["DAG", "GES", "PDAG", "XGES", "FedGES", "FedRCD", "__version__"]

# Imported on first use, so that a subpackage (fedcausal.lingam, say) loads
# without the others.
_HOMES = {
    "DAG": "fedcausal.graph",
    "PDAG": "fedcausal.graph",
    "GES": "fedcausal.search",
    "XGES": "fedcausal.search",
    "FedGES": "fedcausal.fedges",
    "FedRCD": "fedcausal.lingam",
}


def __getattr__(name: str) -> Any:
    if name in _HOMES:
        value = getattr(importlib.import_module(_HOMES[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'fedcausal' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
