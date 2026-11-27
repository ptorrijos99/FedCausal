"""Federated causal structure learning, on top of causalbase.

Two estimators cover the common cases:

* :class:`FedGES` learns a causal structure from discrete data held by several
  sites, which exchange only structures and scores.
* :class:`FedRCD` learns a linear non-Gaussian model from continuous data held
  by several sites, which exchange only moment sums.

Each follows scikit-learn's conventions: hyperparameters go to the
constructor, ``fit`` learns, and results end in an underscore. Learning from
data in one place (GES, XGES, the scores, graphs and metrics the federated
methods build on) is the job of the ``causalbase`` package.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fedcausal.fedges import FedGES
    from fedcausal.lingam import FedRCD

__version__ = "1.0.0"

__all__ = ["FedGES", "FedRCD", "__version__"]

# Imported on first use, so that a subpackage (fedcausal.lingam, say) loads
# without the others.
_HOMES = {
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
