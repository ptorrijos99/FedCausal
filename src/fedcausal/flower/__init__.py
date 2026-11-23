"""Both method families over Flower (``pip install fedcausal[flower]``).

- :mod:`fedcausal.flower.fedges`: FedGES, several rounds of build and fuse.
- :mod:`fedcausal.flower.lingam`: the FedRCD family, one round of statistics
  (``p - 1`` for ``fedrcd_exact``).

The ``run_flower_*`` helpers run a whole federation in one process on Flower's
simulation engine (``fedcausal[flower-simulation]``). A real deployment needs
neither them nor Ray; see ``docs/deployment.md``.
"""

from __future__ import annotations

try:
    import flwr as _flwr  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without flwr
    raise ModuleNotFoundError(
        "fedcausal.flower requires the optional 'flwr' dependency. "
        "Install it with: pip install fedcausal[flower]"
    ) from exc

from fedcausal.flower.fedges import run_flower_fedges
from fedcausal.flower.lingam import run_flower_rcd
from fedcausal.flower.loaders import InMemoryPartitions, NodeConfigLoader, PartitionLoader
from fedcausal.flower.serde import (
    control_config,
    dag_to_record,
    names_fingerprint,
    record_to_dag,
    record_to_sums,
    schema_fingerprint,
    sums_to_record,
)

__all__ = [
    "InMemoryPartitions",
    "NodeConfigLoader",
    "PartitionLoader",
    "control_config",
    "dag_to_record",
    "names_fingerprint",
    "record_to_dag",
    "record_to_sums",
    "run_flower_fedges",
    "run_flower_rcd",
    "schema_fingerprint",
    "sums_to_record",
]
