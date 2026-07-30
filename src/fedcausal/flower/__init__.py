"""Flower transports for the federated methods in fedcausal.

``flwr`` is an optional dependency: importing this subpackage requires it.
Nothing under :mod:`fedcausal.fedges` or :mod:`fedcausal.lingam` imports it, so
the algorithms work without it; ``pip install fedcausal[flower]`` adds the
transport.

One module per method family, both built on the same plumbing:

- :mod:`fedcausal.flower.fedges` -- FedGES over discrete data, several rounds of
  build and fuse.
- :mod:`fedcausal.flower.lingam` -- the FedRCD family over continuous data, one
  round of cumulant statistics (``p - 1`` for ``fedrcd_exact``).

Both take their data through a :class:`~fedcausal.flower.loaders.PartitionLoader`
and pack their messages with :mod:`fedcausal.flower.serde`. Adding a third
family means writing a module beside these two, not a new transport.

The ``run_flower_*`` helpers below run a whole federation in one process through
Flower's Ray-based simulation engine (``fedcausal[flower-simulation]``). A real
deployment needs neither them nor Ray; see ``docs/deployment.md``.
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
from fedcausal.flower.lingam import run_flower_rcd, run_flower_rcd_exact
from fedcausal.flower.loaders import InMemoryPartitions, NodeConfigLoader, PartitionLoader
from fedcausal.flower.serde import (
    control_config,
    dag_to_record,
    names_fingerprint,
    record_to_dag,
    record_to_stats,
    schema_fingerprint,
    stats_to_record,
)

__all__ = [
    "InMemoryPartitions",
    "NodeConfigLoader",
    "PartitionLoader",
    "control_config",
    "dag_to_record",
    "names_fingerprint",
    "record_to_dag",
    "record_to_stats",
    "run_flower_fedges",
    "run_flower_rcd",
    "run_flower_rcd_exact",
    "schema_fingerprint",
    "stats_to_record",
]
