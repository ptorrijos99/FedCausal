"""Flower transport for FedGES (``ClientApp`` / ``ServerApp`` Message API).

``flwr`` is an optional dependency: importing this subpackage requires it. The
in-process engine in :mod:`fedcausal.federated` never imports ``flwr``, so the
package works without it; ``pip install fedcausal[flower]`` adds the transport.

Simulation additionally needs Flower's Ray-based engine
(``fedcausal[flower-simulation]``), which the deployment path does not.
"""

from __future__ import annotations

try:
    import flwr as _flwr  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without flwr
    raise ModuleNotFoundError(
        "fedcausal.flower requires the optional 'flwr' dependency. "
        "Install it with: pip install fedcausal[flower]"
    ) from exc

from fedcausal.flower.app import build_client_app, build_server_app
from fedcausal.flower.data import InMemoryPartitions, NodeConfigLoader, PartitionLoader
from fedcausal.flower.serde import (
    control_config,
    dag_to_record,
    record_to_dag,
    schema_fingerprint,
)
from fedcausal.flower.simulation import run_flower_fedges

__all__ = [
    "build_client_app",
    "build_server_app",
    "run_flower_fedges",
    "PartitionLoader",
    "InMemoryPartitions",
    "NodeConfigLoader",
    "dag_to_record",
    "record_to_dag",
    "control_config",
    "schema_fingerprint",
]
