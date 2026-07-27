"""How a ``ClientApp`` gets hold of its own data.

A supernode must be able to load its shard without the server, or anyone else,
handing it the whole dataset. That is the difference between a simulation and a
deployment, so it is a seam rather than an assumption:
:class:`PartitionLoader` is what :func:`~fedcausal.flower.build_client_app`
takes, and two implementations ship with it.

- :class:`InMemoryPartitions` indexes a list of datasets by ``partition-id``.
  Simulation only -- every actor holds every partition.
- :class:`NodeConfigLoader` reads a path out of the supernode's own
  ``node_config``, so each node loads its shard and nothing else. This is the
  deployment path; see ``docs/flower-deployment.md``.

Anything matching the protocol works, so loading from a database, an object
store, or a hospital's own extract is a matter of writing one function.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fedcausal.data import DiscreteDataset
from fedcausal.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flwr.common import Context

__all__ = ["PartitionLoader", "InMemoryPartitions", "NodeConfigLoader"]


@runtime_checkable
class PartitionLoader(Protocol):
    """Resolves a supernode's :class:`~fedcausal.data.DiscreteDataset`."""

    def __call__(self, context: Context) -> DiscreteDataset:
        """Return the dataset this supernode should learn from."""
        ...


def _partition_id(context: Context) -> int:
    """Return the supernode's ``partition-id``.

    Raises
    ------
    ConfigurationError
        If the key is absent, which means the app is running outside a
        simulation that assigns partition ids.
    """
    if "partition-id" not in context.node_config:
        raise ConfigurationError(
            "node_config has no 'partition-id'; InMemoryPartitions only works "
            "in simulation. Use NodeConfigLoader for a real deployment."
        )
    return int(context.node_config["partition-id"])


class InMemoryPartitions:
    """Serve one of several in-memory datasets, indexed by ``partition-id``.

    For simulation. Every ``ClientApp`` actor closes over the whole list, which
    is fine at study scale and impossible in a deployment.

    Parameters
    ----------
    partitions : Sequence[DiscreteDataset]
        One dataset per partition id.

    Raises
    ------
    ConfigurationError
        If `partitions` is empty.
    """

    def __init__(self, partitions: Sequence[DiscreteDataset]) -> None:
        if not partitions:
            raise ConfigurationError("InMemoryPartitions needs at least one partition")
        self._partitions = list(partitions)

    def __call__(self, context: Context) -> DiscreteDataset:
        """Return the partition matching this supernode's ``partition-id``.

        Raises
        ------
        ConfigurationError
            If the id is out of range for the partitions held.
        """
        pid = _partition_id(context)
        if not 0 <= pid < len(self._partitions):
            raise ConfigurationError(
                f"partition-id {pid} is out of range for {len(self._partitions)} partitions"
            )
        return self._partitions[pid]


class NodeConfigLoader:
    """Load a supernode's own CSV, named by a key in its ``node_config``.

    The deployment path: start each supernode with
    ``--node-config "data-path='/path/to/my/shard.csv'"`` and it reads that file
    and nothing else. Gzipped CSVs are read directly.

    Parameters
    ----------
    key : str, optional
        Which ``node_config`` key holds the path (default ``"data-path"``).
    """

    def __init__(self, key: str = "data-path") -> None:
        self.key = key

    def __call__(self, context: Context) -> DiscreteDataset:
        """Load and return this supernode's dataset.

        Raises
        ------
        ConfigurationError
            If the key is missing from ``node_config`` or names a path that
            does not exist.
        """
        raw = context.node_config.get(self.key)
        if raw is None:
            raise ConfigurationError(
                f"node_config has no {self.key!r}; start the supernode with "
                f"--node-config \"{self.key}='/path/to/shard.csv'\""
            )
        path = Path(str(raw))
        if not path.exists():
            raise ConfigurationError(f"{self.key}={path} does not exist on this supernode")
        return DiscreteDataset.from_csv(path)
