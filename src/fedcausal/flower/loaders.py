"""How a ``ClientApp`` finds its own data.

- :class:`InMemoryPartitions` picks one of several in-memory datasets by
  ``partition-id``. Simulation only: every actor holds every partition.
- :class:`NodeConfigLoader` reads a CSV named in the supernode's own
  ``node_config``, so each site opens its own file and nothing else.

Any callable matching :class:`PartitionLoader` works in their place, for
example one that reads from a database.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable

from fedcausal.data import ContinuousDataset, DiscreteDataset
from fedcausal.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from flwr.common import Context

__all__ = ["InMemoryPartitions", "NodeConfigLoader", "PartitionLoader"]

DatasetT = TypeVar("DatasetT", DiscreteDataset, ContinuousDataset)
DatasetT_co = TypeVar("DatasetT_co", covariant=True)


@runtime_checkable
class PartitionLoader(Protocol[DatasetT_co]):
    """Resolves the dataset a supernode should work from."""

    def __call__(self, context: Context) -> DatasetT_co:
        """Return this supernode's dataset."""
        ...


class InMemoryPartitions(Generic[DatasetT]):
    """Serve one of several in-memory datasets, indexed by ``partition-id``.

    Parameters
    ----------
    partitions : Sequence[DatasetT]
        One dataset per partition id.

    Raises
    ------
    ConfigurationError
        If `partitions` is empty.
    """

    def __init__(self, partitions: Sequence[DatasetT]) -> None:
        if not partitions:
            raise ConfigurationError("InMemoryPartitions needs at least one partition")
        self._partitions: list[DatasetT] = list(partitions)

    def __call__(self, context: Context) -> DatasetT:
        """Return the partition matching this supernode's ``partition-id``.

        Raises
        ------
        ConfigurationError
            If ``partition-id`` is absent from ``node_config``, or out of range
            for the partitions held.
        """
        if "partition-id" not in context.node_config:
            raise ConfigurationError(
                "node_config has no 'partition-id'; InMemoryPartitions only works "
                "in simulation. Use NodeConfigLoader for a real deployment."
            )
        pid = int(context.node_config["partition-id"])
        if not 0 <= pid < len(self._partitions):
            raise ConfigurationError(
                f"partition-id {pid} is out of range for {len(self._partitions)} partitions"
            )
        partition: DatasetT = self._partitions[pid]
        return partition


class NodeConfigLoader(Generic[DatasetT]):
    """Load a supernode's own CSV, named by a key in its ``node_config``.

    Start each supernode with ``--node-config "data-path='/path/to/shard.csv'"``.
    Gzipped CSVs work too.

    Parameters
    ----------
    dataset_cls : type[DatasetT]
        ``DiscreteDataset`` for FedGES, ``ContinuousDataset`` for FedRCD.
    key : str, optional
        The ``node_config`` key holding the path. Default ``"data-path"``.
    """

    def __init__(self, dataset_cls: type[DatasetT], key: str = "data-path") -> None:
        self.dataset_cls: type[DatasetT] = dataset_cls
        self.key = key

    def __call__(self, context: Context) -> DatasetT:
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
        loaded: DatasetT = self.dataset_cls.from_csv(path)
        return loaded
