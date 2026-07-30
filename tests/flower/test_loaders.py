"""The loaders, held to one contract whatever dataset they build."""

from __future__ import annotations

import pytest

pytest.importorskip("flwr")

from fedcausal.data import ContinuousDataset, DiscreteDataset  # noqa: E402
from fedcausal.exceptions import ConfigurationError  # noqa: E402
from fedcausal.flower.loaders import InMemoryPartitions, NodeConfigLoader  # noqa: E402

from ._stubs import continuous_chain, discrete_chain, node_context  # noqa: E402

# One CSV per dataset type, written to the same contract: two columns, a header
# row, and rows the loader has to count correctly.
_CASES = [
    (DiscreteDataset, "A,B\nyes,no\nno,yes\nyes,yes\nno,no\n", ("A", "B"), 4),
    (ContinuousDataset, "x0,x1\n1.0,2.0\n3.0,4.0\n5.0,6.0\n", ("x0", "x1"), 3),
]


@pytest.mark.parametrize(("dataset_cls", "csv_text", "names", "n_rows"), _CASES)
def test_node_config_loader_reads_the_supernodes_own_shard(  # type: ignore[no-untyped-def]
    tmp_path, dataset_cls, csv_text, names, n_rows
) -> None:
    csv = tmp_path / "shard.csv"
    csv.write_text(csv_text, encoding="utf-8")

    loaded = NodeConfigLoader(dataset_cls)(node_context(**{"data-path": str(csv)}))

    assert loaded.names == names
    assert loaded.n_rows == n_rows


@pytest.mark.parametrize("dataset_cls", [DiscreteDataset, ContinuousDataset])
def test_node_config_loader_says_what_is_missing(dataset_cls) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ConfigurationError, match="--node-config"):
        NodeConfigLoader(dataset_cls)(node_context())


@pytest.mark.parametrize("dataset_cls", [DiscreteDataset, ContinuousDataset])
def test_node_config_loader_rejects_a_missing_file(dataset_cls) -> None:  # type: ignore[no-untyped-def]
    context = node_context(**{"data-path": "/nope/absent.csv"})
    with pytest.raises(ConfigurationError, match="does not exist"):
        NodeConfigLoader(dataset_cls)(context)


def test_node_config_loader_honours_a_custom_key(tmp_path) -> None:  # type: ignore[no-untyped-def]
    csv = tmp_path / "shard.csv"
    csv.write_text("x0,x1\n1.0,2.0\n3.0,4.0\n", encoding="utf-8")

    loaded = NodeConfigLoader(ContinuousDataset, key="site-csv")(
        node_context(**{"site-csv": str(csv)})
    )

    assert loaded.n_rows == 2


@pytest.mark.parametrize("data", [discrete_chain(), continuous_chain(0, n=200)])
def test_in_memory_partitions_indexes_by_partition_id(data) -> None:  # type: ignore[no-untyped-def]
    loader = InMemoryPartitions([data, data])

    assert loader(node_context(1)) is data


@pytest.mark.parametrize("data", [discrete_chain(), continuous_chain(0, n=200)])
def test_in_memory_partitions_rejects_an_out_of_range_id(data) -> None:  # type: ignore[no-untyped-def]
    loader = InMemoryPartitions([data])

    with pytest.raises(ConfigurationError, match="out of range"):
        loader(node_context(7))


def test_in_memory_partitions_rejects_an_empty_list() -> None:
    with pytest.raises(ConfigurationError, match="needs at least one partition"):
        InMemoryPartitions([])


def test_in_memory_partitions_says_it_is_simulation_only() -> None:
    """Without a partition id there is no simulation, so the guard names the deployment path."""
    loader = InMemoryPartitions([discrete_chain()])
    context = node_context()
    del context.node_config["partition-id"]

    with pytest.raises(ConfigurationError, match="NodeConfigLoader"):
        loader(context)
