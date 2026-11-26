"""Where a site's data comes from, and how the server tells sites apart."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr")

from causalbase.data import ContinuousDataset, DiscreteDataset  # noqa: E402

from fedcausal.exceptions import ConfigurationError  # noqa: E402
from fedcausal.flower.loaders import InMemoryPartitions, NodeConfigLoader  # noqa: E402
from fedcausal.flower.serde import names_fingerprint, schema_fingerprint  # noqa: E402

from ._stubs import discrete_chain, node_context  # noqa: E402


def test_node_config_loader_reads_the_supernodes_own_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    for cls, text, rows in [
        (DiscreteDataset, "A,B\nyes,no\nno,yes\nyes,yes\nno,no\n", 4),
        (ContinuousDataset, "A,B\n1.0,2.0\n3.0,4.0\n5.0,6.0\n", 3),
    ]:
        csv = tmp_path / "shard.csv"
        csv.write_text(text, encoding="utf-8")
        loaded = NodeConfigLoader(cls)(node_context(**{"data-path": str(csv)}))
        assert (loaded.names, loaded.n_rows) == (("A", "B"), rows)
        assert NodeConfigLoader(cls, key="site")(node_context(site=str(csv))).n_rows == rows
        with pytest.raises(ConfigurationError, match="--node-config"):
            NodeConfigLoader(cls)(node_context())
        with pytest.raises(ConfigurationError, match="does not exist"):
            NodeConfigLoader(cls)(node_context(**{"data-path": str(tmp_path / "absent.csv")}))


def test_in_memory_partitions_serve_simulations_only() -> None:
    data = discrete_chain()
    assert InMemoryPartitions([data, data])(node_context(1)) is data
    with pytest.raises(ConfigurationError, match="out of range"):
        InMemoryPartitions([data])(node_context(7))
    with pytest.raises(ConfigurationError, match="at least one partition"):
        InMemoryPartitions([])
    context = node_context()
    del context.node_config["partition-id"]  # no partition id: a deployment
    with pytest.raises(ConfigurationError, match="NodeConfigLoader"):
        InMemoryPartitions([data])(context)


def test_fingerprints_see_the_variables_and_not_the_rows() -> None:
    def dataset(names: list[str], cards: list[int]) -> DiscreteDataset:
        levels = [[str(i) for i in range(c)] for c in cards]
        return DiscreteDataset(np.zeros((4, len(names))), cards, names, levels)

    base = dataset(["X", "Y"], [2, 3])
    other_rows = dataset(["X", "Y"], [2, 3])
    other_rows.values[0, 0] = 1
    assert schema_fingerprint(base) == schema_fingerprint(other_rows)
    for other in (
        dataset(["X", "Z"], [2, 3]),
        dataset(["X", "Y"], [2, 4]),
        dataset(["Y", "X"], [3, 2]),
    ):
        assert schema_fingerprint(other) != schema_fingerprint(base)
    assert names_fingerprint(["a", "b"]) != names_fingerprint(["b", "a"])
