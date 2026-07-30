"""``NodeConfigContinuousLoader``: the continuous-side deployment loader.

The continuous counterpart of `tests/flower/test_client_handler.py`'s
`NodeConfigLoader` tests -- same structure, same assertions, so an operator
running either family does the same thing and a reviewer can tell the two
loaders are held to the same contract.
"""

from __future__ import annotations

import pytest

pytest.importorskip("flwr")

from flwr.common import Context, RecordDict  # noqa: E402

from fedcausal.exceptions import ConfigurationError  # noqa: E402
from fedcausal.lingam.flower import NodeConfigContinuousLoader  # noqa: E402


def _context(**node_config: object) -> Context:
    return Context(run_id=1, node_id=1, node_config=node_config, state=RecordDict(), run_config={})


def test_node_config_continuous_loader_reads_the_supernodes_own_shard(tmp_path) -> None:  # type: ignore[no-untyped-def]
    csv = tmp_path / "shard.csv"
    csv.write_text("x0,x1\n1.0,2.0\n3.0,4.0\n5.0,6.0\n", encoding="utf-8")

    loaded = NodeConfigContinuousLoader()(_context(**{"data-path": str(csv)}))

    assert loaded.names == ("x0", "x1")
    assert loaded.n_rows == 3


def test_node_config_continuous_loader_says_what_is_missing() -> None:
    with pytest.raises(ConfigurationError, match="--node-config"):
        NodeConfigContinuousLoader()(_context())


def test_node_config_continuous_loader_rejects_a_missing_file() -> None:
    context = _context(**{"data-path": "/nope/absent.csv"})
    with pytest.raises(ConfigurationError, match="does not exist"):
        NodeConfigContinuousLoader()(context)


def test_node_config_continuous_loader_honours_a_custom_key(tmp_path) -> None:
    csv = tmp_path / "shard.csv"
    csv.write_text("x0,x1\n1.0,2.0\n3.0,4.0\n", encoding="utf-8")

    loaded = NodeConfigContinuousLoader(key="site-csv")(_context(**{"site-csv": str(csv)}))

    assert loaded.n_rows == 2
