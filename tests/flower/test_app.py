"""The Flower app in flower_app/ runs both families end to end, over a stub grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("flwr")

from flwr.common import Context, RecordDict  # noqa: E402

from fedcausal.data import contiguous_split  # noqa: E402
from fedcausal.flower import fedges, lingam  # noqa: E402
from fedcausal.flower.loaders import InMemoryPartitions  # noqa: E402

from ._stubs import StubGrid, continuous_chain, discrete_chain  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "flower_app"


@pytest.fixture(scope="module")
def app():  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(APP))
    try:
        import fedcausal_site.app as module

        yield module
    finally:
        sys.path.remove(str(APP))


def _server(tmp_path: Path, **run_config: object) -> Context:
    config = {"result-path": str(tmp_path / "result.json"), **run_config}
    return Context(run_id=0, node_id=0, node_config={}, state=RecordDict(), run_config=config)


def test_the_app_runs_fedrcd_and_fedges_and_writes_the_result(app, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    parts = contiguous_split(continuous_chain(1), 3)
    for method in ("deflation", "exact"):
        grid = StubGrid(lingam.handle_train_message, [InMemoryPartitions(parts)] * 3)
        app._run_fedrcd(grid, _server(tmp_path, method=method), 3, None)
        result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
        assert result["causal_order"] == [0, 1, 2, 3]
        assert len(result["coef"]) == 4
    parts = contiguous_split(discrete_chain(), 3)
    grid = StubGrid(fedges.handle_train_message, [InMemoryPartitions(parts)] * 3)
    app._run_fedges(grid, _server(tmp_path, **{"num-server-rounds": 3}), 3, None)
    edges = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["edges"]
    assert {frozenset(e) for e in edges} == {frozenset((0, 1)), frozenset((1, 2))}
