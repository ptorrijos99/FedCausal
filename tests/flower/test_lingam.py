"""FedRCD over Flower: the same answer as in one process, and the runs it refuses."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flwr")

from fedcausal.data import ContinuousDataset, contiguous_split  # noqa: E402
from fedcausal.exceptions import (  # noqa: E402
    ClientFailureError,
    ConfigurationError,
    SchemaMismatchError,
)
from fedcausal.flower.lingam import handle_train_message, run_server_loop  # noqa: E402
from fedcausal.flower.loaders import InMemoryPartitions  # noqa: E402
from fedcausal.lingam import cumulants_from_sums, fedrcd, fedrcd_exact, moment_sums  # noqa: E402

from ._stubs import StubGrid, continuous_chain, permuted_continuous_chain  # noqa: E402


def _grid(data: ContinuousDataset, n: int = 3, **kwargs: object) -> StubGrid:
    parts = contiguous_split(data, n)
    return StubGrid(handle_train_message, [InMemoryPartitions(parts) for _ in range(n)], **kwargs)  # type: ignore[arg-type]


def test_every_method_gives_the_in_process_answer() -> None:
    """The permuted chain makes the source a different column at every exact round."""
    for data in (continuous_chain(1), permuted_continuous_chain(1, perm=[3, 2, 1, 0])):
        parts = np.array_split(data.values, 3)
        for method, order in [("deflation", 4), ("ratio", 3), ("sort", 3), ("exact", 4)]:
            if method == "exact":
                expected = fedrcd_exact(parts)
            else:
                sums = [moment_sums(p, order) for p in parts]
                expected = fedrcd(cumulants_from_sums(sums), method=method, order=order)  # type: ignore[arg-type]
            grid = _grid(data)
            actual = run_server_loop(grid, n_clients=3, method=method, order=order)  # type: ignore[arg-type]
            assert actual.causal_order == expected.causal_order, method
            np.testing.assert_allclose(actual.coef, expected.coef, rtol=1e-10, atol=1e-10)
            assert grid.call_count == (data.n_vars - 1 if method == "exact" else 1)


def test_the_server_refuses_what_would_give_a_wrong_answer() -> None:
    three, four = continuous_chain(0, n_vars=3), continuous_chain(0, n_vars=4)
    values = np.random.default_rng(0).normal(size=(500, 3))
    forward = ContinuousDataset(values, ["a", "b", "c"])
    # Same names in another order: the arrays have the same shapes, so only the
    # fingerprint catches it.
    swapped = ContinuousDataset(values[:, [1, 0, 2]], ["b", "a", "c"])
    for a, b in [(three, four), (forward, swapped)]:
        for method in ("deflation", "exact"):
            grid = StubGrid(handle_train_message, [lambda _c, a=a: a, lambda _c, b=b: b])  # type: ignore[list-item]
            with pytest.raises(SchemaMismatchError, match="disagree on their variables"):
                run_server_loop(grid, n_clients=2, method=method)  # type: ignore[arg-type]
    # A dropout must not quietly shrink the ensemble, unless asked to tolerate it.
    for method in ("deflation", "exact"):
        with pytest.raises(ClientFailureError, match="needs 3"):
            run_server_loop(_grid(three, fail_nodes={1}), n_clients=3, method=method)  # type: ignore[arg-type]
    assert run_server_loop(_grid(three, fail_nodes={1}), n_clients=3, min_clients=2)  # type: ignore[arg-type]
    for kwargs, match in [
        ({"min_clients": 0}, "min_clients"),
        ({"method": "bogus"}, "unknown method"),
        ({"order": 5}, "order must be 3 or 4"),
        ({"pruning": "lasso"}, "pruning"),
    ]:
        with pytest.raises(ConfigurationError, match=match):
            run_server_loop(_grid(three, 2), n_clients=2, **kwargs)  # type: ignore[arg-type]


@pytest.mark.slow
def test_the_flower_simulation_gives_the_in_process_answer() -> None:
    pytest.importorskip("ray", reason="Flower's simulation engine needs Ray")
    from fedcausal.flower.lingam import run_flower_rcd

    data = continuous_chain(1)
    parts = np.array_split(data.values, 3)
    expected = fedrcd(cumulants_from_sums([moment_sums(p) for p in parts]))
    actual = run_flower_rcd(contiguous_split(data, 3))
    assert actual.causal_order == expected.causal_order
    np.testing.assert_allclose(actual.coef, expected.coef, rtol=1e-12, atol=1e-12)
    exact = run_flower_rcd(contiguous_split(data, 3), method="exact")
    assert exact.causal_order == fedrcd_exact(parts).causal_order
