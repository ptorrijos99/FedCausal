"""The bundled networks load, and their ground truth lines up with their data."""

from __future__ import annotations

import pytest

from fedcausal.datasets import (
    NetworkBundle,
    list_networks,
    load_alarm,
    load_asia,
    load_cancer,
    load_child,
    load_insurance,
    load_network,
)
from fedcausal.graph import DAG
from fedcausal.io import DiscreteBN

ALL_NETWORKS = list(list_networks())


def test_list_networks_is_ordered_smallest_first() -> None:
    sizes = list(list_networks().values())
    assert sizes == sorted(sizes)


def test_list_networks_returns_a_copy() -> None:
    """Callers must not be able to mutate the catalogue out from under the loaders."""
    listing = list_networks()
    listing["not_a_network"] = 0
    assert "not_a_network" not in list_networks()


@pytest.mark.parametrize("name", ALL_NETWORKS)
def test_every_bundled_network_loads(name: str) -> None:
    bundle = load_network(name)

    assert isinstance(bundle, NetworkBundle)
    assert bundle.name == name
    assert bundle.data.n_rows == 5000
    assert bundle.data.n_vars == list_networks()[name]
    assert bundle.truth.num_nodes == bundle.data.n_vars
    assert bundle.truth.edges()


@pytest.mark.parametrize("name", ALL_NETWORKS)
def test_truth_names_the_same_edges_as_the_network(name: str) -> None:
    """`truth` must describe the same edges the network does, by name."""
    bundle = load_network(name)

    by_xbif_order = {
        (bundle.network.names[u], bundle.network.names[v]) for u, v in bundle.network.dag.edges()
    }
    by_data_order = {(bundle.data.names[u], bundle.data.names[v]) for u, v in bundle.truth.edges()}

    assert by_xbif_order == by_data_order


@pytest.mark.parametrize("name", ALL_NETWORKS)
def test_bundled_xbif_and_csv_agree_on_column_order(name: str) -> None:
    """Pin the precondition that makes the test above weaker than it looks.

    For all five bundled networks the .xbif declares its variables in the same
    order the CSV does, so `load_network`'s re-indexing is the identity and
    dropping it would change nothing. This records that, so the next network
    added with a differing order fails here loudly rather than making
    `test_truth_survives_a_reordered_xbif` the only thing standing between a
    user and a silently wrong comparison.
    """
    bundle = load_network(name)
    assert list(bundle.network.names) == list(bundle.data.names)


def test_truth_survives_a_reordered_xbif(monkeypatch: pytest.MonkeyPatch) -> None:
    """A .xbif declaring its variables in another order must still give the same truth.

    The bundled files happen to agree with their CSVs, so this forces the
    disagreement the re-indexing exists for. Without it the returned `truth`
    would carry the .xbif's node numbering, and every comparison against a
    graph learned from `data` would silently score the wrong nodes.
    """
    real = load_cancer().network
    order = list(reversed(range(len(real.names))))
    shuffled_names = [real.names[i] for i in order]
    position = {name: new for new, name in enumerate(shuffled_names)}
    shuffled = DiscreteBN(
        dag=DAG.from_edges(
            real.dag.num_nodes,
            [(position[real.names[u]], position[real.names[v]]) for u, v in real.dag.edges()],
        ),
        names=shuffled_names,
        # `load_network` builds `truth` from `names` and `dag` alone; the
        # remaining fields ride along untouched and are not permuted here.
        outcomes=real.outcomes,
        parent_order=real.parent_order,
        cpts=real.cpts,
    )
    monkeypatch.setattr("fedcausal.datasets.load_xbif", lambda _path: shuffled)

    bundle = load_network("cancer")

    expected = {(real.names[u], real.names[v]) for u, v in real.dag.edges()}
    got = {(bundle.data.names[u], bundle.data.names[v]) for u, v in bundle.truth.edges()}
    assert got == expected


@pytest.mark.parametrize("name", ALL_NETWORKS)
def test_truth_is_acyclic(name: str) -> None:
    truth = load_network(name).truth
    assert len(truth.topological_order()) == truth.num_nodes


def test_unknown_network_names_the_available_ones() -> None:
    with pytest.raises(ValueError, match="unknown network 'munin'") as excinfo:
        load_network("munin")

    # A bare "unknown network" leaves the caller guessing; the message has to
    # carry the catalogue or it is not actionable.
    for name in ALL_NETWORKS:
        assert name in str(excinfo.value)


@pytest.mark.parametrize(
    ("loader", "name"),
    [
        (load_cancer, "cancer"),
        (load_asia, "asia"),
        (load_child, "child"),
        (load_insurance, "insurance"),
        (load_alarm, "alarm"),
    ],
)
def test_named_loaders_match_load_network(loader, name: str) -> None:  # type: ignore[no-untyped-def]
    assert loader().name == load_network(name).name


def test_loaders_do_not_share_mutable_state() -> None:
    """A caller that mutates its dataset must not corrupt the next caller's.

    Guards against the obvious future optimisation of caching the parsed
    bundle, which would hand every caller the same numpy arrays.
    """
    first = load_cancer()
    original = int(first.data.values[0, 0])
    first.data.values[0, 0] = original + 1

    assert int(load_cancer().data.values[0, 0]) == original
