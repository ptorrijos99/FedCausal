import math

import numpy as np

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG
from fedcausal.scores import BDeuScore


def _brute_force_bdeu_likelihood(values, cards, node, parents, ess):
    """Independent reference: dict-of-counts + math.lgamma (no shared code with the kernel)."""
    r = int(cards[node])
    q = int(np.prod([cards[p] for p in parents])) if parents else 1
    counts: dict[tuple, np.ndarray] = {}
    for row in values:
        key = tuple(row[p] for p in parents)
        counts.setdefault(key, np.zeros(r))[row[node]] += 1
    a_j, a_jk = ess / q, ess / (q * r)
    total = 0.0
    for n_k in counts.values():
        total += math.lgamma(a_j) - math.lgamma(a_j + n_k.sum())
        total += sum(math.lgamma(a_jk + c) - math.lgamma(a_jk) for c in n_k)
    return total


def _brute_force_prior_term(num_parents, structure_prior, n_rows):
    """Independent reference port of Tetrad's getPriorForStructure."""
    e = structure_prior
    if e == 0:
        return 0.0
    vm = n_rows - 1
    return num_parents * math.log(e / vm) + (vm - num_parents) * math.log(1.0 - (e / vm))


def test_kernel_matches_brute_force() -> None:
    rng = np.random.default_rng(33)
    values = rng.integers(0, 3, size=(200, 5)).astype(np.int32)
    ds = DiscreteDataset.from_arrays(
        values,
        np.full(5, 3, np.int32),
        [f"X{i}" for i in range(5)],
        [["0", "1", "2"]] * 5,
    )
    s0 = BDeuScore(ds, sample_prior=10.0, structure_prior=0.0)  # prior term is 0 at sp=0
    for node in range(5):
        for parents in [(), (0,), (1, 3), (0, 2, 4)]:
            if node in parents:
                continue
            expected = _brute_force_bdeu_likelihood(values, ds.cardinalities, node, parents, 10.0)
            got = s0.local_score(node, tuple(sorted(parents)))
            prior = got - expected  # must be count-independent
            got2 = s0.local_score(node, tuple(sorted(parents)))
            assert math.isclose(got, got2)
            assert math.isclose(got - prior, expected, rel_tol=1e-9)
            assert math.isclose(prior, 0.0, abs_tol=1e-9)


def test_matches_brute_force_with_nonzero_priors() -> None:
    rng = np.random.default_rng(7)
    values = rng.integers(0, 3, size=(150, 4)).astype(np.int32)
    ds = DiscreteDataset.from_arrays(
        values,
        np.full(4, 3, np.int32),
        [f"X{i}" for i in range(4)],
        [["0", "1", "2"]] * 4,
    )
    s = BDeuScore(ds, sample_prior=10.0, structure_prior=0.001)
    for node in range(4):
        for parents in [(), (0,), (1, 2), (0, 1, 3)]:
            if node in parents:
                continue
            lik = _brute_force_bdeu_likelihood(values, ds.cardinalities, node, parents, 10.0)
            prior = _brute_force_prior_term(len(parents), 0.001, ds.n_rows)
            got = s.local_score(node, tuple(sorted(parents)))
            assert math.isclose(got, lik + prior, rel_tol=1e-9)


def test_prior_cancels_between_equal_sized_parent_sets() -> None:
    rng = np.random.default_rng(11)
    values = rng.integers(0, 2, size=(80, 5)).astype(np.int32)
    ds = DiscreteDataset.from_arrays(
        values,
        np.full(5, 2, np.int32),
        [f"X{i}" for i in range(5)],
        [["0", "1"]] * 5,
    )
    s = BDeuScore(ds, sample_prior=10.0, structure_prior=0.05)
    # Two equal-sized parent sets: the structure-prior term cancels in the difference.
    diff_scored = s.local_score(0, (1, 2)) - s.local_score(0, (3, 4))
    lik_a = _brute_force_bdeu_likelihood(values, ds.cardinalities, 0, (1, 2), 10.0)
    lik_b = _brute_force_bdeu_likelihood(values, ds.cardinalities, 0, (3, 4), 10.0)
    assert math.isclose(diff_scored, lik_a - lik_b, rel_tol=1e-9)


def test_sparse_path_matches_brute_force() -> None:
    # High-cardinality parents so that q * r exceeds the dense threshold (4 * n_rows),
    # forcing the unique()-based sparse counting path.
    rng = np.random.default_rng(42)
    cards = np.array([4, 8, 8, 8], np.int32)
    n_rows = 60
    values = np.column_stack([rng.integers(0, int(c), size=n_rows) for c in cards]).astype(np.int32)
    ds = DiscreteDataset.from_arrays(
        values,
        cards,
        [f"X{i}" for i in range(4)],
        [[str(k) for k in range(int(c))] for c in cards],
    )
    node, parents = 0, (1, 2, 3)
    q = 8 * 8 * 8
    r = 4
    assert q * r > 4 * n_rows  # confirm we are on the sparse path
    s = BDeuScore(ds, sample_prior=10.0, structure_prior=0.0)
    expected = _brute_force_bdeu_likelihood(values, ds.cardinalities, node, parents, 10.0)
    got = s.local_score(node, parents)
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_defaults() -> None:
    """The defaults are the ones documented, and structure_prior is on."""
    values = np.zeros((4, 2), np.int32)
    ds = DiscreteDataset.from_arrays(values, np.full(2, 2, np.int32), ["A", "B"], [["0", "1"]] * 2)
    s = BDeuScore(ds)
    assert s.sample_prior == 1.0
    assert s.structure_prior == 1.0


def test_reference_priors_are_still_reachable() -> None:
    """The original implementation's effective priors remain one argument away.

    The default changed because it recovers better structure, but reproducing
    published FedGES numbers must stay possible, and the regression tests rely
    on it.
    """
    values = np.zeros((4, 2), np.int32)
    ds = DiscreteDataset.from_arrays(values, np.full(2, 2, np.int32), ["A", "B"], [["0", "1"]] * 2)
    s = BDeuScore(ds, sample_prior=1.0, structure_prior=0.0)
    assert (s.sample_prior, s.structure_prior) == (1.0, 0.0)


def test_cache_and_order_invariance() -> None:
    rng = np.random.default_rng(1)
    values = rng.integers(0, 2, size=(50, 4)).astype(np.int32)
    ds = DiscreteDataset.from_arrays(
        values, np.full(4, 2, np.int32), list("WXYZ"), [["0", "1"]] * 4
    )
    s = BDeuScore(ds)
    assert s.local_score(0, (2, 1)) == s.local_score(0, (1, 2))
    hits, misses = s.cache_stats()
    assert hits >= 1


def test_cache_disabled() -> None:
    rng = np.random.default_rng(2)
    values = rng.integers(0, 2, size=(30, 3)).astype(np.int32)
    ds = DiscreteDataset.from_arrays(values, np.full(3, 2, np.int32), list("ABC"), [["0", "1"]] * 3)
    s = BDeuScore(ds, cache=False)
    a = s.local_score(0, (1,))
    b = s.local_score(0, (1,))
    assert a == b
    hits, misses = s.cache_stats()
    assert hits == 0


def test_score_dag_sums_local_scores() -> None:
    rng = np.random.default_rng(5)
    values = rng.integers(0, 3, size=(120, 4)).astype(np.int32)
    ds = DiscreteDataset.from_arrays(
        values,
        np.full(4, 3, np.int32),
        [f"X{i}" for i in range(4)],
        [["0", "1", "2"]] * 4,
    )
    s = BDeuScore(ds)
    # 0 -> 2, 1 -> 2, 2 -> 3
    dag = DAG.from_edges(4, [(0, 2), (1, 2), (2, 3)])
    expected = (
        s.local_score(0, ())
        + s.local_score(1, ())
        + s.local_score(2, (0, 1))
        + s.local_score(3, (2,))
    )
    assert math.isclose(s.score_dag(dag), expected, rel_tol=1e-12)
