from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fedcausal.io import forward_sample, load_xbif

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
ASIA_XBIF = FIXTURES / "networks" / "asia.xbif"


def test_asia_has_eight_nodes_and_edges() -> None:
    bn = load_xbif(ASIA_XBIF)
    assert bn.dag.num_nodes == 8
    assert bn.dag.num_edges == 8
    assert len(bn.names) == 8
    assert len(bn.outcomes) == 8
    assert len(bn.cpts) == 8


def test_asia_names_are_the_xbif_variable_names() -> None:
    bn = load_xbif(ASIA_XBIF)
    assert set(bn.names) == {
        "lung",
        "smoke",
        "bronc",
        "xray",
        "tub",
        "asia",
        "dysp",
        "either",
    }


def test_asia_smoke_to_lung_edge_present_by_name() -> None:
    bn = load_xbif(ASIA_XBIF)
    smoke_idx = bn.names.index("smoke")
    lung_idx = bn.names.index("lung")
    assert bn.dag.has_edge(smoke_idx, lung_idx)


def test_asia_outcomes_preserve_xbif_order() -> None:
    bn = load_xbif(ASIA_XBIF)
    for outcomes in bn.outcomes:
        # asia.xbif declares every OUTCOME as "yes" then "no" -- must NOT be
        # resorted into ["no", "yes"].
        assert outcomes == ["yes", "no"]


def test_asia_cpt_rows_sum_to_one() -> None:
    bn = load_xbif(ASIA_XBIF)
    for j, cpt in enumerate(bn.cpts):
        row_sums = cpt.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"variable {bn.names[j]!r} has CPT rows not summing to 1: {row_sums}"
        )


def test_asia_cpt_shapes_match_parent_configs() -> None:
    bn = load_xbif(ASIA_XBIF)
    smoke_idx = bn.names.index("smoke")
    lung_idx = bn.names.index("lung")
    assert bn.parent_order[smoke_idx] == ()
    assert bn.cpts[smoke_idx].shape == (1, 2)
    assert bn.parent_order[lung_idx] == (smoke_idx,)
    assert bn.cpts[lung_idx].shape == (2, 2)

    either_idx = bn.names.index("either")
    dysp_idx = bn.names.index("dysp")
    assert bn.cpts[either_idx].shape == (4, 2)
    assert bn.cpts[dysp_idx].shape == (4, 2)


def test_asia_lung_given_smoke_cpt_values() -> None:
    bn = load_xbif(ASIA_XBIF)
    smoke_idx = bn.names.index("smoke")
    lung_idx = bn.names.index("lung")
    assert bn.parent_order[lung_idx] == (smoke_idx,)
    # <TABLE>0.1 0.9 0.01 0.99</TABLE>: row 0 = P(lung | smoke=yes), row 1 =
    # P(lung | smoke=no), columns in declared outcome order [yes, no].
    np.testing.assert_allclose(bn.cpts[lung_idx], [[0.1, 0.9], [0.01, 0.99]])


def test_asia_dysp_cpt_values_pin_parent_order() -> None:
    bn = load_xbif(ASIA_XBIF)
    bronc_idx = bn.names.index("bronc")
    either_idx = bn.names.index("either")
    dysp_idx = bn.names.index("dysp")
    # dysp's <DEFINITION> declares <GIVEN>bronc</GIVEN> then
    # <GIVEN>either</GIVEN>, so parent_order must preserve that order.
    assert bn.parent_order[dysp_idx] == (bronc_idx, either_idx)

    # <TABLE>0.9 0.1 0.8 0.2 0.7 0.3 0.1 0.9</TABLE> reshapes to (4, 2)
    # row-major with the first declared GIVEN (bronc) slowest and the
    # second (either) fastest; outcome index 0 = "yes" for both parents
    # (asia.xbif declares every OUTCOME as "yes" then "no"). These four
    # rows are the published Lauritzen & Spiegelhalter (1988) asia values
    # for dysp | (bronc, either): P(dysp=yes | bronc, either) is 0.9, 0.8,
    # 0.7, 0.1 for (yes,yes), (yes,no), (no,yes), (no,no) respectively.
    # Crucially the (yes,no) and (no,yes) rows differ (0.8 vs 0.7), so this
    # test can only pass under the correct row-major/first-parent-slowest
    # convention -- a transposed convention would swap those two rows and
    # fail the assertions below. (Verified by temporarily reversing the
    # parent tuple used in the row-index computation and confirming this
    # test fails.)
    parents = bn.parent_order[dysp_idx]
    cards = [len(bn.outcomes[p]) for p in parents]

    def row_of(bronc_outcome: int, either_outcome: int) -> int:
        config = (bronc_outcome, either_outcome)
        idx = 0
        for outcome, card in zip(config, cards, strict=True):
            idx = idx * card + outcome
        return idx

    cpt = bn.cpts[dysp_idx]
    # bronc=yes (0), either=yes (0) -> P(dysp=yes)=0.9, P(dysp=no)=0.1
    np.testing.assert_allclose(cpt[row_of(0, 0)], [0.9, 0.1])
    # bronc=yes (0), either=no (1) -> P(dysp=yes)=0.8, P(dysp=no)=0.2
    np.testing.assert_allclose(cpt[row_of(0, 1)], [0.8, 0.2])
    # bronc=no (1), either=yes (0) -> P(dysp=yes)=0.7, P(dysp=no)=0.3
    np.testing.assert_allclose(cpt[row_of(1, 0)], [0.7, 0.3])
    # bronc=no (1), either=no (1) -> P(dysp=yes)=0.1, P(dysp=no)=0.9
    np.testing.assert_allclose(cpt[row_of(1, 1)], [0.1, 0.9])


def test_asia_either_is_deterministic_or() -> None:
    bn = load_xbif(ASIA_XBIF)
    lung_idx = bn.names.index("lung")
    tub_idx = bn.names.index("tub")
    either_idx = bn.names.index("either")
    # either's <DEFINITION> declares <GIVEN>lung</GIVEN> then
    # <GIVEN>tub</GIVEN>.
    assert bn.parent_order[either_idx] == (lung_idx, tub_idx)

    # <TABLE>1.0 0.0 1.0 0.0 1.0 0.0 0.0 1.0</TABLE> reshapes to (4, 2)
    # row-major with lung slowest, tub fastest: either is a deterministic
    # OR of its parents (P(either=yes)=1.0 unless both lung and tub are
    # "no"). This is order-insensitive (OR is symmetric in its parents,
    # unlike dysp above) but still pins the exact values for this second
    # multi-parent node.
    cpt = bn.cpts[either_idx]
    configs = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (lung_outcome, tub_outcome)
    for row, (lung_outcome, tub_outcome) in enumerate(configs):
        expected_yes = 1.0 if lung_outcome == 0 or tub_outcome == 0 else 0.0
        np.testing.assert_allclose(cpt[row], [expected_yes, 1.0 - expected_yes])


def test_load_xbif_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_xbif(FIXTURES / "does_not_exist.xbif")


def test_forward_sample_returns_discrete_dataset() -> None:
    bn = load_xbif(ASIA_XBIF)
    ds = forward_sample(bn, 100, seed=0)
    assert ds.n_rows == 100
    assert ds.n_vars == 8


def test_forward_sample_uses_sorted_categories() -> None:
    bn = load_xbif(ASIA_XBIF)
    ds = forward_sample(bn, 100, seed=0)
    for cats in ds.categories:
        assert list(cats) == sorted(cats)
    assert list(ds.cardinalities) == [2] * 8


def test_forward_sample_deterministic_under_fixed_seed() -> None:
    bn = load_xbif(ASIA_XBIF)
    ds1 = forward_sample(bn, 500, seed=42)
    ds2 = forward_sample(bn, 500, seed=42)
    assert np.array_equal(ds1.values, ds2.values)
    assert ds1.categories == ds2.categories


def test_forward_sample_different_seeds_differ() -> None:
    bn = load_xbif(ASIA_XBIF)
    ds1 = forward_sample(bn, 500, seed=1)
    ds2 = forward_sample(bn, 500, seed=2)
    assert not np.array_equal(ds1.values, ds2.values)


def test_forward_sample_asia_marginal_matches_known_ground_truth() -> None:
    # Known ground truth for the asia network: P(asia = yes) = 0.01.
    bn = load_xbif(ASIA_XBIF)
    ds = forward_sample(bn, 50_000, seed=0)
    asia_idx = ds.names.index("asia")
    yes_code = ds.categories[asia_idx].index("yes")
    p_yes = float(np.mean(ds.values[:, asia_idx] == yes_code))
    assert p_yes == pytest.approx(0.01, abs=0.005)
