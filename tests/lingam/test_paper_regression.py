"""The ported solvers must reproduce the original FedLiNGAM implementation.

The fixtures come from the code the PGM 2026 paper links to; see
tests/fixtures/lingam/README.md. A failure here means the port changed the
paper's results and must be fixed, not re-baselined.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fedcausal.lingam import (
    fedhc,
    fedishc,
    fedrcd,
    fedrcd_exact,
    fedrcd_noniterative,
    hc,
    hc_lingam,
    ishc_beta,
)
from fedcausal.lingam.cumulants import aggregate_and_compute_cumulants, compute_local_raw_stats

FIXTURE = Path(__file__).parent.parent / "fixtures" / "lingam" / "rcd_baseline.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
# The cumulants belong to the problem, not the method, so one record per problem.
PROBLEMS = list({case["name"]: case for case in CASES}.values())

# Federated solvers: consume aggregated cumulant stats.
SOLVERS = {
    "fedrcd": fedrcd,
    "fedrcd_noniterative": fedrcd_noniterative,
    "ishc": ishc_beta,
    "fedishc": fedishc,
    "fedhc": fedhc,
}

# Centralised solvers: consume the pooled data matrix directly, not stats.
CENTRALISED_SOLVERS = {
    "hc_lingam": hc_lingam,
    "hc": hc,
}


def make_lingam_data(seed: int, n_samples: int, n_vars: int) -> np.ndarray:
    """Regenerate a recorded problem.

    Must reproduce ``tests/fixtures/lingam/record_baseline.py``'s generator
    exactly: the RNG is stateful, so the *sequence* of draws matters, and
    batching or reordering them produces different data.
    """
    rng = np.random.default_rng(seed)
    b_true = np.zeros((n_vars, n_vars))
    for child in range(1, n_vars):
        for parent in range(child):
            if rng.random() < 0.5:
                b_true[child, parent] = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
    noise = rng.laplace(0.0, 1.0, size=(n_samples, n_vars))
    x = np.zeros((n_samples, n_vars))
    for j in range(n_vars):
        x[:, j] = x @ b_true[j] + noise[:, j]
    return x


def _data(case: dict) -> np.ndarray:
    return make_lingam_data(case["seed"], case["n_samples"], case["n_vars"])


def _clients(case: dict) -> list[np.ndarray]:
    return list(np.array_split(_data(case), case["n_clients"], axis=0))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['name']}-{c['method']}")
def test_port_reproduces_the_original(case):
    method = case["method"]
    if method == "fedrcd_exact":
        result = fedrcd_exact(_clients(case))
    elif method in CENTRALISED_SOLVERS:
        # hc_lingam/hc work on the pooled data matrix, not aggregated stats.
        result = CENTRALISED_SOLVERS[method](_data(case))
    else:
        stats = aggregate_and_compute_cumulants(
            [compute_local_raw_stats(c, order=4) for c in _clients(case)]
        )
        result = SOLVERS[method](stats)

    assert list(result.causal_order) == case["causal_order"]

    if case["B"] is None:
        # fedhc's original (`hc()` in fedishc/core.py, iterative=False)
        # returns None for B: the no-deflation ablation deliberately leaves B
        # estimation to the caller, so there is nothing recorded from the
        # original to bit-check the port's B against. Only the causal order
        # is pinned for this method.
        return

    np.testing.assert_allclose(result.b, np.array(case["B"]), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("case", PROBLEMS, ids=lambda c: c["name"])
def test_cumulants_match_the_recording(case):
    """Pin the aggregated cumulants themselves, not just the order they induce.

    Once the causal order is fixed, ``B`` is re-estimated from ``cov`` alone
    -- no solver ever reads ``c3``/``c4``/etc. again. So a subtle bias-
    correction bug in ``cumulants.py`` only shows up in
    ``test_port_reproduces_the_original`` if it happens to be large enough to
    flip the tau ranking; a smaller error would pass that gate silently. This
    test checks the eight aggregated arrays directly, with no regression step
    in between to absorb a discrepancy.
    """
    stats = aggregate_and_compute_cumulants(
        [compute_local_raw_stats(c, order=4) for c in _clients(case)]
    )
    for key in ("cov", "c3", "c12", "c21", "c4", "c13", "c31", "c22"):
        np.testing.assert_allclose(
            stats[key], np.array(case["cumulants"][key]), rtol=1e-12, atol=1e-12
        )
