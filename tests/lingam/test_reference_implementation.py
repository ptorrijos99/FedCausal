"""fedcausal.lingam reproduces the original FedLiNGAM and HC-LiNGAM implementations.

The fixtures come from that implementation; see tests/fixtures/lingam/README.md.
If this fails, fix the code: never re-record the fixtures from it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fedcausal.lingam import fedrcd, fedrcd_exact
from fedcausal.lingam.cumulants import cumulants_from_sums, moment_sums

FIXTURE = Path(__file__).parent.parent / "fixtures" / "lingam" / "rcd_baseline.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
# The cumulants belong to the problem, not the method, so one record per problem.
PROBLEMS = list({case["name"]: case for case in CASES}.values())

# Each recorded method as a setting of the unified solvers, all with OLS
# coefficients (pruning=None), as the original estimated them. FedISHC is left out:
# the original divides Cum(x_j, x_j, x_s) by Cum(x_s, x_s, x_s), which estimates
# the square of the coefficient, and fedrcd(method="ratio") uses the coefficient.
SETTINGS = {
    "fedrcd": {},
    "fedrcd_noniterative": {"method": "sort"},
    "ishc": {"order": 3},
    "fedhc": {"method": "sort", "order": 3},
}


def make_lingam_data(seed: int, n_samples: int, n_vars: int) -> np.ndarray:
    """Regenerate a recorded problem, drawing in exactly the recorder's order."""
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


CHECKED = [case for case in CASES if case["method"] != "fedishc"]


@pytest.mark.parametrize("case", CHECKED, ids=lambda c: f"{c['name']}-{c['method']}")
def test_port_reproduces_the_original(case):
    method = case["method"]
    if method == "fedrcd_exact":
        result = fedrcd_exact(_clients(case), pruning=None)
    elif method == "hc":
        result = fedrcd_exact([_data(case)], pruning=None)
    elif method == "hc_lingam":
        pooled = cumulants_from_sums([moment_sums(_data(case))])
        result = fedrcd(pooled, method="sort", pruning=None)
    else:
        stats = cumulants_from_sums([moment_sums(c, order=4) for c in _clients(case)])
        result = fedrcd(stats, pruning=None, **SETTINGS[method])

    assert list(result.causal_order) == case["causal_order"]

    if case["B"] is None:
        # The original returns no B for fedhc.
        return

    np.testing.assert_allclose(result.coef, np.array(case["B"]), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("case", PROBLEMS, ids=lambda c: c["name"])
def test_cumulants_match_the_recording(case):
    """The aggregated cumulants match, not just the order they lead to.

    B is estimated from the covariance alone, so a small error in the higher
    cumulants could pass the other test unless it flipped the order.
    """
    stats = cumulants_from_sums([moment_sums(c, order=4) for c in _clients(case)])
    for key in ("cov", "c3", "c12", "c21", "c4", "c13", "c31", "c22"):
        np.testing.assert_allclose(
            stats[key], np.array(case["cumulants"][key]), rtol=1e-12, atol=1e-12
        )
