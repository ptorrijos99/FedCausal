"""The ported solvers must reproduce the original FedLiNGAM implementation.

The fixtures come from the code the PGM 2026 paper links to; see
tests/fixtures/lingam/README.md. A failure here means the port changed the
paper's results and must be fixed, not re-baselined.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fedcausal.lingam import fedrcd, fedrcd_exact, fedrcd_noniterative, ishc_beta
from fedcausal.lingam.cumulants import aggregate_and_compute_cumulants, compute_local_raw_stats

FIXTURE = Path(__file__).parent.parent / "fixtures" / "lingam" / "rcd_baseline.json"
CASES = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]

SOLVERS = {
    "fedrcd": fedrcd,
    "fedrcd_noniterative": fedrcd_noniterative,
    "ishc": ishc_beta,
}


def make_lingam_data(seed: int, n_samples: int, n_vars: int) -> np.ndarray:
    """Regenerate a recorded problem.

    This must reproduce ``tools/record_lingam_baseline.py``'s generator exactly.
    The RNG is stateful, so the *sequence* of draws is what matters: one
    ``rng.random()`` per (child, parent) pair, followed by ``rng.uniform`` then
    ``rng.choice`` only when that draw is below 0.5, and a single ``rng.laplace``
    call after the loop finishes. Batching or reordering any of those silently
    produces different data. ``test_regenerated_data_matches_the_recording``
    below is what makes such a desync say so, instead of surfacing as a
    mysterious solver mismatch.
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


def _digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()[:16]


def _clients(case: dict) -> list[np.ndarray]:
    return list(np.array_split(_data(case), case["n_clients"], axis=0))


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['name']}-{c['method']}")
def test_regenerated_data_matches_the_recording(case):
    """Fail here, and the generator drifted -- not the solvers."""
    assert _digest(_data(case)) == case["x_sha16"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['name']}-{c['method']}")
def test_port_reproduces_the_original(case):
    clients = _clients(case)
    if case["method"] == "fedrcd_exact":
        result = fedrcd_exact(clients)
    else:
        stats = aggregate_and_compute_cumulants(
            [compute_local_raw_stats(c, order=4) for c in clients]
        )
        result = SOLVERS[case["method"]](stats)

    assert list(result.causal_order) == case["causal_order"]
    np.testing.assert_allclose(result.b, np.array(case["B"]), rtol=1e-9, atol=1e-9)
