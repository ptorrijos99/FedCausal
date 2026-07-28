"""Record FedRCD reference outputs from the original FedLiNGAM implementation.

Run this once, against the FedLiNGAM repository the PGM 2026 paper links to,
*before* any porting work begins. The resulting fixture is what proves the port
did not change the paper's results, so it must never be regenerated from ported
code.

Usage::

    python tools/record_lingam_baseline.py --fedlingam /path/to/FedLiNGAM
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def make_lingam_data(seed: int, n_samples: int, n_vars: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linear non-Gaussian SEM in a fixed causal order 0..n_vars-1."""
    rng = np.random.default_rng(seed)
    b_true = np.zeros((n_vars, n_vars))
    for child in range(1, n_vars):
        for parent in range(child):
            if rng.random() < 0.5:
                b_true[child, parent] = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
    # Laplace noise: non-Gaussian, which is what makes the order identifiable.
    noise = rng.laplace(0.0, 1.0, size=(n_samples, n_vars))
    x = np.zeros((n_samples, n_vars))
    for j in range(n_vars):
        x[:, j] = x @ b_true[j] + noise[:, j]
    return x, b_true


def split(x: np.ndarray, n_clients: int) -> list[np.ndarray]:
    """Contiguous horizontal split, matching the experiments' partitioning."""
    return [part for part in np.array_split(x, n_clients, axis=0)]


def digest16(x: np.ndarray) -> str:
    """Return a 16-hex-char SHA-256 digest of the generated matrix.

    Lets a later regeneration of the same synthetic data (e.g. by a test that
    reimplements ``make_lingam_data``) prove it produced byte-identical data
    before comparing solver outputs, instead of silently desyncing the RNG
    call sequence and failing in a way that looks like a solver bug.
    """
    return hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()[:16]


CASES = [
    {"name": "p5_n2000_c3", "seed": 0, "n_samples": 2000, "n_vars": 5, "n_clients": 3},
    {"name": "p8_n5000_c4", "seed": 1, "n_samples": 5000, "n_vars": 8, "n_clients": 4},
    {"name": "p10_n3000_c2", "seed": 2, "n_samples": 3000, "n_vars": 10, "n_clients": 2},
]


def main() -> None:
    """Record FedRCD baselines from the original FedLiNGAM checkout to a JSON fixture."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fedlingam", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/lingam/rcd_baseline.json"))
    args = parser.parse_args()

    sys.path.insert(0, str(args.fedlingam))
    from src.common.cumulants import (  # noqa: PLC0415
        aggregate_and_compute_cumulants,
        compute_local_raw_stats,
    )
    from src.fedrcd.core import (  # noqa: PLC0415
        fedrcd,
        fedrcd_exact,
        fedrcd_noniterative,
        ishc,
    )

    records = []
    for case in CASES:
        x, _ = make_lingam_data(case["seed"], case["n_samples"], case["n_vars"])
        x_sha16 = digest16(x)
        clients = split(x, case["n_clients"])
        raw = [compute_local_raw_stats(part, order=4) for part in clients]
        stats = aggregate_and_compute_cumulants(raw)

        for method, order_and_b in [
            ("fedrcd", fedrcd(stats)),
            ("fedrcd_noniterative", fedrcd_noniterative(stats)),
            ("ishc", ishc(stats)),
            ("fedrcd_exact", fedrcd_exact(clients)),
        ]:
            causal_order, b_est = order_and_b
            records.append(
                {
                    **case,
                    "method": method,
                    "causal_order": [int(v) for v in causal_order],
                    "B": np.asarray(b_est, dtype=float).tolist(),
                    "x_sha16": x_sha16,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cases": records}, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
