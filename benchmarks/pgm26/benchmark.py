"""PGM 2026 benchmark runner -- "Federated Causal Discovery via Regression-Directed Cumulants".

Ports the two modes the paper's README documents as its reproduction commands
onto the ``fedcausal.lingam`` solvers:

``main``
    p x K grid at n=2000, sweeping all eleven noise distributions over five
    seeds each (the paper's main results table).
``stratification``
    A fixed (p=20, n=2000, K=10) cell swept over two noise families --
    ``symmetric`` (zero third-order cumulant: uniform, laplace, logistic, t)
    and ``asymmetric`` (exponential, gamma, pareto) -- which is how the paper
    separates the regime where the third-order tau discriminator has signal
    from the regime where it does not.

Every record is one JSON line appended to `--out` (default
``benchmarks/results/pgm26.jsonl``), carrying at least ``method``, ``n_vars``,
``n_clients``, ``seed``, ``shd``, and ``order_correct``.

Usage (from the repository root)::

    pip install -e ".[lingam-experiments]"
    python benchmarks/pgm26/benchmark.py --mode main
    python benchmarks/pgm26/benchmark.py --mode stratification
    python benchmarks/pgm26/benchmark.py --mode main --quick   # seconds, what CI runs

See ``README.md`` in this directory for expected runtimes.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (  # noqa: E402
    ALL_DISTRIBUTIONS,
    ASYMMETRIC_DISTRIBUTIONS,
    SYMMETRIC_DISTRIBUTIONS,
    generate_lingam_dataset,
    true_dag,
)
from fedcausal.data import ContinuousDataset, contiguous_split  # noqa: E402
from fedcausal.lingam import (  # noqa: E402
    RCDResult,
    aggregate_and_compute_cumulants,
    compute_local_raw_stats,
    fedhc,
    fedishc,
    fedrcd,
    fedrcd_exact,
    fedrcd_noniterative,
    hc,
    hc_lingam,
    ishc_beta,
)
from fedcausal.metrics import edge_scores, shd  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "results" / "pgm26.jsonl"

# ── Main-grid defaults (paper Table 1: p x K, n fixed) ──────────────────────
P_VALUES: tuple[int, ...] = (10, 20, 30, 50)
K_VALUES: tuple[int, ...] = (1, 5, 10, 20, 50)
DEFAULT_N = 2_000
GRAPH_DENSITY = 0.2
N_SEEDS = 5

# ── Stratification-diagnostic defaults ───────────────────────────────────────
STRAT_P = 20
STRAT_N = 2_000
STRAT_K = 10
STRAT_DENSITY = 0.2
STRAT_SEEDS = 30

# Below this magnitude, an estimated coefficient is treated as "no edge" when
# thresholding an `RCDResult` into a `DAG` for SHD. All solvers below are run
# with `use_lasso=True, lmbda_type="fixed"`, which already drives most
# non-edges to exactly zero; this only guards the solvers whose B comes from
# inverting a mixing matrix (`fedishc`) rather than a sparse regression.
EDGE_THRESHOLD = 0.05

# `fedishc` never accepts `use_lasso`/`lmbda_type` (its B always comes from
# inverting the accumulated mixing matrix); the rest do.
StatsMethod = Callable[[dict[str, Any]], RCDResult]
ClientMethod = Callable[[Iterable[npt.NDArray[np.float64]]], RCDResult]
CentralMethod = Callable[[npt.NDArray[np.float64]], RCDResult]

#: Methods that consume aggregated cumulant statistics (one communication round).
STATS_METHODS: dict[str, StatsMethod] = {
    "FedHC": functools.partial(fedhc, use_lasso=True, lmbda_type="fixed"),
    "FedISHC": fedishc,
    "ISHC-Beta": functools.partial(ishc_beta, use_lasso=True, lmbda_type="fixed"),
    "FedRCD": functools.partial(fedrcd, use_lasso=True, lmbda_type="fixed"),
    "FedRCD-NI": functools.partial(fedrcd_noniterative, use_lasso=True, lmbda_type="fixed"),
}

#: Methods that need the raw (per-client) data directly, not aggregated stats.
CLIENT_METHODS: dict[str, ClientMethod] = {
    "FedRCD-X": functools.partial(fedrcd_exact, use_lasso=True, lmbda_type="fixed"),
}

#: Centralised baselines, run once per (n_vars, dist, seed) cell at n_clients=1.
CENTRAL_METHODS: dict[str, CentralMethod] = {
    "HC": functools.partial(hc, use_lasso=True, lmbda_type="fixed"),
    "HC-LiNGAM": functools.partial(hc_lingam, use_lasso=True, lmbda_type="fixed"),
}


def _optional_lingam_methods() -> dict[str, CentralMethod]:
    """Return DirectLiNGAM/ICALiNGAM baselines if the ``lingam`` package is installed.

    Imported lazily, and only from here, so the rest of this module -- and the
    ``--quick`` smoke test -- runs without the ``lingam-experiments`` extra.
    Only these two comparison baselines are skipped when it is absent.

    Returns
    -------
    dict[str, CentralMethod]
        Empty when ``lingam`` cannot be imported.
    """
    try:
        import lingam
    except ImportError:
        return {}

    def _wrap(model_cls: type) -> CentralMethod:
        def _fit(x: npt.NDArray[np.float64]) -> RCDResult:
            model = model_cls().fit(x)
            order = tuple(int(i) for i in model.causal_order_)
            b = np.asarray(model.adjacency_matrix_, dtype=float)
            return RCDResult(order, b)

        return _fit

    return {
        "DirectLiNGAM": _wrap(lingam.DirectLiNGAM),
        "ICALiNGAM": _wrap(lingam.ICALiNGAM),
    }


def _order_is_consistent(b_true: npt.NDArray[np.float64], causal_order: tuple[int, ...]) -> bool:
    """Return whether `causal_order` respects every directed edge of `b_true`.

    True iff, for every nonzero ``b_true[child, parent]``, `parent` precedes
    `child` in `causal_order`. This is weaker than an exact permutation match:
    pairs with no true edge between them may appear in either relative order,
    which is the right notion of "the order is correct" for a partially
    connected DAG (``density < 1`` leaves most pairs unordered by the truth).

    Parameters
    ----------
    b_true : numpy.ndarray
        Ground-truth mixing matrix, ``b_true[child, parent]`` convention.
    causal_order : tuple[int, ...]
        Estimated causal order, source first.

    Returns
    -------
    bool
        Whether `causal_order` is a valid topological order of `b_true`.
    """
    position = {v: i for i, v in enumerate(causal_order)}
    child_idx, parent_idx = np.nonzero(b_true)
    return all(
        position[p] < position[c]
        for c, p in zip(child_idx.tolist(), parent_idx.tolist(), strict=True)
    )


def _record(
    method: str,
    result: RCDResult,
    elapsed: float,
    b_true: npt.NDArray[np.float64],
    truth: Any,
    *,
    n_vars: int,
    n_clients: int,
    n_samples: int,
    dist: str,
    density: float,
    seed: int,
    family: str | None = None,
) -> dict[str, Any]:
    """Build one JSONL record for `method`'s result on one cell.

    Parameters
    ----------
    method : str
        Method name, as it appears in `STATS_METHODS` / `CLIENT_METHODS` /
        `CENTRAL_METHODS`.
    result : RCDResult
        The method's estimated causal order and coefficient matrix.
    elapsed : float
        Wall-clock seconds the method took.
    b_true : numpy.ndarray
        Ground-truth mixing matrix, for `_order_is_consistent`.
    truth : DAG
        Ground-truth DAG, for `shd` and `edge_scores`.
    n_vars : int
        Number of variables in the cell.
    n_clients : int
        Number of federated clients (``1`` for a centralised baseline).
    n_samples : int
        Total number of rows in the cell.
    dist : str
        Noise distribution name.
    density : float
        Graph density used to generate the cell.
    seed : int
        Seed used to generate the cell.
    family : str, optional
        Noise family (``"symmetric"`` / ``"asymmetric"``), set only by the
        stratification mode.

    Returns
    -------
    dict[str, Any]
        A JSON-serialisable record.
    """
    est = result.to_dag(threshold=EDGE_THRESHOLD)
    scores = edge_scores(truth, est)
    record: dict[str, Any] = {
        "method": method,
        "n_vars": n_vars,
        "n_clients": n_clients,
        "n_samples": n_samples,
        "dist": dist,
        "density": density,
        "seed": seed,
        "shd": shd(truth, est),
        "order_correct": _order_is_consistent(b_true, result.causal_order),
        "precision": scores.precision,
        "recall": scores.recall,
        "f1": scores.f1,
        "runtime_s": elapsed,
    }
    if family is not None:
        record["family"] = family
    return record


def _timed(fn: Callable[..., RCDResult], arg: Any) -> tuple[RCDResult, float]:
    """Call ``fn(arg)``, returning ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    result = fn(arg)
    return result, time.perf_counter() - t0


def run_main_cell_group(
    p: int, dist: str, seed: int, n_samples: int, density: float, k_values: Iterable[int]
) -> list[dict[str, Any]]:
    """Run every method on one (p, dist, seed) cell, across all of `k_values`.

    Generates the data once (matching every `k` to the same underlying
    dataset, as the paper's grid does) and runs the centralised baselines
    once, then splits into each client count in turn for the federated
    methods.

    Parameters
    ----------
    p : int
        Number of variables.
    dist : str
        Noise distribution, one of `data.ALL_DISTRIBUTIONS`.
    seed : int
        Seed for data generation.
    n_samples : int
        Total number of rows.
    density : float
        Graph density.
    k_values : Iterable[int]
        Client counts to sweep.

    Returns
    -------
    list[dict[str, Any]]
        One record per (method, k) pair, plus one per centralised baseline.
    """
    rng = np.random.default_rng(seed)
    x, b_true, _true_order = generate_lingam_dataset(rng, n_samples, p, density, dist)
    truth = true_dag(b_true)

    records: list[dict[str, Any]] = []

    central_methods = {**CENTRAL_METHODS, **_optional_lingam_methods()}
    for name, fn in central_methods.items():
        result, elapsed = _timed(fn, x)
        records.append(
            _record(
                name,
                result,
                elapsed,
                b_true,
                truth,
                n_vars=p,
                n_clients=1,
                n_samples=n_samples,
                dist=dist,
                density=density,
                seed=seed,
            )
        )

    names = [f"x{i}" for i in range(p)]
    for k in k_values:
        ds = ContinuousDataset.from_arrays(x, names)
        clients = [c.values for c in contiguous_split(ds, k)]
        raw = [compute_local_raw_stats(c, order=4) for c in clients]
        stats = aggregate_and_compute_cumulants(raw)

        for name, stats_fn in STATS_METHODS.items():
            result, elapsed = _timed(stats_fn, stats)
            records.append(
                _record(
                    name,
                    result,
                    elapsed,
                    b_true,
                    truth,
                    n_vars=p,
                    n_clients=k,
                    n_samples=n_samples,
                    dist=dist,
                    density=density,
                    seed=seed,
                )
            )

        for name, client_fn in CLIENT_METHODS.items():
            result, elapsed = _timed(client_fn, clients)
            records.append(
                _record(
                    name,
                    result,
                    elapsed,
                    b_true,
                    truth,
                    n_vars=p,
                    n_clients=k,
                    n_samples=n_samples,
                    dist=dist,
                    density=density,
                    seed=seed,
                )
            )

    return records


def run_stratification_cell(
    family: str, dist: str, seed: int, p: int, n_samples: int, k: int, density: float
) -> list[dict[str, Any]]:
    """Run every federated method (plus DirectLiNGAM, if available) on one cell.

    Parameters
    ----------
    family : str
        ``"symmetric"`` or ``"asymmetric"``.
    dist : str
        Noise distribution within `family`.
    seed : int
        Seed for data generation.
    p : int
        Number of variables.
    n_samples : int
        Total number of rows.
    k : int
        Number of federated clients.
    density : float
        Graph density.

    Returns
    -------
    list[dict[str, Any]]
        One record per method.
    """
    rng = np.random.default_rng(seed)
    x, b_true, _true_order = generate_lingam_dataset(rng, n_samples, p, density, dist)
    truth = true_dag(b_true)

    names = [f"x{i}" for i in range(p)]
    ds = ContinuousDataset.from_arrays(x, names)
    clients = [c.values for c in contiguous_split(ds, k)]
    raw = [compute_local_raw_stats(c, order=4) for c in clients]
    stats = aggregate_and_compute_cumulants(raw)

    records: list[dict[str, Any]] = []
    for name, stats_fn in STATS_METHODS.items():
        result, elapsed = _timed(stats_fn, stats)
        records.append(
            _record(
                name,
                result,
                elapsed,
                b_true,
                truth,
                n_vars=p,
                n_clients=k,
                n_samples=n_samples,
                dist=dist,
                density=density,
                seed=seed,
                family=family,
            )
        )

    for name, client_fn in CLIENT_METHODS.items():
        result, elapsed = _timed(client_fn, clients)
        records.append(
            _record(
                name,
                result,
                elapsed,
                b_true,
                truth,
                n_vars=p,
                n_clients=k,
                n_samples=n_samples,
                dist=dist,
                density=density,
                seed=seed,
                family=family,
            )
        )

    direct_lingam = _optional_lingam_methods().get("DirectLiNGAM")
    if direct_lingam is not None:
        result, elapsed = _timed(direct_lingam, x)
        records.append(
            _record(
                "DirectLiNGAM",
                result,
                elapsed,
                b_true,
                truth,
                n_vars=p,
                n_clients=1,
                n_samples=n_samples,
                dist=dist,
                density=density,
                seed=seed,
                family=family,
            )
        )

    return records


def _append_records(records: Iterable[dict[str, Any]], out_path: Path) -> int:
    """Append `records` to `out_path` as JSON lines, returning the count written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record))
            fh.write("\n")
            n += 1
    return n


def run_main(*, quick: bool, out_path: Path) -> int:
    """Run the main p x K x dist x seed grid, appending records to `out_path`.

    Parameters
    ----------
    quick : bool
        If ``True``, shrink the grid to one small cell group so the whole
        run finishes in a couple of seconds.
    out_path : Path
        JSONL file to append records to.

    Returns
    -------
    int
        Number of records written.
    """
    if quick:
        p_values: tuple[int, ...] = (10,)
        k_values: tuple[int, ...] = (1, 5)
        dists: tuple[str, ...] = ("exponential", "uniform")
        seeds: Iterable[int] = (0,)
        n_samples = 200
    else:
        p_values = P_VALUES
        k_values = K_VALUES
        dists = ALL_DISTRIBUTIONS
        seeds = range(N_SEEDS)
        n_samples = DEFAULT_N

    total = 0
    for p in p_values:
        for dist in dists:
            for seed in seeds:
                records = run_main_cell_group(p, dist, seed, n_samples, GRAPH_DENSITY, k_values)
                total += _append_records(records, out_path)
    print(f"main{' (quick)' if quick else ''}: wrote {total} records -> {out_path}")
    return total


def run_stratification(*, quick: bool, out_path: Path) -> int:
    """Run the symmetric/asymmetric noise-family diagnostic, appending to `out_path`.

    Parameters
    ----------
    quick : bool
        If ``True``, shrink to one distribution per family, one seed, and a
        smaller (p, n) so the whole run finishes in a couple of seconds.
    out_path : Path
        JSONL file to append records to.

    Returns
    -------
    int
        Number of records written.
    """
    if quick:
        p, n_samples, k = 8, 200, 3
        families = {"symmetric": ("uniform",), "asymmetric": ("exponential",)}
        seeds: Iterable[int] = (0,)
    else:
        p, n_samples, k = STRAT_P, STRAT_N, STRAT_K
        families = {"symmetric": SYMMETRIC_DISTRIBUTIONS, "asymmetric": ASYMMETRIC_DISTRIBUTIONS}
        seeds = range(STRAT_SEEDS)

    total = 0
    for family, dists in families.items():
        for dist in dists:
            for seed in seeds:
                records = run_stratification_cell(
                    family, dist, seed, p, n_samples, k, STRAT_DENSITY
                )
                total += _append_records(records, out_path)
    print(f"stratification{' (quick)' if quick else ''}: wrote {total} records -> {out_path}")
    return total


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested benchmark mode."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=("main", "stratification"),
        default="main",
        help="main: p x K x dist grid (Table 1). stratification: symmetric vs asymmetric noise.",
    )
    parser.add_argument("--quick", action="store_true", help="tiny grid, finishes in seconds")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSONL output path")
    args = parser.parse_args(argv)

    if args.mode == "main":
        run_main(quick=args.quick, out_path=args.out)
    else:
        run_stratification(quick=args.quick, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
