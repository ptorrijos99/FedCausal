# FedRCD: federated causal discovery over continuous data

`fedcausal.lingam` learns a causal order and a linear coefficient matrix from continuous, horizontally-partitioned data, using higher-order cumulants instead of the discrete-side's model fusion. Where FedGES federates by exchanging graphs and scores, this family federates by exchanging sufficient statistics: each client reduces its local rows to a handful of moment sums, the server aggregates those sums into unbiased cumulant estimates, and a causal order and a coefficient matrix (`RCDResult.causal_order`, `RCDResult.b`) fall out of a tau discriminator applied to the aggregated cumulants. It is the LiNGAM family's federated counterpart the way FedGES is GES's: same statistical model, split across sites that never pool their rows.

The method assumes a linear, non-Gaussian structural equation model (SEM): `x = Bx + e`, with `B` acyclic and each `e_i` non-Gaussian. Non-Gaussianity is what makes the causal order identifiable at all -- under Gaussian noise the joint distribution alone cannot distinguish a DAG from its Markov-equivalent alternatives, which is why the benchmark generator excludes `gaussian` from its noise catalogue (see [`benchmarks/pgm26/README.md`](../benchmarks/pgm26/README.md)).

## What actually crosses the network

Never a row of data. Each client computes `compute_local_raw_stats(x, order=4)` on its own local matrix and sends the result; nothing else leaves the client. That function returns nine additive moment-sum arrays (plus the sample count `n`, a scalar, not an array):

| Array | Shape | What it sums |
|---|---|---|
| `sum_x` | `(p,)` | Σ xᵢ |
| `sum_xx` | `(p, p)` | Σ xᵢxⱼ |
| `sum_x3` | `(p,)` | Σ xᵢ³ |
| `sum_x1y2` | `(p, p)` | Σ xᵢxⱼ² |
| `sum_x2y1` | `(p, p)` | Σ xᵢ²xⱼ |
| `sum_x4` | `(p,)` | Σ xᵢ⁴ |
| `sum_x1y3` | `(p, p)` | Σ xᵢxⱼ³ |
| `sum_x3y1` | `(p, p)` | Σ xᵢ³xⱼ |
| `sum_x2y2` | `(p, p)` | Σ xᵢ²xⱼ² |

Every one of these is additive across disjoint row subsets, which is the only property the server needs: it sums each array across clients (`aggregate_and_compute_cumulants`), centres by the pooled mean, and applies bias-corrected k-statistic formulas to get unbiased 3rd- and 4th-order cumulant estimates. The server never sees an individual row, and a client's contribution cannot be recovered from its nine arrays alone (they are aggregate statistics over the whole local sample, not a reversible encoding of it). That is the exchange for every solver in this family except `fedrcd_exact`, which repeats it once per round -- see below.

Over Flower (`fedcausal.lingam.flower`, requires the `flower` extra), this is one round-trip for every solver except `fedrcd_exact`: the server broadcasts a `train` message, each supernode computes its nine arrays and replies with them, its row count, and a fingerprint of its variable names, and the server aggregates, checks the fingerprints agree, and runs the selected solver.

## One-shot versus multi-round

| Function | Rounds | What it does |
|---|---|---|
| `fedrcd_noniterative` | 1 | Computes tau once from the aggregated cumulants and sorts by row sum. No deflation at all -- the federated counterpart of HC-LiNGAM. Fastest, but ignores the residual structure a source leaves behind once removed. |
| `fedrcd` | 1 | Iteratively identifies the most exogenous remaining variable via the tau discriminator, then *algebraically* deflates the aggregated cumulant arrays in place using OLS-estimated beta coefficients, and repeats over the shrinking active set. Exact under the true LiNGAM model; accumulates numerical error under finite samples or model misspecification, because each deflation step operates on already-deflated statistics rather than on fresh data. |
| `ishc_beta` | 1 | Same shape as `fedrcd`'s loop, but only ever uses the 3rd-order tau discriminator (kept for comparison against the 4th-order default). |
| `fedrcd_exact` | p - 1 | Runs one full communication round per variable removed. Each round: clients deflate their *actual local rows* by the broadcast `(source, beta)` and send fresh raw moment sums computed from the residuals, rather than the server updating its own aggregated arrays algebraically. No approximation accumulates, because every round's cumulants come from real data, not from propagating an earlier round's estimate. This is the exact federated counterpart of centralised HC. |

The extra communication is worth it when finite-sample noise or a mildly misspecified model would let `fedrcd`'s algebraic deflation drift: each of its update steps is exact in the population limit, but on real samples the error in an early deflation carries into every later one. `fedrcd_exact` pays `p - 1` rounds of O(p²) statistics (O(p³) total) to avoid that drift entirely, recomputing cumulants from the clients' actual deflated data at every step instead of trusting an algebraic update. For small `p` or noise close to the Gaussian boundary, where the tau signal is already weak, that extra round trip is where the difference tends to show up; for well-separated non-Gaussian noise at moderate `p` the one-shot solvers are usually good enough and much cheaper. `benchmarks/pgm26/README.md`'s `main` grid is the place to see the actual numbers across noise families and client counts, not a rule of thumb here.

`fedrcd_noniterative`, `fedrcd`, `ishc_beta`, `fedishc`, and `fedhc` all take the aggregated `Cumulants` dict directly and never touch client data themselves -- they are pure functions of the statistics, so anything that can produce that dict (the in-process aggregator, the Flower transport, a stub in a test) can drive them. `fedrcd_exact` is the exception: it takes the list of per-client arrays and manages the deflation rounds itself, because each round needs fresh statistics from each client's *current* data, not just the previous round's aggregate.

## Baselines, and where they come from

`fedcausal.lingam` ports several algorithms besides the FedRCD family itself, for comparison in the PGM 2026 experiments:

- **`fedishc` / `fedhc`** -- Chen, W., et al., "FedISHC: Federated Causal Discovery via Higher-Order Cumulants" (AAAI / arXiv, 2025). `fedishc` deflates iteratively using the alpha-ratio (rather than `fedrcd`'s OLS beta); `fedhc` is the no-deflation ablation, sorting once by the initial 3rd-order tau row sums.
- **`hc_lingam` / `hc`** -- Chen, W., et al., "Higher Order Cumulants-Based Method for Direct and Efficient Causal Discovery" (IEEE TNNLS, 2025). Centralised, single-machine baselines: `hc_lingam` sorts once by a global 4th-order tau matrix, `hc` iterates the same deflate-and-resort loop as `fedrcd` but directly on centred data rather than on federated cumulant sums. Both are ported line-for-line from the original single-machine implementation, including their default OLS coefficient estimator.
- **`DirectLiNGAM` / `ICALiNGAM`** -- external, non-federated comparison points via the [`lingam`](https://github.com/cdt15/lingam) package. Not part of `fedcausal.lingam`; pulled in only by `benchmarks/pgm26/benchmark.py` under the `lingam-experiments` extra, and silently skipped if that extra is not installed.

Every one of these functions returns the same `RCDResult` (`causal_order`, `b`, `to_dag(threshold)`), so they are interchangeable in code that only cares about the recovered structure.

## Reproducing the paper's numbers

[`benchmarks/pgm26/README.md`](../benchmarks/pgm26/README.md) reproduces the two experiments the PGM 2026 paper's own repository documents as its reproduction commands: the main `p x K x noise-distribution` grid, and the variance-stratification diagnostic that separates the noise regimes where the 3rd-order and 4th-order tau discriminators do and do not have signal. It also documents a known deviation from the original: this port's client split is deterministic contiguous blocks rather than the original's shuffle-then-split, which is statistically equivalent for freshly generated i.i.d. rows but means runs are not bit-identical to the paper's.
