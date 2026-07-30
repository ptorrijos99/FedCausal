# FedRCD: federated causal discovery over continuous data

`fedcausal.lingam` learns a causal order and a linear coefficient matrix from continuous data spread across sites. Each client reduces its rows to a handful of moment sums, the server aggregates those into unbiased cumulant estimates, and a causal order and coefficient matrix (`RCDResult.causal_order`, `RCDResult.b`) fall out of a tau discriminator applied to them. FedGES federates by exchanging graphs; this family federates by exchanging sufficient statistics.

The method assumes a linear, non-Gaussian structural equation model: `x = Bx + e`, with `B` acyclic and each `e_i` non-Gaussian. Non-Gaussianity is what makes the causal order identifiable at all. Under Gaussian noise the joint distribution cannot distinguish a DAG from its Markov-equivalent alternatives, which is why the benchmark generator leaves `gaussian` out of its noise catalogue (see [`benchmarks/pgm26/README.md`](../benchmarks/pgm26/README.md)).

## What actually crosses the network

Never a row of data. Each client computes `compute_local_raw_stats(x, order=4)` on its own matrix and sends the result; nothing else leaves. That function returns nine additive moment-sum arrays, plus the sample count `n`:

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

Every one of them is additive across disjoint row subsets, which is the only property the server needs. It sums each array across clients (`aggregate_and_compute_cumulants`), centres by the pooled mean, and applies bias-corrected k-statistic formulas to get unbiased third- and fourth-order cumulant estimates. A client's contribution cannot be recovered from its nine arrays: they are aggregates over its whole local sample, not a reversible encoding of it.

Over Flower (`fedcausal.flower.lingam`, needs the `flower` extra) that is one round trip for every solver except `fedrcd_exact`. The server broadcasts a `train` message, each supernode replies with its nine arrays, its row count and a fingerprint of its variable names, and the server checks the fingerprints agree before aggregating.

## Deployment

The client side loads its data through `NodeConfigLoader(ContinuousDataset)`: each supernode reads a CSV path out of its own `node_config` and that file and nothing else. `InMemoryPartitions` is the simulation counterpart, indexing a list of in-memory datasets by `partition-id`, which is what `run_flower_rcd` and `run_flower_rcd_exact` use internally.

Both are the same classes the discrete side uses, so [Deployment](deployment.md) covers this family too, and the runnable app in [`examples/flower_app/`](../examples/flower_app/) serves both.

## One-shot versus multi-round

| Function | Rounds | What it does |
|---|---|---|
| `fedrcd_noniterative` | 1 | Computes tau once from the aggregated cumulants and sorts by row sum. No deflation at all, the federated counterpart of HC-LiNGAM. Fastest, and it ignores the residual structure a source leaves behind once removed. |
| `fedrcd` | 1 | Identifies the most exogenous remaining variable via the tau discriminator, then *algebraically* deflates the aggregated cumulant arrays in place using OLS-estimated betas, and repeats over the shrinking active set. |
| `ishc_beta` | 1 | The same loop as `fedrcd`, but only ever using the third-order tau discriminator. Kept for comparison against the fourth-order default. |
| `fedrcd_exact` | p - 1 | One full communication round per variable removed. Clients deflate their *own rows* by the broadcast `(source, beta)` and send fresh moment sums computed from the residuals, so every round's cumulants come from real data. The exact federated counterpart of centralised HC. |

`fedrcd`'s deflation is exact in the population limit. On real samples it is not: each step operates on already-deflated statistics, so the error in an early deflation carries into every later one. `fedrcd_exact` pays `p - 1` rounds of O(p²) statistics to avoid that drift, and the extra traffic earns its keep when finite-sample noise or a mildly misspecified model would let the drift matter, which tends to mean small `p` or noise close to the Gaussian boundary where the tau signal is weak to begin with. For well-separated non-Gaussian noise at moderate `p` the one-shot solvers are usually good enough and much cheaper. For the actual numbers rather than a rule of thumb, see `benchmarks/pgm26/README.md`'s `main` grid.

`fedrcd_noniterative`, `fedrcd`, `ishc_beta`, `fedishc` and `fedhc` take the aggregated `Cumulants` dict directly and never touch client data. They are pure functions of the statistics, so anything that can produce that dict can drive them: the in-process aggregator, the Flower transport, a stub in a test. `fedrcd_exact` is the exception, since each of its rounds needs fresh statistics from each client's *current* data rather than the previous round's aggregate.

## Baselines, and where they come from

`fedcausal.lingam` ports several algorithms besides the FedRCD family itself, for comparison in the PGM 2026 experiments:

- **`fedishc` / `fedhc`** from Chen, W., et al., "FedISHC: Federated Causal Discovery via Higher-Order Cumulants" (AAAI / arXiv, 2025). `fedishc` deflates iteratively using the alpha-ratio rather than `fedrcd`'s OLS beta; `fedhc` is the no-deflation ablation, sorting once by the initial third-order tau row sums.
- **`hc_lingam` / `hc`** from Chen, W., et al., "Higher Order Cumulants-Based Method for Direct and Efficient Causal Discovery" (IEEE TNNLS, 2025). Centralised, single-machine baselines: `hc_lingam` sorts once by a global fourth-order tau matrix, `hc` iterates the same deflate-and-resort loop as `fedrcd` but directly on centred data. Both are ported line for line from the original implementation, including its default OLS coefficient estimator.
- **`DirectLiNGAM` / `ICALiNGAM`**, external non-federated comparison points via the [`lingam`](https://github.com/cdt15/lingam) package. Not part of `fedcausal.lingam`; pulled in only by `benchmarks/pgm26/benchmark.py` under the `lingam-experiments` extra, and skipped if that extra is not installed.

All of them return the same `RCDResult` (`causal_order`, `b`, `to_dag(threshold)`), so they are interchangeable in code that only cares about the recovered structure.

## Reproducing the paper's numbers

[`benchmarks/pgm26/README.md`](../benchmarks/pgm26/README.md) reproduces the two experiments the PGM 2026 paper's own repository documents as its reproduction commands: the main `p x K x noise-distribution` grid, and the variance-stratification diagnostic that separates the noise regimes where the third- and fourth-order tau discriminators do and do not have signal. It also documents one known deviation from the original: this port splits clients into deterministic contiguous blocks rather than shuffling first, which is statistically equivalent for freshly generated i.i.d. rows but means runs are not bit-identical to the paper's.
