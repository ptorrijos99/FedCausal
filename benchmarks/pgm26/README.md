# PGM 2026 -- Federated Causal Discovery via Regression-Directed Cumulants

Reproduces the two experiments the paper's original repository documents as its
reproduction commands, ported onto `fedcausal.lingam`'s solvers.

```bash
pip install -e ".[lingam-experiments]"
python benchmarks/pgm26/benchmark.py --mode main
python benchmarks/pgm26/benchmark.py --mode stratification
```

Both append JSON-lines records to `benchmarks/results/pgm26.jsonl` (override with
`--out`). Add `--quick` for a small grid that finishes in a couple of seconds, which
is what `tests/bench/test_pgm26_smoke.py` runs -- and what CI runs.

The `lingam-experiments` extra is only needed for the `DirectLiNGAM`/`ICALiNGAM`
comparison baselines (via the `lingam` package). Every `fedcausal.lingam` method runs
without it; those two rows are silently skipped if the package is not importable.
`bnlearn`, `matplotlib`, and `seaborn` are declared in the same extra for other modes
of the original experiment suite (real-graph structures, plotting) that this port does
not include -- see "What this port leaves out" below.

## Modes

**`main`** -- the paper's Table 1. A grid of `p in {10, 20, 30, 50}` variables x
`K in {1, 5, 10, 20, 50}` federated clients x all eleven noise distributions
(`gaussian` is excluded: LiNGAM is not identifiable under Gaussian noise) x 5 seeds,
at a fixed `n=2000` and `density=0.2`. That is 1100 data-generation cells, each
scored against nine methods (five federated single-round solvers, one federated
multi-round solver, HC, HC-LiNGAM, and -- if installed -- DirectLiNGAM/ICALiNGAM), so
roughly 9900 records for a full run.

**`stratification`** -- the variance-stratification diagnostic. A fixed
(`p=20, n=2000, K=10, density=0.2`) cell swept over two noise families:
`symmetric` (`uniform`, `laplace`, `logistic`, `t` -- zero third-order cumulant) and
`asymmetric` (`exponential`, `gamma`, `pareto`), 30 seeds each. This is how the paper
separates the regime where the third-order tau discriminator has signal from the one
where it does not: on symmetric noise, methods that rely on 4th-order cumulants
(`FedRCD`, `FedRCD-NI`, `FedRCD-X`) should keep recovering the causal order while the
third-order-only baselines (`FedHC`, `FedISHC`, `ISHC-Beta`) degrade toward chance.

## Expected runtime

`--quick` (either mode): a few seconds -- one small cell group, `p<=10`, `n<=200`.

A full `main` run is the expensive one: 1100 cells, several of them at `p=50` and
`K=50`, each running nine methods (`FedRCD-X` alone costs `p-1` re-aggregation rounds).
Budget on the order of hours on a single machine; there is no parallelism or
resume-from-partial-results in this port (see below). A full `stratification` run is
much smaller (7 distributions x 30 seeds x one fixed cell size) and should finish in
minutes.

## What this port leaves out, and why

The original `experiments/pgm26/benchmark.py` has nine modes (`main`, `n_ablation`,
`density`, `bnlearn`, `hetero`, `real_sachs`, `stratification`, `smoke`, `all`); this
port covers only the two the published README documents as reproduction commands.
Specifically not ported:

- **`bnlearn`/`real_sachs` modes** -- real graph structures and the Sachs protein
  dataset. Out of scope for `--mode {main,stratification}`; would need `bnlearn` and
  bundled data files this repository does not carry.
- **Heteroscedastic noise** (`_hetero` distributions) -- used only by the `hetero`
  mode. `data.py`'s generator always uses unit noise scale, matching how `main` and
  `stratification` actually call the original generator.
- **Per-method dual Lasso reconstruction** (`lasso_fixed` vs `lasso_bic`, plus the
  extra/missing/reversed edge breakdown) -- the original recomputed B twice per method
  for a finer-grained comparison. This port uses each solver's own `RCDResult.b`
  (already estimated with `use_lasso=True, lmbda_type="fixed"` where the solver
  accepts it) and reports SHD and precision/recall/F1 via `fedcausal.metrics`, per the
  porting brief's directive to use one metric authority rather than reimplement the
  original's private helpers.
- **Resume-from-partial-CSV** -- the original tracked completed `(cell, seed, method)`
  keys so a killed run could pick up where it stopped. This port always appends; a
  full run does not currently need to survive being interrupted, and appending is
  simpler. Delete or move `pgm26.jsonl` before a fresh full run if you do not want a
  previous partial run's records mixed in.
- **`FedISHC-4`** -- the 4th-order-tau ablation of ISHC from the original suite. Not
  part of the `fedcausal.lingam` public API this port builds on.

`order_correct` is this port's own field, not a `fedcausal.metrics` function: it is
`True` iff the estimated causal order places every true edge's parent before its
child (weaker than an exact permutation match, which would also constrain pairs the
ground-truth graph never connects). See `_order_is_consistent` in `benchmark.py`.
