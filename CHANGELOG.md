# Changelog

This project follows [Semantic Versioning](https://semver.org/). While the version is below 1.0, the public API may change between minor releases.

## 0.1.0

First public release.

### Structure learning

- `GES`, greedy equivalence search over CPDAGs, with an incremental engine that re-scores only the candidate pairs whose neighbourhood changed. Byte-identical to the full scan everywhere the two have been compared, and several times faster; see [Benchmarks](docs/benchmarks.md).
- `BDeuScore` with a vectorised counting kernel and a memoised local-score cache.
- Runs the whole benchmark ladder up to `munin` (1041 variables, 5000 rows), single-threaded and in pure Python. The optional `accel` extra adds numba for the dominant path check on large graphs.
- Operator score deltas are compared against a tolerance that scales with the scores, so accumulated `lgamma` rounding cannot be mistaken for an improvement. Without it, the search inserted edges that existed only in the rounding on the largest networks, and a platform whose `lgamma` rounded differently would have produced a different graph.

### Federation

- `run_fedges` runs the whole loop on one dataset. `Client`, `Server`, and `GESLearner` are available for finer control.
- Five convergence criteria, composable through `MultipleConvergence`.
- Structural fusion following Puerta et al. (2021): the consensus union plus the budgeted variants `MaxEdges`, `MaxFrequency`, `Consensus`, `MaxParents`, and `Intersection`.
- The loop is transport-agnostic. The server holds no client data, scores are reported by the site that computed them, and evaluation happens outside the loop through an observer.

### Flower

- `ClientApp` and `ServerApp` on the Message API, supporting both simulation and deployment against a real SuperLink.
- Where a client's data comes from is a seam: `NodeConfigLoader` reads one shard from the supernode's own config, `InMemoryPartitions` indexes a list for simulation, and anything matching `PartitionLoader` works in their place.
- Clients report a schema fingerprint and the server refuses a federation whose members disagree on their variables.
- A failed client aborts the round rather than silently shrinking the ensemble; `min_clients` opts into tolerating dropouts.
- Both transports are asserted to produce identical global models round by round.

### Graphs, data, and metrics

- `DAG` and `PDAG`, Meek rules, DAG and CPDAG conversion, moralisation, maximum-cardinality triangulation, treewidth.
- `DiscreteDataset` over integer-coded categorical data, stored column-major so the score kernel's gathers are contiguous.
- `ContinuousDataset` over real-valued data, the input format for `fedcausal.lingam`.
- `contiguous_split` and `dirichlet_split` for simulated sites, generic over `DiscreteDataset` and `ContinuousDataset`.
- `shd`, `shd_skeleton`, `smhd`, `smhd_graphs`, `edge_scores`.
- XMLBIF loading and forward sampling.

### `fedcausal.lingam`

- The FedRCD family, for continuous data under a linear non-Gaussian (LiNGAM) model: `fedrcd`, `fedrcd_noniterative`, and `fedrcd_exact` recover a causal order and coefficient matrix from cumulant statistics rather than raw rows. `fedrcd` and `fedrcd_noniterative` need one communication round; `fedrcd_exact` runs `p - 1` rounds of exact deflation to avoid the algebraic error the one-shot solvers can accumulate under finite samples. `ishc_beta` is a related third-order-only variant.
- `compute_local_raw_stats` and `aggregate_and_compute_cumulants`: the client-side and server-side halves of the statistics exchange. Nine additive moment-sum arrays per client cross the network, never a row of data.
- `fedishc`, `fedhc` (Chen et al.'s FedISHC baselines) and `hc_lingam`, `hc` (Chen et al.'s centralised HC-LiNGAM baselines), ported for comparison in the PGM 2026 benchmark.
- A Flower transport in `fedcausal.lingam.flower`, requiring the `flower` extra but not otherwise imported by the subpackage, so `fedcausal.lingam` itself needs only numpy and scipy.
- See [FedRCD](docs/fedrcd.md) and [`benchmarks/pgm26/`](benchmarks/pgm26/README.md).
