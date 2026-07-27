# Changelog

This project follows [Semantic Versioning](https://semver.org/). While the version is below 1.0, the public API may change between minor releases.

## 0.1.0

First public release.

### Structure learning

- `GES`, greedy equivalence search over CPDAGs, with an incremental engine that re-scores only the candidate pairs whose neighbourhood changed. Verified byte-identical to the full scan on every benchmark network, and roughly 13x faster at 223 variables.
- `BDeuScore` with a vectorised counting kernel and a memoised local-score cache.
- Completes `munin` (1041 variables, 5000 rows) in about 25 minutes, single-threaded and in pure Python. The optional `accel` extra adds numba for roughly 4x on the dominant path check.

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
- `contiguous_split` and `dirichlet_split` for simulated sites.
- `shd`, `shd_skeleton`, `smhd`, `smhd_graphs`, `edge_scores`.
- XMLBIF loading and forward sampling.
