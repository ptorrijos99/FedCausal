# Changelog

This project follows [Semantic Versioning](https://semver.org/). While the version is below 1.0, the public API may change between minor releases.

## 0.2.0

A high-level API, a much faster search, and none of the original Java implementation's quirks. Results are not bit-for-bit those of 0.1.0.

### Estimators

- `fedcausal.GES`, `fedcausal.FedGES` and `fedcausal.FedRCD` follow scikit-learn's conventions: hyperparameters in the constructor, `fit` learns, results end in an underscore, and `get_params`/`set_params` make them work with `sklearn.base.clone` and parameter searches, without depending on scikit-learn.
- They take DataFrames, arrays or datasets. `FedGES.fit` and `FedRCD.fit` take one table per site; levels are pooled across sites.
- `FedGES(n_jobs=...)` spreads the sites over worker processes, each keeping its sites and their score caches for the whole run.
- `FedGES.history_` records every local and global model per round.
- `required` and `forbidden` adjacencies in `GES` and `FedGES`, by variable name or index, held by the search and imposed after every fusion.
- `criterion="bdeu"`, `"bic"` or `"gaussian_bic"` in `GES` and `FedGES`: BDeu or BIC over categorical data, or the BIC of a linear Gaussian model over continuous data (`BICScore`, `GaussianBICScore`). `penalty_discount` scales the BIC penalty, and the `expected_parents` prior applies to all three.
- `max_parents` in `GES` and `FedGES` caps the parents an Insert may give a node.
- `GES.score(X)` is the mean held-out log likelihood per row, so `GridSearchCV` can tune `GES`; `fedcausal.scores.log_likelihood` gives it for any DAG.
- The fusion operators are frozen dataclasses, so `FedGES(global_fusion=Consensus(30))` prints as such and two equal operators compare equal.
- A `DenseFusionWarning` when a fused global model has more than twice the sites' mean number of edges: the sign that sites disagree on their v-structures and the consensus transform is adding edges no site has.

### Speed

- numba is a required dependency. BDeu counting, the semi-directed path check, CPDAG completion and the consensus order of the fusion operators are compiled.
- The search evaluates operators one child at a time, deciding the path condition of every candidate parent with one traversal and counting families that share parents together, in parallel.
- Twenty to forty times faster on the same data and the same structures: `andes` (223 variables) takes 0.7 s against 25 s, `pigs` (441) 2.7 s against 114 s; see [Benchmarks](docs/benchmarks.md).
- Insert scores are cached as one array per family, indexed by the added parent; the graph is completed to its CPDAG in place, changing only the pairs that differ; and the Dor-Tarsi extension and Chickering's edge labelling walk neighbour and parent lists instead of rows of the matrix. Together they cut `munin` (1041 variables) from 192 s to 21 s.
- The fusion operators' consensus order and transform are compiled. Fusing one round of `andes` over ten sites, server and sites together, went from about twenty minutes to under fifteen seconds.
- BDeu is exact for any number of parents: configurations are relabelled when they outgrow a 64-bit code, where 0.1.0 overflowed.
- The score cache keys each family by one integer, a bit per parent, instead of a set: 26 times less memory for the keys on dense searches.
- A deployed site that reads its file every round keeps its score cache between rounds.
- FedRCD's sites reduce their rows with four matrix products, the deflation updates every remaining variable at once, and the adaptive lasso is compiled and warm-started along its penalty path. With 30 variables, 20,000 rows and five sites a fit takes 0.02 s against 0.65 s, and `method="exact"` 0.14 s against 2.5 s.
- `import fedcausal` loads each estimator's subpackage on first use, and pandas only when a DataFrame is built: importing an estimator takes 0.6 s against 1.1 s.

### Changed

- The structure prior is a Bernoulli prior on each possible parent, `expected_parents=1.0` by default. The previous prior took the number of rows where the number of variables belongs.
- `run_fedges`, `Client`, `Server`, `GESLearner` and the observers are replaced by `FedGES`, `Site` and `run_rounds`. The Flower transport runs the same `run_rounds`.
- Convergence criteria: `AnyOf` replaces `MultipleConvergence`; `NoneConvergence` and `IterationEquality` are gone (`AnyOf()` never stops early).
- `refinement` takes `None`, `"forward"`, `"backward"` or `"both"`.
- `run_flower_fedges`, `run_flower_rcd` and `run_flower_rcd_exact` take a list of site datasets.
- `fedrcd(stats, method=..., order=...)` and `fedrcd_exact(sites, order=...)` run every FedRCD variant and the published baselines; `FedRCD` takes the same `method` and `order`. With `order=3` the sites send only their third-order sums, over Flower too.
- FedRCD's sites send their column means and moment sums centred on them (`mean`, `m11`, `m21`, `m31`, `m22`) instead of raw power sums, and the server merges them exactly; the Flower messages change accordingly.
- `Consensus(pct)` keeps an edge when at least `pct`% of the inputs have it, rounding the number of votes up; it used to round down.
- `shd` takes PDAGs as well as DAGs and counts the node pairs whose marks differ.
- `ContinuousDataset.from_arrays` is gone; the constructor does the same. scipy is no longer a dependency.

### Fixed

- FedRCD's cumulants are accurate wherever the data are located. Built from raw power sums, they lost every significant digit once the mean was about 10⁴ standard deviations from zero, and the causal order with them.
- Sites that send moment sums of different orders are refused; their sums used to be added up as if they matched.
- The consensus transform placed some nodes one position too early in the order it starts from, so the transformed DAGs, and every fusion built on them, could carry edges a minimal I-map does not need.
- `Consensus(50)` over three sites kept every edge one site had: the vote threshold was truncated to one.
- `shd` between two CPDAGs could be negative.
- `required` and `forbidden` reject indices that are not columns of the data, instead of failing later or truncating `1.7` to `1`.
- `ges` started from a PDAG completes it to its CPDAG first.
- The bundled `insurance` network keeps a level its sample never takes, as `forward_sample` does.
- `FedGES(n_jobs=...)` keeps every site's model in its place when one site has none yet, and shutting the workers down neither hides the error that stopped them nor waits forever.
- FedISHC estimates the coefficient of a source as `Cum(x_s, x_s, x_j) / Cum(x_s, x_s, x_s)`; the port divided `Cum(x_j, x_j, x_s)`, the coefficient's square.
- `apply_meek_rules(..., prevent_cycles=True)` leaves an edge undirected when orienting it would close a directed cycle, which voted PDAGs without a consistent extension otherwise allowed.

### Removed

- The options that reproduced the Java implementation: `h_pool`, `empty_t_gate`, `incremental`, and the full-scan engine. The Insert and Delete operators follow Chickering (2002) exactly.
- The `accel` extra, now part of the base install, and the reference tests pinned against the Java outputs.
- `fedrcd_noniterative`, `ishc_beta`, `fedishc`, `fedhc`, `hc_lingam` and `hc`, each a setting of `fedrcd` or `fedrcd_exact` (see [FedRCD](docs/fedrcd.md)), and the single-machine cumulant helpers `unbiased_cumulant3`, `unbiased_cumulant4`, `cross_cumulants3`, `cross_cumulants4` and `tau_matrix_4th_from_data`.

## 0.1.0

First public release.

### Structure learning

- `GES`, greedy equivalence search over CPDAGs, with an incremental engine that re-scores only the candidate pairs whose neighbourhood changed. Byte-identical to the full scan everywhere the two have been compared, and several times faster; see [Benchmarks](docs/benchmarks.md).
- `BDeuScore` with a vectorised counting kernel and a memoised local-score cache.
- Runs the whole benchmark ladder up to `munin` (1041 variables, 5000 rows), single-threaded and in pure Python. The optional `accel` extra adds numba for the dominant path check on large graphs.
- Operator score deltas are compared against a tolerance that scales with the scores, so accumulated `lgamma` rounding cannot be mistaken for an improvement. Without it, the search inserted edges that existed only in the rounding on the largest networks, and a platform whose `lgamma` rounded differently would have produced a different graph.

### `fedcausal.fedges`

- `run_fedges` runs the whole FedGES loop on one dataset. `Client`, `Server`, and `GESLearner` are available for finer control.
- Five convergence criteria, composable through `MultipleConvergence`.
- Structural fusion following Puerta et al. (2021): the consensus union plus the budgeted variants `MaxEdges`, `MaxFrequency`, `Consensus`, `MaxParents`, and `Intersection`.
- The loop is transport-agnostic. The server holds no client data, scores are reported by the site that computed them, and evaluation happens outside the loop through an observer.

### `fedcausal.flower`

- `ClientApp` and `ServerApp` on the Message API for both method families, in simulation and deployed against a real SuperLink. One module per family over shared plumbing, so a third one is a module rather than a second transport.
- Where a client's data comes from is a seam: `NodeConfigLoader` reads one shard from the supernode's own config, `InMemoryPartitions` indexes a list for simulation, and anything matching `PartitionLoader` works in their place. All of them are generic over the dataset type.
- Clients report a schema fingerprint and the server refuses a federation whose members disagree on their variables.
- A failed client aborts the round rather than silently shrinking the ensemble; `min_clients` opts into tolerating dropouts.
- Both transports are asserted to produce identical results to the in-process ones, round by round.

### Graphs, data, and metrics

- `DAG` and `PDAG`, Meek rules, DAG and CPDAG conversion, moralisation, maximum-cardinality triangulation, a treewidth bound.
- `DiscreteDataset` over integer-coded categorical data, stored column-major so the score kernel's gathers are contiguous.
- `ContinuousDataset` over real-valued data, the input format for `fedcausal.lingam`.
- `contiguous_split` and `dirichlet_split` for simulated sites, generic over `DiscreteDataset` and `ContinuousDataset`.
- `shd`, `shd_skeleton`, `smhd`, `smhd_graphs`, `edge_scores`.
- XMLBIF loading and forward sampling.

### `fedcausal.lingam`

- The FedRCD family, for continuous data under a linear non-Gaussian (LiNGAM) model: `fedrcd` and `fedrcd_exact` recover a causal order and coefficient matrix from cumulant statistics rather than raw rows. `fedrcd` needs one communication round; `fedrcd_exact` runs `p - 1` rounds of exact deflation to avoid the algebraic error the one-round methods can accumulate under finite samples.
- `compute_local_raw_stats` and `aggregate_and_compute_cumulants`: the client-side and server-side halves of the statistics exchange. Moment sums cross the network, never a row of data.
- FedISHC, FedHC, HC-LiNGAM and HC (Chen et al.) as settings of `fedrcd` and `fedrcd_exact`, for comparison against FedRCD.
- Every solver is a pure function of the aggregated statistics, so `fedcausal.lingam` itself needs only numpy. Its Flower transport is `fedcausal.flower.lingam`, under the `flower` extra.
- See [FedRCD](docs/fedrcd.md).
