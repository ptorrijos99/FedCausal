# Test fixtures

`toy.csv` and `none_category.csv` are hand-written fixtures for the dataset loader's own tests, the second covering `None` as a genuine category label rather than a missing value.

`lingam/` holds the recorded baseline the FedRCD solvers are pinned against.

The Bayesian networks the rest of the suite learns from are not here: they ship inside the package under `src/fedcausal/datasets/data/` and are reached through `fedcausal.datasets`, so the examples and the documentation can use the same five networks the tests do.

The larger networks used by the scale tests and the benchmarks (up to munin's 1041 variables) are not shipped at all. Point `FEDCAUSAL_NETWORKS` at a directory holding them to run those; see [`docs/benchmarks.md`](../../docs/benchmarks.md).
