# Test fixtures

`networks/` holds five Bayesian networks from the public [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/), each as a network definition (`<net>.xbif`) and one sampled dataset (`<net>.0.csv.gz`):

| Network | Variables | Rows |
|---|---:|---:|
| cancer | 5 | 5 000 |
| asia | 8 | 5 000 |
| child | 20 | 5 000 |
| insurance | 27 | 5 000 |
| alarm | 37 | 5 000 |

They keep the test suite self-contained: `pytest` needs no downloads and no environment variables. The CSVs are gzipped because categorical data of this shape compresses about twentyfold, which keeps all five under 200 KB. `DiscreteDataset.from_csv` reads `.gz` directly.

The larger networks used by the scale tests and the benchmarks (up to munin's 1 041 variables) are not shipped. Point `FEDCAUSAL_NETWORKS` at a directory holding them to run those; see [`docs/benchmarks.md`](../../docs/benchmarks.md).

`toy.csv` and `none_category.csv` are hand-written fixtures for the dataset loader's own tests, the second covering `None` as a genuine category label rather than a missing value.
