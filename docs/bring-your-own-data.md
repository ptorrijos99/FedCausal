# Bring your own data

`GES` and `FedGES` take fully observed categorical data; `FedRCD` takes numeric data. Neither handles missing values.

## Tables the estimators accept

A pandas DataFrame, a 2-D array or a `DiscreteDataset` all work:

```python
import pandas as pd

from fedcausal import GES

GES().fit(pd.read_csv("cohort.csv", dtype=str))
GES().fit(array)                     # columns are named x0, x1, ...
```

Every distinct value of a column is one level, so `1, 2, 3` are three categories, not a quantity. A DataFrame with missing values is rejected, naming the column; impute or drop those rows first.

## From a CSV, without guessing

```python
from fedcausal.data import DiscreteDataset

data = DiscreteDataset.from_csv("cohort.csv")  # .csv.gz works too
```

`from_csv` reads every value as a label, including `None`, `NA` and `NaN`, which several standard benchmark networks use as real levels. Only an empty or blank field counts as missing, and that raises. Pandas' own reader, by contrast, turns those strings into missing values unless told otherwise (`keep_default_na=False`).

## Encoding several sites at once

`FedGES.fit` pools levels across sites before encoding, so a value seen at only one site is a level everywhere and every site scores over the same domains. To do the same yourself:

```python
from fedcausal.data import as_discrete_sites

sites = as_discrete_sites([frame_a, frame_b, frame_c])
```

The tables must share their columns, in the same order.

## How levels are encoded

Within each column, levels are numbered in sorted order of their label, not in order of first appearance. So `["yes", "no"]` encodes `no` as 0 and `yes` as 1, whatever order they show up in the file.

This is a reproducibility choice. Two sites with the same variables but different row orders must agree on the encoding, or their graphs mean different things when combined. Sorting the labels is the simplest rule that guarantees it.

The consequence is that level indices need not match another tool's. Anything computed from the contingency tables (counts, BDeu scores, learned structures) is unaffected, since those are invariant to which integer names which level.

## Splitting data across simulated sites

```python
from fedcausal.data import contiguous_split, dirichlet_split

parts = contiguous_split(data, 5)  # five contiguous blocks
parts = dirichlet_split(data, 5, alpha=0.5, seed=0)  # unequal, non-IID sites
```

`contiguous_split` keeps rows in their original order and slices them into blocks, which is the honest default when rows are already unordered. `dirichlet_split` draws site sizes from a Dirichlet distribution, so a low `alpha` gives you a few large sites and several small ones. Use that when you want to know how the method behaves under imbalance.

Both are for experiments. In a real deployment each site already has its own data and you never split anything; see [Deployment](deployment.md).

## What every site must agree on

Structures travel between sites as bare column indices. Every participant must therefore hold the same variables, in the same order, with the same set of levels. Row counts can and should differ.

The Flower server checks this before it trusts any edge: each client sends a fingerprint of its names and cardinalities, and a mismatch stops the run with a `SchemaMismatchError`. If you are wiring sites yourself, do the same check, or line up the columns before you start.

## Loading a network definition

```python
from fedcausal.io import load_xbif, forward_sample

bn = load_xbif("alarm.xbif")
data = forward_sample(bn, n=5000, seed=0)
```

`load_xbif` reads XMLBIF, the format the [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/) publishes. `forward_sample` draws from the network, which is how the recovery tests generate data with a known answer.

Watch the variable order here: `bn.names` comes from the file and need not match a CSV of the same network. Re-index before comparing, as [Quickstart](quickstart.md) shows.
