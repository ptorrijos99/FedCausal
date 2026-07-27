# Bring your own data

fedcausal works on fully observed categorical data. No continuous variables, no missing values.

## From a CSV

```python
from fedcausal.data import DiscreteDataset

data = DiscreteDataset.from_csv("cohort.csv")     # .csv.gz works too
```

The file needs a header row of variable names, then comma-separated category labels. Every value is read as a string, so `0` and `1` become the labels `"0"` and `"1"` rather than numbers. That is deliberate: a column coded `1, 2, 3` is a set of three categories, not a quantity, and silently treating it as one would change the score.

Two things the loader will not do quietly.

It does not sniff for missing-value tokens. `None`, `NA`, `NULL`, and `NaN` are read as ordinary category labels, because several standard benchmark networks use `None` as a real level. If your file uses one of those strings to mean "missing", replace it before loading.

It rejects genuinely empty fields. A blank or whitespace-only cell raises `ValueError` naming the column, rather than guessing. Impute or drop those rows first; fedcausal has no opinion on which.

## From a DataFrame or an array

```python
import pandas as pd

data = DiscreteDataset.from_pandas(pd.read_csv("cohort.csv", dtype=str))
```

Or build one directly, which is what you want when the encoding is already done:

```python
import numpy as np
from fedcausal.data import DiscreteDataset

data = DiscreteDataset(
    values=np.array([[0, 1], [1, 1], [0, 0]], dtype=np.int32),
    cardinalities=np.array([2, 2], dtype=np.int32),
    names=["smoking", "cancer"],
    categories=[["no", "yes"], ["no", "yes"]],
)
```

## How levels are encoded

Within each column, levels are numbered in sorted order of their label, not in order of first appearance. So `["yes", "no"]` encodes `no` as 0 and `yes` as 1, whatever order they show up in the file.

This is a reproducibility choice. Two sites with the same variables but different row orders must agree on the encoding, or their graphs mean different things when combined. Sorting the labels is the simplest rule that guarantees it.

The consequence is that level indices need not match another tool's. Anything computed from the contingency tables (counts, BDeu scores, learned structures) is unaffected, since those are invariant to which integer names which level.

## Splitting data across simulated sites

```python
from fedcausal.data import contiguous_split, dirichlet_split

parts = contiguous_split(data, 5)                    # five contiguous blocks
parts = dirichlet_split(data, 5, alpha=0.5, seed=0)  # unequal, non-IID sites
```

`contiguous_split` keeps rows in their original order and slices them into blocks, which is the honest default when rows are already unordered. `dirichlet_split` draws site sizes from a Dirichlet distribution, so a low `alpha` gives you a few large sites and several small ones. Use that when you want to know how the method behaves under imbalance.

Both are for experiments. In a real deployment each site already has its own data and you never split anything; see [Flower deployment](flower-deployment.md).

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
