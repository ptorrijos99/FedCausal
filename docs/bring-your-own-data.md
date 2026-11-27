# Bring your own data

`GES` and `FedGES` take fully observed categorical data; `FedRCD` takes numeric data. Neither handles missing values.

Each site's table is read and encoded as [causalbase](https://ptorrijos99.github.io/causalbase/bring-your-own-data.html) reads any table (its guide covers tables, CSV files and how levels are numbered). What federation adds is below.

## Encoding several sites at once

`FedGES.fit` pools levels across sites before encoding, so a value seen at only one site is a level everywhere and every site scores over the same domains. To do the same yourself:

```python
from causalbase.data import as_discrete_sites

sites = as_discrete_sites([frame_a, frame_b, frame_c])
```

The tables must share their columns, in the same order.

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
