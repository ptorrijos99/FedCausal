"""
Using your own data
===================

The bundled networks are for learning the API. Real work starts from your table,
and there are three things about the loader worth knowing before you trust what
comes out: it treats every value as a category, it will not guess at missing
values, and it fixes the level encoding by sorting.

fedcausal works on fully observed categorical data. No continuous columns, no
missing values.
"""

# %%
# From a DataFrame. ``from_csv`` takes the same shape from a file, including
# ``.csv.gz``.

import pandas as pd

from fedcausal.data import DiscreteDataset

frame = pd.DataFrame(
    {
        "smoking": ["no", "yes", "yes", "no", "yes", "no"],
        "tar": ["low", "high", "high", "low", "high", "low"],
        "cancer": ["no", "yes", "no", "no", "yes", "no"],
    }
)
data = DiscreteDataset.from_pandas(frame)

print(f"{data.n_rows} rows, {data.n_vars} variables: {', '.join(data.names)}")
print(f"cardinalities: {[int(c) for c in data.cardinalities]}")

# %%
# Levels are numbered in sorted order of their label, not order of first
# appearance. So ``smoking`` encodes ``no`` as 0 and ``yes`` as 1 whatever order
# they show up in the file.
#
# This is a reproducibility choice, and it matters most in the federated setting:
# two sites with the same variables but different row orders have to agree on the
# encoding, or their graphs mean different things when they are combined.

for name, levels in zip(data.names, data.categories, strict=True):
    print(f"  {name:<8} {dict(enumerate(levels))}")

# %%
# Numbers are categories too. A column coded ``1, 2, 3`` is three levels, not a
# quantity, and every value is read as a string to keep it that way. Silently
# treating it as a number would change the score.

numeric = DiscreteDataset.from_pandas(pd.DataFrame({"stage": [3, 1, 2, 1]}))
print(f"stage levels: {numeric.categories[0]}")

# %%
# What the loader refuses. A blank cell raises rather than guessing, naming the
# column so you know where to look.

try:
    DiscreteDataset.from_pandas(pd.DataFrame({"smoking": ["yes", "", "no"]}))
except ValueError as error:
    print(f"ValueError: {error}")

# %%
# Missing-value tokens are *not* sniffed for. ``None``, ``NA``, ``NULL`` and
# ``NaN`` are read as ordinary category labels, because several standard
# benchmark networks use ``None`` as a real level. If your file uses one of them
# to mean "missing", replace it before loading; fedcausal has no opinion on
# whether you impute or drop.

with_none = DiscreteDataset.from_pandas(pd.DataFrame({"result": ["None", "yes", "no"]}))
print(f"'None' is kept as a level: {with_none.categories[0]}")

# %%
# Splitting across simulated sites, for experiments. In a real deployment each
# site already holds its own data and you never split anything.

from fedcausal.data import contiguous_split, dirichlet_split
from fedcausal.datasets import load_child

child = load_child().data

even = contiguous_split(child, 5)
uneven = dirichlet_split(child, 5, alpha=0.5, seed=0)

print(f"contiguous_split: {[s.n_rows for s in even]}")
print(f"dirichlet_split:  {[s.n_rows for s in uneven]}")

# %%
# ``contiguous_split`` slices rows in their original order, the honest default
# when rows are already unordered. ``dirichlet_split`` draws site sizes from a
# Dirichlet, so a low ``alpha`` gives a few large sites and several small ones.
# Use it when you want to know how a method behaves under imbalance.
#
# Whatever the source, every participating site must hold the same variables in
# the same order with the same levels, because structures travel between sites as
# bare column indices. The Flower server checks this on the first round and stops
# the run with a ``SchemaMismatchError`` rather than trusting an edge.
