"""
Learn one structure from one table
==================================

The starting point: no federation, one dataset, one graph out. Everything the
federated methods do is built on this call.

We use ``asia``, the smallest useful network bundled with fedcausal, and score
the result against the structure the data was actually sampled from.
"""

# %%
# A bundle carries the sampled rows, the true structure, and the network itself.
# The truth arrives already re-indexed onto the dataset's column order, which is
# the step that silently ruins comparisons when it is done by hand.

from fedcausal.datasets import load_asia
from fedcausal.metrics import edge_scores, shd, smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

bundle = load_asia()
data, truth = bundle.data, bundle.truth

print(f"{data.n_rows} rows, {data.n_vars} variables: {', '.join(data.names)}")

# %%
# Run the search. ``GES`` takes a score and returns a CPDAG, not a DAG.

cpdag = GES(BDeuScore(data)).fit(data)

# %%
# A CPDAG separates what the data settles from what it does not. A directed edge
# means every DAG in the equivalence class agrees on that direction; an
# undirected edge means the data genuinely cannot tell the two apart. That is a
# statement about the data, not a sign that the search gave up.

print("Compelled edges (every DAG in the class agrees):")
for u, v in cpdag.directed_edges():
    print(f"  {data.names[u]} -> {data.names[v]}")

print("\nReversible edges (the data cannot tell the direction):")
for u, v in cpdag.undirected_edges():
    print(f"  {data.names[u]} - {data.names[v]}")

# %%
# Scoring against the truth needs a concrete orientation, so pick any member of
# the class with ``to_dag()`` and treat the choice among reversible edges as
# arbitrary.
#
# Prefer SMHD over SHD here. It moralises both graphs first, so it does not
# punish an arbitrary orientation choice the way SHD does.

learned = cpdag.to_dag()
scores = edge_scores(truth, learned)

print(f"SHD {shd(truth, learned)}, SMHD {smhd(truth, learned)}")
print(f"Skeleton precision {scores.precision_skel:.3f}, recall {scores.recall_skel:.3f}")
