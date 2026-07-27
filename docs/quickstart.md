# Quickstart

## Install

```bash
pip install git+https://github.com/ptorrijos99/fedcausal
```

Nothing else is required. The extras add the Flower transport (`[flower]`), numba acceleration for large graphs (`[accel]`), and causal-learn for the comparison benchmark (`[compare]`).

## Learn a structure from one table

```python
from fedcausal.data import DiscreteDataset
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

data = DiscreteDataset.from_csv("survey.csv")
cpdag = GES(BDeuScore(data)).fit(data)

print(cpdag.directed_edges())    # compelled: every DAG in the class agrees
print(cpdag.undirected_edges())  # reversible: the data cannot tell them apart
```

`fit` returns a CPDAG, not a DAG. That distinction matters: an undirected edge means the data is genuinely indifferent to the direction, not that the search gave up. Call `cpdag.to_dag()` when you need a concrete orientation, and treat the choice among reversible edges as arbitrary.

Nodes are integer indices. Their names live on the dataset:

```python
names = data.names
for u, v in cpdag.directed_edges():
    print(f"{names[u]} -> {names[v]}")
```

## Measure it against a known network

```python
from fedcausal.graph import DAG
from fedcausal.io import load_xbif
from fedcausal.metrics import shd, smhd, edge_scores

bn = load_xbif("asia.xbif")

# The .xbif declares its own variable order, which need not match the CSV's.
# Re-index onto the dataset's order, or every comparison below is silently
# comparing the wrong nodes.
index = {name: i for i, name in enumerate(data.names)}
truth = DAG.from_edges(
    data.n_vars,
    [(index[bn.names[u]], index[bn.names[v]]) for u, v in bn.dag.edges()],
)

learned = cpdag.to_dag()
print(shd(truth, learned), smhd(truth, learned))   # 3 2
print(edge_scores(truth, learned).f1_skel)         # 0.875
```

Prefer SMHD when comparing search results. It moralises both graphs first, so it does not punish an arbitrary orientation choice among reversible edges the way SHD does. [Validation](validation.md) explains why.

## Run the federation in one process

```python
from fedcausal.federated import run_fedges

global_model, stats = run_fedges(data, n_clients=5, n_iterations=20, truth=truth)

for row in stats:
    if row.stage == "server":
        print(f"round {row.iteration}: {row.edges} edges, SHD {row.shd}")
```

The data is split into five contiguous blocks, each treated as one site's private holding. Nothing about the loop assumes the sites are in one process, which is what lets the same code run over Flower unchanged.

## Change how the sites combine their graphs

```python
from fedcausal.fusion import MaxParents

global_model, stats = run_fedges(
    data,
    n_clients=5,
    n_iterations=20,
    server_fusion=MaxParents(3),   # cap the consensus structure's density
    refinement="GES",              # re-search the differences after fusing
)
```

## Next

Real data has quirks the synthetic examples do not: [Bring your own data](bring-your-own-data.md). Real federations run on more than one machine: [Flower deployment](flower-deployment.md).
