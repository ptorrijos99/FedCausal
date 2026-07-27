"""Learn one Bayesian network structure and score it against the truth.

Uses the asia network bundled with the test fixtures, so it runs with no setup:

    python examples/01_learn_structure.py
"""

from __future__ import annotations

from pathlib import Path

from fedcausal.data import DiscreteDataset
from fedcausal.graph import DAG
from fedcausal.io import load_xbif
from fedcausal.metrics import edge_scores, shd, smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

NETWORKS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "networks"


def main() -> None:
    """Learn asia's structure from 5000 samples and report how close it is."""
    data = DiscreteDataset.from_csv(NETWORKS / "asia.0.csv.gz")
    print(f"{data.n_rows} rows, {data.n_vars} variables: {', '.join(data.names)}\n")

    cpdag = GES(BDeuScore(data)).fit(data)

    print("Compelled edges (every DAG in the class agrees):")
    for u, v in cpdag.directed_edges():
        print(f"  {data.names[u]} -> {data.names[v]}")
    print("\nReversible edges (the data cannot tell the direction):")
    for u, v in cpdag.undirected_edges():
        print(f"  {data.names[u]} - {data.names[v]}")

    # The .xbif declares its own variable order, so re-index onto the data's.
    bn = load_xbif(NETWORKS / "asia.xbif")
    index = {name: i for i, name in enumerate(data.names)}
    truth = DAG.from_edges(
        data.n_vars,
        [(index[bn.names[u]], index[bn.names[v]]) for u, v in bn.dag.edges()],
    )

    learned = cpdag.to_dag()
    scores = edge_scores(truth, learned)
    print(f"\nSHD {shd(truth, learned)}, SMHD {smhd(truth, learned)}")
    print(f"Skeleton precision {scores.precision_skel:.3f}, recall {scores.recall_skel:.3f}")


if __name__ == "__main__":
    main()
