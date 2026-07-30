"""Run the federated loop in one process and watch it converge.

Splits the alarm dataset across five simulated sites and prints, per round,
what each site learned locally and what the consensus looks like:

    python examples/02_fedges_inprocess.py
"""

from __future__ import annotations

from pathlib import Path

from fedcausal.data import DiscreteDataset
from fedcausal.fedges import run_fedges
from fedcausal.graph import DAG
from fedcausal.io import load_xbif
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

NETWORKS = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "networks"
N_CLIENTS = 5


def main() -> None:
    """Federate alarm across five sites and compare against learning centrally."""
    data = DiscreteDataset.from_csv(NETWORKS / "alarm.0.csv.gz")
    bn = load_xbif(NETWORKS / "alarm.xbif")
    index = {name: i for i, name in enumerate(data.names)}
    truth = DAG.from_edges(
        data.n_vars,
        [(index[bn.names[u]], index[bn.names[v]]) for u, v in bn.dag.edges()],
    )

    print(f"alarm: {data.n_vars} variables, {data.n_rows} rows across {N_CLIENTS} sites")
    print(f"each site therefore sees about {data.n_rows // N_CLIENTS} rows\n")

    global_model, stats = run_fedges(data, n_clients=N_CLIENTS, n_iterations=20, truth=truth)

    print(f"{'round':>6} {'stage':<8} {'site':>5} {'edges':>6} {'SHD':>5} {'SMHD':>5}")
    for row in stats:
        site = "-" if row.client_id < 0 else str(row.client_id)
        print(
            f"{row.iteration:>6} {row.stage:<8} {site:>5} {row.edges:>6} {row.shd:>5} {row.smhd:>5}"
        )

    # What one site would have managed alone, and what pooling everything would.
    from fedcausal.data import contiguous_split

    one_site = contiguous_split(data, N_CLIENTS)[0]
    local = GES(BDeuScore(one_site)).fit(one_site).to_dag()
    central = GES(BDeuScore(data)).fit(data).to_dag()

    print("\nSMHD against the true network:")
    print(f"  one site alone   {smhd(truth, local)}")
    print(f"  federated        {smhd(truth, global_model.dag)}")
    print(f"  all data pooled  {smhd(truth, central)}")
    print(f"\nSHD federated: {shd(truth, global_model.dag)}")


if __name__ == "__main__":
    main()
