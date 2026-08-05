"""
Federate across sites, in one process
=====================================

The question federation has to answer: is pooling *graphs* across sites better
than each site working alone? Here we split ``alarm`` into five simulated sites,
run the FedGES loop, and compare the result against both extremes.

One process, so this is what you want for experiments. Nothing in the loop
assumes the sites share a process, which is what lets the same code run over
Flower unchanged.
"""

# %%

from fedcausal.data import contiguous_split
from fedcausal.datasets import load_alarm
from fedcausal.fedges import run_fedges
from fedcausal.metrics import shd, smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

N_CLIENTS = 5

bundle = load_alarm()
data, truth = bundle.data, bundle.truth

print(f"alarm: {data.n_vars} variables, {data.n_rows} rows across {N_CLIENTS} sites")
print(f"each site therefore sees about {data.n_rows // N_CLIENTS} rows")

# %%
# Run the loop. Passing ``truth`` attaches an observer that scores every round;
# a real federation has no ground truth and simply passes nothing, which is why
# evaluation lives outside the loop rather than inside the server.

global_model, stats = run_fedges(data, n_clients=N_CLIENTS, n_iterations=20, truth=truth)

# %%
# Each round has two stages. In ``build`` every site re-learns locally, warm
# started from the consensus it was last sent; in ``server`` the coordinator
# fuses those local graphs into one global structure.

print(f"{'round':>6} {'stage':<8} {'site':>5} {'edges':>6} {'SHD':>5} {'SMHD':>5}")
for row in stats:
    site = "-" if row.client_id < 0 else str(row.client_id)
    print(f"{row.iteration:>6} {row.stage:<8} {site:>5} {row.edges:>6} {row.shd:>5} {row.smhd:>5}")

# %%
# The comparison that matters. One site alone sees a fifth of the rows; pooling
# all the data is the unreachable ideal, since in a real federation the data
# cannot be moved. Federation should land between the two, closer to the ideal.

one_site = contiguous_split(data, N_CLIENTS)[0]
local = GES(BDeuScore(one_site)).fit(one_site).to_dag()
central = GES(BDeuScore(data)).fit(data).to_dag()

print("SMHD against the true network (lower is better):")
print(f"  one site alone   {smhd(truth, local)}")
print(f"  federated        {smhd(truth, global_model.dag)}")
print(f"  all data pooled  {smhd(truth, central)}")
print(f"\nSHD federated: {shd(truth, global_model.dag)}")
