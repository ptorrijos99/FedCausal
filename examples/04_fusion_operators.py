"""
Choosing how sites combine their graphs
=======================================

The coordinator's only real decision. Every fusion operator transforms the
client graphs into one shared topological order and unions the parent sets,
which is what guarantees the result is acyclic. They differ in how much of that
union they keep, and that is a precision-versus-recall dial.

This runs the dial end to end on ``alarm`` split across ten sites, so the
trade-off is measured rather than asserted.
"""

# %%
# Ten sites, so each sees 500 rows of a 37-variable network. That is little
# enough that the local graphs genuinely disagree, which is the situation the
# operators exist for.

from fedcausal.data import contiguous_split
from fedcausal.datasets import load_alarm
from fedcausal.fusion import Consensus, ConsensusUnion, Intersection, MaxParents
from fedcausal.metrics import edge_scores, smhd
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

N_CLIENTS = 10

bundle = load_alarm()
data, truth = bundle.data, bundle.truth

shards = contiguous_split(data, N_CLIENTS)
client_dags = [GES(BDeuScore(s)).fit(s).to_dag() for s in shards]

print(f"{N_CLIENTS} sites, {data.n_rows // N_CLIENTS} rows each")
print(f"edges learned per site: {[len(d.edges()) for d in client_dags]}")
print(f"true network has {len(truth.edges())} edges")

# %%
# Fuse the same ten graphs five different ways.
#
# ``ConsensusUnion`` keeps every edge any site found; ``Intersection`` keeps only
# what all ten agree on. ``Consensus(pct)`` is the dial between them, and
# ``MaxParents`` caps density instead of agreement.

operators = {
    "ConsensusUnion()": ConsensusUnion(),
    "Consensus(30)": Consensus(30),
    "Consensus(50)": Consensus(50),
    "MaxParents(3)": MaxParents(3),
    "Intersection()": Intersection(),
}

print(f"{'operator':<20} {'edges':>6} {'SMHD':>6} {'prec':>6} {'recall':>7}")
for label, operator in operators.items():
    fused = operator.fuse(client_dags)
    scores = edge_scores(truth, fused)
    print(
        f"{label:<20} {len(fused.edges()):>6} {smhd(truth, fused):>6} "
        f"{scores.precision_skel:>6.3f} {scores.recall_skel:>7.3f}"
    )

# %%
# Read the table down the precision and recall columns. Keeping everything finds
# the most true edges and pays for it in false ones; keeping only unanimous edges
# inverts that. ``Consensus(50)`` is the server default because the middle of
# that dial is usually where SMHD bottoms out.
#
# Every result above is acyclic by construction, whatever the operator:

for operator in operators.values():
    fused = operator.fuse(client_dags)
    assert len(fused.topological_order()) == fused.num_nodes
print("every fused structure is acyclic")
