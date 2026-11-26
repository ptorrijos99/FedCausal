"""
Learn one structure from data held by several sites
===================================================

:class:`~fedcausal.fedges.FedGES` learns from discrete data that stays where it is.
Each round every site runs GES on its own rows, a server fuses the sites'
structures into a global one, and each site fuses that back into its own
model. Only structures and scores leave a site.

Here ``alarm``'s 5000 rows are split into five sites of 1000.
"""

from causalbase import GES
from causalbase.datasets import load_alarm
from causalbase.metrics import shd

from fedcausal import FedGES
from fedcausal.data import contiguous_split

bundle = load_alarm()
sites = contiguous_split(bundle.data, 5)

model = FedGES(n_rounds=10).fit(sites)
print(f"{model.n_rounds_} rounds, {model.dag_.num_edges} edges")

# %%
# ``history_`` has one row per model built: each site's local model with its
# own score, and the global model (``site == -1``), which the server cannot
# score because it holds no data.

model.history_.pivot(index="round", columns="site", values="edges")

# %%
# The federation against one site on its own and against pooling everything,
# which is what federation exists to avoid.

one_site = GES().fit(sites[0]).dag_
pooled = GES().fit(bundle.data).dag_
print(f"SHD, one site alone:  {shd(bundle.truth, one_site)}")
print(f"SHD, FedGES:          {shd(bundle.truth, model.dag_)}")
print(f"SHD, pooled data:     {shd(bundle.truth, pooled)}")

# %%
# With ``n_jobs`` the sites learn in separate processes, each keeping its own
# score cache between rounds. It pays off once a site's search takes longer
# than starting a process, at a few hundred variables.
