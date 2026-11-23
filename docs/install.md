# Install

```bash
pip install fedcausal
```

Python 3.10 or newer. The install pulls in numpy, pandas and numba, and that is all the estimators, the fusion operators and the bundled networks need, federated runs in one process included. The compiled kernels are built on first use and cached next to the package, so the first fit in a fresh environment takes a few seconds longer than the rest.

## Extras

```bash
pip install "fedcausal[federated]"             # the Flower transport, for deployment
pip install "fedcausal[federated-simulation]"  # adds Ray, needed by `flwr run`
```

Only `fedcausal.flower`, the transport across processes and machines, needs an extra; importing it without one fails with a message that names it. Simulation is a separate extra from the transport because it needs Ray, which has no wheel for every Python version and platform. Deploying against a real SuperLink needs only `[federated]`.

The development version installs from GitHub:

```bash
pip install git+https://github.com/ptorrijos99/fedcausal
```

## Check it worked

```python
from fedcausal import GES
from fedcausal.datasets import load_asia

print(GES().fit(load_asia().data).adjacency_)
```
