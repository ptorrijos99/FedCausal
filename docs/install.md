# Install

```bash
pip install git+https://github.com/ptorrijos99/fedcausal
```

Python 3.10 or newer. The base install pulls in numpy, scipy and pandas, and nothing else is required: the search, the fusion operators, both method families and the bundled example networks all work without any extra.

## Extras

```bash
pip install "fedcausal[flower]"             # the Flower transport, for deployment
pip install "fedcausal[flower-simulation]"  # adds Ray, needed by `flwr run`
pip install "fedcausal[accel]"              # numba, worth it past a few hundred variables
pip install "fedcausal[compare]"            # causal-learn, for the comparison benchmark
```

Simulation is a separate extra from the transport because it needs Ray, which has no wheel for every Python version and platform. Deploying against a real SuperLink needs only `[flower]`.

The `accel` extra JITs the semi-directed path check, the dominant cost on large graphs. It is bit-identical to the pure-Python path, and below a variable-count threshold the pure-Python walk wins and is used instead, so on small networks it changes nothing.

## Check it worked

```python
from fedcausal.datasets import load_asia
from fedcausal.scores import BDeuScore
from fedcausal.search import GES

data = load_asia().data
print(GES(BDeuScore(data)).fit(data).directed_edges())
```
