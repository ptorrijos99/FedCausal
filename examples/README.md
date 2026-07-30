# Examples

The three scripts run with no setup, against the networks bundled in `tests/fixtures/networks/`.

```bash
python examples/01_learn_structure.py     # learn asia, score it against the truth
python examples/02_fedges_inprocess.py    # federate alarm across five sites
python examples/03_fedrcd.py              # recover a causal order from continuous data
```

The second one is the interesting one. It prints the per-round trace and then compares three things: what one site recovers from its own fifth of the data, what the federation recovers, and what learning centrally on everything recovers.

[`flower_app/`](flower_app/) is a complete Flower app for both method families, which runs either as a simulation in one process or as a federation across real machines. [docs/deployment.md](../docs/deployment.md) is the walkthrough.
