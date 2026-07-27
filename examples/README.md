# Examples

Both scripts run with no setup, against the networks bundled in
`tests/fixtures/networks/`.

```bash
python examples/01_learn_structure.py     # learn asia, score it against the truth
python examples/02_fedges_inprocess.py    # federate alarm across five sites
```

The second one is the interesting one. It prints the per-round trace and then
compares three things: what one site recovers from its own fifth of the data,
what the federation recovers, and what learning centrally on everything
recovers.

[`flower_deployment/`](flower_deployment/) is a complete Flower app for a
federation across real machines, where each site loads only its own file. See
[docs/flower-deployment.md](../docs/flower-deployment.md) for the walkthrough.
