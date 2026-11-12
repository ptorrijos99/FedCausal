# Validation

What the test suite checks, so you can decide how much to trust it.

```bash
pip install -e ".[dev,flower]"
pytest
```

No downloads and no environment variables: everything the tests need ships with the package.

| Layer | How it is checked |
|---|---|
| BDeu score | Against an independent count-and-`lgamma` implementation, including parent sets with more configurations than any integer code can hold |
| GES | Recovers the bundled networks; stops where no single Insert or Delete improves the score; never ends below the score of the graph it starts from |
| CPDAG completion | Two random DAGs share a CPDAG exactly when they have the same skeleton and v-structures (Verma & Pearl), and every CPDAG extends back into its own class |
| Fusion operators | Acyclicity, coverage of every input, and budget limits, over random DAGs |
| FedGES | Sequential and parallel sites learn the same models; every refinement keeps the site's score; convergence rules |
| FedRCD | The solvers against the original implementation's recorded output |
| Flower transports | The same models as the in-process loop, round by round, over a stub grid that hands each message to the real site handler; failed sites and mismatched variables stop the run |
| Documentation | Every `from fedcausal... import ...` shown in the docs runs |

Search results are compared through scores and equivalence classes rather than edge lists where it matters. A search returns one member of a Markov equivalence class, and the choice among its members is arbitrary; BDeu and the CPDAG are the same for every member.

## Test selection

```bash
pytest -m "not slow"        # skip the long-running ones
```

The slow tests drive a real Flower simulation and need Ray (`fedcausal[flower-simulation]`).
