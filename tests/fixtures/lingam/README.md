# LiNGAM regression fixtures

`rcd_baseline.json` holds causal orders and B matrices produced by the
**original** FedLiNGAM implementation (github.com/ptorrijos99/FedLiNGAM), the
code the PGM 2026 paper links to, on three fixed synthetic problems.

It exists so the port into `fedcausal.lingam` can be shown not to have changed
the paper's results. **Do not regenerate it from ported code** — that would make
the test compare the port against itself. Regenerate only with
`record_baseline.py`, in this directory, pointed at the original repository.

Each case carries eight method records: the four FedRCD-family solvers
(`fedrcd`, `fedrcd_noniterative`, `ishc` — beta/OLS-deflation ISHC,
`fedrcd_exact`) and the four comparison baselines (`fedishc` — the published
alpha-ratio-deflation ISHC, `fedhc`, `hc_lingam`, `hc`). `fedishc` and `fedhc`
are recorded from `src.fedishc.core.ishc` and `src.fedishc.core.hc`
(aggregated-stats input, like the FedRCD family); `hc_lingam` and `hc` are
recorded from `src.hclingam.HCLiNGAM(method='hc-lingam' | 'hc').fit(x)`
against the **pooled** data matrix directly, since those two are centralised
and never see a federated aggregation step. `fedhc`'s original — `hc()` with
its default `iterative=False` — returns `None` for B (the no-deflation
ablation leaves B estimation to the caller), so that one method's `B` field
is recorded as `null` and only its `causal_order` is checked against the
port.

Each record also carries `x_sha16`, a 16-hex-char SHA-256 digest of the full
generated data matrix (before the client split). A consumer that reimplements
`make_lingam_data` — for example a test that regenerates these same synthetic
problems to check the ported solvers — should hash its own matrix the same way
and compare against `x_sha16` first. A mismatch means the RNG call sequence
desynced (wrong draw order, batched calls, etc.) and the data itself differs,
so any downstream discrepancy is not a solver bug.

Each record also carries `cumulants`, the aggregated statistics
(`cov`, `c3`, `c12`, `c21`, `c4`, `c13`, `c31`, `c22`, each as nested lists)
that `aggregate_and_compute_cumulants` produced from the original
implementation for that problem. This field is per-*case*, not per-method: all
four method records sharing a `name` carry an identical `cumulants` block,
the same way they share `x_sha16`. It exists because once a causal order is
fixed, every solver re-estimates `B` from `cov` alone and never reads the
higher-order cumulants again — so a small bias-correction error in
`cumulants.py` (a slightly wrong constant, a dropped term that mostly
cancels) would only surface in the causal-order/`B` comparison if it happened
to be large enough to flip the tau ranking. Pinning the eight arrays directly
closes that gap: it catches a cumulant regression even when it is too small
to change any solver's output.
