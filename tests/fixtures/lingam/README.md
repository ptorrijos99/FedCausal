# LiNGAM regression fixtures

`rcd_baseline.json` holds causal orders and B matrices produced by the
**original** FedLiNGAM implementation (github.com/ptorrijos99/FedLiNGAM), the
code the PGM 2026 paper links to, on three fixed synthetic problems.

It exists so the port into `fedcausal.lingam` can be shown not to have changed
the paper's results. **Do not regenerate it from ported code** — that would make
the test compare the port against itself. Regenerate only with
`tools/record_lingam_baseline.py` pointed at the original repository.

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
