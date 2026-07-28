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
