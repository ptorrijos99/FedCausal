# LiNGAM regression fixtures

`rcd_baseline.json` holds causal orders and B matrices produced by the
**original** FedLiNGAM implementation (github.com/ptorrijos99/FedLiNGAM), the
code the PGM 2026 paper links to, on three fixed synthetic problems.

It exists so the port into `fedcausal.lingam` can be shown not to have changed
the paper's results. **Do not regenerate it from ported code** — that would make
the test compare the port against itself. Regenerate only with
`tools/record_lingam_baseline.py` pointed at the original repository.
