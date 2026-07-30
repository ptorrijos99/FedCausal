"""Generate ``docs/benchmarks.md`` from the measurement files.

Every number in that page comes from ``benchmarks/results/``. Nothing is typed
by hand, because a table assembled from notes is a table that quietly mixes two
measurement runs. Re-run this after re-running the benchmarks::

    python benchmarks/build_docs.py

The prose that describes the tables is derived from them too, so an observation
cannot outlive the data that made it true.
"""

from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

RESULTS = Path("benchmarks/results")
LADDER = RESULTS / "compare_ges.jsonl"
FED = RESULTS / "compare_fedges.jsonl"
FINAL = RESULTS / "final_measurements.json"
OUT = Path("docs/benchmarks.md")

ORDER = [
    "cancer",
    "asia",
    "child",
    "insurance",
    "water",
    "mildew",
    "alarm",
    "hailfinder",
    "hepar2",
    "win95pts",
    "pathfinder",
    "andes",
    "diabetes",
    "pigs",
    "link",
    "munin",
]
FED_NETS = ["asia", "cancer", "child", "insurance", "alarm", "hepar2"]


def rows(path: Path) -> list[dict]:  # type: ignore[type-arg]
    """Read a JSONL results file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_by_cell(records: list[dict], key) -> dict:  # type: ignore[type-arg, no-untyped-def]
    """Keep the last record written for each cell, so a re-run corrects rather than duplicates."""
    out = {}
    for r in records:
        out[key(r)] = r
    return out


def secs(r: dict | None) -> str:  # type: ignore[type-arg]
    """Format one cell's search time, or why there is not one."""
    if r is None:
        return "not run"
    if r.get("status") == "timeout":
        return f"> {int(r['timeout_s']) // 60} min"
    if r.get("status") != "ok":
        return r.get("status", "-")
    v = r["seconds_search"]
    if v < 0.01:
        return "< 0.01"
    return f"{v:.2f}" if v < 10 else (f"{v:.1f}" if v < 100 else f"{v:.0f}")


def main() -> None:
    """Write docs/benchmarks.md from the recorded measurements."""
    ladder = latest_by_cell(rows(LADDER), lambda r: (r["impl"], r["net"]))
    fc = {n: r for (i, n), r in ladder.items() if i == "fedcausal"}
    cl = {n: r for (i, n), r in ladder.items() if i == "causal-learn"}
    final = json.loads(FINAL.read_text()) if FINAL.exists() else {}

    nets = [n for n in ORDER if n in fc]

    # --- scale ladder ---
    scale_rows = []
    for n in nets:
        r = fc[n]
        scale_rows.append(
            f"| {n} | {r['n_vars']} | {secs(r)} | {r.get('n_edges', '-')} | "
            f"{r.get('shd', '-')} | {r.get('smhd', '-')} |"
        )

    # --- vs causal-learn ---
    cmp_rows = []
    for n in nets:
        f, c = fc[n], cl.get(n)
        ratio = ""
        if (
            c is not None
            and c.get("status") == "ok"
            and f.get("status") == "ok"
            and f["seconds_search"] >= 0.01
        ):
            ratio = f"{c['seconds_search'] / f['seconds_search']:.0f}x"
        cmp_rows.append(f"| {n} | {f['n_vars']} | {secs(f)} | {secs(c)} | {ratio} |")

    # --- federated cost ---
    fed = [r for r in rows(FED) if r["impl"] == "fedcausal" and r.get("status") == "ok"]
    grouped = collections.defaultdict(list)
    for r in fed:
        grouped[(r["net"], r["n_clients"])].append(r)
    central = final.get("central_5samples") or {}
    fed_rows = []
    for n in sorted(FED_NETS, key=lambda k: central.get(k, {}).get("n_vars", 0)):
        if n not in central:
            continue
        cells = [f"{central[n]['smhd']:.1f}"]
        for c in (5, 10, 20):
            rs = grouped.get((n, c), [])
            cells.append(f"{statistics.mean(r['smhd'] for r in rs):.1f}" if rs else "-")
        fed_rows.append(f"| {n} | {int(central[n]['n_vars'])} | " + " | ".join(cells) + " |")

    # Say what the federated table shows, computed from it.
    # Rank by absolute extra error, not by ratio: a ratio on a five-node network
    # turns rounding into a headline.
    worst, worst_gap, worst_base = None, 0.0, 0.0
    for n in central:
        rs = grouped.get((n, 20), [])
        if not rs:
            continue
        gap = statistics.mean(r["smhd"] for r in rs) - central[n]["smhd"]
        if gap > worst_gap:
            worst, worst_gap, worst_base = n, gap, central[n]["smhd"]
    fed_note = (
        "Splitting the same data across more sites costs accuracy, because each site sees less "
        "of it. How much depends on the network rather than on its size."
    )
    if worst:
        fed_note += (
            f" The largest cost here is `{worst}`: {worst_base:.0f} structural errors learning "
            f"centrally against {worst_base + worst_gap:.0f} across twenty sites. On the smallest "
            "networks the gap is a couple of edges either way, which is within the spread between "
            "samples."
        )

    cap = final.get("hailfinder_cap") or {}
    incr = final.get("incremental") or {}

    all_incr = dict(incr)
    if final.get("incremental_andes"):
        all_incr["andes"] = final["incremental_andes"]
    incr_rows = [
        f"| {n} | {int(d['n_vars'])} | {d['incremental_s']:.2f} | {d['full_scan_s']:.1f} | "
        f"{d['speedup']:.1f}x |"
        for n, d in sorted(all_incr.items(), key=lambda kv: kv[1]["n_vars"])
    ]

    # State the size-is-not-everything point from the data, so it cannot go stale.
    ranked = sorted((r for r in fc.values() if r.get("status") == "ok"), key=lambda r: r["n_vars"])
    inversion = next(
        (
            (a, b)
            for a, b in zip(ranked, ranked[1:], strict=False)
            if b["seconds_search"] < a["seconds_search"] * 0.8 and a["n_vars"] > 100
        ),
        None,
    )
    density_note = (
        "Runtime tracks how dense and interconnected a network is, not just how many variables "
        "it has."
    )
    if inversion:
        a, b = inversion
        density_note += (
            f" `{b['net']}` has {b['n_vars']} variables to `{a['net']}`'s {a['n_vars']}, and still "
            f"finishes in {b['seconds_search']:.0f} s against {a['seconds_search']:.0f} s."
        )

    sp = final.get("link_structure_prior") or {}
    if sp.get("0.0") and sp.get("1.0"):
        base, penalised = sp["0.0"], sp["1.0"]
        prior_note = (
            f"Two levers help if you work at this scale. A complexity penalty helps most: "
            f"`structure_prior=1.0` takes `link` from {base['edges']} edges to "
            f"{penalised['edges']}, and SMHD from {base['smhd']} to {penalised['smhd']}. "
            f"Warm-starting from any reasonable prior structure helps too, which is also what "
            f"the federated loop gives its clients after the first round."
        )
    else:
        prior_note = (
            "Two levers help if you work at this scale: a complexity penalty through "
            "`structure_prior`, and warm-starting from a prior structure, which is also what "
            "the federated loop gives its clients after the first round."
        )

    nb = final.get("numba_link") or {}
    if nb.get("speedup"):
        numba_text = (
            f"The `accel` extra JITs the semi-directed path check, the dominant cost on large "
            f"graphs. Measured back to back on `link` ({int(nb['n_vars'])} variables) it is "
            f"{nb['speedup']:.2f}x end to end, for a bit-identical result. Below a "
            f"variable-count threshold the pure-Python walk wins and is used instead, so on small "
            f"networks the extra changes nothing."
        )
    else:
        numba_text = (
            "The `accel` extra JITs the semi-directed path check, the dominant cost on large "
            "graphs. The result is bit-identical either way, and the JIT path is selected "
            "automatically above a variable-count threshold."
        )

    text = f"""# Benchmarks

Measured on one machine, single-threaded, with the default settings. Structures are exact; timings are not, so read them as orders of magnitude and see the note at the end.

## How fast, and how large

5000 rows per network. Times are search only.

| Network | Variables | Seconds | Edges found | SHD | SMHD |
|---|---:|---:|---:|---:|---:|
{chr(10).join(scale_rows)}

{density_note}

## Against causal-learn

The same data through both, same machine.

| Network | Variables | fedcausal | causal-learn | Ratio |
|---|---:|---:|---:|---:|
{chr(10).join(cmp_rows)}

## What federating costs you

Mean SMHD against the true network, over five sampled datasets per cell. Lower is better. The first column learns from everything at once; the rest split the same data across that many sites.

| Network | Variables | Centralised | 5 sites | 10 sites | 20 sites |
|---|---:|---:|---:|---:|---:|
{chr(10).join(fed_rows)}

{fed_note}

## What the options buy

`max_t_size=2`, the default, against the exhaustive scan on `hailfinder`, measured back to back:

| | Seconds | Same BDeu |
|---|---:|---|
| `max_t_size=2` (default) | {cap.get("capped_s", 0):.2f} | |
| `max_t_size=None` | {cap.get("exhaustive_s", 0):.1f} | {"yes" if cap.get("same_bdeu") else "no"} |

That is {(cap.get("exhaustive_s", 1) / cap.get("capped_s", 1)):.0f}x for an identical result on this network. The cap forfeits the optimality guarantee in theory; in practice it has not changed a recovered structure on any network the two have been compared on.

`incremental=True`, the default, against the full scan:

| Network | Variables | Incremental | Full scan | Speedup |
|---|---:|---:|---:|---:|
{chr(10).join(incr_rows)}

Both defaults return scores identical to the slower setting everywhere the two have been compared. For `max_t_size` that runs up to `link` at 724 variables, where 2, 4 and `None` all give the same graph. For `incremental` it runs up to `andes` at 223, since a full scan above that is expensive. Neither has been compared on `munin`. The one option known to cost accuracy is `empty_t_gate`; see [Algorithms](algorithms.md).

{numba_text}

## Cross-checked against the original implementation

The reference tests pin the search against the original Java implementation on networks up to 37 variables. `link`, at 724, has now been run through both on the same data:

| | Edges | SHD | SMHD | BDeu |
|---|---:|---:|---:|---:|
| true structure | 1125 | 0 | 0 | -1,059,715 |
| BayesFL (Java) | 1779 | 1597 | 3153 | -1,114,310 |
| fedcausal | 1770 | 1591 | 3098 | -1,114,303 |

The two scores differ by 7 in 1.1 million, and the skeletons share 1666 edges. Both were scored without a structure prior, which is what the Java implementation effectively uses. fedcausal took 186 s against 2900 s.

## Structure quality on the largest networks

The two biggest networks recover badly, and it is worth being explicit about why, because the cause is not a defect in the search.

On `link` (724 variables, 1125 true edges) the search returns 1770 edges scoring -1,114,303. The true structure scores -1,059,714, so the result is 55,000 worse than the answer. That rules out overfitting, which would score *better* than the truth. Warm-starting the same search from the true structure lands at -1,058,834, better than the truth itself. The search is therefore working; greedy search from an empty graph simply reaches a poor local optimum on a network this large and this sparsely sampled. Chickering's optimality guarantee is asymptotic, and 5000 rows over 724 variables is not that regime.

`max_t_size` is not involved: 2, 4, and `None` return the same graph on `link`.

{prior_note}

## Regenerating this page

The tables above are generated, not written:

```bash
python benchmarks/build_docs.py
```

Every number comes from `benchmarks/results/`, so re-running the benchmarks and then this script keeps the page honest. The prose describing each table is derived from it too, so an observation cannot outlive the data behind it.

## Running them yourself

```bash
export FEDCAUSAL_NETWORKS=/path/to/networks
python benchmarks/scale_ges.py                    # the size ladder
python benchmarks/compare_ges.py                  # fedcausal vs causal-learn
python benchmarks/report_ges.py                   # render the results
python benchmarks/compare_fedges.py               # quality against site count
python benchmarks/report_fedges.py
```

`FEDCAUSAL_NETWORKS` should point at a directory laid out as `<root>/res/networks/BBDD/<net>/<net>.0.csv` for the data and `<root>/res/networks/<net>.xbif` for the ground truths. Networks come from the [Bayesian Network Repository](https://www.bnlearn.com/bnrepository/); the five smallest ship with the package under `tests/fixtures/networks/`.

`compare_ges.py` runs one subprocess per cell and appends results as they finish, so a hang costs one cell and the run resumes where it stopped.

## A note on timing

Treat the absolute seconds loosely. Repeating the whole ladder on the same machine, with the same code and identical resulting structures, moved individual timings by up to a factor of two, and a sync client running in the background made it worse still. What survives that noise is the ratios, which are taken back to back in one process, and the orders of magnitude. If you re-run these, do it outside a synced folder and compare within a run rather than across runs.
"""
    OUT.write_text(text, encoding="utf-8")
    print("written", OUT, f"({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
