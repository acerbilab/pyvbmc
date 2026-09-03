# Golden-trace reference population

`baseline/` holds the JSON sidecars (one per run: config, seed, options,
git SHA, final metrics) and the `summary.md` of the reference population of
`dev/scripts/golden_trace.py`, so that

```console
python dev/scripts/golden_trace.py compare dev/golden/baseline <new_dir>
```

works from a fresh checkout. The full `.npz` traces stay gitignored under
`dev/scripts/runs/golden/`.

**Status (2026-09-03):** empty. The first population (220 runs made on
2026-09-03) was withdrawn the same morning: its start points and plausible
boxes were derived from the true posteriors (see the audit in
`dev/plans/benchmark-suite-and-golden-traces.md`). Regenerate with
`bash dev/scripts/regenerate_baseline.sh` (one process, about 10 hours),
which refills this directory at the end.
