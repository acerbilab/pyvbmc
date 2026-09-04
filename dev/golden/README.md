# Golden-trace reference population

`baseline/` holds the JSON sidecars (one per run: config, seed, options,
git SHA, final metrics) and the `summary.md` of the reference population of
`dev/scripts/golden_trace.py`, so that

```console
python dev/scripts/golden_trace.py compare dev/golden/baseline <new_dir>
```

works from a fresh checkout. The full `.npz` traces stay gitignored under
`dev/scripts/runs/golden/`.

**Status (2026-09-04):** the 280 sidecars and `summary.md` of population
`baseline_20260903`: the `golden` suite (14 configurations, D = 2 to 10,
two of them noisy) × seeds 0–19, run on 2026-09-03/04 on the code of
`5020879` (one process, BLAS single-threaded, about 10 h of compute in two
sessions; 280 of 280 succeeded; even-vs-odd null check clean over 56 KS
tests). The 258 sidecars of the second session say `dirty: true` because a
worklog note under `dev/plans/` was uncommitted at the time; no code
differed. The first population (220 runs of 2026-09-03) was withdrawn the
same morning because its start points and plausible boxes were derived from
the true posteriors (see the audit in
`dev/plans/benchmark-suite-and-golden-traces.md`). Regenerate with
`bash dev/scripts/regenerate_baseline.sh golden` (resumable; refills this
directory at the end); extend with `python dev/scripts/golden_trace.py run
--suite golden --seeds 20-49 --workers 1 --out
dev/scripts/runs/golden/baseline_20260903` and copy the new sidecars and
`summary.md` here.
