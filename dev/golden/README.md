# Golden-trace reference population

`baseline/` holds the JSON sidecars (one per run: config, seed, options,
git SHA, final metrics) and the `summary.md` of the reference population of
`dev/scripts/golden_trace.py`, so that

```console
python dev/scripts/golden_trace.py compare dev/golden/baseline <new_dir>
```

works from a fresh checkout. The full `.npz` traces (the per-iteration
record of every run, which `dev/scripts/golden_replay.py` needs for its
per-iteration verdict; about 60 KB per run) stay gitignored under
`dev/scripts/runs/golden/`; from the 1.5 release on they are published as a
release asset, one zip per reference population, to be unpacked into that
directory. Until then they exist only on the machine that ran the
population and can be regenerated from the sidecars' code SHA, seeds and
options.

**Status (2026-09-06):** the 280 sidecars and `summary.md` of population
`item7_20260906`: the `golden` suite (14 configurations, D = 2 to 10, two
of them noisy) × seeds 0–19, run on 2026-09-06 00:16–05:01 on the code of
`bdaf322` (the end of Stage 2 of the modernization plan; one process, BLAS
single-threaded, 4.75 h; 280 of 280 succeeded; even-vs-odd null check
clean over 56 KS tests, `promotion_20260906/null_check_even_vs_odd.md`).
The sidecars say `dirty: true` because record files under `dev/` were
uncommitted when it started; no code differed. It replaced the previous
reference, `baseline_20260903` (code `5020879`, 2026-09-03/04, 9.92 h),
after comparing with it: no config flagged over the 56 KS tests
(`promotion_20260906/compare_vs_baseline.md`; also `compare_vs_item1.md`
against the intermediate population of 2026-09-05 and
`peak_rss_vs_item1.md`, the runner's memory high-water mark per config).
The previous reference's sidecars and summary are in git history before
the promotion commit; its traces stay under
`dev/scripts/runs/golden/baseline_20260903/` on the generating machine.
Regenerate with `bash dev/scripts/regenerate_baseline.sh golden`
(resumable; refills this directory at the end); extend with `python
dev/scripts/golden_trace.py run --suite golden --seeds 20-49 --workers 1
--out dev/scripts/runs/golden/item7_20260906` on the same code (`bdaf322`
or a later commit that the replay reports `identical` against these
traces) and copy the new sidecars and `summary.md` here.
