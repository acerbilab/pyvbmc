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
The 280 sidecars record `18a236c` with `dirty: false`, a documentation-only
descendant of `bdaf322`. The runner records Git metadata after each run;
the numerical source is unchanged between those commits. It replaced the previous
reference, `baseline_20260903` (code `5020879`, 2026-09-03/04, 9.92 h),
after comparing with it: no config flagged over the 56 KS tests
(`promotion_20260906/compare_vs_baseline.md`; also `compare_vs_item1.md`
against the intermediate population of 2026-09-05 and
`peak_rss_vs_item1.md`, the runner's memory high-water mark per config).
The previous reference's sidecars and summary are in git history before
the promotion commit; its traces stay under
`dev/scripts/runs/golden/baseline_20260903/` on the generating machine.
Regenerate with `bash dev/scripts/regenerate_baseline.sh golden`
(resumable; refills this directory at the end).

**Extension status (in progress):** Stage 3 is implemented on
`dev-next-stage3` at code `4ee612d` with records/docs `285cd74`; smoke run
`34043031387` and full-matrix run `34043071150` are green (all nine matrix jobs).
It has not been merged into `dev-next`, the integrated gates have not
run, and no freeze commit is recorded. After both CI runs pass, merge Stage 3 into
`dev-next`; on that integrated checkout require
`make_oracle_fixtures.py --check --exact`, an `identical` default
`golden_replay.py`, and the full local suite. Freeze the passing integrated
commit for both extension nights and keep `vectorized_target=False` in
every reference configuration.

Night 1 extends the 14 original configurations with seeds 20–49 (about
7 h); night 2 adds `cigar_D8` and `student_D8` at seeds 0–49 and
`cigar_D15_exhaust` at seeds 0–9 (about 5 h). Use the same frozen commit
for both. They may run as one chain only when the PI explicitly authorizes
its start; batch 2 starts only after batch 1 and its checks succeed. The
exact commands, null checks and publication steps are in
roadmap pickup 3f. The original 280 sidecars and `.npz` traces remain
unaltered records of `18a236c` (numerical code `bdaf322`); the newly generated sidecars must retain
the actual frozen integrated code SHA. Numerical identity is established
by the replay, not by changing historical provenance. Copy the combined
sidecars and `summary.md` here after each extension. No latent bug fix,
including a trajectory-neutral fix, lands until both nights finish.
