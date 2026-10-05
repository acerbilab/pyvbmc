# The noisy Rosenbrock target of the release gate: the end of warm-up

Written 2026-10-05. The release gate's comparison of the release code with
`f91fdf0` ([its record](../experiments/release_gate_20261002_assessment/comparison.md)) rejects no
confirmatory test, but one configuration, `rosenbrock_D2_noise3_production`,
moves the same way on every measure. This note records why, and the
decision it leaves to the PI.

## What the comparison shows

At seeds 0–99 the release code's runs of `rosenbrock_D2_noise3_production`
are usable 72 times against the before arm's 87, their median MMTV is
0.115 against 0.081, MMTV is worse in 65 of 100 paired seeds (p = 0.002
before the Holm correction, 0.22 after) and gsKL in 61, and the runs make
15 fewer evaluations (median 170 against 185), which the KS screen flags.
At noise 1 the runs shorten too (135 to 125 evaluations), with no loss of
accuracy. The other noisy configurations keep their evaluation counts;
`student_D8_noise3` and `logreg_D5_noise3` are better in the release
(usable 49 to 57 and 78 to 87), `lumpy_D10_noise3` a little worse in gsKL,
and the eight noisy configurations together are usable 583 times against
579.

## Where the evaluations go

The traces of both arms (the draft releases' archives) show that the
release code's runs end warm-up about three iterations earlier: at noise 3,
median iteration 6 against 9 (35 evaluations against 50); at noise 1,
iteration 4 against 7, the earliest the check allows. The rest of the run
is as long as before (27 iterations at noise 3). Two fixes of the port
correctness review moved the end of warm-up to MATLAB's: W2-1 (`c918d12`,
the "recent improvement" window, five iterations in the port against
MATLAB's three) and W2-2 (`8e591ff`, MATLAB's recomputed LCB maxima, never
wired up in the port). The review's
[ledger](2026-09-23-port-correctness-review.md) lists both among the fixes that
move default trajectories and that the benchmark had not read for accuracy.
Within the before arm, runs whose warm-up happened to end by iteration 6
were usable 9 times in 12 (0.75), against 48 in 53 (0.91) when it ended at
iteration 9 or later.

## The experiment

On the developer's machine, one process at a time with one BLAS thread,
`rosenbrock_D2_noise3_production` at seeds 0–49 with the release code and
with the release code whose end of warm-up is `f91fdf0`'s (W2-1's window
undone in the worker's process, `recompute_lcb_max=False`), each run as the
population harness runs it (`golden_trace.run_task` with the harness's
options). The runs (`warmup_exp/`), the script that made them
(`scripts/warmup_exp.py`) and its logs are under
`dev/scripts/runs/release_gate_20261002/` on the machine whose
`dev/scripts/runs/LOCAL.md` lists them ("The release gate's hand-back and
Phase 9").

| Seeds 0–49 | Release code | Release, old warm-up | Cluster before arm | Cluster after arm |
|---|---|---|---|---|
| Usable | 37 | 41 | 40 | 38 |
| MMTV, median | 0.121 | 0.090 | 0.085 | 0.102 |
| gsKL, median | 0.269 | 0.096 | | |
| Evidence error, median | 0.214 | 0.139 | | |
| Evaluations, median | 170 | 185 | 185 | 170 |
| Warm-up ends, median iteration | 6 | 8 | | |

The old warm-up gives back the before arm's evaluation count and about its
accuracy: better MMTV in 28 of 50 seeds (signed-rank p = 0.09), gsKL in 30
(p = 0.07), evidence error in 27 (p = 0.08); usable in 11 seeds where the
release code's run is not, and not in 7 where it is. After warm-up the two
codes' trajectories part, so the pairing is weak. The two warm-up fixes are
the main cause of the shorter runs on this target and of most of its loss
of accuracy, at a significance that stays borderline at 50 seeds. The
cluster's gap at seeds 0–49 is 40 against 38 usable runs; the rest of the
87 against 72 lies in seeds 50–99.

## The decision (PI, open)

- **A. Keep MATLAB's end of warm-up** (the analyst's recommendation). The
  fixes are faithful to MATLAB, the cost falls on one hard noisy target,
  and the population, the fingerprints and Arm 0 stand. The cost is
  written into the promotion record, and an item outside 1.5 tests a longer
  warm-up for noisy targets of few dimensions on the benchmark.
- **B. Keep the longer warm-up as a deliberate difference from MATLAB.**
  It needs the benchmark first, to show that nothing else gets worse, then
  the release code's population again on the cluster (about 165
  CPU-hours), Phase 9 again, and Arm 0 waits for it (its harness commit
  holds the release code's package).
