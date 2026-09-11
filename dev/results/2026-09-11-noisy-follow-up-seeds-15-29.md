# Noisy follow-up: seeds 15–29 of Rosenbrock SD3 and logistic SD3

## Findings

The shifts that the first stage found on seeds 0–14 of the two difficult
noisy configurations do not replicate on the independent seeds 15–29 of
the same frozen treatment. On noisy Rosenbrock SD3, where seeds 0–14 had
shown a fourfold larger median gsKL, all three posterior metrics move in
the candidate's favor on seeds 15–29 (11 or 12 of 15 seeds improve) and
usability rises from 10 to 14 of 15. On noisy logistic SD3, where seeds
0–14 had lost two usable runs, seeds 15–29 gain five and lose one. None
of the eight pre-specified confirmatory tests rejects; the smallest
Holm-adjusted p-value is 0.44, and it belongs to an improvement.

Pooled over seeds 0–29, the paired median changes of every metric in both
configurations lie within ±0.025 of zero, and usability is 27/30 against
23/30 (Rosenbrock) and 25/30 against 23/30 (logistic). The pooled 303-run
candidate population has 287 usable runs against 273 on the matching
reference seeds, with no rejection in the 76-test KS screen or the 95-test
paired family. The first-stage finding was seed-level variability of hard
noisy targets, not a population regression, and nothing here identifies a
numerical change to investigate.

No further sampling is recommended. Whether to accept this evidence and
promote the 303-run population to the new reference is the remaining
decision; the reference and its identity are preserved either way.

## Design

The extension ran as a separate campaign of the same treatment
(`dev/scripts/runs/population_extension_20260911/`), reusing the overnight
campaign's frozen `source/` (`68a43db`, numerical code `de686f7`) and
`gpyreg/` (`a2f8ddc`) checkouts. The launcher's recorded identity is
identical to the overnight campaign's: commits, import paths, dependency
versions (Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1, gpyreg 1.1.0, cma
4.4.4), thread settings and options. Seeds 15–29 of `logreg_D5_noise3`
and `rosenbrock_D2_noise3` have reference runs (50 and 30 seeds), so every
new case pairs with a reference case.

The confirmatory analysis was fixed in the launch manifest before the run:
seeds 15–29 paired by seed with the reference, exact two-sided signed-rank
tests of evidence error, gsKL and MMTV and exact McNemar tests of usability
for each configuration, eight tests under one Holm correction at alpha
0.05, with Rosenbrock gsKL and MMTV as the primary outcomes. Seeds 0–14
generated the hypothesis of a deterioration; seeds 15–29 test it. The
signed-rank null assumes independent seed differences symmetric about
zero, as in the first-stage report.

All 30 cases completed with validated completion records, no execution
errors and finite metrics. All 30 boosts were attempted and accepted. The
pooled assessment revalidated all 303 candidate cases against their
campaigns' launch records, all 870 reference sidecars against the
published hashes, and every boost decision against the production guard.

## Confirmatory results

Positive changes mean worse accuracy. Medians are of the individual
candidate-minus-reference differences. The seeds 0–14 column repeats the
first-stage tests for the same configurations for comparison; the Holm
column applies to the seeds 15–29 family only.

| Configuration and metric | Seeds 0–14: median change, improved / worsened, raw p | Seeds 15–29: median change, improved / worsened, raw p | Holm, seeds 15–29 |
| --- | --- | --- | ---: |
| Rosenbrock SD3, gsKL | +0.307, 5 / 10, 0.107 | −0.217, 11 / 4, 0.095 | 0.57 |
| Rosenbrock SD3, MMTV | +0.052, 5 / 10, 0.055 | −0.046, 11 / 4, 0.055 | 0.44 |
| Rosenbrock SD3, evidence error | +0.009, 7 / 8, 0.330 | −0.045, 12 / 3, 0.073 | 0.51 |
| Rosenbrock SD3, usability | 1 gain, 1 loss, 1.0 | 5 gains, 1 loss, 0.219 | 1 |
| Logistic SD3, gsKL | +0.043, 7 / 8, 0.934 | −0.034, 8 / 7, 0.762 | 1 |
| Logistic SD3, MMTV | +0.027, 7 / 8, 0.359 | −0.014, 9 / 6, 0.489 | 1 |
| Logistic SD3, evidence error | +0.058, 6 / 9, 0.639 | +0.008, 7 / 8, 0.890 | 1 |
| Logistic SD3, usability | 1 gain, 3 losses, 0.625 | 5 gains, 1 loss, 0.219 | 1 |

The Rosenbrock reference seeds 15–29 are harder than its seeds 0–14
(10/15 usable and 12/15 converged, against 13/15 and 15/15), whereas the
candidate is similar on both halves (13/15 and 14/15 usable). The
first-stage shift and the follow-up reversal are of comparable size and
opposite sign, which is what seed-level variability on a target with a
200-evaluation budget looks like.

### Noisy Rosenbrock D2, SD 3

| | Reference, all 30 seeds | Reference, seeds 15–29 | Candidate, seeds 15–29 | Candidate, seeds 0–29 |
| --- | ---: | ---: | ---: | ---: |
| Usable | 23/30 | 10/15 | 14/15 | 27/30 |
| Converged | 23/30 | 12/15 | 14/15 | 25/30 |
| Median gsKL | 0.109 | 0.319 | 0.114 | 0.142 |
| Median MMTV | 0.076 | 0.128 | 0.069 | 0.095 |
| Median evidence error | 0.233 | 0.239 | 0.196 | 0.214 |
| Median evaluations | 185 | 175 | 185 | 182.5 |

Pooled paired changes over seeds 0–29: gsKL −0.013 (16 improved / 14
worsened, p 0.78), MMTV −0.002 (16 / 14, p 0.89), evidence error −0.014
(19 / 11, p 0.64), usability 6 gains and 2 losses (p 0.29). The candidate's
pooled cohort medians remain above the reference's while its usability and
paired changes are better: each side has three runs with gsKL above 1
(reference seeds 8, 23 and 25; candidate seeds 6, 11 and 21), on
different seeds. The one follow-up loss is seed 21, where the candidate
converges at 165 evaluations with gsKL 1.231 and MMTV 0.208 while the
reference reached its budget with gsKL 0.059. The gains are seeds 17, 23,
24, 25 and 26; seed 23 moves from gsKL 1.177 to 0.051 and seed 25 from
1.232 to 0.411. The candidate's one budget termination on these seeds is
seed 25, which the reference also did not converge; the reference has
three (seeds 21, 25 and 29).

The candidate's pre-boost posteriors on seeds 15–29 already have median
gsKL 0.147 and MMTV 0.078, better than the reference's returned posteriors
on the same seeds, so the first-stage suggestion that the main-loop
solution had degraded is not supported.

### Noisy logistic regression D5, SD 3

| | Reference, all 50 seeds | Reference, seeds 15–29 | Candidate, seeds 15–29 | Reference, seeds 0–29 | Candidate, seeds 0–29 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Usable | 39/50 | 10/15 | 14/15 | 23/30 | 25/30 |
| Converged | 50/50 | 15/15 | 15/15 | 30/30 | 30/30 |
| Median gsKL | 0.418 | 0.385 | 0.351 | 0.417 | 0.392 |
| Median MMTV | 0.150 | 0.157 | 0.144 | 0.157 | 0.151 |
| Median evidence error | 0.191 | 0.263 | 0.391 | 0.234 | 0.246 |
| Median evaluations | 242.5 | 250 | 265 | 245 | 252.5 |

Pooled paired changes over seeds 0–29: gsKL +0.0003 (15 / 15, p 0.95),
MMTV −0.008 (16 / 14, p 0.87), evidence error +0.024 (13 / 17, p 0.66),
usability 6 gains and 4 losses (p 0.75). The follow-up loss is seed 27
(gsKL 0.365 to 0.827, MMTV 0.123 to 0.269); the gains are seeds 18, 19,
25, 26 and 28; in the reference, seeds 18, 19 and 25 had missed the MMTV
threshold by at most 0.012, seeds 26 and 28 by 0.048 and 0.030. The candidate's median evidence error on seeds 15–29 is higher
than the reference's, with a 7 / 8 split; the pooled paired change is
+0.024. Evaluations are higher on these seeds (4,010 against 3,820).

## Pooled population

| Outcome | Full reference | Reference, matching seeds | Candidate |
| --- | ---: | ---: | ---: |
| Runs | 870 | 303 | 303 |
| Converged | 853 | 293 | 295 |
| Usable | 825 (94.8%) | 273 (90.1%) | 287 (94.7%) |
| Target evaluations | 131,375 | 49,465 | 48,825 |
| Summed optimizer time | 21.638 h | 8.741 h | 8.604 h |

There are 22 same-seed transitions into usability and 8 out of it. The
recomputed KS screen (76 tests, 19 configurations, Holm alpha 0.05) flags
no configuration; its smallest p-value remains the exhaust cohort's
evidence-error improvement (0.028). The recomputed paired family (95
tests) rejects nothing; its smallest raw p-value remains Student D4 MMTV
(0.0256, all 15 runs usable). The final boost accepts 298 of 303
candidates and rejects the same five as in the first stage; usability
rises from 284 pre-boost to 287 returned, with no boost-induced loss. On
the 30 new cases, boosting turns one logistic run usable and leaves the
other 27 usable runs usable.

The exact signed-rank tests now enumerate the sign-flip distribution by
dynamic programming over midranks, which handles ties exactly at any
sample size; on the first-stage data it reproduces the committed results
bit for bit.

## Timing

The supervisor started at 12:35:47 UTC+03 and finished at 14:41:23, after
2 h 5 min 36 s. Summed optimizer time was 123.0 minutes against 96.1 for
the same reference seeds (ratio 1.22 on logistic, 1.40 on Rosenbrock),
whereas the first stage had run these configurations 6–12% faster than the
reference. Evaluation counts are comparable (6,785 against 6,480), so the
difference is machine speed: the laptop ran slower throughout, including
the last 45 minutes when nothing else was running. During the first 35
minutes the assessment tooling was exercised on the same machine (artifact
validation and a dry run on copied cases). Trajectories are deterministic
given the seed and single-threaded BLAS, so accuracy metrics and
evaluation counts are unaffected; the wall-clock figures are observations,
not a speed comparison.

## Reproduction and artifacts

Run `.venv/Scripts/python.exe dev/scripts/analyze_population_run.py` from
the repository root. Its defaults read both campaigns, revalidate all 303
cases and the 870 reference sidecars, recompute the pooled comparison and
write
[assessment.json](../experiments/population_extension_20260911/assessment.json),
[comparison.md](../experiments/population_extension_20260911/comparison.md)
and both campaign manifests under `dev/experiments/population_extension_20260911/`.
The JSON's `follow_up` entry holds the extension's per-seed changes,
confirmatory tests, the first-stage tests for the same configurations,
and its boost summary; the top-level keys hold the pooled population in
the first-stage report's layout.

Raw traces, captures, completion records and logs remain under
`dev/scripts/runs/population_extension_20260911/results/`, with the frozen
checkouts under `dev/scripts/runs/population_overnight_20260910/`. Launch
provenance is recorded in the
[population plan](../plans/final-population-benchmark.md); the first stage
is reported in the
[overnight assessment](2026-09-11-overnight-population.md).
