# Real-data reference extension, 2026-09-12

This extension adds **120 runs** on the real-data benchmark targets of
[benchmark-realistic-targets.md](../../plans/benchmark-realistic-targets.md):
the Bayesian timing model and the multisensory causal-inference model on two
subjects, as four configurations at seeds 0–29 each. The combined reference
`reference_990_20260912` has **990 JSON/NPZ pairs across 23 configurations**,
including **250 noisy runs across seven configurations**. The 870 pairs of
the previous reference are unchanged; their records are
[`noisy_extension_20260907`](../noisy_extension_20260907/README.md) and
[`extension_20260907`](../extension_20260907/README.md).

| New configuration | Target | D | Noise SD | Budget |
|---|---|---:|---:|---:|
| `multisensory_s1_D6` | multisensory causal inference, subject 1 | 6 | – | 400 |
| `multisensory_s1_D6_noise1.3` | multisensory causal inference, subject 1 | 6 | 1.3 | 400 |
| `multisensory_s2_D6_noise1.3` | multisensory causal inference, subject 2 | 6 | 1.3 | 400 |
| `timing_D5_noise2.2` | Bayesian time-interval reproduction | 5 | 2.2 | 350 |

The noise levels are the 2020 noisy-VBMC paper's (the IBS noise at the MAP)
and the budgets its 50 (D + 2) evaluations. The ground truths are the
importance-weighted populations of `dev/scripts/data/truths/`, generated
on 2026-09-11/12 and described in the plan. Runs use the standard VIQR
acquisition for the noisy configurations and the current defaults
throughout; no option changed for these additions.

## Results

All 120 runs produced complete traces and finite comparison metrics; no run
produced an error file. The launcher validated every case's artifact set
(trace, sidecar, boost capture, boost report) against a hash-verified
completion record.

| New configuration | Runs | Converged | Reached budget | Usable | Median evals | Mean optimizer time |
|---|---:|---:|---:|---:|---:|---:|
| `multisensory_s1_D6` | 30 | 30 | 0 | 28/30 | 162 | 1.35 min |
| `multisensory_s1_D6_noise1.3` | 30 | 30 | 0 | 5/30 | 205 | 3.09 min |
| `multisensory_s2_D6_noise1.3` | 30 | 30 | 0 | 22/30 | 208 | 3.32 min |
| `timing_D5_noise2.2` | 30 | 29 | 1 | 28/30 | 215 | 4.10 min |

“Usable” is the summary's criterion: evidence error < 1, gsKL < 1 and
MMTV < 0.2. The noisy runs terminate on the reliability index well inside
their budgets rather than spending them; one timing run reached its 350
evaluations. Noisy multisensory subject 1 is the hardest configuration of
the whole reference in posterior accuracy (median gsKL 2.4, median MMTV
0.22), with the evidence error at its usual level (median 0.37): the
reference records this outcome rather than treating every completed run as
an accurate fit. The [baseline summary](../baseline/summary.md) has the
full distributions.

The returned posteriors are under-dispersed on every real-data target.
The ratio of VBMC's marginal SD to the truth's, median over the 30 seeds
with quartiles:

| Configuration | Parameter | SD ratio (VBMC / truth) | Mean error / true SD |
|---|---|---|---|
| `timing_D5_noise2.2` | `w_s` | 0.68 [0.64, 0.78] | +0.16 |
| | `w_m` | 0.76 [0.70, 0.79] | −0.07 |
| | `mu_p` | 0.85 [0.81, 0.89] | −0.09 |
| | `sigma_p` | 0.72 [0.66, 0.79] | +0.16 |
| | `lambda` | 0.91 [0.88, 0.95] | −0.08 |
| `multisensory_s1_D6` | `sigma_vest` | 0.65 [0.61, 0.86] | +0.28 |
| | `sigma_vis` (low, medium, high) | 0.81, 0.83, 0.90 | −0.14, −0.16, −0.12 |
| | `kappa`, `lambda` | 0.94, 0.98 | −0.01, −0.02 |
| `multisensory_s1_D6_noise1.3` | `sigma_vest` | 0.38 [0.34, 0.49] | +0.61 |
| | `sigma_vis` (low, medium, high) | 0.66, 0.63, 0.88 | −0.46, −0.51, −0.37 |
| | `kappa`, `lambda` | 0.94, 0.93 | +0.00, −0.08 |
| `multisensory_s2_D6_noise1.3` | `sigma_vest` | 0.76 [0.69, 0.82] | +0.00 |
| | `sigma_vis` (low, medium, high) | 0.73, 0.92, 0.95 | −0.29, −0.21, −0.03 |
| | `kappa`, `lambda` | 0.81, 0.83 | −0.18, +0.30 |

On timing, `w_s` and `sigma_p` come out about 30 % too narrow across all
30 seeds, which the four smoke runs of 2026-09-11 had suggested; the ELBO
sits at the true log evidence (median ELBO − lnZ +0.08 [−0.07, +0.26]).
On both multisensory subjects the ELBO is about 0.4 below the true log
evidence (median −0.39, −0.37 and −0.37 for the three configurations), and
the noise narrows the vestibular-noise marginal of subject 1 to 0.38 of its
true SD while shifting its mean by 0.6 SD. These are findings about the
current defaults on these targets, recorded for the noisy-acquisition work
that is assessed on this reference.

## Source, certification and environment

All 120 runs used a detached worktree of `dev-next` at
`fc50ee1a0dd9f9252239e53d86da699f30f3500c` and the gpyreg checkout at
`a2f8ddce867f502e29717959cf0ff3529f598618`, both clean. The launcher
verified before every case that PyVBMC, gpyreg and the helper modules were
imported from those checkouts and that the dependency versions, Python
version and thread settings matched the manifest: Python 3.12.6, NumPy
2.5.2, SciPy 1.18.1, gpyreg 1.1.0, cma 4.4.4; one worker, one BLAS thread,
`vectorized_target=False`, `performance_calibration="off"`,
`tol_elcbo_boost=0.1`. Every sidecar records `fc50ee1`, not dirty.

The numerics of the default path at `fc50ee1` equal those of the frozen
treatment `68a43db` of the population campaigns, which the PI accepted as
the numerics of 1.5: the commits between them add S-VBMC (a separate
subpackage imported on request) and opt-in noisy-acquisition options, none
of them active here. Two gates certified this before launch, both run from
the frozen checkout:

- the oracle check `make_oracle_fixtures.py --check --exact` reported 11 of
  11 fixtures bit-exact against the committed references;
- `golden_replay.py` on the five default cases (Gaussian, banana,
  half-normal, cigar and noise-SD-1 Rosenbrock at seed 0) against the
  `68a43db` traces of `population_overnight_20260910` reported identical
  stored loop and final and identical initial design for every case, zero
  flags, 3.7 minutes ([frozen_checkout_replay.md](frozen_checkout_replay.md)).

The [launch manifest](launch_manifest.json) embeds both records with the
allocation, options and dependency versions.

The supervisor ran from 10:11:48 to 16:16:39 local UTC+03 on the benchmark
laptop, **6 h 05 min**, with the four configurations in sequence
(noiseless subject 1, noisy subject 1, noisy subject 2, timing). Summed
optimizer time was 5.93 h. Light interactive use was permitted; the
timings are observations, not a controlled speed comparison. The plan's
estimate of 2 to 3 minutes per timing run was low: a run spends 13
seconds on the target and the rest in the VIQR acquisition.

| Cohort | Pairs | Recorded code | Dirty flag |
|---|---:|---|---|
| Historical original | 280 | `18a236c` | false |
| First extension | 420 | `7314a6a` | false |
| Second extension | 110 | `7314a6a` | true; historical documentation changes |
| Noisy extension | 60 | `623f5cd` | false |
| Real-data extension | 120 | `fc50ee1` | false |

## Join, checks and artifacts

`dev/scripts/reference_join.py join` produced the combined reference. It
repeated the launcher's completion check on every new case, verified the
870 previous pairs and their summary against the tracked
[2026-09-07 manifest](../noisy_extension_20260907/sha256_manifest.json),
copied both populations byte for byte into
`dev/scripts/runs/golden/reference_990_20260912/` (every copy hashed
against its source and, for the new pairs, against the completion record;
all 990 archives passed their ZIP integrity checks), wrote the combined
summary, the new [SHA256 manifest](sha256_manifest.json) (990 pairs and
the summary; JSON and Markdown hashed with line endings normalized to LF,
NPZ as raw bytes; every entry of the previous manifest preserved), copied
the 120 new sidecars and the summary into the tracked
[`baseline/`](../baseline/), and wrote the [validation record](validation.json),
which also pins the frozen launcher, the frozen helper modules and the
four artifacts of every new case, since the boost captures stay with the
gitignored campaign.

The [even/odd null check](even_vs_odd.md) on the combined population
passed: no configuration flagged across 92 KS tests, Holm alpha 0.05. The
new configurations have 15 seeds per half; this null check does not
establish equivalence or remove the low power of the smaller cohorts.

The [final replay](final_replay.md) of seed 0 of each new configuration,
run from the frozen checkout against the combined traces, reported
identical stored loop and final and identical initial design for all four
cases, zero flags. So the replay gate is bit-exact on the real-data
configurations with the current code, while the older configurations
still part early because their traces predate the 1.5 numerics. The
returned posterior's transformer is not stored by any trace and remains
uncertifiable.

Local artifacts on the benchmark machine (gitignored) are under
`dev/scripts/runs/population_realdata_20260912/`: the frozen `source/`
worktree, `certification/` (the oracle log and the five-case replay),
`launch_manifest.json`, `process.json`, `launcher.*.log`, `results/` with
the traces, sidecars, boost captures, per-case logs and completion
records, `final_replay/`, the join logs and the width analysis
(`analyze_widths.py`, `widths.json`). The gpyreg checkout is
`population_overnight_20260910/gpyreg/`. The combined traces are under
`dev/scripts/runs/golden/reference_990_20260912/`; the previous
`reference_870_20260907/` remains unchanged. No release asset is
published by this extension.
