# Arm 3: the port's end of warm-up on the release gate's cases

Created 2026-10-05. Status: **decided by the PI; the pairing across
harness commits, the arm's branch and the brief open**.

## Purpose

The release gate's comparison of the release code with `f91fdf0`
([its record](../experiments/release_gate_20261002_assessment/comparison.md))
found the release code's runs of `rosenbrock_D2_noise3_production`
shorter and less accurate than the before arm's. Two fixes of the port
correctness review, W2-1 (`c918d12`) and W2-2 (`8e591ff`), end warm-up as
MATLAB does, about three iterations earlier than the port did
([the note](../results/2026-10-05-noisy-rosenbrock-warmup.md)). Arm 3 is
the release code with the port's end of warm-up, `f91fdf0`'s, run on the
release gate's population cases, pools and stacking in one batch with
Arm 0 ([its plan](arm-1.0.4-comparison.md)), so that the choice between
MATLAB's end of warm-up and the port's rests on every configuration of the
cluster's campaigns, by a rule fixed before the runs.

## Decisions (PI, 2026-10-05)

1. **The warm-up decision waits for Arm 3, and Arm 0 does not.** Arm 0
   runs from a commit whose package is the after arm's, as its plan
   requires; if the port's end of warm-up is adopted, Arm 0's runs stand
   and only their comparison changes.
2. **Arm 3's code** is the release code with `f91fdf0`'s end of warm-up:
   the window of the warm-up's improvement check as it was before W2-1
   (the last five iterations at the defaults, against MATLAB's three), and
   `recompute_lcb_max` off by default, which leaves the recorded maxima of
   the lower confidence bound to the check as before W2-2. Nothing else
   changes: the branch is cut from the commit that runs Arm 0. gpyreg
   `v1.4.0`, the after arm's environment and node family.
3. **Arm 3 runs the population, the pools and the stacking**: the
   `production` suite at seeds 0–99, the pools of the `svbmc_pool` suite
   with the release gate's allocation, and the stacking comparison with
   its two analyses, as the release gate's campaigns ran them (the
   [Slurm plan](slurm-benchmark-support.md), Phase 8). Arm 0 runs the
   population alone.
4. **Fresh seeds.** `rosenbrock_D2_noise3_production` at seeds 100–199,
   with the release code and with Arm 3, for the rule's main test: the
   after arm's seeds 0–99 are the runs that raised the question.
5. **The rule**, applied to the noisy and the noiseless configurations
   apart. The end of warm-up acts in every run, so Arm 3 changes both; a
   version that took the port's end for noisy targets alone would make
   the after arm's runs on the noiseless configurations and Arm 3's on the
   noisy ones, and the two arms judge it with no further arm. Better and
   worse are in accuracy (evidence error, gsKL, MMTV) and usability; the
   evaluation counts are reported as a cost and decide nothing.
   - **Noisy targets** take the port's end of warm-up only if all of:
     on the fresh seeds, Arm 3's MMTV is lower (one-sided exact signed-rank
     test on the paired seeds, α = 0.05) and its usability not worse (exact
     McNemar test, two-sided, α = 0.05, not rejecting with Arm 3 the
     worse); no noisy configuration of the population is worse in Arm 3 on
     the release gate's confirmatory tests (the default family of
     `population_run.confirmatory_family`, under Holm); no noisy pool
     condition is worse (below); and Arm 3's stacking meets the campaign's
     criteria as the after arm's did (below).
   - **Noiseless targets** keep MATLAB's end of warm-up unless Arm 3 is
     better on some noiseless configuration of the population on those
     confirmatory tests and worse on none, nor on a noiseless pool
     condition.
   - **The pools**: for each condition, the runs paired by seed over the
     seeds that both pools ran and verified, the exact two-sided
     signed-rank test on evidence error, gsKL and MMTV, and the exact
     McNemar test on passing the pool's filters, in one Holm family over
     the conditions and tests at α = 0.05; a condition is worse when the
     family rejects one of its tests with Arm 3 the worse.
   - **The stacking**: Arm 3's stacking meets criteria 1, 2 and 4 of the
     [campaign plan](svbmc-benchmark-campaign.md) on every condition, and
     fails criterion 3 on no condition but `student_D8_noise3_svbmc`.
   - If the port's end is adopted for a kind of target, it enters the
     package as a deliberate difference from MATLAB (its entry in the
     catalogue of `pyvbmc/vbmc/README.md`, the changelog), Phase 9 of the
     Slurm plan runs again from the new code, and the promotion of the new
     reference takes its populations from the arms the decision names;
     otherwise MATLAB's end stays and the promotion record states the
     cost on the noisy Rosenbrock target, with an item outside 1.5 that
     tests a longer warm-up for noisy targets of few dimensions.
6. **The S-VBMC ELBO headline waits for Arm 3's stacking.** The headline
   of noisy stacks becomes the two-level shrinkage estimate
   (`two_level_full`) if, in Arm 3's single-run analysis
   (`single_run/added.md`), its added bias lies within −0.25 to +0.15 nats
   at every noisy condition and `M`, the range of the stage D and release
   pools, and lies closer to zero than the capped headline's on
   `student_D8_noise3_svbmc` at every `M`; the headline of noiseless stacks
   becomes the raw ELBO if its added bias lies within ±0.1 nats at every
   noiseless condition and `M`. Each bound is read on the medians as the
   analysis prints them, to two decimals.
7. **Arm 3's data.** If the port's end of warm-up is adopted, Arm 3's
   population and pools are the release's public archives in place of the
   after arm's; otherwise they stay internal, as Arm 0's do.

## Design

### The pairing across harness commits

Two arms of the harness's own code pair seed by seed (`prepare --pair`,
`analyze_population_run.py --arms`), and the arm that rescores is the
candidate, whose code rescores both. The release gate's arms shared one
harness checkout, and the before arm ran other code from a tree that
`PYVBMC_SOURCE` named. Arm 3 cannot share the after arm's checkout: its
package is the checkout's own, at a later commit than the after arm's,
and the after arm, finished and recorded, does not rescore again. So a
pair may have harness checkouts at different commits, on the terms of the
comparison of a legacy arm: both identities hold the files that build and
score a run (`RUN_FILES`), equal, the same imported versions and
different package commits, and the arms share the allocation, the
options, the confirmatory family, the node family and the environment.
The candidate (Arm 3) rescores its own cases, which must reproduce their
in-run metrics exactly, and the reference's; the after arm's rescored
metrics, made by Arm 3's code, show whether the two codes score alike,
since their package differs in the end of warm-up alone, which no metric
calls.

### The campaigns

| Campaign | Harness checkout | Pairs with | Code |
|---|---|---|---|
| `population_v104` | Arm 0's commit | `--compare population_after` | 1.0.4 |
| `population_arm3` | Arm 3's commit | `--pair population_after` | Arm 3 |
| `pools_arm3` | Arm 3's commit | | Arm 3 |
| `stacking_arm3` | Arm 3's commit | `--pool pools_arm3` | Arm 3 |
| `fresh_release` | Arm 0's commit | | release |
| `fresh_arm3` | Arm 3's commit | `--pair fresh_release` | Arm 3 |

The after arm's directory is restored from its archive where it is gone,
as for Arm 0, and is never finished again. The pools and the stacking take
the release gate's allocation and grid, and the two analyses of the pools
follow the stacking, from the Arm 3 checkout. The two checkouts are a
clone and a detached worktree of it, each at its commit from its
campaigns' first submission to their hand-back. Arm 0's `rescore-arms`
rescores Arm 3 beside the after arm and Arm 0 (Arm 0's plan, "Rescoring
across harness commits"). The operator's guide (`dev/scripts/hpc/README.md`,
"Arm 3: the port's end of warm-up" and "The second batch's hand-back")
gives the commands, and the [brief](release-gate-handoff-2.md) the order,
the commits and the limits.

### The comparisons

On the tracked copies, after the hand-back: `analyze_population_run.py
--arms` with the after arm as the reference and `population_arm3` as the
candidate, and likewise `fresh_release` and `fresh_arm3`. The rule's tests
that the analysis does not report as such are computed from its paired
data: the one-sided signed-rank test of the fresh seeds' MMTV, and the
pools' family, which reads the verified runs of the two pools, paired by
seed, from the campaigns' archives in their draft releases (the pools'
tracked copies hold the selection and the summary, not each run's
metrics), on the developer's machine, where nothing of them is committed.

### Cost

From the release gate's accounting: Arm 3's population about 175
CPU-hours (the after arm's, its noisy runs a few iterations longer), its
pools about 220 and its stacking about 17; the fresh seeds 200 runs of a
two-dimensional noisy target, under 10. With Arm 0's 500, about 925.

## Live checklist

- [x] The PI's decisions (2026-10-05).
- [ ] The pairing across harness commits: `prepare --pair`, the analysis,
  their tests and documentation, on a feature branch merged into
  `dev-next`.
- [ ] Arm 3's branch: the end of warm-up and the package's tests that pin
  it, cut from the commit that runs Arm 0.
- [ ] A local pair across the two commits (a few seeds), through the
  finish's `rescore` and the analysis.
- [ ] The operator's guide and the brief for the batch (Arm 0 and Arm 3),
  with the commits and the limits.
- [ ] Review (fresh reviewers, read-only) and its fixes.
- [ ] The batch on the cluster, the hand-back.
- [ ] The comparisons: after against Arm 3, the fresh seeds, the pools,
  the stacking; the rule applied; the PI's decision.

## Worklog

- 2026-10-05: the PI's decisions above, after the noisy Rosenbrock
  investigation (the note) and the release pools' reading (`dev/TODO.md`,
  the S-VBMC ELBO headline selection).
