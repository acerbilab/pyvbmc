# Arm 3: the port's end of warm-up on the release gate's cases

Created 2026-10-05. Status: **decided by the PI; the pairing across
harness commits, Arm 3's branch, the guide and the brief written, checked
locally and reviewed; the commits to be named and the batch to run**.

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
   version that took the port's end for noisy targets alone would run as
   the after arm on the noiseless configurations and as Arm 3 on the noisy
   ones, and the two arms judge it with no further arm. A configuration or
   condition is noisy when the targets module gives it a noise level
   (`noise_sd`), which is where VBMC's noise handling is on. Better and
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
     criteria on the noisy stacking conditions (below).
   - **Noiseless targets** keep MATLAB's end of warm-up unless Arm 3 is
     better on some noiseless configuration of the population on those
     confirmatory tests and worse on none, nor on a noiseless pool
     condition, and its stacking meets the criteria on the noiseless
     stacking conditions.
   - **The pools**: for each condition, the runs paired by seed over the
     seeds that both pools ran and verified, the exact two-sided
     signed-rank test on evidence error, gsKL and MMTV, and the exact
     McNemar test on passing the pool's filters, in one Holm family over
     the conditions and tests at α = 0.05; a condition is worse when the
     family rejects one of its tests with Arm 3 the worse.
   - **The stacking**: on the conditions of the kind of target judged,
     Arm 3's stacking meets criteria 1, 2 and 4 of the
     [campaign plan](svbmc-benchmark-campaign.md), read as "The
     comparisons" says, and fails criterion 3 on no condition but
     `student_D8_noise3_svbmc`.
   - **What the tests cannot see**: a case that failed or was given up in
     either arm leaves its seed's pair out, and a test that cannot be
     computed enters its family at p = 1; both are listed with the reading
     and go to the PI before the rule is applied.
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
   (`single_run/added.md`), its added bias (the column `two_level_full
   added`) lies within −0.25 to +0.15 nats, bounds included, at every noisy
   condition and `M`, and its absolute value is smaller than the capped
   headline's (`capped_I_median added`) on `student_D8_noise3_svbmc` at
   every `M`; the headline of noiseless stacks stays the raw ELBO, which it
   is, if its added bias (the median of `raw added [CI]`) lies within ±0.1
   nats, bounds included, at every noiseless condition and `M`. Each value
   is the median as the analysis prints it, to two decimals. If a clause
   fails, that headline stays as it is and the reading goes to the PI. The
   release pools gave −0.23 to +0.15 for the two-level estimate and −0.01
   to +0.05 for the raw noiseless ELBO at `M` = 2 to 32; the stage D pools
   gave the same ranges through `M = 16`, and at `M = 32` up to +0.19 and
   +0.11, which these bounds refuse (open with the PI).
7. **Arm 3's data.** If the port's end of warm-up is adopted, Arm 3's
   population and pools are the release's public archives in place of the
   after arm's; otherwise they stay internal, as Arm 0's do.

## Design

### Arm 3's code

The branch `dev-arm3-port-warmup` holds two changes over the commit that runs
Arm 0: the improvement window of `_check_warmup_end_conditions`
(`pyvbmc/vbmc/vbmc.py`) as it was before W2-1, and `recompute_lcb_max` off in
`advanced_vbmc_options.ini`, with the tests that pin either. At the default
options warm-up then ends when it ended at `f91fdf0`, the before arm's code,
and the noisy Rosenbrock experiment of the
[note](../results/2026-10-05-noisy-rosenbrock-warmup.md) made these two changes
alone, so that at the default options warm-up ends by `f91fdf0`'s rule. What
happens once warm-up ends stays the release code's: the warping clocks that
start at its end (W2-3, `fb8a12e`, which the port review's ledger lists among
the fixes that move default trajectories), the reset of the running covariance
of the GP hyperparameters (W3-9, `f954c63`, which no number of a run at the
default options reads) and the order of ties in the trim (W3-16, `2ff2dfe`). So
do two guards of the check that act only where `f91fdf0` would raise: maxima
that pass over NaN (from `8e591ff`) and the empty window (W2-14, `54f0f19`).
The branch's changelog and the catalogue of differences from MATLAB in
`pyvbmc/vbmc/README.md` still describe the release code, and the description of
`recompute_lcb_max` its value in Arm 3 alone; all three are rewritten if the
port's end of warm-up is adopted. If it is not, the branch leaves the working
line as `retain/arm3-port-warmup`, cut at its last commit, which the records
cite (`AGENTS.md`, "Branches").

### The pairing across harness commits

Two arms of the harness's own code pair seed by seed (`prepare --pair`,
`analyze_population_run.py --arms`), and the arm that rescores is the
candidate, whose code rescores both. The release gate's arms shared one
harness checkout, and the before arm ran other code from a tree that
`PYVBMC_SOURCE` named. Arm 3 cannot share the after arm's checkout: its
package is the checkout's own, at a later commit than the after arm's,
and the after arm, finished and recorded, does not rescore again. So a
pair may have harness checkouts at different commits, on the terms of the
comparison of a legacy arm: both checkouts are clean, both identities
hold the files that build and score a run (`RUN_FILES`), equal, and the
same imported versions, their package code differs
(`package_numerics_differ`), and the arms share the allocation, the
options, the confirmatory family, the node family and the environment.
The pairing does not compare the arms' gpyreg checkouts, which are both
`v1.4.0` here; the analysis records each.
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
candidate, and likewise `fresh_release` and `fresh_arm3`, which run from a
checkout that holds the arms' commits. Each test's paired change is the
candidate's value less the reference's. The signed-rank test compares the
sums of the ranks of the positive and the negative changes (zeros left
out, ties at their mean rank), so its direction is theirs and not the
median's: a test of an error metric that its Holm family rejects finds
Arm 3 worse when the positive changes' rank sum is the larger and better
when it is the smaller, and a McNemar test of usability finds it worse
when the usability it loses outnumbers what it gains. The p-values the
rule reads are the analysis's unadjusted `pvalue` for a single test and
its Holm decision for a family.

The rule's tests that the analysis does not report as such:

- **The fresh seeds' MMTV.** The one-sided p-value is half the exact
  two-sided `pvalue` that the analysis reports for the MMTV of
  `rosenbrock_D2_noise3_production`, when the positive changes' rank sum
  is the smaller, as computed from the paired changes; otherwise the test
  does not reject. The analysis's `statistic` is the smaller of the two
  rank sums, whichever way the changes go, and gives no direction.
- **The pools' family**, from the metrics and the filter verdict that each
  verified run's completion record holds (`svbmc_pool_run.py`: `metrics`,
  `verdict.passes`), read from the two pools' archives in their draft
  releases (`release-gate-pools-20261002` and Arm 3's) on the developer's
  machine, since the pools' tracked copies hold the selection and the
  summary alone; nothing of the archives is committed. The family holds 32
  tests, four for each of the eight conditions, fixed in size: a test with
  no finite pair enters it at p = 1, and the pairs with a metric that is
  not finite in either arm are left out of that metric's test and counted.
  The metrics are the in-run ones, not rescored: Arm 3's rescoring of the
  after arm counts the cases that Arm 3's code scores as the release code
  did.
- **The stacking's criteria**, from Arm 3's stacking `summary.md`, read as
  the release pools' stacking was (`dev/TODO.md`, the final large-scale
  check): criterion 2 by its column of flagged cells, criterion 4 by the
  runtime ratio, criterion 3 by its gate column at the `M` where both
  arms ran, and criterion 1, which the summary prints no verdict for, by
  each condition's median of the largest weight difference against the
  release pools' (open with the PI).
- **The S-VBMC headline's bounds** (decision 6) read
  `single_run/added.md` of Arm 3's analyses, an asset of the draft
  release `release-gate-stacking-arm3-<date>`, which holds the cluster's
  details and stays out of the repository.

The script that applies these tests is written and committed before the
batch's results reach the developer's machine (open with the PI).

Arm 0's `rescore-arms` rescores Arm 3 with the release code, for a
comparison of Arm 0 with Arm 3 should the port's end of warm-up be
adopted. The analysis of a legacy reference refuses a rescoring that is
not made by the candidate's package code, so that comparison needs it to
accept a rescoring from a package that differs from the candidate's in
code that no metric calls, when every candidate case's rescored metrics
equal its in-run metrics, which the analysis already requires of the
candidate. If the port's end is adopted, the promotion of the new
reference needs more than `reference_promote.py` holds: it takes one
after arm, whose code the package must equal, and an accepted assessment
whose candidate is that arm.

If the port's end is adopted for one kind of target alone, which public
archives the release attaches, the after arm's, Arm 3's or each for its
own configurations, is the PI's to name with the decision; both arms'
public assets exist.

### Cost

From the release gate's accounting: Arm 3's population about 175
CPU-hours (the after arm's, its noisy runs a few iterations longer), its
pools about 220 and its stacking about 17; the fresh seeds 200 runs of a
two-dimensional noisy target, under 10. With Arm 0's 500, about 925.

## Live checklist

- [x] The PI's decisions (2026-10-05).
- [x] The pairing across harness commits: `prepare --pair`, the analysis,
  their tests and documentation, on the branch `feat-arm3-pairing`.
- [x] Arm 3's branch, `dev-arm3-port-warmup`: the end of warm-up and the
  package's tests that pin it.
- [x] A local pair across the two commits, through the finish's `rescore`
  and the analysis; the harness's test modules on Arm 3's code.
- [x] The operator's guide and the brief for the batch.
- [x] Review (two fresh reviewers, read-only) and its fixes.
- [ ] `feat-arm3-pairing` merged into `dev-next`, its merge Arm 0's
  commit; Arm 3's branch cut again from it; both pushed; the brief names
  the two commits, which the PI confirms at the launch.
- [ ] The batch on the cluster, the hand-back.
- [ ] The script of the pools' family, and the comparisons: after
  against Arm 3, the fresh seeds, the pools, the stacking; the rule
  applied; the PI's decision.
- [ ] If the port's end of warm-up is adopted: the analysis's extension
  for comparing Arm 0 with Arm 3, the promotion's for an adopted arm ("The
  comparisons"), and the branch's documentation; if it is not, the branch
  retained.

## Worklog

- 2026-10-05: the PI's decisions above, after the noisy Rosenbrock
  investigation (the note) and the release pools' reading (`dev/TODO.md`,
  the S-VBMC ELBO headline selection).
- 2026-10-05: the pairing across harness commits (`a4b2c0e6` on
  `feat-arm3-pairing`), with tests of `pair_differences`, of `prepare --pair`
  and of the analysis across two commits; the release gate's arm comparison,
  run again on its tracked copies with that code, gives the same
  `assessment.json` byte for byte. Arm 3's branch, first cut from `a4b2c0e6` in
  the worktree `../pyvbmc-arm3` (`57a5826d`, since cut again): its warm-up test
  modules and the oracles pass. A local pair across the two commits,
  `rosenbrock_D2_noise3_production` at seeds 100 to 102 with the release code
  and with Arm 3, each from a clean checkout of its own: `prepare --pair`
  accepted it, both arms verified, Arm 3's `rescore` reproduced its own cases
  and scored the release code's three as that code had, and the analysis
  validated the pair; Arm 3's runs made 180 to 200 evaluations against the
  release code's 170 to 175 (the runs and logs on the machine that
  `dev/scripts/runs/LOCAL.md` lists, "Arm 3's pairing and branch"). From the
  developer's main checkout the pair was refused: its ignored
  `dev/scripts/data/truths/chains/` enters the data directory's SHA-256, which
  the guide now warns of. The cluster's environment-check modules on Arm 3's
  code pass, but the stacking module's, which skip without the original S-VBMC
  checkout. The guide and the brief (`ca935600`). Two fresh reviewers,
  read-only: no fault in the campaigns or analyses that the batch runs; fixed
  after them the hand-back's order (`population_v104`'s copies hold the
  rescoring of `rescore-arms`, so every copy is written after it), what the
  rescoring job exports, the public assets' own directory, the commands the
  guide lacked, the pairing's refusal of the same package code at two commits,
  the test of the node families, the pin of Arm 3's default of
  `recompute_lcb_max`, and the reading of the rule's tests ("The comparisons").
