# Arm 3's reading: the end of warm-up and the S-VBMC headline

Written 2026-10-06. The release gate's second batch ran Arm 3, the release
code with the end of warm-up that the port had before two of the port
review's fixes moved it to MATLAB's, beside the release code
([Arm 3's plan](../plans/arm-3-warmup-comparison.md);
[the note that raised the question](2026-10-05-noisy-rosenbrock-warmup.md)).
This report records the reading of its results with the guide fixed before
the runs, and the PI's two decisions of 2026-10-06: **MATLAB's end of
warm-up stays, for noisy and noiseless targets**, and **the headline ELBO of
a noisy S-VBMC stack becomes the two-level shrinkage estimate**, the raw
ELBO staying the headline of a noiseless one.

## What was read

The campaigns are the first batch's after arm (the release code, `ff3ed014`)
and pools (`dev/experiments/release_gate_20261002/`), and the second batch's
`population_arm3`, `fresh_release`, `fresh_arm3`, `pools_arm3` and
`stacking_arm3` (Arm 3 at `ad63dd5a`, gpyreg `v1.4.0`;
`dev/experiments/release_gate_20261005/`, whose README describes the runs).
Every case of them verified, and none failed.

`dev/scripts/arm3_guide.py`, committed on 2026-10-05 before the batch ran,
applies the guide (the plan's decisions 5 and 6 and "The comparisons") to
five parts:

- **the population**: `analyze_population_run.py --arms` on the tracked
  copies, the after arm against `population_arm3` (24 configurations at seeds
  0–99);
- **the fresh seeds**: the same on `fresh_release` and `fresh_arm3`
  (`rosenbrock_D2_noise3_production` at seeds 100–199);
- **the pools**: each condition's runs paired by seed between the two pools,
  from each verified run's completion record in the pools' archives;
- **the stacking**: Arm 3's stacking `summary.md`;
- **the S-VBMC headline**: `single_run/added.md` of Arm 3's two analyses of
  its pools.

The pools and the analyses are assets of the draft releases
`release-gate-pools-20261002`, `release-gate-pools-arm3-20261005` and
`release-gate-stacking-arm3-20261005`; each archive matched the SHA-256 that
its hand-back README gives, and nothing of them is committed, since they hold
the cluster's details. The guide ran on the developer's machine at
`cd842264`, each command exiting 0, and reported nothing for the PI to see
before the reading: no case failed, no test was left uncomputed, no pair was
left out, and in every pool condition the paired seeds are all the seeds
that both pools verified (350, and 480 for the ring). The population, the
fresh seeds and the stacking need the tracked copies alone; a reading of
those three parts made before this one, from a clean clone, gave the same
numbers, which this reading checked before it read the other two. The
outputs
(`guide.md`, `guide.json` and the two comparisons of arms) are under
`dev/scripts/runs/guide_arm3_20261005/` on the machine whose
`dev/scripts/runs/LOCAL.md` lists them.

## The end of warm-up

The guide reads **MATLAB's end of warm-up** for both kinds of target.

**Noisy targets.** Arm 3 takes the port's end only if all four of its
clauses hold, and two fail:

- *The fresh seeds* (the rule's main test): Arm 3's runs are usable 84 times
  against 73 (19 gains, 8 losses, exact McNemar p = 0.052), and the ranks of
  their lower MMTVs outweigh those of their higher ones (rank sums 2283
  against 1722; median 0.091 against 0.098), but not significantly
  (one-sided signed-rank p = 0.127): the clause, lower MMTV at α = 0.05,
  fails. The runs make 185 evaluations against 170 (medians).
- *No noisy configuration worse* fails on `student_D8_noise3_production`,
  whose evidence error is worse in Arm 3 (median 1.09 against 0.83 nats;
  signed-rank p = 0.0076, Holm within the configuration 0.030); its usable
  runs fall from 57 to 46 (p = 0.061) and its gsKL and MMTV lean worse.
  `logreg_D5_noise3_production` leans worse too (MMTV p = 0.019, Holm 0.076;
  usable 87 to 79), and `multisensory_s1_D6_noise1.3_production`'s gsKL
  (p = 0.047, Holm 0.19). The noisy Rosenbrock target that raised the
  question is better in Arm 3 on every measure but not significantly
  (usable 72 to 81, p = 0.108). Over the eight noisy configurations Arm 3's
  runs are usable 574 times against 579.
- *No noisy pool condition worse* holds: no test of any pool condition
  rejects in either direction (the smallest Holm p is 0.13, Rosenbrock's
  gsKL, in Arm 3's favour).
- *The stacking meets the criteria on the noisy conditions* holds: criteria
  1, 2 and 4 on every noisy condition (the median largest weight difference
  0.008 to 0.030; stacking 0.19 to 0.49 of the original's time), criterion 3
  failing on `student_D8_noise3_svbmc` alone, which the guide exempts.

**Noiseless targets.** Arm 3 would need to be better on some noiseless
configuration and worse on none. No test of a noiseless configuration or
pool condition rejects in either direction, so the clause that some
noiseless configuration be better fails; the stacking meets the criteria on
both noiseless conditions. Arm 3's noiseless runs are a little longer (the
KS screen flags the evaluation counts of `cigar_D4`, 125 to 135, and
`corr_D5`, 100 to 105, medians), at the same accuracy. Many pairs have equal
metrics in the two arms: the signed-rank tests of the pools rank 86 nonzero
differences in 350 pairs on the noiseless Gaussian mixture and 272 on the
noiseless multisensory target, 243 (Student) to 341 in 350 on the noisy
conditions and 451 in 480 on the ring.

**The PI's decision (2026-10-06):** MATLAB's end of warm-up stays for noisy
and noiseless targets, as the guide reads. The analyst recommended it on
reading these results, the same day: the port's end would trade a gain on
the noisy Rosenbrock target that the fresh seeds do not establish for a
significant loss on the noisy Student target, and would add a deliberate
difference from MATLAB. What follows (the plan's decisions 5 and 7, and
`AGENTS.md`, "Branches"):

- The release code is unchanged, and so are the reference populations that
  the promotion of the new golden reference takes: the after arm, with the
  assessment of the first batch; Phase 9's replay fingerprints and seeded
  gate runs (2026-10-04, made with the after arm's code) stand.
- The promotion's record states the cost on the noisy Rosenbrock target: in
  the release code its runs end warm-up about three iterations earlier than
  the port's did and are usable 72 times in 100 against the before arm's 87
  ([the note](2026-10-05-noisy-rosenbrock-warmup.md)).
- An item outside 1.5 tests a longer warm-up for noisy targets of few
  dimensions (`dev/TODO.md`).
- Arm 3's branch leaves the working line as `retain/arm3-port-warmup`, cut at
  `ad63dd5a`, which the records cite, and Arm 3's population and pools stay
  internal (decision 7): their public assets stay in their draft release.
- The changelog's speed of noisy runs against 1.0.4 stays 2.2 times
  ([the speed record](2026-10-06-speed-against-1.0.4.md)).

## The S-VBMC headline

Decision 6 reads the added bias of each estimate, a stack's bias against its
own Monte Carlo ELBO less the mean bias of its input runs, in the medians
over cells that `single_run/added.md` prints at two decimals. Its three
clauses hold:

- the two-level estimate (`two_level_full`) adds −0.23 to +0.14 nats at
  every noisy condition and `M` from 2 to 32, inside −0.25 to +0.20;
- on `student_D8_noise3_svbmc` its absolute added bias is below the capped
  headline's at every `M`;
- the raw ELBO of a noiseless stack adds −0.01 to +0.08 nats, inside ±0.15.

| Noisy condition | `M` = 2 | 3 | 4 | 5 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| `multisensory_s1_D6_noise3_svbmc` | −0.19 | −0.13 | −0.09 | −0.06 | −0.08 | +0.01 | +0.11 |
| `multisensory_s1_D6_noise1.3_svbmc` | −0.13 | −0.09 | −0.08 | −0.06 | +0.02 | +0.08 | +0.14 |
| `rosenbrock_D2_noise3_svbmc` | −0.08 | +0.01 | −0.05 | −0.00 | −0.04 | −0.01 | +0.05 |
| `gmm_D2_noise3_svbmc` | −0.23 | −0.23 | −0.16 | −0.14 | −0.04 | −0.13 | +0.03 |
| `ring_D2_noise3_svbmc` | −0.13 | −0.11 | −0.12 | −0.11 | −0.14 | −0.14 | −0.17 |
| `student_D8_noise3_svbmc` | −0.23 | −0.16 | −0.09 | −0.12 | −0.09 | +0.07 | +0.08 |
| `student_D8_noise3_svbmc`, capped | −0.71 | −0.89 | −0.92 | −1.11 | −1.10 | −1.36 | −1.48 |

The added bias of the two-level estimate, and on the last row that of the
capped headline, medians over 20 cells (10 at `M` = 16 and 32). Over the
same cells the raw ELBO adds +0.02 to +0.82 nats on the noisy conditions,
more at larger `M`.

| Noiseless condition | `M` = 2 | 3 | 4 | 5 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| `gmm_D2_svbmc` | −0.01 | −0.01 | +0.00 | +0.01 | +0.02 | +0.00 | +0.02 |
| `multisensory_s1_D6_svbmc` | +0.01 | +0.01 | +0.01 | +0.02 | +0.01 | +0.03 | +0.08 |

The added bias of the raw ELBO, medians over the same cells.

Arm 3's pools are the release code's in everything but the end of warm-up,
and the first batch's pools, made with the release code, give the same
picture in the same medians (their analyses' `single_run/added.md`, an asset
of the draft release `release-gate-stacking-20261002`, read on 2026-10-04):
over the conditions and `M` = 2 to 32, the two-level estimate adds −0.23 to
+0.15 nats on the noisy conditions, the capped headline −1.41 to −0.04 and
the raw ELBO +0.01 to +0.86, and on the noiseless ones the raw ELBO adds
−0.01 to +0.05. Taken with the stage D pools of September, the two-level
estimate spans −0.23 to +0.19 and the raw noiseless ELBO −0.01 to +0.11,
which the bounds of decision 6 hold with a margin.

**The two compositions of the shrinkage.** The implemented estimate adds
each run's between-run shift to its within-shrunk components, so that a
run's own-weighted level ends at its shrunk level plus whatever the
within-run stage did to it; the alternative pins each run's level to its
shrunk value (the variant `two_level_anchored` of
`dev/scripts/svbmc_shrink_elbo.py`). The port review found the difference
(finding N1 F3 of [its ledger](2026-09-23-port-correctness-review.md)), and
the PI kept the implemented composition (2026-09-19), whose within-run
correction of the level addresses the optimism of a single run that the
shrinkage across runs cannot see, for the release pools to confirm. On the
stage D pools the anchored variant added −0.02 to +0.21 nats at `M` = 3 to 5
and +0.02 to +0.42 at `M = 16`
([the stage D report](2026-09-15-svbmc-pool-comparison.md), "The inputs'
own bias, and what stacking adds"). On the noisy conditions it adds +0.00 to
+0.36 nats in the first batch's pools and +0.01 to +0.36 in Arm 3's, more at
larger `M` on every condition but the ring, against the implemented
estimate's −0.23 to +0.15 and −0.23 to +0.14. The anchored variant leaves
part of what stacking adds, so the release pools confirm the implemented
composition that the PI kept on 2026-09-19.

**The PI's decision (2026-10-06):** a noisy stack reports the two-level
shrinkage estimate in its implemented composition
(`elbo_details["shrunk_two_level"]`, the analyses' `two_level_full`) as
`stacked.elbo`, and a noiseless one the raw ELBO, as before. The change of
the noisy headline was merged into `dev-next` with this report (PR #185,
`edda2ae0`): where the shrinkage is numerically undefined, a noisy stack
reports the capped value with a warning, and
`elbo_details` keeps the raw and capped values and names the selected one
(`headline_method`). The estimate targets the optimism that choosing the
weights on noisy estimates adds; it can also remove part of the bias that
each run's ELBO carries from VBMC, and residual bias can remain, as the
`SVBMC` documentation says. The
[headline note](../2026-09-15-svbmc-headline-shrinkage.md) holds the
derivation of the choice and its open questions.

## Limits

- The guide is the analyst's working of the PI's outline, fixed before the
  runs; the PI read the results with it and was not bound by it (the plan's
  decision 5).
- Each clause of no harm is tested within its configuration or condition,
  under Holm over that family's four tests; among the eight noisy
  configurations a chance refusal is about one in five.
- The fresh seeds are one configuration's 100 pairs, and after warm-up the
  two codes' trajectories part, so the pairing is weak.
- The added bias is measured against the mean bias of the stack's input runs,
  the yardstick of the S-VBMC campaign (its plan's decision 11), not against
  the true log evidence.
