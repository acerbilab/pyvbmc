# Golden reference promotion, 2026-10-07

`reference_2400_20261007` holds 2400 runs: the 24 configurations of the
`production` suite at seeds 0–99, made with the release code of PyVBMC
1.5. They are the after arm of the release gate's population campaigns,
which ran on a Slurm cluster on 2026-10-02 from PyVBMC `ff3ed014` with
gpyreg `682585f` (`v1.4.0`), one case per task
([the Slurm plan](../../plans/slurm-benchmark-support.md), Phase 8). Their
redacted records are
[`dev/experiments/release_gate_20261002/population_after/`](../../experiments/release_gate_20261002/population_after),
and the [README of the release gate's records](../../experiments/release_gate_20261002/README.md)
names the campaign's archive, which holds the traces. Every case verified,
so no case is left out of the reference. The reference replaces
`reference_990_20260913`, 990 runs of the `golden` suite made with the
code from before the fixes of the port correctness review, several of
which move default trajectories; its
[promotion record](../promotion_20260913/README.md) and traces are kept.

`previous_reference_README.md` is `dev/golden/README.md` as it stood before
this promotion, copied verbatim by `prepare`. Its "Current reference" is
`reference_990_20260913`, and its relative links resolve from
`dev/golden/`, not from this directory.

## The assessment

The PI accepted the assessment on 2026-10-07:
[`assessment.json`](../../experiments/release_gate_20261002_assessment/assessment.json),
whose SHA-256 with LF line endings is
`2625fc50cb035ccabe7e5ec50bc6ebc7743c515eec08240c15b361a14420898d`, and its
[comparison](../../experiments/release_gate_20261002_assessment/comparison.md).
It compares the population seed by seed with the same 2400 runs of the
code at the start of the port correctness review (`f91fdf0`, with gpyreg
`9e70e6b`), the before arm of the same campaigns, both rescored with the
release code.

| Matched assessment | Before | Reference |
|---|---:|---:|
| Complete cases | 2400 | 2400 |
| Converged | 2299 | 2299 |
| Usable | 2152 | 2157 |
| Target evaluations | 458,095 | 453,010 |
| Summed optimizer hours | 163.2 | 153.4 |

Usability requires evidence error < 1, gsKL < 1 and MMTV < 0.2; there are
125 paired gains and 120 losses. The confirmatory family, fixed before the
runs, holds 96 tests: exact signed-rank tests of the evidence error, gsKL
and MMTV and exact McNemar tests of usability, for each configuration,
under one Holm correction at 0.05. None rejects. The KS screen flags the
evaluation counts of `cigar_D4` and `rosenbrock_D2_noise3_production`,
whose runs stop earlier (median ratios 0.93 and 0.92).

The noisy Rosenbrock target is worse in the reference on every measure:
its runs are usable 72 times in 100 against the before arm's 87. Two fixes
of the port review, W2-1 and W2-2, end its warm-up about three iterations
earlier, as MATLAB VBMC does
([the note](../../results/2026-10-05-noisy-rosenbrock-warmup.md)). Arm 3,
the release code with the port's earlier end of warm-up, run on every
configuration of the release gate, did not establish the gain on fresh
seeds and made the noisy Student target worse, so MATLAB's end of warm-up
stays (PI, 2026-10-06; [the reading](../../results/2026-10-06-arm-3-reading.md)).
This reference therefore records that cost. A longer warm-up for noisy
targets of few dimensions is a question for after 1.5 (`dev/TODO.md`).

## The replay fingerprints and the gate runs

Exact replay depends on the machine and its BLAS, so the traces that
`golden_replay.py` compares with by default are the reference's replay
fingerprints: one run at seed 0 of each configuration, made on
2026-10-04 on the machine that `dev/scripts/runs/LOCAL.md` lists, from
`dce4e18c`, whose package and run code are `ff3ed014`'s, with gpyreg
`682585f`, one BLAS thread and historical calibration budgets, in 43
minutes. Each was judged against the population's accuracy envelope
([the report](fingerprint_replay.md)). One lies outside its envelope,
`corr_D5`, on its evidence error, gsKL and MMTV (0.048, 0.012 and 0.018),
with 105 evaluations, inside the population's range. The population's own
runs lie outside their envelopes at a rate of 0.0375, at which one or more
of 24 has a probability of 0.60, above the promotion's threshold of 0.01.
None of the 24 equals the cluster's run of seed 0 in every semantic final
field, as expected across machines.

The port review's six seeded gate runs, two of them with a prior object
(`dev/scripts/seeded_gate_runs.py`), were recorded twice on that machine
from the same commit and reproduced in all 138 arrays
([their record](gate_runs.json)).

The fingerprints' traces and the gate runs' files are local, under
`dev/scripts/runs/golden/reference_2400_20261007_fingerprints/` on that
machine (gitignored); the [manifest](sha256_manifest.json) hashes them
with the 2400 sidecars.

## Checks of the promotion

`prepare` checked that `dev/golden/baseline/` held `reference_990_20260913`
as its manifest says, and its traces against that manifest; the after
arm's redaction record, allocation, options, clean source trees,
verification report and rescored metrics; the assessment against the
accepted SHA-256; the fingerprints and the gate runs as above. The
[even/odd check](even_vs_odd.md) of the population has no flags in its 96
KS tests. The [replay of the five defaults](final_replay.md) against the
fingerprints, at `948bac0d`, is identical in every non-timer NPZ loop and
final array, semantic final-result field and initial design, in 3.7
minutes. No trace stores the returned posterior's transformer, so replay
reports that field as uncertifiable. The [validation record](validation.json)
holds the provenance and the results of every gate.

The procedure is [promote.py](promote.py), `dev/scripts/reference_promote.py`
as it ran, from the repository root of the machine of the fingerprints:

```console
python -u dev/scripts/reference_promote.py prepare \
    --after dev/experiments/release_gate_20261002/population_after \
    --assessment dev/experiments/release_gate_20261002_assessment \
    --accepted-assessment 2625fc50cb035ccabe7e5ec50bc6ebc7743c515eec08240c15b361a14420898d \
    --fingerprints <fingerprints> --gate-runs <gate runs> \
    --record dev/golden/promotion_20261007
python -u dev/scripts/reference_promote.py replay \
    --record dev/golden/promotion_20261007 --out <replay>
python -u dev/scripts/reference_promote.py publish \
    --record dev/golden/promotion_20261007 --final-replay <replay>
```

`publish` replaced the sidecars and the summary in `dev/golden/baseline/`
with the reference's and rewrote the documents that name the current
reference. This promotion changes no solver numerics and publishes no
release asset.
