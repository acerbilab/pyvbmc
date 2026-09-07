# Reference extension completed 2026-09-07

The authorized two-batch chain added 530 runs to `item7_20260906` and passed
every required check. The reference now contains **810 JSON/NPZ pairs**:
16 configurations at seeds 0–49 and `cigar_D15_exhaust` at seeds 0–9.
All runs produced finite comparison metrics and complete, readable traces;
there were no error files. The original 280 pairs remain byte-identical.
The 800 ordinary runs reported convergence; the 10 exhaust runs reached their
intended 750-evaluation limit. Successful execution does not mean every
posterior meets the summary's separate accuracy-based `usable` criterion.

## Execution and provenance

The chain ran from 2026-09-06 20:01:12 to 2026-09-07 08:55:36, local
Europe/Kiev time (UTC+03), including both replays and checks: 12 h 54 min.
One numerical worker ran at a time, with BLAS single-threaded. Light laptop
use was permitted; these timings are observations, not a controlled speed
comparison.

| Segment | Runs added | First start | Last finish | Summed run time |
|---|---:|---|---|---:|
| Original 14 configurations, seeds 20–49 | 420 | Sep 6 20:04:41 | Sep 7 03:26:04 | 7.30 h |
| `cigar_D8`, `student_D8`, seeds 0–49 | 100 | Sep 7 03:26:49 | Sep 7 06:10:20 | 2.71 h |
| `cigar_D15_exhaust`, seeds 0–9 | 10 | Sep 7 06:10:26 | Sep 7 08:52:40 | 2.70 h |

All new numerical work used detached commit
`7314a6afe158c5673775ca79601b88b3913ae383`, still named by the non-moving
`reference/stage3-20260906` branch. The original `.venv/Scripts/python.exe`
was retained: Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1, gpyreg 1.1.0,
cma 4.4.4; editable gpyreg was clean at
`a2f8ddce867f502e29717959cf0ff3529f598618`. Every population invocation
explicitly passed `{"vectorized_target": false}` as `--options` through a
Python subprocess argument list, with `--workers 1` and
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.

| Cohort | Recorded SHA | Dirty | Explanation |
|---|---|---|---|
| Historical 280 | `18a236c` | false | Original metadata retained; numerical code `bdaf322` |
| First-batch 420 | `7314a6a` | false | Clean frozen checkout |
| Second-batch 110 | `7314a6a` | true | First-batch baseline publication changed tracked documentation |

The dirty flag on batch 2 and the final replay is truthful: the first-batch
summary was published before batch 2 started. The launcher checked before
and after commands that HEAD stayed frozen, only `dev/golden/baseline/`
had tracked changes, no changes were staged, and the reference dependencies
and clean gpyreg checkout were retained. No provenance was rewritten.
Concurrent documentation work advanced `dev-next` elsewhere; the launcher
returned to its latest commit `807d015` after the final checks, preserving
those release decisions and the generated baseline files.

## Checks and permanent evidence

| Check | Result | Report |
|---|---|---|
| Fresh preflight replay, five default seed-0 cases | All trajectories and initial designs identical | [Preflight replay](preflight_replay.md) |
| First batch: even versus odd seeds | No configuration flagged; 56 KS tests, Holm alpha 0.05 | [Batch 1 split](batch1_even_vs_odd.md) |
| First batch: seeds 0–19 versus 20–49 | No configuration flagged; 56 KS tests, Holm alpha 0.05 | [Old versus new seeds](batch1_old_vs_new.md) |
| Final population: even versus odd seeds | No configuration flagged; 68 KS tests, Holm alpha 0.05 | [Final split](final_even_vs_odd.md) |
| Final replay against the expanded reference | All five trajectories and initial designs identical | [Final replay](final_replay.md) |

Each command's exit code gated the next command; comparison failures were
not masked. Complete pair sets were checked at 700, 800 and 810 runs, with
matching labels/seeds, frozen SHA, requested batching option, dependency and
thread metadata, finite comparison metrics, and NPZ ZIP integrity. SHA256
checks confirmed that all 560 original files were unchanged. These artifact
checks were repeated before committing the records. Both baseline publication
steps were verified against the source files, including the summary.

[Validation metadata](validation.json) records counts, environment and steps.
[SHA256 manifest](sha256_manifest.json) fingerprints all 810 JSON/NPZ pairs and
the final summary. Text hashes normalize line endings to LF so Git checkouts
on different platforms can verify them; NPZ hashes use the raw file bytes.
Machine-readable replay reports are alongside the Markdown
reports. The [first-batch summary](batch1_summary.md) preserves the intermediate
700-run population; the [current summary](../baseline/summary.md) covers all 810.

The initial supervisor stopped after the successful preflight because it
compared the rendered verdict to exactly `identical`; that text also includes
initial-design details. The launcher was corrected to check the structured
`identical`, `ok`, and `flagged` fields and the five expected seed-0 identities.
It resumed at 20:04:39 from the freshly completed replay after rechecking the
environment and historical pairs. No numerical run was repeated, no failed
numerical check was bypassed, and all subsequent stages completed normally.

## Reproduction and local artifacts

The three population commands used `dev/scripts/golden_trace.py run` with
`--suite golden`, `--workers 1`, explicit batching-false options, and
`--out dev/scripts/runs/golden/item7_20260906`. The `--only`/`--seeds` arguments
were:

| Invocation | `--only` | `--seeds` |
|---|---|---|
| Batch 1 | `normal_D5,corr_D5,halfnormal_D2,rosenbrock_D2,banana_D2,banana_D6,banana_D10,cigar_D4,lumpy_D4,lumpy_D10,student_D4,logreg_D5,rosenbrock_D2_noise1,logreg_D5_noise3` | `20-49` |
| Batch 2, D8 | `cigar_D8,student_D8` | `0-49` |
| Batch 2, exhaust | `cigar_D15_exhaust` | `0-9` |

Both replays used `dev/scripts/golden_replay.py` with defaults except for
their output directories. The null checks used `golden_trace.py compare
--split <population>` and, for the historical/new cohorts, `compare
<seeds_0_19> <seeds_20_49>` on copies of the sidecars. The legacy
`regenerate_baseline.sh` does not reproduce this seed allocation and masks
comparison failure; it was not used for this extension.

The full traces remain gitignored under
`dev/scripts/runs/golden/item7_20260906/` (55.9 MiB of NPZ files). Copy that
directory to use exact replay in another checkout; release-asset publication
remains part of the 1.5 release work. The JSON sidecars (2.42 MiB) and summary
are tracked in `dev/golden/baseline/`, 811 files in total.

Local logs and operational state remain under
`dev/scripts/runs/reference_extension_20260906/`,
`dev/scripts/runs/golden_grow_20260906.log`,
`dev/scripts/runs/golden_extend_20260906.log`, and
`.venv/reference_extension_20260906.*.log`. The temporary supervisor and its
keep-awake request have ended. There is no job or watcher to reattach to.

The reference boundary is complete. Next is the planned latent-fix work
(roadmap pickup 9), with each trajectory-moving fix replayed and the final
release code checked against this expanded reference population.
