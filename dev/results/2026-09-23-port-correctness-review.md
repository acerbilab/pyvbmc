# The port correctness review of PyVBMC and gpyreg: closing ledger

Completed 2026-09-23. The review ran from 2026-09-19 to 2026-09-23 under the
plan [`plans/port-correctness-review.md`](../plans/port-correctness-review.md),
which holds its design, the PI's decisions, the reviewer brief, the working
rules and the worklog. This ledger lists every finding of its eight waves
with its class, its dating, its evidence, the PI's disposition and its fix
commit. It points at the per-wave ledgers under
[`experiments/port_review_20260919/verification/`](../experiments/port_review_20260919/verification/),
which hold each finding's full statement, its reproduction and the gates of
its pass, and at gpyreg 1.3.0, 1.3.1 and 1.3.2, which carry the review's
fixes to gpyreg. The catalogue of
the deliberate differences from MATLAB that the review settled is in the
porting log [`pyvbmc/vbmc/README.md`](../../pyvbmc/vbmc/README.md). The
findings that the close left without a ruling, and the items that the
independent check of the wave-6 pass had left for later, were ruled on the
same day, after the close, and fixed where the ruling says so; fresh
reviewers then checked that work, and what they found was ruled on and fixed
in its turn ("Findings ruled after the close").

## Scope and method

Fresh Opus reviewers, none of whom saw another's report, read PyVBMC and its
GP backend gpyreg for errors and latent defects before the code is frozen
for the final release benchmark. The code was split into slices (plan,
"Slices"). A slice with a MATLAB counterpart had two reviewers: one checked
internal correctness against the docstrings, the papers and the
mathematics, and one compared the code line by line with MATLAB VBMC. The
subsystems without a MATLAB counterpart (S-VBMC, the PyMC adapter and the
posterior exports, the machine-local calibration) had the internal track
alone. Slice M checked the MATLAB commits made since the port began, and
four third readers (O1 to O4) re-derived the critical numerical paths and
compared them with MATLAB.

Every finding was verified, by a small script, a finite-difference check
or a second reading of the MATLAB source by another agent. No MATLAB was run
(PI, 2026-09-20): a statement about MATLAB rests on its source at `396d649`,
or on a Python transcription of it. The PI ruled on the verified findings
before their fixes, except in wave 4, where the PI ruled on the
orchestrator's recommendations before the verification, which amended them
(`verification/wave4.md`); a few fixes were made during a pass for the PI to
confirm or strike (W5-7, W6-38, and the items found during the passes). Each
fix is one commit per finding, with a test written against the contract
where it changes code. The gates of a fix pass grew with the review. From
wave 2 on they were the exact oracle check (`make_oracle_fixtures.py --check
--exact`), the four seeded gate runs, the whole suite in the default
environment, the tests of the optional integrations in the environments that
hold them (selections of them; in wave 6, the extras cell of the CI matrix
alone) and the full CI matrix; wave 0 was gated by the tests of each module,
and wave 1 by the tests of the modules it touched and the exact oracle
check. From wave 3 on, a pass that moves default trajectories was read for
accuracy on benchmark targets whose truth is known, but for W5-1 and W5-6,
which move a draw now and then. The fix passes of waves 1 to 7 were each read
afterwards by fresh reviewers without the session's context, and what they
found was ruled on and fixed in its turn. The fixes of wave 0 had no such
check.

| Code | Repository | At the start | At the close |
|---|---|---|---|
| PyVBMC | this repository | `f91fdf0` (`dev-next`, 2026-09-19) | `c228cc2e` (`dev-port-review`, equal to `dev-next`), which holds every fix of the eight waves |
| gpyreg | `acerbilab/gpyreg` | `9e70e6b` (1.2.1) | `v1.3.1` (`1dbbfc5`); PyVBMC requires `v1.3.2` (`29b868c`), released after the close |
| MATLAB VBMC, the comparison target | `acerbilab/vbmc` | `396d649` (`master`, 2023-05-03) | unchanged |

| Record | Path under `experiments/port_review_20260919/` |
|---|---|
| The reviewers' reports, saved verbatim | `reviews/` (33 reports), and `prep_report.md` for the preparatory agent |
| The per-wave ledgers and the verifiers' reports | `verification/wave0.md` to `wave7.md`, with `wave1_P6.md`, `wave1_M_P2.md`, `wave1_options.md`, `wave2_B_loop.md`, `wave2_C_setup.md`, `wave2_warmup_history.md`, `wave3_P5.md`, `wave3_P8.md`, `wave4_P3.md`, `wave4_P4.md`, `wave5_P2.md`, `wave5_P7.md`, `wave5_P9.md`, `wave6_G1.md`, `wave6_G2.md` and `wave7_O1.md` to `wave7_O4.md` |
| The verification scripts | `verification/scripts/`; their logs, and the raw reports of the independent checks, are on the machine that ran them (`dev/scripts/runs/LOCAL.md`, "Port correctness review") |
| The reports of the fix agents | `fixes/` |
| The known-differences sheet and the counterpart map | `known_differences.md`, `counterpart_map.md` |
| Defects found on the MATLAB side | `matlab_side_defects.md`: 63 numbered defects, then the questionable points shared by both implementations and the state written or declared and never read |

### How to read the tables

One row per finding of a per-wave ledger; where that ledger groups findings
in one row, so does this one.

- **id**: the per-wave ledger's. Wave 0 names its findings by report (`N1 F1`;
  `N2r` is the rerun of the N2 review, `N2s` the static one). Wave 1 uses the
  labels of `wave1.md` and the rows `O-1` to `O-6` of `wave1_options.md`.
  Items that a per-wave ledger lists without a number carry labels given
  here: `W3-FPn` to `W7-FPn` for what was found during a fix pass or its
  round, `C4-n` for the findings of the independent check of the wave-4
  pass. In the tables of the checks, R1 to R6, F1 to F3, G, P and R are the
  check's reviewers as its paragraph lists them; in wave 7's, O1, O2 and S1
  to S3 label its reviewers' findings, not slices.
- **where**: the Python and the MATLAB locations as the per-wave ledger or
  the report cites them. Python and gpyreg line numbers are those of the
  code the wave read, which the head of each wave below names where the
  records do; MATLAB lines are at `396d649`. `gpt.py` is `pyvbmc/vbmc/gaussian_process_train.py`.
- **category**: the reviewer report's own, "—" where it gives none.
- **class; dating**: the verifier's class, and the dating from both histories
  as the records give it.
- **evidence**: the verification file and the scripts under
  `verification/scripts/` that settled the row, or "reading".
- **disposition**: the PI's ruling or the recorded outcome, with later
  changes. "As proposed" means that the PI took the orchestrator's proposal
  as it stood.
- **fix**: the commits. A gpyreg commit is marked "gpyreg". The checks read
  and reported; "check:" marks a commit made on the findings of the
  independent check of the pass, and in wave 6 "round:" one made on the
  findings of the check of that first round and "that check:" one made on
  the check for substantial errors. In wave 1, "check table:" marks a commit
  that `wave1.md` lists under its check for this finding without naming it
  in the dispositions table, and "ruling:" one made on the PI's rulings after
  the check. A row's id followed by a colon ("W7-16:") names the commit of
  that row. "(git)" and "git `abc`" mark a commit that no record names, found
  in the history. Every PyVBMC commit cited in the tables of the waves is an
  ancestor of `c228cc2e`, but for `29c822d`, a fix agent's commit that the
  branch took as `e48fade`, and every gpyreg commit cited there is an
  ancestor of `v1.3.1`. The commits of the section "Findings ruled after the
  close" come after those, on `dev-port-review` and, in gpyreg, in `v1.3.2`.

## Summary

| Wave | Dates | Slices | Findings reported | Per-wave ledger | Fixes | Into `dev-next` |
|---|---|---|---|---|---|---|
| 0 | 2026-09-19 | N1, N2, N3 (internal track), and the preparatory agent's sheet and map | 24: N1 5, N2 9 over two reports, N3 10 | `wave0.md`, 23 rows | 18 commits, 16 of them found in git | `57cbfb4`, with waves 1 and 2 |
| 1 | 2026-09-19; check 09-22 | M; P6 on both tracks; P2 on the comparison track | 44: M 5, P6 13 and 10, P2 16 | `wave1.md`, with `wave1_P6.md`, `wave1_M_P2.md` and `wave1_options.md` | 35 commits after `6c151cb`, with `cb2d513` before them; the check's round 22 (19 by its fix agents), then 7 on the PI's rulings | `57cbfb4`; the check merged into `dev-port-review` as `e86bbb1c` |
| 2 | 2026-09-20; check 09-21 | P1a, P1b, both tracks | 43 | `wave2.md`, W2-1 to W2-30 and 14 minor observations | 47 commits in all, the CI fixes, W2-30 and records included; check 18 | `57cbfb4`; the check with wave 5, `03a8d46` |
| 3 | 2026-09-20; check 09-21 | P5, P8, both tracks | 48 | `wave3.md`, W3-1 to W3-36 | 34 commits; check 11, `b731fac` included | `82aee90`; check `5acd382` |
| 4 | 2026-09-20; check 09-21 | P3, P4, both tracks | 29, 5 minor observations and an adjacent one (W4-17) | `wave4.md`, W4-1 to W4-22 | 12 commits; check 5 | `9d9c01b` |
| 5 | 2026-09-21 | P7, P9, both tracks; P2, internal track | 54 | `wave5.md`, W5-1 to W5-40 | 32 commits; check 13, and `5c4fc87`, `6d492a2`, `6dcd027`, `f9ab814` | `03a8d46`; `1563053` |
| 6 | 2026-09-21 to 09-23 | G1, G2 (gpyreg), both tracks | 36, and 52 minor observations | `wave6.md`, W6-1 to W6-38 | gpyreg: 56 commits in `acerbilab/gpyreg#50`; PyVBMC: 8 | gpyreg 1.3.0; `dev-next` fast-forwarded |
| 7 | 2026-09-23 | O1 to O4, third readers | 14, two observations, and a verifier's candidate (W7-17) | `wave7.md`, W7-1 to W7-17 | PyVBMC: 19, then 3 in the round of its check and `8921845f`; gpyreg: 12 | gpyreg 1.3.1; `dev-next` fast-forwarded |
| after the close | 2026-09-23 | the 17 findings that the close left without a ruling and the 10 items that the check of the wave-6 pass left for later; then the independent check of that work, six reviewers | 14 more found while the rulings were carried out; the check's findings | this ledger, "Findings ruled after the close" | PyVBMC: 27, then 27 in the round of the check; gpyreg: 16, then 16, in 1.3.2 | fast-forward |

What the review changed:

- **Defects fixed.** Most rows below end in a fix; the largest, by reach, are
  the level-1 noise model of the GP (W3-1), which had silently been level
  0's; the warm-up end, the LCB maxima and the minimum-iteration guard of the
  main loop (W2-1 to W2-5); the inputs of the GP hyperparameter fit (W3-2 to
  W3-5, W3-7) and the statistics of gpyreg's bound recommendations, pooled over
  the dimensions (W6-1); the remainder of the balanced draw of the
  variational posterior (W5-1); the CMA-ES search's per-coordinate scales
  and the size of the initial design (wave 1); and the saved files that
  ended the interpreter under another Python version (W2-30).
- **Stricter interfaces.** A value that used to be accepted and to fail later,
  or to be ignored, is refused where it is given (plan, "Decisions",
  2026-09-20): unknown option names, options read only at construction given
  to `load`, unported samplers, mean functions and features, priors that do
  not cover the hard bounds, starting points without spread, non-finite
  prior arguments in gpyreg.
- **gpyreg 1.3.0, 1.3.1 and 1.3.2**: 1.3.0 and 1.3.1 released on 2026-09-23
  with the fixes of waves 6 and 7, 1.3.2 on 2026-09-24 with those of the
  rulings after the close and of their check (below); PyVBMC requires
  1.3.2.
- **Default trajectories.** The fixes listed under "The fixes that move
  default trajectories" change what a run at the shipped options computes, so
  the golden references and the production-reference runs describe the code
  from before them (`dev/TODO.md`, "The golden references after the port
  review").
- **The catalogue of deliberate differences.** The known-differences sheet
  grew from 67 entries to 134 and was consolidated into the porting log
  `pyvbmc/vbmc/README.md`; the list of MATLAB-side defects holds 63 entries,
  as material for the MATLAB repository.

What it leaves, all in `dev/TODO.md`: the regeneration of the golden
references; an oracle state at uncertainty level 1; a seeded gate run with a
prior; and, from the rulings of wave 0, `save` and `load` for an `SVBMC`
object and the comparison of the two compositions of the S-VBMC shrinkage.

## The fixes that move default trajectories

| Wave | Commit | Finding | What moves |
|---|---|---|---|
| 1 | `cb2d513` | P2 F2 = M F1 | runs with `D >= 10` (`fun_eval_start` is `10*ceil((D+1)/10)`) |
| 1 | `4d1d984`, `d617d99` | P2 F1, P2 F5 | every CMA-ES search with `D > 1` (per-coordinate scales, no noise handler) |
| 1 | `bd47856`, `4892fe3` | O-3 to O-5 | runs that end without a stable iteration (the ranking criterion and its counts), and the posterior an input warp starts from |
| 1 | `7331841` | P6 int F3 | runs with `K = 1`: the reported ELBO, the ranking of the optimizations, the pruning, the draws of the Monte Carlo estimate |
| 1 | `be38570` | P6 int F1 + cmp F1 | `D > 1` with two slow candidates (the type-2 starting widths) |
| 1 | `cb8a51d` | P2 F9 | `D = 1` (the bounded scalar search) |
| 1 | `181a63b` | P2 F3, F4 | an `x0` longer than `fun_eval_start` |
| 1 | `510a493` | P2 F8 = M F3 | every acquired point of a noisy run that is not a repeat, equal to a full recomputation to rounding |
| 1 | `118626e` | P2 F10 | the GP refit inside active sampling on noisy targets |
| 1, check | `6590ea4` | P2 F10, verifier item 4 | noisy runs (`N` and `n_eff` refreshed after every evaluation) |
| 1, check | `831b024` | P6 int F3 | noisy runs whose old posterior has one component |
| 2 | `c918d12` | W2-1 | the end of warm-up (the recent-improvement window) |
| 2 | `8e591ff` | W2-2 | the end of warm-up (the recomputed LCB maxima) |
| 2 | `fb8a12e` | W2-3 | the first warp and the earliest stable termination |
| 2 | `8cd4bbc` | W2-4 | the first variational optimization of a run whose `x0` the transform moves |
| 2 | `567444f` | W2-5 | runs on which `min_iter` binds |
| 3 | `14a01e2`, `6e135f2`, `1268d69`, `720c416`, `992f7cb` | W3-2, W3-3, W3-4, W3-5, W3-7 | the design and the starting points of every fit that draws a design, the sampler widths of the fits that sample (W3-4), and the cap of the noise (W3-7) |
| 5 | `23d962a` | W5-1 | the remainder of every balanced draw with `K > 1`: `sKL` and `r_index` of every run, and a trajectory now and then, where a moved candidate is acquired |
| 5 | `8e2977a`, `94638fe` | W5-6, R3-1 | runs with more starting points than the initial design takes, where the sieve's rounded shares meet a half |
| 6 | gpyreg `f76eca2` | W6-1 | every sampled GP fit of every run (the mean's boxes, the design, the optimizer's box, the sampler's widths) |

No fix of waves 0, 4 and 7, nor any made after the close, moves a default
trajectory; W3-1 (`095c82c`)
changes runs at uncertainty level 1, which is not a default. The moving
fixes of the wave-1 and wave-2 passes were not read for accuracy, the
benchmark sweep becoming a gate in wave 3; the release benchmark is their
measure. The later ones were read on benchmark targets whose truth is known:
the wave-3 pass on eight targets and 32 seeded pairs, with differences both
ways (`verification/wave3.md`, "Gates"); W6-1 on eight targets and 59 seeded
pairs, better on 24 and worse on 35 by the ELBO error, better on 27 and worse
on 30 by gsKL (two tied), better on 30 and worse on 29 by MMTV, its means
worse on `corr_D5`, `cigar_D4` and the noisy Rosenbrock on all three metrics
and on `lumpy_D4` in the ELBO error, kept after the ruling was reopened and
the mean's bounds were probed (`verification/wave6.md`, "Gates"); and the
merge of the wave-1 check, of whose commits `6590ea4` and `831b024` move
noisy runs, better by the ELBO error on 21 of 30 seeds and worse on
`rosenbrock_D2_noise3` by 0.025 nats of mean ELBO error
(`verification/wave1.md`, "The merge into `dev-port-review`"). W5-1 and W5-6
were not swept: they move a draw now and then.

## Oracle re-baselines

Each made from the stored states with the generator's targeted mode, the
reason recorded in the fixture's `.json`, every other reference asserted
bit-identical.

| Wave | Commit | Oracle | Reason |
|---|---|---|---|
| 1 | `dd89374` | `active_sample_step`, on the seven states that hold a reference | the CMA-ES search at the per-coordinate scales and without cma's noise handler (P2 F1, F5); every `acq_*` reference unchanged |
| 3 | `0ccaf76`; `2828fd3` | `gp_fit`, `gp_fit_history` and the log prior of `gp_nlZ`; the noisy fit-history capture, replayed through the mode `--rebaseline-gp-fit-history` (`c60834d`) | the bounds of W3-5 and W3-7 and the pool of W3-2 and W3-3 |
| 4 | `2dc98ce` | `acq_AcqFcnIMIQR` on `rosenbrock_D2_noise1_viqr` | the starting point of IMIQR's chains drawn by importance weight (W4-1) |
| 6 | `0a6682fc` | the three fit-history captures, audit entries only (every output moved by zero) | the `logp` key removed from `hyp_dict` (W6-23) |
| 6 | `3123135e` | `gp_fit` and `gp_fit_history` on the seven states that sample, and the three captures | the recommendations per column (W6-1) |
| 7 | `83a592c6` | `vp_pdf` on `cigar_D4_boosted`, `cigar_D4_largeK`, `corr_D5_warped`, at the rows whose density was below the smallest normal double | the log density by log-sum-exp there (W7-3) |

## gpyreg 1.3.0, 1.3.1 and 1.3.2

**1.3.0** (2026-09-23): the annotated tag `v1.3.0` on `0186d89`, the merge of
`acerbilab/gpyreg#51`, which dates the release notes; the code came in
`#50` (`port-review-wave6`, merged as `a4c2cc0`): 56 commits, the 35 of the
wave-6 pass, the 16 of the round after its independent check, the 4 of the
round after the check of that round, and `286b595`. Its GitHub release
uploaded the wheel and the source archive to PyPI. It carries W6-1, W6-3
to W6-18, W6-21, W6-22, W6-26 to W6-29, W6-31 to W6-35 and W6-38, the
documentation of W6-24, and the check findings fixed in the two rounds
(below); its release notes have "Upgrading" points for the inputs it
refuses, the Metropolis step it removes, `df_base` after `fit`, the kernels'
diagonal with the gradient and the exception types that changed. PyVBMC moved its pin to `a4c2cc0` (`4196649a`), then to the tag,
with `gpyreg >= 1.3.0` (`2dc2a6c3`).

**1.3.1** (2026-09-23): the annotated tag `v1.3.1` on `1dbbfc5`, the merge of
`acerbilab/gpyreg#52` (`w7-fixes`, 13 commits: the 12 of the wave-7 pass and
its round, and the dating of the release notes, `d6054b2`), uploaded to
PyPI by its GitHub release. It carries W7-14, W7-15 and W7-16, the design of
`fit` in a prior's upper tail and the refusal of an inverted smooth box
(found in the fix round), the design of a fixed coordinate with a prior
(found by the check), and the tests of the O4 test notes. PyVBMC moved its
pin to the tag with `gpyreg >= 1.3.1` (`9068f2d0` (git)).

**1.3.2** (2026-09-24): the annotated tag `v1.3.2` on `29b868c`, the merge of
`acerbilab/gpyreg#53` (`w6-leftovers`, 33 commits: the 16 of the rulings
after the close, the 16 of the round of their check, and the dating of the
release notes, `755a4b3`), uploaded to PyPI by its GitHub release (the
wheel and the source archive). It carries the gpyreg fixes of "Findings
ruled after the close". Its release notes have an "Upgrading" point for
each change that can stop a script written for 1.3.1, and for the
predictions of the low-noise representation, which change, PyBADS's among
them. None of its changes moves a number that PyVBMC computes: PyVBMC's
gates against the branch give the oracles and the seeded runs bit for bit.
The first run of the pull request's checks failed on macOS with Python
3.11: a test of the branch expected the `ValueError` with which `update`
refuses a starting point of NaN, where the LAPACK of those runners reports
the failure of factoring a covariance that holds NaN first (the check's
table, below). Before the tag, the PI's session on PyBADS ran PyBADS's
tests against the branch and against 1.3.1, three times each: no refusal
fires and no test breaks, 88 pass on both, and `test_he_noisy_sphere_opt`,
which fails on 1.3.1 on its tolerance, passes on 1.3.2, the low-noise
change having moved the path of a seeded test before it, which leaves
NumPy's global stream in another state. PyVBMC moved its minimum and its
pin to the tag (`AGENTS.md`).

## Findings ruled after the close

The close left seventeen verified findings and items without a ruling, the
code they describe unchanged at `c228cc2e`, and the independent check of the
wave-6 pass had left ten items for later in `dev/TODO.md`. None of them
changes what a run at the shipped options computes. The PI ruled on all of
them on 2026-09-23, after the close, on the orchestrator's recommendations:
the first in three batches, the second in one. Opus agents made the fixes in
worktrees, one commit per finding, with a test seen to fail on the code
before it where the commit changes code (a test of code that was already
right was seen to fail on a planted break), and the orchestrator read each
diff before taking it onto `dev-port-review`. Two commits change code
without a test, `1541b891` and `59bf7a4b`, which remove a fallback that no
saved file needs and a check that could not fire; five change comments or
documents alone. Their gates are the row "after the close" of "Gates, CI
and merges". The rows of waves 0 and 2 below carry the same rulings. Fresh
reviewers then checked the whole of this work; what they found, and what
became of it, closes the section ("The check of the rulings, and its
round").

| Wave | id | Finding | Class | Ruling (PI, 2026-09-23) | Fix |
|---|---|---|---|---|---|
| 0 | N3 F2 | the calibration workload `active_d4_k20` times the Monte Carlo entropy with gradients, where the package uses that count without | defect (performance only) | fix: the workload is timed without gradients, in the value group; recipe revision `machine-calibration-v2` | `bc6180b6` |
| 0 | N3 F4 | a `ValueError` from `make_record` or `write_record` discards a finished calibration campaign | defect, latent | fix: a report that fails validation gives the fallback profile with status `"invalid"`, and a record that cannot be built or written leaves the result in memory | `58228a6a` |
| 0 | N3 F5 | the held-out gate of a calibration needs all four rounds, `ceil(0.80*4) = 4` | as specified: the plan's rule of at least `ceil(0.8 R)` winning rounds, which it applies to the four held-out rounds (`plans/machine-local-calibration.md`); conservative. The class recorded at the close, "the plan is silent", was wrong | left: the rule errs toward the default, which costs speed at most; a comment at the line says what it asks for | comment `56a8cf6f` |
| 0 | N3 F6 | an incomplete calibration campaign discards the groups that finished; a test pins it | design choice | left: the plan prescribes it ("do not persist a partial winner as a successful calibration"); the five-minute watchdog stands against a campaign of tens of seconds, and an incomplete campaign returns the previous profile or the defaults, which cost speed at most | — |
| 0 | N3 F8 | `sieve_gradient` times a gradient call of 8192 rows that the package makes one point at a time | defect (performance only) | fix: the workload is removed; the package's one gradient call of the density takes one point per evaluation, whose layout no budget changes (since `bf1e87ef`, below, which drops the gradient from the screen of `vp.mode`'s starting points) | `5cd8ccc3` |
| 0 | N3 F9 | the key of the calibration cache omits the package version | design choice | left: the plan's decision ("invalidate on explicit kernel/recipe revision rather than every docs commit"); a key on the version would discard every cache at each release and at each reinstall of a development checkout, and since N3 F1 every budget gives the same numbers, so a stale profile costs speed at most. Comments beside the blocking of both kernels say that a change to it bumps `KERNEL_REVISION` and updates the campaign's copy of the layout | comments `56a8cf6f`, `2369ea2f` |
| 2 | C-M8 | `VBMC(x0=...)` with a list or a float raises `AttributeError` from `x0.ndim` (`verification/wave2_C_setup.md`) | Python-only defect, a hard failure with an unhelpful message | fix: `x0` is converted to an array before its shape is read, as the bounds are | `facbae61` |
| 2 | C-C7 | `Options.eval` calls a callable option by keyword where MATLAB's `evaloption_vbmc` calls it positionally, so `ns_ent` given as `lambda n: ...` raises `TypeError` at its first use (`verification/wave2_C_setup.md`) | Python-only interface difference | fix: a callable evaluated with one parameter receives it by position, as in MATLAB; with several, by keyword | `e83f00b6` |
| 2 | C-C2, its second half | MATLAB warns when `NoiseSize` is given with `SpecifyTargetNoise` (`misc/setupoptions_vbmc.m:139-140`); PyVBMC ports the error on the contradictory configuration (`1a9f37a`) and not the warning | Python-only difference | fix: the warning, at construction and in `load` | `a7944920` |
| 2 | C-M2, its second half | `log_file_level` refuses every level outside `{0, 10, 20, 30, 40, 50}`; `658ecb8` fixed the handler loop of the same row alone | Python-only defect, leftover-grade | fix: any non-negative integer level; the log file is written whenever it is named (`logging.NOTSET` wrote none); any other value is refused | `f5ec2dc7` |
| 2, check | left for the PI | the option functions of a saved run are not rebuilt from the `.ini` files in `load`, which would let a run with an importable target continue under another minor version of Python | open question | left: rebuilding them needs a rule for a default that changes between releases, since a continued run would change its settings partway, and the two-interpreter experiment of W2-30; a target that cannot be imported by name is stored as bytecode all the same, and the `save` and `load` docstrings and the FAQ say what a saved run allows under another Python version | — |
| 2, check | left for the PI | `vbmc.x0` stays in the inference space of construction when the run warps; `AGENTS.md` and the class docstring say so | documented | left: documented; `vbmc.x0_orig` gives the caller's coordinates, and `load`'s back-fill of `x0_orig` for older files (`ce59fbe`) rests on `x0` staying in the space of construction | — |
| 2, check | left for the PI | `optim_state["last_warmup"] = 0` for a run without warm-up is MATLAB's 1-based value in a 0-based field, masked at every default; the fallback of `.get("last_successful_warping", 0)` is an iteration index where the key starts at minus infinity | latent | fix: `-1`, the index before the first iteration, so that a run without warm-up takes the update after every active sample in its first `active_sample_full_update_past_warmup` iterations, as in MATLAB; the fallback is removed, every release since 0.9.0 setting the key | `ab52337a`, `1541b891` |
| 2, check | left for the PI | the main loop sizes its sieve at `self.vp.K`, which with `variable_means` off is the count of the iteration before | non-default option | left: with free means, too, `self.vp.K` is the count of the posterior entering the fit, `update_K` returning the new count without changing the posterior; the porting log's entry on the sieve's count, a deliberate difference, says that it holds for fixed means as well | entry `9bc3974f` |
| 2, check | left for the PI | `validate_run_limits` checks neither the sign nor the integrality of `min_iter`; the guard of the inert options runs one way; the warnings of `Options` go to the root logger and miss `log_file_name` | input checks, logging | fix: `min_iter` must be a finite non-negative integer, infinity refused since the minimum holds back every termination, the budget's included unless the run accounts for evaluations made before it (`precomputed_evaluations` or an `initialization_cost`), whose budget can end it below the minimum; the reads of the undeclared `warp_nonlinear`, `varactivesample` and `skip_elbo_variance` are removed and the guard runs both ways. Left: the warnings of construction come before the handler of the log file exists, the logger being built from the checked options, and reach the console through the root logger, as the warnings of the reshaping and casting of the bounds do | `769d166d`, `68e76f8e`, `38e1e311` |
| 2, check | left for the PI | `_normalized_hard_bound` of the `PyMCTarget` route does not replicate a scalar bound; the tests of `_recompute_lcb_max` reach neither its wiring into the loop, nor the noisy branch, nor `add_noise=False` | input check, tests | fix: a single-valued bound holds for every variable; tests of the noisy branch, of the latent variance and of the wiring. The noisy branch cannot change the latent bound, in gpyreg as in `gplite_pred`: the per-point noise enters through the conditioning of the GP | `ba4ceb0c`, `2885df70`, `6ede754d` |
| 2, check | left for the PI | `CHANGELOG.md` has no entry for W2-16 and W2-11, no "Upgrading" line for the copies that `determine_best_vp` and `final_boost` work on or for the frozen `pop` and `del`, and says of the budget keys of the results that each is present "only when its argument was used" | documentation | fix: the two Upgrading lines, the clause on the budget keys and a `Fixed` line for W2-16. W2-11 left: nothing reads the running averages of the moments, so no result or reported field changes | `44ad800e` |

What the independent check of the wave-6 pass left for later
(`verification/wave6.md`):

| Code | Item | Ruling (PI, 2026-09-23) | Fix |
|---|---|---|---|
| PyVBMC | `load` refuses an option that only construction reads even when its value equals the stored one, so a run's own options given back with a new `max_fun_evals` raise | fix: a value that states the stored one, read as construction reads the option, is taken and changes nothing; another value is refused as before | `a86c7331` |
| PyVBMC | raising `ns_gp_max` from zero in `load` is stored and leaves the hyperparameters unsampled, `_init_optim_state` deriving `stop_sampling` from it | fix: `load`, and the continuation of a finished run, which restores the state of its last iteration, set `stop_sampling` from the option as construction does, unless the sampling stopped in the stable regime | `a7dd32d5` |
| PyVBMC | the three fit-history captures hold the arrays `capture/ref/fit/hyp_dict_logp`, which no sidecar names | left, as the item says, to the next rewrite of a capture: rewriting now would re-baseline a platform-bound oracle for nothing | — |
| gpyreg | `RationalQuadraticARD.get_bounds_info` carries a copy of the bounds helper of `covariance_functions.py`; `GP.update` and `GP.quad` each recover the noise scale of a posterior pickled without it | fix, bit-identical: the kernel calls the helper and adds the entries of its shape; one method of `Posterior` recovers the scale | gpyreg `fbdbf5f`, `7750a2f` |
| gpyreg | a `fit` on a single training point, or on points that share one coordinate, ends with L-BFGS-B's `KeyError`, the length-scale bounds being `(-inf, -inf)` (entry 49 of the MATLAB-side list) | fix, stricter: the bound recommendation refuses a column without spread unless the caller gives the length scales it concerns, and the scale of `NegativeQuadratic`, a finite lower bound (so narrowed after the check of the rulings, below; the first form asked for finite lower and upper bounds, and refused pairs that 1.3.1 fitted). PyVBMC reaches a column without spread only in the high-posterior-density subset, with `integer_vars` or when the best points of an initial design share a value, whose recommendations `_gp_hyp` reads through the components and not through this method; its fit is bit-identical | gpyreg `f10819a`, `5ca8bab` |
| gpyreg | `get_priors` returns `None` for a Student's t block with mixed degrees of freedom, and, for a smooth box with a NaN `sigma`, a block that `set_priors` refuses | fix: `set_priors(get_priors())` reproduces the priors, and `get_priors` refuses a state that `set_priors` refuses; after the check of the rulings, `set_priors` marks the GP as having priors only where a coordinate holds one, so that the round trip keeps that mark too | gpyreg `c613552`, `f3860c5` |
| gpyreg | `set_priors` sets `no_prior` before its checks; `np.abs(sigma)` in the prior code is dead; the `Raises` section of `fit` is incomplete; `set_bounds` takes an inverted pair | fix: the flag, the `Raises` section and the refusal of an inverted pair. The absolute value left: it is not dead, since a GP pickled by gpyreg 1.2.1, or one whose `hyper_priors` are written directly, can hold a negative `sigma` | gpyreg `b8cbd31`, `5e6d05f`, `7d19206` |
| gpyreg | `test_split_update` fixes the number of hyperparameter samples; the test of the documented prior densities runs without bounds | fix: one sample and two; bounded cases against truncated densities computed by quadrature | gpyreg `5d60af5`, `a1b5d94` |
| gpyreg | in the low-noise representation of the posterior (a noise variance below 1e-6), `predict` and `predict_full` form the predictive covariance from the explicit inverse, as `gplite_pred.m` does: at a noise standard deviation of 1e-6 the variances inside the data are off by up to 1.9e-3 where they are about 1e-12 (the configuration of the item; about 1e-4 in that of gpyreg's release notes) | fix: the posterior keeps its Cholesky factor, from which the predictions and the rank-one update form the covariance; the Cholesky representation is bit-identical. No PyVBMC run at the shipped `tol_gp_noise` reaches the representation, and PyVBMC's own counterparts of the same expressions keep the inverse (the check of the rulings, below, G1); the GPs of PyBADS reach it, and their predictive variances change. Entry 62 of the MATLAB-side list | gpyreg `b99cee4` |
| gpyreg | `fit` with a whole float `thin`, 2.0 for one, raises `TypeError` | fix: a whole number of either type; any other value refused | gpyreg `ee309a2` |

Found while the rulings were carried out:

| Code | Finding | Ruling (PI, 2026-09-23) | Fix |
|---|---|---|---|
| PyVBMC | `noise_size` given as `None`, a list, an empty array or a NumPy number raised at the first GP fit of a run at level 0 or 1 (a NumPy number with recent versions of NumPy; older ones ignored it), and a value that is not positive was replaced by `tol_gp_noise` without a word, where MATLAB refuses it | fix, stricter: an empty value leaves the option unset, a positive finite number is used, and any other value is refused at construction and in `load` | `54de679c`, `b24a9238` |
| PyVBMC | `active_importance_sampling_mcmc_samples` given as a function worked with VIQR and raised with IMIQR, whose MCMC branch read the option raw; MATLAB takes a string expression in its variational branch alone, a difference the porting log did not list | fix: both branches evaluate the option and round it up, a value that is not a finite number is refused, and the porting log has the entry | `a318aa17` |
| PyVBMC | `vp.mode(orig_flag=False)` evaluated the density with its gradient on the 100,000 points of its screen and dropped the gradient | fix: the screen takes the values alone, which are bit-identical, so the mode is too; the call took 0.40 s instead of 0.69 s in the measurement of the commit | `bf1e87ef` |
| PyVBMC | four of the peaks that the calibration's memory test quotes were measured on an earlier kernel | fix: the peaks of the current kernel | `6aa36592` |
| PyVBMC | the calibration plan's request to verify the gradients of the density read as open once the gradient workload was gone | fix: the plan's note on recipe revision v2 answers it | the records |
| PyVBMC | two dated records name the removed `skip_elbo_variance` guard (`dev/2026-09-02-modernization-discussion.md`, `dev/plans/stage2-gp-log-joint-einsum.md`), and `dev/scripts/eta_bound_comparison.py` lists it among the options it records | left to the release's sweep of statements made stale (`dev/TODO.md`, "Release documentation and validation") | — |
| gpyreg | `GP.quad` formed the variance of an integral in the low-noise representation from the explicit inverse, as `gplite_quad.m` does | fix: from the Cholesky factor, as the predictions do | gpyreg `0aa76e2` |
| gpyreg | `fit`'s `n_samples`, `opts_N` and `init_N`, and the `N` of `SliceSampler.sample`, raised `TypeError` on a whole float, and some ran a negative value or zero without a word | fix: whole numbers of either type; any other value refused | gpyreg `0fac152` |
| gpyreg | `fit` on a GP without training data raised `AttributeError` from a component | fix: a `ValueError` that says so, before anything changes | gpyreg `dedf90b` |
| gpyreg | PyBADS's known-noise path (`fit_lik` false) sets the noise prior `("delta", noise_mu)`, a type that gpyreg does not implement and `set_priors` refuses | a matter for PyBADS, handed to the work on PyBADS that the PI runs from its repository | — |
| gpyreg | `SliceSampler.sample` took a bool as `thin` or `burn`, where its `N` and the counts of `fit` refuse one | fix: refused | gpyreg `3eccef5` |
| gpyreg | `get_priors` refuses a GP that holds a negative `sigma`, pickled by gpyreg 1.2.1 or written into `hyper_priors` directly, whose log prior the kept absolute value still computes | left: `get_priors` returns the form that `set_priors` takes, `set_priors` refuses a negative `sigma`, and such a GP still fits and predicts; the release notes of 1.3.2 give it an Upgrading point (after the check of the rulings) | gpyreg `e09ab8b` (the notes) |
| PyVBMC | with `integer_vars`, the high-posterior-density subset can share one integer value in a coordinate, and the lower bound of that length scale is then `-inf` | left at first: the fit completes, its upper bound coming from the whole training set, and the option is experimental. The check of the rulings found that reason wrong: the `-inf` comes from PyVBMC's own bounds on the subset (MATLAB bounds the length scales from the whole training set), and the first fit of a run, which starts from `log 0`, fails, with or without `integer_vars`; the PI ruled a fix (below, F1) | `456dce6c` |
| PyVBMC | two of the three run-time checks of the VIQR sample count could not fire once the count is read through `_sample_count` | fix: the check that the count is positive stays | `59bf7a4b` |

### The check of the rulings, and its round

Before anything of this work was pushed, the PI asked for a check of all of
it by fresh reviewers. Six read-only Opus reviewers read PyVBMC
`3aba4fc5..093342ed` and gpyreg `1dbbfc5..9280a75` on 2026-09-23, one part
each: R1 the calibration; R2 the options, construction and `load` (the one
of them allowed test files and capped runs); R3 the main loop, the
importance sampling and the posterior; R4 gpyreg's low-noise
representation; R5 gpyreg's input checks, priors and tests; R6 the records
and documents. None found a defect that had to be fixed before anything
else: every fix did what its ruling asked, and none moved a number that a
run at the shipped options computes (22,287 arrays of gpyreg's Cholesky
representation compared with 1.3.1, none different). Their reports are on
the orchestrator's machine (`dev/scripts/runs/LOCAL.md`, "Port correctness
review"). The PI ruled on everything they found on the orchestrator's
recommendations, on 2026-09-23 and, for the items the fix agents noticed, on
2026-09-24: each item is fixed or left with its reason, none deferred.
From then on the fix agents reported only substantial issues beyond
their items (PI, 2026-09-24).
Opus fix agents made the fixes in worktrees, one commit per item, with a
test seen to fail first where the commit changes code (the test of the
calibration's layout, of code that was right, on planted drifts), and the
orchestrator read each diff and wrote the records. The id is the row of
the report, or the item of the orchestrator's list to the PI (A to G).

| Code | id | Finding | Ruling (PI) | Fix |
|---|---|---|---|---|
| gpyreg | A1 (R5 S1) | the refusal of a column without spread also refused a caller's finite lower bound with an upper bound of `+inf` or left unset, which 1.3.1 fitted with finite predictions | fix: refused only where the lower bound the fit uses is not finite, `+inf` included (1.3.1 ended there with `KeyError`); a finite lower bound fits as in 1.3.1, bit for bit | gpyreg `5ca8bab` |
| gpyreg | A2 (R5 S3) | `set_priors(get_priors())` turned off the mark of priors of a GP whose only prior is a block with no prior in any coordinate, and the docs named the wrong family for such a block | fix: `set_priors` marks the GP as having priors only where a coordinate holds one; the docstrings and the notes say, family by family, what `get_priors` returns for such a block | gpyreg `f3860c5` |
| gpyreg | A3 (R5 S4a) | a count given as a 0-d array, which 1.3.1 ran bit for bit as the integer, was refused | fix: a 0-d array holding a whole number is that number, for the counts of `fit` and for `N`, `thin` and `burn` of `SliceSampler.sample` (which refused one in 1.3.1 too) | gpyreg `51d4f09` |
| gpyreg | A4 (R4 S2) | the `eigh` reference of `test_low_noise_predictions_at_the_training_inputs` was less accurate than the test's tolerance, which seeds 1, 2 and 4 exceeded | fix: an exact reference in rational arithmetic | gpyreg `1eb6c5f` |
| gpyreg | A5 (R5) | `fit` checked `burn` only inside the sampler, after the optimization, and not at all without samples | fix: checked with the counts, before anything changes | gpyreg `18b6bd2` |
| gpyreg | B1 (R4 S1) | the notes said that the draws of `random_function` change at the level of rounding after single-point updates; they change by O(1) | fix: they change with the posterior | gpyreg `a274143` |
| gpyreg | B2 (R5 S2) | the notes gave 1.3.1 "an infinite length scale" where its log length scale was `-inf` | fix | gpyreg `7db965f` |
| gpyreg | B3 (R5 S4, R6 S8) | Upgrading points missing: a NaN `init_N`; an inverted pair from `set_bounds` followed by a fit given its own bounds; `get_priors` on a GP pickled by 1.2.x with a negative or NaN `sigma`; the family of a Student's t block with degrees of freedom 0 and NaN | fix | gpyreg `e09ab8b` |
| gpyreg | B4 (R4) | the numbers of the low-noise entry depend on a configuration the notes did not state; the sentence on PyBADS did not say what changes for it | fix: the configurations and orders of magnitude; PyBADS's predictive means are unchanged and its variances change | gpyreg `8072a51` |
| gpyreg | B5 (R4 S3, R6 S4) | gpyreg's `AGENTS.md` and several docstrings did not describe the kept factor, the users of gpyreg or what `fit` raises | fix | gpyreg `107c829` |
| gpyreg | item 4 (fix agent G) | `predict`'s `ValueError` for `return_lpd` without `y_star`, and the `TypeError` of `_convert_shapes` for `s2`, are in no `Raises` section | fix | gpyreg `52a04f9` |
| gpyreg | item 5 (fix agent G) | the docstring of `test_low_noise_rank_one_updates_match_a_full_recomputation` quoted numbers of its configuration that nobody had checked | fix | gpyreg `8df7a94` |
| gpyreg | fix agent H, 1 | the `Raises` sections of `update`, `predict` and `predict_full` did not name the `ValueError` of the check of the shapes of the data, which `fit`'s names | fix | gpyreg `7200191` |
| gpyreg | fix agent H, 2 | `update` given `y_new` or `s2_new` without `X_new` appended the targets to a GP that holds data and then failed, leaving it without a posterior, and raised `AttributeError` on a GP without data (1.3.1 too) | fix, then widened (PI, 2026-09-24): `update` refuses, before it changes anything, a call that would make the number of targets, or of noise variances, differ from the number of inputs, `X_new` without `y_new` on a GP that holds targets included; targets given alone for inputs held without targets, and variances given alone for data held without variances, are taken as 1.3.1 takes them, bit for bit; counts that already differ are not checked | gpyreg `2c90b9f` |
| gpyreg | fix agent H, 3 | a complex noise variance gets the `TypeError` of `float()`, not the check's own message | left: the error is clear | — |
| gpyreg | fix agent I | `fit` given inputs of another number without the targets, or without the noise variances that its noise function reads, that the GP holds stored them, raised from its objective and left the GP with new inputs beside old posteriors (1.3.1 too) | fix: refused before anything changes, by `update`'s rule; variances that the noise function does not read are not counted, and counts that already differ are not checked. On a GP of a single training point 1.3.1 ran such calls of both methods by broadcasting the one target or variance; they are refused, with an Upgrading point | gpyreg `38e8ada` |
| gpyreg | the checks of `#53` | `test_fit_raises_what_it_documents` expected the `ValueError` of `update` for a starting point of NaN, and failed on macOS with Python 3.11, whose LAPACK reports the failure of factoring a covariance that holds NaN, so that the objective at the starting point raises `LinAlgError` first | fix (the orchestrator, within the PI's go for the release): the test takes either exception and checks the message of each, and `fit`'s `Raises` section says that its `LinAlgError` can come from the objective; with scipy's Cholesky made to fail on NaN, as on those runners, gpyreg's suite passes (696) | gpyreg `1cbab6d` |
| PyVBMC | C1 (R1 S1) | `calibrate` still raised for a complete campaign whose settings fail validation | fix: validated in the same `try` as the report, status `"invalid"` | `92b93aa9` |
| PyVBMC | C1b (R1) | a refused result printed "could not complete", and a record that could not be built was reported as a failed write | fix: each message says what happened | `60621f80` |
| PyVBMC | C1c (R1) | nothing tied the campaign's copy of the kernels' blocking to the kernels | fix, beyond the ruling on N3 F9: a test over a grid of shapes and budgets, and comments that name each copy | `6bdfbaa7` |
| PyVBMC | item 3 (fix agent B) | a campaign that its own checks call invalid printed "could not complete" | fix: it says that its results cannot be used, and why | `1cd736d3`, worded as below by `68f47251` |
| PyVBMC | C2 (R2 S2) | the positional call of a single evaluation parameter changed some functions that 1.0.4 took without an error (`lambda scale=100, K=1: ...`) and broke others (`functools.partial`) | fix: by keyword where the function takes an argument of that name, which is 1.0.4's call, by position otherwise, which is MATLAB's. A function for `adaptive_k` that takes 1.0.4's keyword `unkn` alone (`06f9b035`, unreleased, renamed it `K`) still changes, as the Upgrading line says; the orchestrator kept that and told the PI | `b935bc64` |
| PyVBMC | C3 (R2 S3, S4) | `load` took back a construction-only option in fewer forms than it said (`bounded_transform` aliases, flags read by their truth), and a run saved by 1.0.4 with `uncertainty_handling=[1]` could not take back its own option | fix: compared as construction reads it; `load` rewrites a 1.0.4 form of `uncertainty_handling` or `specify_target_noise` into the form that states the run's level, as it rewrites `integer_vars` | `abe2706c` |
| PyVBMC | item 2 (fix agent A) | `load` compared `uncertainty_handling` without `specify_target_noise`, so a pair that construction reads alike could be refused | fix: the pair is compared as construction reads it | `9b544e73` |
| PyVBMC | C4 (R3) | `active_importance_sampling_mcmc_samples` was checked only at the first importance sampling | fix: a value that is not callable is checked at construction and in `load` | `de069cae` |
| PyVBMC | item 6 (fix agent A) | the same option at or below zero is refused only at the first importance sampling, and only with VIQR | left: the value is valid for IMIQR, which then leaves out the MCMC step, and VIQR's refusal names the option | — |
| PyVBMC | C5 (R3) | `load` left a stored `last_warmup = 0` of a run without warm-up saved by an earlier release | fix: mapped to -1 | `e394b26a` |
| PyVBMC | C6 (R6) | `gp_sample_thin` was not checked, and gpyreg 1.3.2 refuses some of its values only at the first GP fit | fix: checked at construction and in `load`, whatever `ns_gp_max` | `9159f342` |
| PyVBMC | C7 (R2, R3, R6) | wording in code and in the changelog: a stale comment and test docstring; the reason for refusing an infinite `min_iter`; the reading of a 0-d `noise_size`; the messages of `min_iter` and `noise_size`; `load`'s `Raises` section; the changelog's lines on `log_file_level`, `noise_size` and the refusals that name a remedy; the porting log's entries on the checks of `min_iter` and `gp_sample_thin`, and `AGENTS.md` on what `load` rewrites | fix | `881a6a59`, `c3407f0b`, `6a8c3767`, `9b8ed926`, `44d42838`, `71d8b167`, `85adf33b`, `ac60c513`, `03c6401e` |
| PyVBMC | C7b (R3) | the test of the noisy branch of `_recompute_lcb_max` passes with the branch's variances dropped, which its docstring did not say | fix: the docstring says what it pins | `abbe11f1` |
| PyVBMC | D1 (R3 S2) | the porting log's entry on `VarActiveSample` cited the sampler of another option | fix; the sheet flags its entry | `ed2f57b8` |
| PyVBMC | D2 (R3 S3) | the porting log said MATLAB's MCMC branch reads the sample count as a number; it reads it raw, and the default reaches it as a string. The changelog left out the float counts | fix; MATLAB-side entry 63 | `8eea2981` |
| PyVBMC | D3 (R3, R6) | the faster `vp.mode(orig_flag=False)` had no changelog line | fix | `7c1e2ff7` |
| PyVBMC | F1 (R3 S1, R6 S7) | the first GP fit of a run stops, or returns NaN predictions without samples, when the points of highest density share a value in a coordinate, with or without `integer_vars` (1.0.4 too): the length scale starts at `log 0` | fix: in such a coordinate the start and the lower bound of the length scale, and the start of the scale of the negative quadratic mean, come from the whole training set, as MATLAB's bounds do; the location of the mean starts at the shared value, as in MATLAB. The PI accepted that runs with `integer_vars` whose subset shares a value move at later fits | `456dce6c` |
| PyVBMC | item 1 (fix agent B) | after F1, gpyreg's recommendations on such a subset still printed a warning of a division by zero | fix: silenced around those calls | `6e3a11b8` |
| PyVBMC | fix agent C, 4 | a coordinate whose values in the subset differ by less than about 1e-154 near zero has a spread that underflows to zero, which F1 does not detect, testing for a shared value exactly | left: no transformed space that PyVBMC builds comes near | — |
| PyVBMC | fix agent C, 5 | construction read the uncertainty handling level with its own chain of tests, beside the function that `load` reads it with | fix: construction reads it with the same function | `490d4d8f` |
| PyVBMC | fix agent C, 6 | a pair of the two noise options refused by `load` names both, where one of them alone would be taken | left: the message is true | — |
| PyVBMC | fix agent C, 7 | the calibration's message for a result that cannot be used said "finished" for a campaign that its baseline check stopped part-way | fix: a wording that holds for both | `68f47251` |
| PyVBMC | E (R1 S2, R2 O4, O5, S6, R4 S4, R6 S1-S3, S5-S7, S9, S10) | errors in the records: this ledger's statements of ancestry, a count of the worklog, the reasons of three rulings, the configurations of the low-noise numbers, entry 62 and a line of `wave6.md`, the wording of entry 49, the TODO item of the pickup, the reminder about the captures' arrays, and the notes of PyBADS | fix, in the files that hold them | the records of the round |
| PyVBMC | G1 (R4 S4, R6 S5) | PyVBMC's own counterparts of gplite's low-noise expressions (the variance of `_gp_log_joint`, VIQR, IMIQR, the active importance sampling) form their variances from the explicit inverse | left: they are MATLAB's formulas; a run reaches them only with `tol_gp_noise` below 1e-3; the error grows like the inverse of the noise variance, about 4e-8 at a noise standard deviation of 1e-4, and matters from about 1e-5 down; a fix would touch four hand-derived hot paths that no gate reaches | — |
| PyVBMC | G2 (R1) | the calibration has no workload for the gradient of the entropy at the `ns_ent_active` count | left: speed alone; every budget gives the same numbers | — |
| both | G2 (R2 S7, O8, R5) | the type rules of the new checks differ from one option to another, and a `Fraction` or `10**400` gets `TypeError`, not `ValueError` | left: each check follows its sibling option (`min_iter` follows `max_iter`) | — |
| gpyreg | G2 (R4) | a low-noise posterior pickled by 1.3.1 factors its training covariance at every call | left: one factorization per call, as 1.3.1's `random_function` made | — |
| gpyreg | G3 (R5) | PyBADS's `_robust_gp_fit_` raises the lower bound of its noise after each failed fit, so the pair inverts after the fifth consecutive failure, and its result is unbound if all ten fail | a matter for PyBADS, handed to the work on PyBADS that the PI runs from its repository | — |

## The findings, wave by wave

### Wave 0: S-VBMC, the PyMC adapter and the exports, the calibration

Verified by the orchestrator on 2026-09-19 at `0e4e27a`, whose package code
is that of `f91fdf0` (`verification/wave0.md`); these slices have no MATLAB
counterpart, and the classes are defect, as specified, documentation,
cleanup and design choice. The PI ruled the same day; three Opus agents made
the fixes, whose commits the records name only for the Monte Carlo entropy
(`f3ba8d3`, `7c7972f`); the others were found in `f91fdf0..57cbfb4`. The
rulings also gave `dev/TODO.md` an item on saving and loading an `SVBMC`
object, which `dill` serializes (worklog, 2026-09-19), answering no numbered
finding. The
preparatory agent wrote the known-differences sheet (67 entries) and the
counterpart map (124 rows) before any comparison reviewer ran
(`prep_report.md`).

| id | slice | where | category | finding | class; dating | evidence | disposition | fix |
|---|---|---|---|---|---|---|---|---|
| N1 F1 | N1 | `svbmc/svbmc.py:57` `_validate_posteriors` (effect `_entropy.py:91`); no counterpart | cross-module | Runs whose original-space boxes differ are accepted. Draws outside a narrower run's box give NaN, so the ELBO, entropy and weights are silently NaN. | defect; not dated | wave0.md; `n1_f1_f2_f3_svbmc.py` part 1 | fixed: construction compares every run's original-space hard bounds and raises | `a7cc1f91` (git) |
| N1 F2 | N1 | `svbmc.py:605` reads `self.w`, `:747` overwrites it; no counterpart | state/caching | A second `optimize()` starts its logits from the first solution rather than the runs' own weights, so repeated calls drift (the `"all-weights"` and `"posterior-only"` modes). | defect; not dated | same script, part 2 | fixed: every `optimize()` starts from the runs' own weights; first call bit-identical | `b742a9bb` (git) |
| N1 F3 | N1 | `svbmc/_elbo_shrinkage.py:159`; no counterpart | formula/gradient | The two shrinkage stages do not compose to the shrunk run level: the between-run shift is added to the within-shrunk components. | as specified (step 5 of `plans/svbmc-shrinkage-estimator.md`); not dated | same script, part 3 | PI: stays as devised; a comparison of the two compositions added to the headline-selection item of `TODO.md` | — (TODO text in `7ddf79b0` (git)) |
| N1 F4 | N1 | `svbmc.py:846` (docstring), `:889`; no counterpart | control flow | `sample(balance_flag=True)` does not keep every component count within one draw. The within-run split delegates to `VariationalPosterior.sample`, which places its remainder at random. | documentation; not dated | `n1_f4_balance.py` | the docstring, the API page and Example 7 state the bound that holds; code unchanged | `b8ca3d6a`, `a601b761` (git) |
| N1 F5 | N1 | `svbmc.py:467` (with `:428`, `:552-558`); no counterpart | cross-module | The docstring of `stacked_entropy` says `stacked_ELBO` subtracts the Jacobian corrections. The subtraction happens once, in `__init__`. | documentation; not dated | reading | docstring corrected | `7395a36d` (git) |
| N2r F1 | N2 | `pymc/_target.py:349` (effect `:1133`); no counterpart | formula/gradient (support derivation) + control flow | The support is inferred from the transform kind: a log transform is read as (0, inf). `pm.Wald(alpha=2)` then reports a wrong support, and a run aborts on a non-finite density. | defect; not dated | `n2r_f1_wald_support.py`, `n2r_f1_wald_abort.py` | fixed: a kept log transform is checked by a probe ladder on the variable's own prior density; a shifted support is rejected at construction | `3a9da43e` (git) |
| N2r F2 | N2 | `_target.py:836`, `:859`; no counterpart | control flow | Only two of PyTensor's no-gradient signals are caught. `MethodNotDefined` (`pm.Rice`) escapes construction instead of taking the gradient-free route. | defect; not dated | `n2r_f2_rice_gradient.py` | fixed: a missing pullback is treated as an absent derivative | `12645aa9` (git) |
| N2r F3 | N2 | `variational_posterior/_torch.py:147`; `parameter_transformer.py:515`; no counterpart | formula/gradient | Near a hard bound the Torch export's density differs from `vp.pdf`. The NumPy transformer loses precision there (2e-2 within 2 ulp); the Torch inverse is exact. | defect of the core transformer's precision near a bound (Torch the accurate side); W7-6: at an upper bound with `abs(b) < b - a`, worst at `b = 0`, shared with MATLAB | `n2r_f3_near_bound.py`; W7-6: `wave7_O3_F3_F4_precision.py` | waited for P8 and O3. W3-33 found MATLAB's expressions the same. W7-6: the loss is at an upper bound with `abs(b) < b - a`; left as it is in both | — |
| N2r F4 + N2s F2 | N2 | `_target.py:1093-1102` `to_model_variables`, reached from `to_arviz` `:1252`; no counterpart | control flow (boundary condition) | One backward-map value that rounds onto its finite support bound rejects the whole export batch (found by both reviewers). | as specified (the plan requires strict interior), with a reachable boundary case the plan did not consider; not dated | `n2r_f4_onesided_absorb.py` | fixed: such a coordinate is returned as the adjacent value inside the support; non-finite and out-of-support values are still rejected | `ee67d225` (git) |
| N2s F1 | N2 | `_target.py:664`, `_plausible.py:289-292`, `_target.py:968-984`; no counterpart | defaults | Every default construction takes 4000 prior draws, used only for a location warning when no curvature fallback is needed. | as specified (`plans/pymc-target-adapter.md`); not dated | `n2s_f1_prior_draws.py` | PI: the draws at construction stay; the arguments that avoid them are documented | `acb6424a` (git) |
| N2s F3 | N2 | `_target.py:608-612`, `_compat.py:16`; no counterpart | control flow | `UnsupportedModel` (a `ValueError`) raised inside the mode search is relabelled as a mode-search failure and loses its type. | defect (diagnostics only); not dated | reading | fixed: `UnsupportedModel` passes through unchanged | `007ddaeb` (git) |
| N2s F4 | N2 | `_target.py:1187-1191` against `:619-621`; no counterpart | control flow (diagnostics) | The no-gradient warning names the initial point even when the user supplied `start`. | defect (message only); not dated | reading | fixed: the warning names the starting point the route used | `8e1d1691` (git) |
| N2s F5 | N2 | `_target.py:384`; no counterpart | control flow | The mask `unbounded` is computed and never read. | cleanup; not dated | grep | removed | `c362d06d` (git) |
| N3 F1 | N3 | `calibration/_campaign.py:608` (tolerance `:416`); `entropy/entmc_vbmc.py:120-156`; no counterpart | cross-module | Non-default entropy chunk budgets change the numbers (6.7e-16 in the gradient, 8.9e-16 in the value), so a calibrated profile can move a seeded run. | as specified at kernel level (`plans/machine-local-calibration.md`, rtol = atol = 1e-12); not dated | `n3_f1_entropy_budget.py` | PI decision: the MC entropy made chunk-independent bit for bit; the campaign accepts a budget only on identical output; kernel revision `chunk-kernels-v2` | `f3ba8d3`, `7c7972f`; docs `81371dd2` (git) |
| N3 F2 | N3 | `_campaign.py:204-211`; no counterpart | indexing/shape | Workload `active_d4_k20` times 200 samples per component with gradients, in the `entropy_grad` group. The package uses that count value-only. | defect (performance only); not dated | reading | PI, 2026-09-23, after the close: fix, the workload timed without gradients in the value group | `bc6180b6` |
| N3 F3 | N3 | `_campaign.py:1293-1310`; no counterpart | formula/gradient | The summary reports a rejected candidate's held-out speed-up next to the default selection. | defect (report metadata); not dated | reading | fixed: a speed-up is reported only when the held-out validation accepted the selected budget; otherwise None | `0d3875cf` (git) |
| N3 F4 | N3 | `_api.py:229-239`; no counterpart | control flow | A `ValueError` from `make_record` or `write_record` discards a finished campaign; only `OSError` is caught. | defect, latent (no live trigger); not dated | reading | PI, 2026-09-23, after the close: fix, a fallback profile or the result kept in memory | `58228a6a` |
| N3 F5 | N3 | `_campaign.py:883`; no counterpart | formula/gradient | The held-out win gate needs all four rounds: `ceil(0.80*4) = 4`. | as specified: the plan's rule of `ceil(0.8 R)` winning rounds, applied to its four held-out rounds (the class "the plan is silent" was wrong); conservative; not dated | reading | PI, 2026-09-23, after the close: left, the rule costing speed at most; a comment at the line | comment `56a8cf6f` |
| N3 F6 | N3 | `_campaign.py:1509-1532`; no counterpart | control flow | A watchdog stop or a coverage failure discards every group that did complete; a test pins this. | design choice; not dated | reading | PI, 2026-09-23, after the close: left, as the plan prescribes | — |
| N3 F7 | N3 | `calibration/__init__.py:14-16`, `_api.py:79-99`; no counterpart | defaults | The `calibrate` docstrings promise output the code does not print: selected budgets, held-out numbers, a 30-second estimate. | documentation; not dated | reading | both docstrings describe the printed output; printed text unchanged | `103c9998` (git) |
| N3 F8 | N3 | `_campaign.py:171-178`; no counterpart | indexing/shape | `sieve_gradient` times an 8192-row pdf call with gradients. The package takes that gradient one point at a time (`get_mode`). | defect (performance only); not dated | reading | PI, 2026-09-23, after the close: fix, the workload removed | `5cd8ccc3` |
| N3 F9 | N3 | `_cache.py:269-282`; no counterpart | state/caching | The cache key omits the package version: it holds the schema, kernel and workload revisions plus the machine identity. | design choice; not dated | reading | PI, 2026-09-23, after the close: left, the plan's decision; comments beside the kernels' blocking name `KERNEL_REVISION` | comments `56a8cf6f`, `2369ea2f` |
| N3 F10 | N3 | `_campaign.py:28`, `:353-361`; no counterpart | control flow | `_MAX_WORKLOAD_SLOWDOWN` and `_deduplicate_order` are defined and never used. | cleanup; not dated | grep | the veto compares against `1/_MAX_WORKLOAD_SLOWDOWN` (same value, asserted by a test); the helper is removed | `d676f752` (git) |

### Wave 1: the MATLAB changes since the port (M), the variational optimization (P6), the active-sampling search (P2)

Verified on 2026-09-19: P6 at `e024b1b`, M and P2 at `cb2d513`, the options
at `0d24698`, the package code of `f91fdf0` with the wave-0 fixes, the
chunk-independent entropy and `cb2d513` (`verification/wave1_P6.md` and
`wave1_M_P2.md`, whose headers carry their corrections, and
`wave1_options.md`, a pass over the options surface that grew out of M F2).
`verification/wave1.md` gives the disposition of every finding.

| id | slice | where | category | finding | class; dating | evidence | disposition | fix |
|---|---|---|---|---|---|---|---|---|
| P6 int F1 + cmp F1 | P6 | `vbmc/variational_optimization.py:904-910` `_vb_init` type 2; `misc/vbinit_vbmc.m:32` | indexing/shape (int); formula/gradient (cmp) | `V / lambd0**2` broadcasts `(D,)` against `(D,1)`, giving `mean(V)·mean(1/λ²)` for MATLAB's `mean(V./λ²)`. Type-2 widths are inflated 1.01–3.02× on the oracle states. | confirmed port discrepancy; never matched (Python `9267816`, 2021-08-25; MATLAB line from `28dd1db`, 2018-07-30, per the header correction) | wave1_P6.md (checks not retained) | fixed | `be38570` |
| P6 int F2 + cmp F6 | P6 | `variational_optimization.py:308-309`, `:326`; `set_parameters`; `misc/rescale_params.m:33-37` | state/caching | The returned posterior's `eta` is the endpoint's while `w` is the winning midpoint's, so `softmax(eta) != w`. MATLAB deletes `eta`. | confirmed port discrepancy (latent); MATLAB unchanged | wave1_P6.md | fixed; the same commit keeps the `(1, K)` shape of `eta` in the pruning loop (a verifier observation) | `495a497`; check table: `b34a825`; ruling of 2026-09-22: `422f774` |
| P6 int F13b + cmp F2 | P6 | `variational_optimization.py:1027` (`bounds = None`); `get_bounds`; `misc/vpbounds.m:9-24`, `vbinit_vbmc.m:39`, `vpoptimize_vbmc.m:188` | state/caching | The soft bounds never accumulate: the box is rebuilt from the current training inputs on every call, where MATLAB accumulates it across iterations. | confirmed port discrepancy; since the original port (`9267816`); pinned by a test from `d144a83` (2026-09-05) | wave1_P6.md; ruling: `soft_bounds_trace.py` | PI: the box is rebuilt from the call's training inputs; the accumulation code, which never had a stored box to widen, is removed | `7dd2e3f` |
| P6 int F5 + cmp F8 | P6 | `variational_optimization.py:834-841` `_sieve`, `:168`; `misc/vpsieve_vbmc.m:84-87` | control flow | `_sieve(init_N=0)` returns a bare posterior and an int that `optimize_vp` cannot consume. | confirmed port discrepancy (unreachable in PyVBMC); MATLAB unchanged | wave1_P6.md | fixed | `51b01cb`; check table: `5b7881c` (docs) |
| P6 int F10 + cmp F9 | P6 | `vbmc/minimize_adam.py:81`, `:97`, `:139-140`; `utils/fminadam.m:39` | indexing/shape (int); state/caching (cmp) | `minimize_adam` writes one Adam step into the caller's `x0`. Its trailing-average window wraps for `max_iter` 11 to 19. | confirmed port discrepancy (latent); the window a confirmed Python-only defect (latent); MATLAB unchanged | wave1_P6.md | fixed | `dccbc3d` |
| P6 int F7 + cmp F4 | P6 | `variational_optimization.py:227-241`; `misc/vpoptimize_vbmc.m:76-101` | control flow | The deterministic-entropy branch raises on `not res.success`, has no evaluation cap, and reads `DetEntTolOpt` as a gradient-norm tolerance. MATLAB caps at `50*(D+2)` and continues. | confirmed port discrepancy; MATLAB unchanged since the port | wave1_P6.md | the iterate is kept with a warning; the cap and the tolerance become a sheet entry; a NaN objective is handled by the check | `04d15ff`; check: `43dfb7b` |
| P6 int F3 | P6 | `variational_optimization.py:482` `_eval_full_elcbo`; `misc/vpoptimize_vbmc.m:279`, `:288` | control flow | The full ELCBO evaluation uses the Monte Carlo entropy where the exact one exists (`K == 1`, `entropy_switch`). | paper-versus-code difference shared with MATLAB; not dated | wave1_P6.md | exact entropy for a one-component posterior; under `entropy_switch` shared with MATLAB and kept | `7331841`; check table: `831b024` |
| P6 int F4, F6, F13a | P6 | `minimize_adam.py:87`, `:101`; `variational_optimization.py:1645`; `:484`; `utils/fminadam.m:48`, `:63`; `misc/gplogjoint.m:404`; `misc/vpoptimize_vbmc.m:281` | indexing/shape; formula/gradient; state/caching | F4: `x_tab` and `y_tab` are offset by one. F6: `var_ss` adds an SD of variances to a variance of means. F13a: `"skip_elbo_variance" in options` is always false. | not defects: F4 shared with MATLAB; F6 an exact port (MATLAB line since `a1680e3`, 2018-05-04); F13a a faithful port of a dead MATLAB guard | wave1_P6.md | not defects; F6 documented | `d54266c`; check table: `5b7881c` |
| P6 int F8, F9 | P6 | `variational_optimization.py:551-561`; `:771-775` | state/caching; defaults | F8: `eta` is excluded from the soft-bound loss while `get_bounds` still computes its bounds. F9: `elcbo_beta` is hard-coded to 0. | intentional differences (F8: PI decision of 2026-09-08; F9: unported, inactive at MATLAB defaults) | wave1_P6.md (reading) | already on the sheet | — |
| P6 int F11, F12 | P6 | `minimize_adam.py:63`, `variational_optimization.py:261-268`, `:49-65`; `utils/fminadam.m:23`, `private/updateK.m:16-26` | defaults; defaults/control flow | F11: Adam's `β₂` and step sizes differ from the paper. F12: `update_K` masks the two oldest window entries, with a window of 6 against the paper's 4. | paper-versus-code differences shared with MATLAB | wave1_P6.md | no action | — |
| P6 cmp F3 | P6 | `vbmc/vbmc.py:1504-1506`; `vbmc.m:699` with `:459` | control flow (and defaults) | The sieve's candidate count is `50·vp.K`. MATLAB's main loop evaluates `NSelbo` at a stale `K = Kwarmup`, always 100. | confirmed port discrepancy (PyVBMC's value the more sensible); MATLAB line since `c476ab9` (2019-01-10) | wave1_P6.md (grep) | sheet entry; MATLAB-side defect, entry 8; the warp-branch paragraph corrected by W2-16 | — |
| P6 cmp F5 | P6 | `variational_optimization.py:1476-1479`, `:1499-1503`, `:1588-1592`; `misc/gplogjoint.m:86-90`, `setupvars_vbmc.m:247` | formula/gradient | The GP-smoothing bandwidth (`vp.delta`, `Bandwidth`) is missing from `tau` and `nu`. | intentional difference in effect / unported feature, inactive at MATLAB defaults | wave1_P6.md | sheet (unported) | — |
| P6 cmp F7 | P6 | `variational_optimization.py:45`, `options.py:248-251`; `private/updateK.m:10` | cross-module (and defaults) | `adaptive_k` is evaluated with the keyword `unkn`, so a callable cannot be called; the `round` semantics also differ. | confirmed port discrepancy; MATLAB unchanged | wave1_P6.md | fixed: the keyword by wave 1, the rounding by wave 5 (W5-6) | `06f9b03`; `8e2977a`; check table: `1443b39` |
| P6 cmp F10a, F10b | P6 | `variational_optimization.py:821`, `:313`; `misc/vpsieve_vbmc.m:81`, `vpoptimize_vbmc.m:176` | control flow | The sort is unstable where MATLAB's `sort` is stable, and `argmin` selects a NaN where MATLAB's `min` skips it. | confirmed port discrepancy (ties only; needs a NaN evaluation); MATLAB unchanged | wave1_P6.md | fixed, with the second unstable sort the verifier found; empty slots and the midpoint by the check | `81e21bf`; check: `43dfb7b` |
| P6 cmp sheet notes 1-2 | P6 | sheet entries on `J_sjk` and on sampling of the variational parameters; `misc/vpoptimize_vbmc.m:236-237` | — | The `J_sjk` entry mischaracterizes MATLAB; the `VarParamsBack` entry omits two dead options. | reviewer correct | wave1_P6.md | sheet corrected | `583ef86` |
| P6 cmp sheet note 3 | P6 | `active_sample.py:716`; `private/activesample_vbmc.m:468` | — | The result of the `vp_repo` `np.append` is discarded. | reviewer correct (confirmed Python-only defect, dormant) | wave1_P6.md (reading) | fixed | `1564769` |
| P6 verifier: type 3 permutation | P6 | `variational_optimization.py:969-971`; `misc/vbinit_vbmc.m:86` | — | Type 3 of `_vb_init` draws its permutation only when the means are optimized; MATLAB always draws it. | verifier observation (random-stream difference; non-default) | wave1_P6.md | left for the PI; kept (2026-09-22): the streams are not comparable draw for draw, and the case needs `variable_means=False` | — |
| P6 verifier: `get_bounds` `tol_weight == 0` | P6 | `variational_posterior.py:359-363`; `misc/vpbounds.m:28` | — | A PyVBMC-only special case for `tol_weight == 0`, doubly dead. | PyVBMC addition, not a port | wave1_P6.md | removed | `7dd2e3f` |
| P6 verifier: `get_parameters` cached mode | P6 | `variational_posterior.py:1025-1035`; `misc/rescale_params.m:40` | — | `get_parameters` leaves the cached mode, which MATLAB removes. | unported line, harmless | wave1_P6.md | fixed in wave 5 (W5-13) | `eab21fa` |
| P2 F1 | P2 | `active_sample.py:516-542`; `private/activesample_vbmc.m:275-283`, `utils/cmaes_modded.m:561-567` | defaults (optimizer settings) | CMA-ES gets a scalar `sigma0 = max(insigma)` and loses MATLAB's per-coordinate scaling `insigma/max(insigma)`. | confirmed port discrepancy; unchanged in both histories | wave1_M_P2.md; `cmaes_side_by_side/` | fixed (`CMA_stds`) | `4d1d984`; check table: `3bc079f`; ruling: `e4ffa60` |
| P2 F2 = M F1 | P2, M | `option_configs/advanced_vbmc_options.ini:11`; `vbmc.m:198` | defaults | `fun_eval_start` is `max(D, 10)` where current MATLAB has `10*ceil((D+1)/10)`; they differ for `D >= 10`. | confirmed port discrepancy; MATLAB changed after the port, `c387612` (2022-10-26); Python line since `d72c7df` (2021-05-29) | wave1_M_P2.md | fixed (PI decision) | `cb2d513`; check table: `e56da87` |
| P2 F3, F4 | P2 | `active_sample.py:167-188`, `:867-868`; `misc/initdesign_vbmc.m:29-46`, `activesample_vbmc.m:561-562` | control flow/state; indexing/shape | Surplus starting points are deleted from the cache instead of kept, and the random-point count subtracts the whole cache rather than the points taken. | confirmed port discrepancy (F4 masked by F3); MATLAB unchanged (F3: since 2019) | wave1_M_P2.md | fixed; their reuse after a warp by the check | `181a63b`; check: `e985b95` |
| P2 F5 | P2 | `active_sample.py:539-542`; `utils/cmaes_modded.m:215` | defaults (optimizer settings) | PyVBMC runs CMA-ES with cma's noise handler; MATLAB has `Noise.on = 0`. | confirmed port discrepancy, knowingly retained (Stage 2 plan); handler since the original port (`4949c82`, 2021-11-04) | wave1_M_P2.md; `cmaes_side_by_side/` | PI: dropped after the side-by-side (equivalent in value within noise; about 2.4 extra evaluations per generation) | `d617d99` |
| P2 F6 | P2 | `active_sample.py:544`, `.ini:185`; `cmaes_modded.m:1708-1721`, `activesample_vbmc.m:282-290` | defaults/control flow | `search_cmaes_best` is never read. PyVBMC returns the best-ever solution; MATLAB returns the best of the last generation. | confirmed port discrepancy; MATLAB unchanged | wave1_M_P2.md; `cmaes_side_by_side/` | PI: stays inert (the acquisition is deterministic during the search) | `a63b17a`; sheet, `cma` entry |
| P2 F7 | P2 | `active_sample.py:871-880`, `vbmc.py:1011`; `activesample_vbmc.m:565-568` | control flow | `search_cache_frac > 0` raises on the first active-sampling step, because the search cache starts as `[]`. | confirmed Python-only defect; MATLAB unchanged | wave1_M_P2.md | fixed; since wave 5 (W5-24), a value above 0.25 with the shipped fractions is refused | `9abeb10`; wave 5: `a25fc73` |
| P2 F8 = M F3 | P2, M | `active_sample.py:730-733`; `activesample_vbmc.m:481` (M cited `:468`) | control flow | The rank-one GP update is gated with `and` where MATLAB has `or`. | confirmed port discrepancy; the header says the verdict fails on noisy targets, where MATLAB recomputes (`gplite_post.m:76-79`); since the original port | wave1_M_P2.md; W4-11: `wave4_P4_5_rank_one_update.py`, `wave4_P4_5b_sn2_mult.py` | changed by `510a493` (rank one for a fresh noisy observation, away from MATLAB); the PI ruled in wave 4 (W4-11) that it stays | `510a493`; check table: `e71eab2` |
| P2 F9 | P2 | `active_sample.py:484-488`, `:545-551`; `activesample_vbmc.m:265-316` | control flow/defaults | At `D == 1` the search switches permanently to an unbounded SciPy Nelder-Mead with SciPy's default budget. | confirmed port discrepancy (Python-only substitution); M's stated cause holds (`cma` does not support `D = 1`, and with bounds it raises); `wave1_M_P2.md`'s correction of it does not (its header); since `4949c82` (2021-11-04) | wave1_M_P2.md | fixed (bounded scalar search over its interval); the `"Nelder-Mead"` value removed in wave 5 (W5-25) | `cb8a51d`; check table: `3bc079f`, `73fb89c`; wave 5: `4cc09cc` |
| P2 F10 | P2 | `active_sample.py:293-295`; `activesample_vbmc.m:84`, `funlogger_vbmc.m:279` | state/caching | `active_sample` writes `optim_state["N_eff"]`; its readers use `n_eff`. | confirmed Python-only defect; "no effect at defaults" holds at the noiseless defaults only (header) | wave1_M_P2.md | fixed for the name; the check adds the refresh after every evaluation | `118626e`; check: `6590ea4` |
| P2 F11 | P2 | `active_sample.py:327`, `:733`, `vbmc.py:1073`; `misc/noiseshaping_vbmc.m` | cross-module | `noise_shaping` is a live switch whose shaping is never applied. | confirmed port discrepancy (unported feature, partially live option) | wave1_M_P2.md | refused when true; the check makes the message name the `load` remedy | `bb6ab65`; check: `497d15e` |
| P2 F12 | P2 | `active_sample.py:500-502`; `activesample_vbmc.m:253-254` | indexing/shape | The non-finite fallback builds `(N, D)` search bounds instead of one per coordinate. | confirmed Python-only defect, latent (practically unreachable) | wave1_M_P2.md | fixed | `659e83c`; check table: `3bc079f` |
| P2 F13 | P2 | `active_sample.py:509-553`; `activesample_vbmc.m:264`, `:317-320` | control flow | The local search has no `try/catch` fallback. | confirmed port discrepancy; MATLAB unchanged | wave1_M_P2.md | fixed (the sieve's best candidate is kept) | `795b3b0`; check table: `3bc079f` |
| P2 F14 | P2 | `active_sample.py:665-668`, `.ini:299`; `activesample_vbmc.m:436-443` | defaults | `active_sample_fess_thresh` is declared and never read. | confirmed port discrepancy (inert option) | wave1_M_P2.md | registered as inert | `a63b17a` |
| P2 F15 | P2 | `active_sample.py:715-720`; `activesample_vbmc.m:468` | state/caching | The `vp_repo` append is a discarded expression. | confirmed Python-only defect, inert; the reviewer's reasoning is wrong | wave1_M_P2.md | removed | `1564769` |
| P2 F16 | P2 | `active_sample.py:669-691`, `:192-202`; `activesample_vbmc.m:446-457`, `:533-535`, `initdesign_vbmc.m:50-57` | control flow (diagnostics only) | (a) One `gp_train` timer is unbalanced; (b) the initial design is untimed; (c) the overlapping timers are not subtracted. | confirmed Python-only defect (diagnostics) | wave1_M_P2.md | (a) fixed; (b) and (c) left by a ruling of wave 5 (W5-28) | `a719de6` |
| P2 minor observations (8) | P2 | `active_sample.py:490`, `:415`, `:699`, `:787`, `:481`, `:632-637`; `stats/get_hpd.py:35` | — | The eight: `f_val_old` reused from the batch; half-to-even rounding; unstable `argsort` of the search set; direct option calls; case-sensitive `"none"`; draw order; `gp_length_scale` shape; `skip_logger` left in the cache. | all confirmed (the header corrects "seven" to eight) | wave1_M_P2.md | `f_val_old`: Stage 2 plan. Rounding: W5-6. `argsort`: ruled and fixed. Direct calls: check. `"none"`: W5-25. Draws: sheet. Shape: no action. `skip_logger`: fixed, and an acquired cached point leaves the cache in every case (MATLAB-side defect 6) | `a7f323e`; `8e2977a`; check table: `179406c`; ruling: `3d71b00` |
| P2 §4 sheet notes | P2 | `active_sample.py:28` (`_selection_policy_callback`); sheet | — | The sheet lacks the `_selection_policy_callback` seam and (per P2) any `vp_repo` entry. | — | wave1.md | the seam is a developer hook, inert unless installed; no action. P2's `vp_repo` statement is wrong: the entry existed | — |
| M F2 | M | `.ini:135`, `:231`, `:233`, `:235`, `:305`, `:313`; `vbmc.m` defopts, `misc/setupoptions_vbmc.m` | defaults (option surface) | Six options that MATLAB deleted in 2021 survive as dead `.ini` entries, accepted and ignored. | confirmed port discrepancy (surface only); MATLAB changed after the port, `2044530` (2021-06-18) | wave1_M_P2.md (grep) | registered as inert | `a63b17a` |
| M F4 | M | `vbmc.py:3097-3158` `_create_result_dict`; `vbmc.m:942-956` | cross-module (public output surface) | MATLAB's `samples` output struct has no counterpart in `results`. | confirmed port discrepancy (public surface; line citation off); MATLAB changed after the port, `48c82e0` (2021-03-27) | wave1_M_P2.md | sheet (unported) | — |
| M F5 | M | `whitening/whitening.py:151-158`; `misc/warp_input_vbmc.m:59-63` | control flow | The non-numeric branch of `warp_cov_reg` cannot execute: it uses attribute access and indexes where MATLAB calls. | confirmed Python-only defect; not dated (no MATLAB change in the range caused it) | wave1_M_P2.md | fixed; the check adds the check at construction and in `load` | `1d99936`; check: `cc0ef2f` |
| M sheet notes | M | `counterpart_map.md` rows `initdesign_vbmc.m`, `gplite_pred.m`; sheet | — | `46b6f5e` is misattributed; the `gplite_pred` note; missing entries for the probit default, `SearchOptimizer` values, the D == 1 override, the inert options, `samples`, and two MATLAB-side defects. | all confirmed (the verifier's exception, the cause of the D == 1 override, is withdrawn by the header of `wave1_M_P2.md`) | wave1_M_P2.md | counterpart map corrected; the prediction's log density is slice G2's; the one-dimensional override as P2 F9 | `583ef86`; `wave6.md` |
| verifier items 1-2 | P2 | `cma.fmin` settings; `cmaes_modded.m:207`, `setupoptions_vbmc.m:170` | — | MATLAB evaluates `x0` inside CMA-ES (`EvalInitialX`), and `TolHistFun = 1e-13` against cma's `1e-12`. | settings differences (same class as F1, F5, F6) | wave1_M_P2.md | PI: cma's defaults stay, and so does cma's absolute `tolx` (worklog, 2026-09-19) | — (sheet, `cma` entry) |
| verifier item 3 | P2 | `funlogger_vbmc.m:232-247` with `activesample_vbmc.m:481` | — | On a noiseless repeat, MATLAB's `or` path appends a duplicate GP training row. | MATLAB-side defect | wave1_M_P2.md | MATLAB-side defect, entry 5 | — |
| verifier item 4 | P2 | `function_logger.py:347`; `funlogger_vbmc.m:279` | — | MATLAB refreshes `Neff` after every logged evaluation; the Python refresh is commented out. | the other half of P2 F10 | wave1_M_P2.md | fixed by the check | check: `6590ea4` |
| verifier items 5-6 | P2 | `active_sample.py:397-404`; `activesample_vbmc.m:152-157`; tests | — | The pre-computation order differs, with no stream effect; the reports' test-adequacy claims hold. | no action | wave1_M_P2.md | no action | — |
| O-1 | options | both `.ini` files; `options.py` `INERT_OPTIONS` | — | Declared options that no module reads: the six of M F2, knobs of unported features, and options whose consumer takes an argument. The header corrects the count to 28 (25 after `bd47856`). | confirmed (surface only); not dated | wave1_options.md (scan kept as a test) | registered inert, marked, warned about at validation; `bd47856` made three live; wave 2 (W2-10, W2-22) handled three more | `a63b17a`, `0d9a422`; check table: `2a3b8e8`, `2d931a6` |
| O-2 | options | `vbmc.py`, `active_sample.py`; `noiseshaping_vbmc.m` unported | — | `noise_shaping=True` disables the rank-one update without shaping the noise: a configuration of neither toolbox. | confirmed port discrepancy (P2 F11); not dated | wave1_options.md (grep) | `NotImplementedError` at construction; since `9b5213d`, `load` refuses a saved run holding it (header); the check makes the message name the remedy | `bb6ab65`; check table: `497d15e` |
| O-3 | options | `vbmc.py` call sites of `determine_best_vp`; `vbmc.m` → `best_vbmc` | — | Both call sites passed none of `safe_sd`, `frac_back` and `rank_criterion_flag`, so `rank_criterion = True` never took effect. | confirmed port discrepancy; not dated | wave1_options.md | both call sites pass the options; since the check, a direct call defaults to them | `bd47856`; check table: `e3c6e93`, `c469ba5` |
| O-4 | options | ranking branch of `determine_best_vp` | — | The ranking branch had never executed: it indexed object-dtype arrays, which raise `IndexError` when used as a mask. | confirmed Python-only defect (latent until O-3); not dated | wave1_options.md | reads through `np.asarray`; the check ranks NaN scores last | `bd47856`; check table: `f5eafa7` |
| O-5 | options | `determine_best_vp`; `misc/best_vbmc.m` | — | Two counts use the 0-based `max_idx` where `best_vbmc` uses the number of iterations: the look-back window start and the rank penalty. | confirmed port discrepancy; not dated | wave1_options.md | fixed | `4892fe3` |
| O-6 | options | `optim_state["int_mean_fun"]` | — | `gp_int_mean_fun` is copied into an `optim_state` key that nothing reads. | confirmed (surface only); not dated | wave1_options.md (grep) | none in `wave1_options.md` (the sheet's `intmeanfun` entry); the check registered it as inert, with `proposal_fcn` | check: `375df34` |

**The independent check of the pass** (2026-09-22, `wave1.md`). Six fresh
read-only reviewers on `a3d4a70d`: the active-sampling search; the initial
design, the search cache, the GP updates inside active sampling and the
warp's regularization; the variational optimization; the options and the
selection of the returned posterior; the three ledgers and the worklog; the
changelog, the sheet, the other records and the consequences outside the
changed lines. No later pass had undone a wave-1 fix. The fix round ran on
the branch `dev-port-review-w1check`, cut at `326c7676`, merged into
`dev-port-review` as `e86bbb1c` (fix reports `fixes/wave1_check_agent_A.md`
and `_B.md`).

| finding | statement | disposition | fix |
|---|---|---|---|
| R3: `optimize_vp` | Slots never filled hold `+inf` and NaN parameters. When every evaluation was NaN, `nanargmin` took an empty slot and returned a NaN posterior without an error. | selects among the evaluations made; raises when all are NaN; the midpoint skips NaN as MATLAB's `min` does | `43dfb7b` |
| R4: NaN scores | In the object arrays NaN is incomparable, so a NaN ELCBO or reliability index ranked anywhere, and the look-back took a NaN. | NaN ranks last, the look-back skips it, and all-NaN takes the last iteration (both branches) | `f5eafa7` |
| R6: direct `determine_best_vp()` | Called without its selection arguments, it ignored the run's options. | defaults to the run's options; the `best_safe_sd` test names its branch | `e3c6e93`, `c469ba5` |
| R2: cached point after a warp | W5-7's check compared the candidate with a new one-row transform, which rounds differently after a warp. The stored value was dropped and the target called. | compared with the sieve's own pre-clip row | `e985b95` |
| R2, R5: `N`, `n_eff` | `118626e` refreshed them only at the top of each acquisition, so the in-loop GP refit read them one evaluation behind. Its message wrongly named `optimize_vp` as a reader. | refreshed after every evaluation and after the initial design, as `funlogger_vbmc.m:278-279`; moves noisy trajectories | `6590ea4` |
| R4: `gp_int_mean_fun`, `proposal_fcn` | Each was copied into an unread key, gave no warning, and `load` refused it with unhelpful advice. | registered as having no effect | `375df34` |
| R4, R6: `load` warning | `load(new_options=...)` did not warn of an option without effect, as the changelog promised. | fixed | `2a3b8e8` |
| R4, R6: `noise_shaping` refusal | The refusal did not name the `new_options` that loads a saved run carrying the option. | fixed | `497d15e` |
| R2: budget equal to the design | The first GP fit ended in a division by zero. `cb2d513` had moved that case to `max_fun_evals = 20` for D 10–19. | fixed | `e56da87` |
| R2: `ns_elbo`, `ns_ent_fine_active` | A number given for either raised `TypeError` in the in-loop updates (P2's fourth minor observation, never disposed of). | read through `Options.eval` | `179406c` |
| R2: `warp_cov_reg` | `None`, a string, a complex number or NaN failed deep in the first warp, and `True` was taken as 1. | checked at construction, in `load`, and a function's result at the warp | `cc0ef2f` |
| R3: in-loop entropy | The in-loop comparison scored a one-component posterior by MC entropy, while its reported ELBO has been exact since `7331841`. | exact on both sides | `831b024` |
| R3: `set_parameters` | A zero weight set `eta` with a division-by-zero warning. | fixed | `b34a825` |
| R1, R2: tests | The rank-one variance at levels 1 and 2; the end-to-end test's GP fit on the run's generator; the local-search branches against MATLAB's expressions. | added | `e71eab2`, `3bc079f` |
| R1, R3, R4, R6: documentation | The mixture-test fixture; `_sieve` and `var_ss`; option descriptions; four search comments; the oracle harness on the legacy global state. | corrected | `893d0ba`, `5b7881c`, `1443b39`, `73fb89c`, `2d931a6`, `ea47ff0` |
| records | The wave-1 ledgers and the cmaes README; the worklog (counts, hash `29c822d`→`e48fade`, trajectory list, soft-bound figures); the changelog; the sheet, the MATLAB-side list, the counterpart map and the Stage 2 plan. | corrected, as flags in the headers where a record is kept verbatim | `b8d218b5`, `aed685ff` (git) |
| left: search-set sort | The sort was unstable when a search cache is kept (P2's third minor observation). | PI 2026-09-22: sorted stably | `3d71b00` |
| left: `Options` from another run | Options built from another run took over its user-option set, and changed the first run (predates wave 1). | fixed | `2d8448e` |
| left: non-finite CMA-ES step | cma does not return from such a start; no known path produces one. | not started; fails as a raising search does | `e4ffa60` |
| left: `get_parameters(raw_flag=True)` | Warned on a zero weight. | fixed | `422f774` |
| left: `hpd_frac` | Below 0.15 of ten points, the first GP fit failed. | checked at construction and in `load` | `f18a6c1` |
| left: `f_vals` texts | The texts did not say that `f_vals` are log-joint values. | texts corrected; the values unchanged | `ba0234c` |
| left: PyMC setup probe | The `max(D, 10)` in the probe script. | its source stated; the number stays | `f5b7efc` |
| kept: type-3 permutation | The draw is taken only when the means are optimized. | kept: the streams are not comparable, and the case needs `variable_means=False` | — |
| kept: warp with integer variables | The snap changes an on-grid cached point in its last bits, so the point is evaluated. | kept: a tolerance would reopen W5-7 | — |
| kept: scan test | The `INERT_OPTIONS` test counts quoted names in strings. | kept: an AST scan waits until the test is touched | — |
| kept: `torch_vi_step.py` | Its source hashes are stale. | refreshed if the scaffolding is taken up again | — |
| kept: porting log | `pyvbmc/vbmc/README.md` has no P2 entries. | consolidated at the end of the review | — |
| kept: wave-1 gate evidence | Not kept, except the options-test log. | the check's gates stand in | — |
| kept: accuracy of `6590ea4` | The refreshed counts move noisy trajectories. | benchmark sweep at the merge; PI 2026-09-23: the merge taken as it is | — |

### Wave 2: the main loop (P1a) and the setup, options and state (P1b)

Read at `51451dc` and verified at `d6c3827` on 2026-09-20, the package code
of `2f2bc94`, after the wave-1 fixes (`verification/wave2.md`, with
`wave2_B_loop.md`, `wave2_C_setup.md` and `wave2_warmup_history.md`). Rows
W2-1 to W2-6 fire at the defaults, W2-6 latently (it needs a best posterior
from before an earlier warp).

| id | slice | where | category | finding | class; dating | evidence | disposition | fix |
|---|---|---|---|---|---|---|---|---|
| W2-1 | P1a | `_check_warmup_end_conditions` (`recent_past`); `private/vbmc_warmup.m:60-65` | indexing/shape | The "recent improvement" window starts at `iteration - (T + 1)` where MATLAB's is `iteration + 1 - T` (0-based): five iterations against three at the defaults, so warm-up tends to end later. | port discrepancy; never matched: one iteration too long in `35be58b` (2021-07-07), two since `9267816` (2021-08-25) | reading; history traced (`wave2_warmup_history.md`) | fix; moves default trajectories | `c918d12` |
| W2-2 | both (P1a int F6, cmp F2; P1b int F8) | `_recompute_lcb_max`, `_check_warmup_end_conditions`; `private/recompute_lcbmax.m`, `private/vbmc_warmup.m:46-50` | defaults (P1a int, P1b int); control flow (P1a cmp) | `_recompute_lcb_max` returns an empty array that nothing reads, and the warm-up check lacks MATLAB's preference for the recomputed cumulative maximum, although `recompute_lcb_max` defaults to `True`. | port discrepancy; never matched, an unfinished placeholder (`pass`, then an empty array since `ec94eba`, 2021-06-08; ToDo from the pylint clean-up `39b29e6`) | reading; reads of `lcb_max_vec` searched; history traced | fix, the function and the consuming branch; moves default trajectories | `8e591ff` (the orchestrator added the `NaN` handling); check: `224c7a1` (documents the NaN condition) |
| W2-3 | P1a | `_setup_vbmc_after_warmup`; `private/vbmc_warmup.m:97-102` | state/caching | The end of warm-up sets `last_warmup` only. `last_warping` and `last_successful_warping` stay at −inf, so neither the first warp nor stability termination is held back as in MATLAB. | port discrepancy; omitted in `35be58b` (2021-07-07); the readers came with `4aa4ca9` (2022-03-03) and `0a27ee3` (2023-02-15) | reading; readers and writers searched; history traced | fix; moves default trajectories | `fb8a12e` |
| W2-4 | P1b | `VBMC.__init__`; `misc/setupvars_vbmc.m:65`, `:82` | indexing/shape (wrong coordinate space) | The initial variational means are built from the untransformed `x0`, so they hold original coordinates (bounds 0/10, plausible 4/6, `x0 = 5` → 9.94). | port discrepancy; since `489598a` (2021-06-03); P1b cmp had called it equivalent | `wave2_initial_vp_mu.py` (five cases); reading; `git log -S` | fix; moves default trajectories | `8cd4bbc` |
| W2-5 | both (P1a int F4, cmp F4; P1b cmp F9) | `_check_termination_conditions`; `private/vbmc_termination.m:98-99` | indexing/shape (P1a); control flow (P1b cmp) | The `min_iter` guard compares the 0-based index, so a run on which the minimum binds performs `min_iter + 1` iterations. | port discrepancy; not dated (a test asserted the off-by-one) | `wave2_min_iter_guard.py`; reading | fix; moves default trajectories | `567444f`; check: `2d3f876` |
| W2-6 | P1a | `whitening.warp_input`; `misc/warp_input_vbmc.m:8`, `:133` | cross-module | `warp_input` inverts the search bounds and cache with the transformer of the best recorded posterior, which may predate a warp. The box comes out 8.5 to 12.5 times wider per coordinate in the reproduction. | shared defect; not dated; latent (never on 21 stored runs, 42 warps) | `wave2_stale_transformer_mechanism.py`, `wave2_stale_transformer_frequency.py` | fix | `7d93f60` |
| W2-7 | P1b | `Options.update_defaults`; `misc/setupoptions_vbmc.m:127-163` | defaults (control-flow consequence) | The five noisy-target defaults apply only with `specify_target_noise`, not whenever uncertainty handling is on, so a level-1 run keeps the noiseless defaults. | port discrepancy; since `70325a3` (2022-06-02) | `wave2_noisy_defaults.py`; reading | fix: the defaults apply whenever uncertainty handling is on | `0ba7680` |
| W2-8 | P1b | `VBMC.__init__` (`update_defaults` runs before the `options_path=` file is read) | control flow (and defaults) | `specify_target_noise = True` written in the `options_path=` file gives a level-2 run with the noiseless defaults. | Python-only defect; not dated | `wave2_noisy_defaults.py` | fix | `640ac92` |
| W2-9 | both (P1a int F7; P1b int F3) | `load`, `optim_state["max_fun_evals"]`, GP training options; `misc/get_GPTrainOptions.m:98` | state/caching | `load(new_options={"max_fun_evals": n})` leaves `optim_state["max_fun_evals"]` stale. Past that horizon the GP design schedule falls to its floor of 9 (9 against 759 at 60 evaluations). | Python-only defect; regression of `6769a9a` (2026-09-16), in no release | `wave2_resume_budget.py` | fix | `6fa6073` |
| W2-10 (with B-M12) | both (P1a int F1, cmp F5; P1b int F2, cmp F2) | `optimize` (`vbmc.py:1226`, `:2169`); `vbmc.m:524-525`, `private/vbmc_termination.m:80` | cross-module (int); state/caching (cmp) | `optimize` reads `optim_state["entropy_force_switch"]`, which nothing writes, where MATLAB reads the option. `entropy_switch=True` with `D >= 5` raises `TypeError`, and the stability branch lacks MATLAB's `isfinite` test. | port discrepancy (B-1, B-M12); since `ec94eba` (2021-06-08) | B-1 (`wave2_B_f1_entropy_force_switch.py`); `wave2_two_option_runs.py` | fix, both sites | `e71f97b` |
| W2-11 | both (P1a int F2, cmp F7; P1b int F5) | `optimize` (`vbmc.py:1580-1582`); `vbmc.m:781` | control flow | `len(run_cov == 0)` stands for `len(run_cov) == 0`, so the running average of the variational moments is never taken and `moments_run_weight` has no effect. | port discrepancy (B-2); since `ec94eba` (2021-06-08, per B-2); only the recorded state differs | B-2 (`wave2_B_f2_run_moments_guard.py`) | fix | `ab5c603` |
| W2-12 | P1a | `_create_result_dict`; `private/vbmc_output.m:5-9` | cross-module (int); state/caching (cmp) | `results["problem_type"]` tests the transformed bounds, which are infinite for a bounded variable too, so it always reports "unconstrained". | shared defect (B-3a); the field dates from `c477a7b` (2021-11-01) and always matched MATLAB | B-3a (`wave2_B_f3_result_fields.py`) | fix in PyVBMC (test the original bounds); MATLAB's defect goes on the sheet | `41ea8b1` |
| W2-13 | P1a | `_create_result_dict`; `private/vbmc_output.m:10` | indexing/shape | `results["iterations"]` holds the 0-based index of the last iteration, where MATLAB's `output.iterations` is the count. | port discrepancy (B-3b); since `c477a7b` (2021-11-01), never matched | B-3b (`wave2_B_f3_result_fields.py`) | fix: the field is the count and is documented so; `best_iter` stays an index | `4822ae1`, `bf027a7` (the stored output of example 3); check: `98a8a91` |
| W2-14 | P1a | `_check_warmup_end_conditions`; `private/vbmc_warmup.m:39`, `:87` | indexing/shape | When `tol_stable_warmup <= fun_evals_per_iter`, the check takes the maximum of an empty slice and raises. | shared defect (B-4; that MATLAB raises is inferred, not run); present form since `e38351a` (2022-05-23), exposure older (`35be58b`, `d407462`) | B-4 (`wave2_B_f4_warmup_empty_slice.py`) | fix | `54f0f19` |
| W2-15 | P1a | search-GP branch of `optimize`; `vbmc.m:471`, `:638-648` | state/caching | With `separate_search_gp=True`, the constant-mean search GP trains from and overwrites the main `hyp_dict` and `sn2_hpd`. The run fails in the search GP's own training call, iteration 1. | port discrepancy (B-5); `aa45473` (2021-07-26), never matched | B-5 (`wave2_B_f5_separate_search_gp.py`); `wave2_two_option_runs.py` | remove the branch (a development option of MATLAB VBMC); the option stays declared and inert; sheet entry | `c7a01f3` |
| W2-16 | P1a | warp-undo refit (`vbmc.py:1326-1328`); `vbmc.m:584` | control flow | The warp-undo refit sizes its sieve at `vp.K` where MATLAB uses `Knew`; the two differ only with `variable_means=False`. | port discrepancy (B-6); `2df0d7e` (2022-09-20) replaced the stale `k_warmup` with `vp.K`, although issue 98 named `Knew` | B-6 (`wave2_B_f6_sieve_warp_branch.py`) | fix; correct the sheet entry and `wave1_P6.md` | `d7c7887` |
| W2-17 | P1b | `_init_optim_state` (`vbmc.py:877-892`); `misc/setupvars_vbmc.m:14-24` | indexing/shape (and defaults) | `integer_vars` is read as a length-`D` mask, where the description and MATLAB use indices; a plain list silently marks every variable as integer. | port discrepancy (C-1); since `489598a` (2021-06-03) | C-1 (`wave2_C_f1_integer_vars.py`, `wave2_C_f2_uncertainty_handling.py`) | fix, type-strict: a boolean array is a mask; an integer array holds 0-based indices; a 0/1 integer array of length `D` is refused | `5b3b093` |
| W2-18 | P1b | `vbmc.py:1016-1021`; `misc/setupvars_vbmc.m:230-236` | control flow | `uncertainty_handling` is tested by length: `True`, `1`, `0`, `False` and `None` raise, while `[0]`, `'no'` and `'off'` select level 1. | port discrepancy (C-2); length test since `489598a`, its polarity corrected in `aa45473` (2021-07-26) | C-2 (`wave2_C_f2_uncertainty_handling.py`) | fix, type-strict boolean; other values raise; description rewritten; MATLAB's error when it is off with `specify_target_noise` on | `1a9f37a`; check: `37c32ec` |
| W2-19 (with C-M6) | P1b | `_bounds.py` `_normalize_bounds` (`:136-157`); `misc/boundscheck_vbmc.m:6-10` | defaults (int); indexing/shape (cmp) | Scalar bounds raise for `D > 1`, with an unformatted message, although the docstring and MATLAB replicate them. | port discrepancy (C-3, C-M6); promised by the docstring since `489598a`; message unformatted since `dd0e17e` (2022-11-23) | C-3 (`wave2_C_f3_scalar_bounds.py`) | fix | `477226b`; check: `57c5631` |
| W2-20 | P1b | `Options.validate_option_names`, `load(new_options=)`; MATLAB validates no name (`misc/setupoptions_vbmc.m:9-26`) | control flow | Option names are validated for `options=` but not for an `options_path=` file or `load(new_options=)`, where a misspelt budget is accepted and ignored. | Python-only inconsistency (C-4); `4b94e1e` (2021-07-17), `ad96d50` (2023-03-08) | C-4 (`wave2_C_f4_f5_options_path_and_load.py`) | fix: an unknown name raises on both routes; validated against the two shipped files | `972bbf4` |
| W2-21 | P1b | `load`, `VBMC.__str__` | state/caching | After `load` of a run with history, `vbmc`, `vp` and `function_logger` hold three distinct transformer objects, equal by value, and `vbmc.parameter_transformer` is not the chosen iteration's. `__str__` inverts `x0` with the current transformer, which is wrong after any warp. | Python-only defect (C-5); `load` has restored from the history since `ad96d50` (2023-03-08) | C-5 (`wave2_C_f4_f5_options_path_and_load.py`, `wave2_C_f5b_warp_x0.py`) | fix, the `x0` line of `__str__` included | `6bb129c` (adds `x0_orig`); check: `ce59fbe`, `4f9e3d6`, `f40da0a` |
| W2-22 | both (P1a int F9; P1b int F8; P1b cmp F6, F10) | `INERT_OPTIONS`, the guard test (`test_options.py:183-205`), `whitening.py`; `misc/setupvars_vbmc.m:249-255`, `vbmc.m:805`, `:961` | defaults (int, cmp F10); state/caching (cmp F6) | `temperature`, `diagnostics` and `entropy_force_switch` are read through no options object and are missing from `INERT_OPTIONS`, and the guard test cannot see them. The tempering sheet entry is wrong. | confirmed (C-6); both options are knobs of unported MATLAB features; the registry dates from `a63b17a`/`0d9a422` (2026-09-19) | C-6 (`wave2_C_f6_inert_options.py`) | register `temperature` and `diagnostics` as inert; the guard counts reads through an options object; correct two sheet entries | `82c624c` |
| W2-23 | P1b | `_read_config_file` | defaults | `_read_config_file` cuts each description at its first `=` or `:` (8 of 183 options); two descriptions are also wrong in the file itself. | Python-only defect (C-7); parser since `d72c7df` (2021-05-29) | C-7 (`wave2_C_f7_descriptions.py`) | fix the parser and the two texts | `3722210` (it also restores "on" in 17 texts where a substitution had written "True"); check: `3f85501` |
| W2-24 | P1b | `pyvbmc/rng.py: get_rng`, the `seed` docstring | random draws | `VBMC(seed=None)` draws four integers from NumPy's global state, against the `seed` docstring's "never written". | documentation defect (C-8); the behavior is a recorded decision (`stage1-rng-generator.md` §3; `d02c517`, 2026-09-02) | C-8 (`wave2_C_f8_f11_rng_and_minors.py`) | docstring only, and the phrase in the generator plan | `77575b1` |
| W2-25 | P1b | default `search_acq_fcn`; `vbmc.m:213` | defaults | The default search acquisition is `AcqFcnLog` where MATLAB has `acqf_vbmc`; the two rank identically in exact arithmetic, and the plain form underflows. | intentional difference, missing from the sheet (C-9); `e20d081` (2022-09-21, PR 102) | C-9 (`wave2_C_f9_acq_forms.py`) | keep; sheet entry | — |
| W2-26 | P1b | option validation; `misc/setupoptions_vbmc.m:109-124` | defaults | Non-positive or non-integer `max_fun_evals`/`max_iter` are not rejected, `max_iter` is not raised to `min_iter`, and a low `max_iter` is overridden silently at run time. | port discrepancy, validation only (C-10a); PyVBMC never had the checks | C-10a (`wave2_C_f10_budget_and_maxiter.py`) | fix, together with W2-5 | `17badf9`; check: `b7a2cd0`, `c1e1634` |
| W2-27 | P1b | `_budget_active` | control flow (defaults) | Passing `initialization_cost=0` or `precomputed_evaluations=None` explicitly turns on the budget path, which caps the last batch at the remaining allowance. | Python-only defect (C-10b); `6769a9a` (2026-09-16) | C-10b (`wave2_C_f10_budget_and_maxiter.py`) | fix: the budget path follows the values, not the presence of the arguments | `f9689dc` |

Minor observations, under the labels of `wave2.md`:

| id | slice | where | category | finding | class; dating | evidence | disposition | fix |
|---|---|---|---|---|---|---|---|---|
| B-M1 | P1a | `vbmc.py:1636` after the copy at `:1622`; `vbmc.m:803`, `:811` | — | The recorded posteriors lack the stability flag. | not a defect (faithful port); `9267816` (2021-08-25) | reading | faithful port, not a defect (ledger) | — |
| B-M2 | P1a | warp branch `vp.mu = gp.X.T` (`:1319`); `vbmc.m:576` | — | The warp branch binds `vp.mu` to a view of `gp.X` where the main loop copies; only with `variable_means=False`. | Python-only defect, latent; `359dd90` (2022-08-19) fixed only the twin | reading | fix | `fe5f9dc` |
| B-M3 | P1a | `final_boost` (`:2509`, `:2517`); `misc/finalboost_vbmc.m:40`, `:48` | — | `final_boost` leaves `warmup = False` and `entropy_alpha = 0` on the live `optim_state`, where MATLAB works on a copy. | port discrepancy; `3bfb485` (2021-07-06) | reading | fix | `261b9ae` |
| B-M4 | P1a | `final_boost` docstring (`:2426-2431`) | — | The docstring types `elbo` and `elbo_sd` as `VariationalPosterior`; they are floats. | Python-only defect (documentation) | reading | not among the observations ruled on at first; PI 2026-09-20: fix | `d5b2139` |
| B-M5 | P1a | `determine_best_vp` (`:2741`, `:2744`); `misc/best_vbmc.m:79` | — | With `do_final_boost=False`, `vbmc.vp` is the iteration-history entry itself. | port discrepancy, latent; `0c3b0ef` (2021-07-06) | reading | fix | `b3ad32b` |
| B-M9 | P1a | closing "finalize" line (`:1839`, `:1215`); `vbmc.m:884-913` | — | The closing line prints only when the boost changed the posterior, and its `sKL` compares with the posterior of the start of the last iteration. | port discrepancy, display only; `c477a7b` (2021-11-01) | reading | fix, with one change (PI 2026-09-20): the divergence is estimated on a copy of the generator, in every case after `9ff44c5` | `e9e7803`, `9ff44c5` |
| B-M11 | P1a (to P8) | `whitening.py:203-209`; `misc/warp_input_vbmc.m:112-119` | — | The warp re-transforms only the active rows of the function logger; MATLAB re-transforms every row. | confirmed difference; dated under P8-1 | reading | left to slice P8 (PI 2026-09-20); both P8 reviewers found it unprompted; it became W3-19 | `9ebaa48` (as W3-19) |
| B-M12 | P1a | stability branch (`:2169`); `private/vbmc_termination.m:80` | — | The branch lacks the `isfinite(options.EntropyForceSwitch)` test. | port discrepancy, unreachable because of B-1; `f1e6a0b` (2021-06-30) | reading | fixed with W2-10 | `e71f97b` |
| B-M13 | P1a | result strings, finalize-line branch; `vbmc.m:880` | — | Four string and format divergences, among them no counterpart of `stats.timer(iter).totalruntime`. | confirmed; `"bounded"` since `85ba77b` (2022-10-06) | reading, grep | no disposition in `wave2.md`; the independent check added the total-run-time line to the sheet | — |
| C-M1 | P1b | `VBMC.__str__` | — | `__str__` always prints `None` for the GP, the log-prior and the prior sampler. The fix agent found the same for the log-density line without a separate prior. | Python-only defect; `4f149c5` (2022-09-30) | `wave2_C_f8_f11_rng_and_minors.py` | fix; the log-density line: PI, fix | `9a068c4`, `1c1f2e4` |
| C-M2 | P1b | `_init_logger` | — | It removes handlers while iterating over them, and raises `AttributeError` on a non-file handler of the "VBMC" logger. | Python-only defect; `97d19a4` (2022-03-10) | `wave2_C_f11b_logger_display.py` | fix; the refusal of the levels outside the standard ones, which `658ecb8` kept: PI, 2026-09-23, after the close, fix | `658ecb8`; `f5ec2dc7` |
| C-M5 | P1b | `Options.__delitem__` / `pop` | — | `Options.pop` succeeds on a frozen options object. | Python-only defect; the freeze dates from `4b94e1e` (2021-07-17) | `wave2_C_f8_f11_rng_and_minors.py` | fix; made by the orchestrator, because four tests outside the agent's files use `force` | `5d2e543` |
| C-M6 | P1b | `_bounds.py:155-157` | — | The bounds error passes its template and its argument separately, so the message is never formatted. | Python-only defect; `dd0e17e` (2022-11-23) replaced the f-string of `d34fc15` | `wave2_C_f3_scalar_bounds.py` | fixed with W2-19 | `477226b` |
| C-M9 | P1b | `x0` handling; `misc/setupvars_vbmc.m:7-12` | — | A non-finite entry anywhere in `x0` replaces the whole starting set with the plausible centre. | shared, faithful port; `489598a` | `wave2_C_f8_f11_rng_and_minors.py` | faithful port, not a defect (ledger) | — |

The verifiers' files hold further minor observations that `wave2.md` does
not name: B-M6, B-M7, B-M8 and B-M10 (leftovers or faithful ports); C-M3,
C-M4, C-M7, C-M10 and C-M11 (leftovers, cost only, or shared wording); C-M8
and C-C7 (above, "Findings ruled after the close"); and C-C1 to C-C12 of
the comparison track, of which C-C2's error is ported in `1a9f37a` and its
warning after the close (above), C-C1's refusal of a noise size that is not
positive with the check of `noise_size` found then, C-C3, C-C4, C-C5
and C-C12 are entries 12, 13, 11 and 10 of the MATLAB-side list (C-C5 and
C-C12 on the sheet as well), C-C8 is the sheet's entry on `display` and
C-C11 an item of the MATLAB-side list of state written and never read.

Found during the fix pass:

| id | slice | where | category | finding | class; dating | evidence | disposition | fix |
|---|---|---|---|---|---|---|---|---|
| W2-28 | fix pass (agent B) | `Options.update_defaults` | — | A noisy run with `max_fun_evals = np.inf` raises `OverflowError` from `ceil(inf * 1.5)`, although the user's budget would have been kept. | Python-only defect; older than the pass | reproduced by the fix's test | fix | `1421759` |
| W2-29 | fix pass (agent A) | `final_boost`, `_gp_log_joint`; `misc/finalboost_vbmc.m:6`, `misc/vbinit_vbmc.m:132-136` | — | With `variable_means=False` the boost asks for `max(vp.K, min_final_components)` components, while the sieve keeps `vp.K` fixed means, and raises a broadcast error. | shared defect, by a reading of MATLAB; older than the pass | `wave2_variable_means_boost.py`; reading | fix: the components are placed at the GP's training inputs | `73d2a81`; check: `6aab88a` |
| W2-30 | fix pass (CI matrix) | `ParameterTransformer._set_bounded_transforms` | — | The bounded transforms are nested functions that dill pickles by value. Under another Python minor version, the first use of a saved bounded posterior ends the interpreter. | Python-only defect; older than the pass | `wave2_xver_*.py`, `wave2_pickled_functions*.py` (3.11.9 and 3.12.6, both directions) | fix (PI 2026-09-20): pickle and copy without the functions and rebuild them from `bounded_types`; the `save`/`load` docstrings say what a saved `VBMC` allows | `e610479`, `1c0d0d3`; check: `1f9d872`, `82dbfa5` |

**The independent check of the pass** (2026-09-21, `wave2.md`). Six fresh
read-only reviewers on `a093f2e`, then `0bf7963` (the same package code),
144 commits after the pass: the warm-up, termination and initial posterior;
the rest of the loop, the warp branch and the final boost; the options and
inputs; `load`, the state and the pickling; `wave2.md` and the developer
records; the user-facing records and the consequences outside the changed
lines. All 47 commits of the pass hold. The fixes are 18 commits on the
branch `dev-port-review-w2check`, cut at `0bf7963` and brought onto
`dev-port-review` as a fast-forward (`b1bab4d`), and into `dev-next` with wave
5 (`03a8d46`). On its branch the check was gated by the tests of every
module touched (413 passed), the oracle tests (143) and the exact oracle
check; the whole suite, the seeded runs, the extras and CI ran with the
wave-5 round (`f86d373`, `63a0808`).

| finding | statement | disposition | fix |
|---|---|---|---|
| run limits in `load` (W2-26) | `validate_run_limits` was called only in the constructor, and `9b5213d` left it out of `load`; `load(new_options=)` took `max_iter = 0`, a bad budget and `max_iter < min_iter`. Three reviewers found it independently. | fixed | `b7a2cd0` |
| `x0_orig` back-fill (W2-21) | The back-fill inverted with the restored iteration's map, which is wrong after a warp (`[6.78, 0.66]` for `[4.0, 0.3]`); it now uses the first recorded map. | fixed | `ce59fbe` |
| fixed-means boost (W2-29) | A GP with fewer training inputs than components ended in the sieve's broadcast error; `optimize` cannot reach the case. | fixed: refused | `6aab88a` |
| description of a user-set option (W2-23) | `print(options)` showed `None` for every advanced-file option the user set; older than the pass. | fixed | `3f85501` |
| `specify_target_noise` (W2-18) | `"no"`, `"off"` and `[0]` turned noise handling on; an empty array raised NumPy's error. | made under the PI's rule for input rows, not ruled on by itself | `37c32ec` |
| plot titles (W2-13) | The final plot and the last animation frame printed the index of the last iteration. | fixed | `98a8a91` |
| scalar plausible bounds without `x0` (W2-19) | They raised `AttributeError` or `IndexError`; now an error names the problem, and a list is accepted. | fixed | `57c5631` |
| test of the minimum-iteration guard (W2-5, W2-26) | Its first case set `max_iter < min_iter`, so its assertion held whatever the guard did. | fixed | `2d3f876` |
| `tol_elcbo_boost` back-fill | `load` sets `tol_elcbo_boost = None` for a file saved before the option existed. | fixed | `8afeebf` |
| oracle harness | `build_options` runs the run-limit check; no fixture changes. | fixed | `c1e1634` |
| `save`/`load` note (W2-30) | The note tied bytecode to unpicklable targets, but the nine function-valued options are stored by value in every saved instance. | corrected in the note, `AGENTS.md`, the fixtures sheet and the FAQ | `1f9d872`, `82dbfa5` |
| sheet citations | `refresh_citations.py` misplaced three MATLAB and ini citations (`9f6f0c7`). | script corrected, 112 rewritten citations audited; also the inert count, the `display` range, the `rng_state` citation, the NaN condition, and the B-M13 total-run-time line | `895ced8`, `224c7a1` (NaN condition); others: git `910d250` |
| `counterpart_map.md` | 17 citations in 11 changed rows were up to 120 lines off. | corrected | git `910d250` |
| `matlab_side_defects.md` header | It left out entries 4 and 14 among the inferred ones. | corrected | git `910d250` |
| `wave2.md`; `wave2_B_loop.md` | `wave2.md` misstated where B-M11 was settled and missed the commit without a test; the verifier's inference on W2-11 does not hold. | corrected; flag under the file's header | git `910d250` |
| `CHANGELOG.md` | Upgrading lines were missing for `problem_type`, the run limits and `integer_vars` indices; `x0_orig` was undocumented. | corrected; `x0_orig` is now in the class docstring | `4f9e3d6`; changelog: git `82dbfa5` |
| example 2 output | The stored `print(vbmc)` showed the W2-21 and W2-4 defects. | three lines edited, not rerun | `f40da0a` |
| porting log, script docstring, oracle `n_iterations` | Wrong function name; wrong `:66`/`:84` citations; the pre-W2-13 index. | corrected; a comment at the generator's line | git `910d250`, `b1bab4d` |

Left for the PI, in `wave2.md`, "Left for the PI": rebuilding the option
functions of a saved run in `load`; `vbmc.x0` staying in the space of
construction; `last_warmup = 0`, and the fallback of
`.get("last_successful_warping", 0)`; the main-loop sieve at `self.vp.K`
with fixed means; `validate_run_limits` not checking `min_iter`, the one-way
inert-option guard and the `Options` warnings on the root logger;
`_normalized_hard_bound`, the reach of the `_recompute_lcb_max` tests and
gaps of the changelog. The PI ruled on them after the close (above,
"Findings ruled after the close").

### Wave 3: the GP training policy (P5) and the transformer, warping and function logger (P8)

Read at `96cded7`; verified on 2026-09-20 (`verification/wave3.md`, with
`wave3_P5.md` and `wave3_P8.md`). Rows W3-1 to W3-7 can change what a run
computes. The fixes were made by three Opus agents on worktrees cut at
`fc8e561` (`fixes/wave3_agent_A.md`, `_B.md`, `_C.md`).

| id | slice | where | category | finding | class; dating | evidence | disposition | fix |
|---|---|---|---|---|---|---|---|---|
| W3-1 | P5 (int F3, cmp F1) | `vbmc.py:1091`, `train_gp` (`gpt.py:96-105`), gpyreg `noise_functions.py:36-39`; `misc/setupvars_vbmc.m:279`, `gplite/gplite_noisefun.m:66-70`, `:186-194`, `misc/gptrain_vbmc.m:165` | cross-module (with defaults) | At uncertainty level 1 the GP gets level 0's noise function, because gpyreg reads the scale flag only under `user_provided_add`. The per-point `s2` and the multiplier hyperparameter are dropped. | port discrepancy; never matched: `aa45473` (2021-07-26), written against gpyreg `24592ac` (2021-05-26) | `wave3_A1_level1_noise.py`; reading; histories traced | fix now, checked by the level-1 run; level 1 becomes a first question of the P3 and P4 briefs | `095c82c` |
| W3-2 | P5 (cmp F5) | `gpt.py:139-142`; `misc/gptrain_vbmc.m:39` | indexing | The window of past GPs that seeds the fit's starting points is one GP short of MATLAB's for an even history (170 of 361 collecting calls). | port discrepancy; never matched (`aa45473`) | `wave3_A_stored_runs.py`; reading | fix, all five moving rows in one pass | `14a01e2` (with the guard for a history holding no GP) |
| W3-3 | P5 (cmp F6) | `gpt.py:152-158`; `misc/gptrain_vbmc.m:44` | indexing/rounding | An oversized pool of past samples is subsampled to `ceil(init_N/2)` where MATLAB keeps `floor`. | port discrepancy; `ceil` since `9267816` (2021-08-25), which repaired `aa45473`'s float argument | `wave3_A_stored_runs.py` (7 of 361 calls); `git log -S` | fix, with W3-2 | `6e135f2` |
| W3-4 | P5 (cmp F7) | `vbmc.py:1610`; `misc/gptrain_vbmc.m:65`, `vbmc.m:804`, `:1041` | state/caching | `gp_hyp_full` records the thinned samples where MATLAB records the chain before thinning, so the sampler widths come from a fifth of the draws. | port discrepancy; never matched (`aa45473`; MATLAB's record from 2019, `d9822e3`) | `wave3_A_stored_runs.py` (732 blocks) | fix, with W3-2 | `1268d69` |
| W3-5 | P5 (cmp F3, mean part) | `_gp_hyp` (`gpt.py:389`, `:382`); `misc/gptrain_vbmc.m:174-188`, `gplite_train.m:120-127` | defaults (bound propagation) | The lower bound of `mean_const` is written as `-inf` where MATLAB leaves `min(y)`, so the design stays inside the plausible box and the fit has no lower bound. | port discrepancy; never matched (`aa45473`) | `wave3_A5_mean_const_bound.py`; `git log -L` | fix, with W3-2 | `720c416`, `894a353`; check: `b00fe40` |
| W3-6 | P8 (int F7) | `warp_input` (`whitening.py:154-182`); `misc/warp_input_vbmc.m:52-71`; paper appendix §B.2 | formula | Zeroing correlations of at most 0.05 can leave the covariance indefinite from `D = 3`; the SVD then misses unit variance along one direction (1.01, 0.96, 0.097). | shared by both implementations and by the paper's recipe; not dated; latent (none of the 990 population posteriors) | `wave3_A6_threshold_population.py`, `wave3_A_stored_runs.py`, `wave3_A5_mean_const_bound.py` | the guard: keep the covariance as it was when the thresholded one is not positive definite; sheet entry | `61a7325` |
| W3-7 | P5 (int F6, cmp F3, noise part) | `_gp_hyp:374-375`; `misc/gptrain_vbmc.m:180` | defaults | The noise bounds are written as `(log(tol_gp_noise), +inf)`, which loses the recommended upper bound `log(max(y) - min(y))`. | port discrepancy (P5-6a); never matched (`aa45473`); issue 99 and PR 116 fixed the same confusion elsewhere | P5-6a (`wave3_P5_1_and_6_gp_hyp_bounds.py`) | fix, with W3-2 | `992f7cb` |
| W3-8 | P5 (int F1, cmp F2) | `_gp_hyp:368-373`, `:399-402`; `misc/gptrain_vbmc.m:177-179` | control flow | The length-scale upper bound from `upper_gp_length_factor` is overwritten 26 lines later, so the option has no effect at any value. | port discrepancy (P5-1); regression: matched MATLAB from `aa45473` until `5f09709` (2022-11-11, PR 116) | P5-1 (same script) | as proposed: fix with W3-5, with a test that sets the option | `de7a99b`; check: `b00fe40` |
| W3-9 | P5 (int F4, F2; cmp F9, F8) | `vbmc.py:1657`, `train_gp:195`, `hyp_dict["full"]`; `vbmc.m:831`, `misc/gptrain_vbmc.m:65`, `:83` | state/caching; indexing/shape (int F2) | Three defects of `run_cov`: the reset writes an unread `runcov` key, the guard tests the hyperparameter count rather than the sample count, and a fit that does not sample keeps the last chain. | port discrepancies (P5-2 to P5-4); the key was mistyped in `4949c82` (2021-11-04), the other two date from `aa45473` | P5-2 to P5-4 (`wave3_P5_3_and_4_runcov_and_full.py`) | as proposed: fix the three together | `f954c63`, `bf081c2` |
| W3-10 | P5 (int F5, cmp F14) | `_gp_hyp:427`; `misc/gptrain_vbmc.m:236` | formula (API misuse) | `np.min(a, b)` stands for `np.minimum(a, b)` in the output-dependent noise bound and raises `TypeError`. | Python-only defect, unreachable (P5-5); `aa45473` | P5-5 (`wave3_P5_5_np_min_branch.py`) | as proposed: fix (two characters) | `dd41323` |
| W3-11 | P5 (cmp F4) | `_gp_hyp` (`:331`, `:393-405`); MATLAB leaves these bounds to `gplite_train` | defaults | The lower bounds of the length scales and the output scale come from the HPD subset, where MATLAB uses the full training set. | intentional difference, missing from the sheet; the justification holds (P5-7; issue 99 / PR 116, `5f09709`) | P5-7 (same script) | as proposed: keep; sheet entry | — (sheet, `9c29ce1`) |
| W3-12 | P5 (int F7) | `vbmc.py:1103-1123`; `misc/setupvars_vbmc.m:287` | control flow (validation placement) | Construction accepts 12 `gp_mean_fun` names but `train_gp` implements 3. The other nine fail at the first GP training, and the message misspells `negquad`. | port discrepancy of the validation (P5-8); the list of names since `d9bfe16` (2022-08-28) | P5-8 (`wave3_P5_8_gp_mean_fun.py`) | as proposed: stricter interface; construction rejects the nine; message corrected | `d9e3cc9`; check: `095c29e`; in `load` since W6-37 (`e657eba8`) |
| W3-13 | P5 (cmp F12, F13) | `_get_gp_training_options` (`:603-604`, `:657-673`); `misc/get_GPTrainOptions.m:71-74`, `:112` | control flow / cross-module; formula (F13) | `gp_hyp_sampler` accepts seven values but gpyreg runs only `slicesample`. The `covsample` fall-through and the `slicelite` burn-in also differ from MATLAB. | port discrepancies in code that cannot complete a fit (P5-11, P5-12); `aa45473` | P5-11 (reading); P5-12 (`wave3_P5_12_slicelite_burnin.py`) | reject at construction and remove the branches | `bee1866`, `894a353`; check: `9b5213d` |
| W3-14 | P5 (cmp F10) | `_get_gp_training_options:639`; `misc/get_GPTrainOptions.m:100` | defaults | The design size is floored at 9 where MATLAB floors it at 0, which also stops the collection of past samples. | port discrepancy (P5-9); in the module's first commit | P5-9 (`wave3_P5_9_and_10_training_options.py`) | fix: the floor is 0 | `3b8b26c` |
| W3-15 | P5 (cmp F11) | `:651` (against `:534`); `misc/get_GPTrainOptions.m:5`, `:109` | indexing (0- vs 1-based) | The retrain-threshold branch tests `iteration > 1` where the same function translates the same predicate as `iteration > 0`. | port discrepancy without effect (P5-10); `aa45473` | P5-10 (same script) | as proposed: fix, with a boundary test | `175629a` |
| W3-16 | P5 (cmp F17) | `_estimate_noise:849`, `stats/get_hpd.py:34`; `misc/gptrain_vbmc.m:355`, `misc/gethpd_vbmc.m:9` | indexing | `argsort(y)[::-1]` is neither MATLAB's stable descending order nor its reverse, so a tie at the cut changes the HPD subset. | port discrepancy, stronger than reported (P5-15); `aa45473`, `get_hpd` `2551469` (2021-08-03) | P5-15 (`wave3_P5_15_tie_order.py`) | fix; no stored oracle state holds equal values (checked 2026-09-20) | `0380c5d`; check: `2ff2dfe`, `06bc8d5`, `027e972` |
| W3-17 | P5 (cmp F15) | `train_gp` (`:143-150`, `:159`); `misc/gptrain_vbmc.m:37-50` | control flow | `train_gp` has no counterpart of MATLAB's `try`/`catch` fallback when a recorded GP has another shape. | port discrepancy, latent (P5-13); `aa45473` | P5-13 (`wave3_P5_13_no_try_catch.py`) | as proposed: leave | — |
| W3-18 | P5 (cmp F16) | `reupdate_gp` (`:958`); `misc/gpreupdate.m:8`, `misc/gptrain_vbmc.m:76`, `private/activesample_vbmc.m:484` | state/caching | The evaluation times are not carried onto the GP (`gp.t`). | not a defect (P5-14): nothing reads the field in MATLAB | P5-14 (reading, grep) | as proposed: one-line sheet entry | — (sheet, `9c29ce1`) |
| W3-19 | P8 (cmp F4, int F5 second half) | `warp_input` (`whitening.py:213-220`); `misc/warp_input_vbmc.m:112-119` | indexing/shape (cmp); state/caching (int) | The warp re-transforms only the rows with `X_flag`. Trimmed rows keep `X` and `y` of the previous space (`corr_D5_warped`: off by up to 8.9 and 11.6). | port discrepancy (P8-1); never matched (`4aa4ca9`, 2022-03-03) | P8-1 (`wave3_P8_warp.py`, `wave3_P8_stale_rows_in_stored_state.py`) | fix, with a warp test on a logger that holds evaluations and an inactive row | `9ebaa48`; check: `791c51d` |
| W3-20 | P8 (int F5 first half) | `_record` (`:657`); `misc/funlogger_vbmc.m:220-247` | state/caching | A repeat at a deactivated input is pooled into that row, which stays inactive: the evaluation is paid for and seen by nothing. | shared defect (P8-2); `2527c47` (2021-05-20) | P8-2 (`wave3_P8_logger.py`) | leave (listed as shared in `matlab_side_defects.md`) | — |
| W3-21 | P8 (int F4, cmp F7) | `_record` (`:719-722`), `add`; `misc/funlogger_vbmc.m:187`, `:151` | control flow (int); state/caching (cmp) | A repeat through `add` (time NaN) turns a measured time into NaN for good. `add` with an explicit time also charges `total_fun_eval_time`, which MATLAB never does. | port discrepancy (P8-3); `2527c47`, never matched | P8-3 (`wave3_P8_logger.py`) | as proposed: fix the guard of the average; leave the total | `490ea14`; check: `0dfc282` |
| W3-22 | P8 (int F13, cmp F8) | `_record` (`:715`, `:741`); `misc/funlogger_vbmc.m:243`, `:269` | indexing/shape | The logger returns a one-element array for a repeat and a scalar for a new point, where the docstrings and MATLAB return a scalar. | port discrepancy without effect today (P8-4); `2527c47` | P8-4 (`wave3_P8_logger.py`) | as proposed: fix | `fb02de0` |
| W3-23 | P8 (int F10 second half, cmp F5) | `parameter_transformer.py:135-142`; `shared/warpvars_vbmc.m` types 1 and 2, `misc/boundscheck_vbmc.m:138-143` | state/caching; defaults (int); control flow (cmp) | The public transformer treats a variable with one finite bound as unbounded, and its inverse lands outside the support; MATLAB has log transforms for it. | intentional difference, missing from the sheet (P8-5); the transforms were never ported (`6a247e5`, 2021-02-25, removed only the labels) | P8-5 (`wave3_P8_transformer_ctor.py`) | as proposed: the constructor raises; sheet entry | `7b8cf6a` |
| W3-24 | P8 (int F10 first half) | `parameter_transformer.py:79-80`, `:106-107`, `:42-44` / `:56` | state/caching; defaults | The transformer keeps the caller's arrays by reference, and its docstring names the keyword `bounded_transform_type` where the signature has `transform_type`. | Python-only defects (P8-6), the first latent; the docstring keyword since `4ce90e2` (2022-07-28) | P8-6 (same script) | as proposed: fix both | `7c2f9b6`, `a385d74` |
| W3-25 | P8 (int F1) | `parameter_transformer.py:153-160` | control flow (order of operations) | With `scale` or `rotation_matrix` and tight plausible bounds, the centering is derived after the rotation and applied before it, so the plausible box leaves `[-0.5, 0.5]`. | Python-only defect, latent (P8-7); `4aa4ca9`/`385f9e9` (2022-03) | P8-7 (same script) | fix: the centering is derived before the rotation | `b74c65c` |
| W3-26 | P8 (int F9) | `parameter_transformer.py:79`, `:293-294`; `shared/warpvars_vbmc.m:763-765` | formula / defaults | `scale` is stored unchecked while `log_abs_det_jacobian` adds `log(scale)`: NaN for a negative entry, a singular map for a zero. | shared formula, exposure Python-only (P8-8); `4aa4ca9` | P8-8 (same script) | as proposed: stricter; a finite positive `scale` of length `D` is required | `dea6b29` |
| W3-27 | P8 (int F2, F3) | `unscent_warp` (`whitening.py`); `utils/unscent_warp.m:18-35` | indexing/shape (dtype) | `unscent_warp` truncates for an integer `x`, and raises for a single-row `x` with a many-row `sigma`. | Python-only defect and port discrepancy, both latent (P8-9a, b); `4aa4ca9` | P8-9 (`wave3_P8_unscent_decorator.py`) | as proposed: fix both | `1a204aa` |
| W3-28 | P8 (int F6) | `FunctionLogger.add` (`:528-530`); `misc/funlogger_vbmc.m:159-162` | defaults / cross-module | `add` without an SD records SD 1 at every level, so `f_vals` with `specify_target_noise` enters the GP with noise 1; MATLAB raises before reaching its default. | port discrepancy (P8-10); from the original port | P8-10 (`wave3_P8_logger.py`) | refuse the combination at construction; `add` without an SD raises at level 2 | `fbe25b6`, `e7e27cf`, `74414ce`; check: `a775bb6`, `68f7984` |
| W3-29 | P8 (int F8) | `whitening.py:254-257`; `misc/warp_input_vbmc.m:164-167` | state/caching | The block commented "Reset GP Hyperparameters" clears the running average of the variational moments; the pre-warp hyperparameter statistics stay in use on both sides. | not a defect (P8-11), a verbatim port (`4aa4ca9`) | P8-11 (reading) | as proposed: leave; listed as shared | — |
| W3-30 | P8 (int F11) | `whitening.py:217-218`, `_record` (`:716-718`, `:742-746`); `warp_input_vbmc.m:117`, `funlogger_vbmc.m:243`, `:269` | formula | `warp_input` divides the log-Jacobian by the temperature and `_record` does not; MATLAB divides in both places. | port discrepancy, unreachable (P8-12); `_record` never had the divisor; MATLAB's from `774327b` (2020-11-01) | P8-12 (`wave3_P8_warp.py`) | as proposed: leave to whatever ports tempering; a line in the tempering entry | — (sheet line, `9c29ce1`) |
| W3-31 | P8 (int F12) | `decorators/handle_0D_1D_input.py:46` | indexing/shape | A 0-D input returns a 2-D result; `input_dims` is unbound if a patched argument has a default; one variable serves several patched arguments. | Python-only (P8-13); the first is pinned by eight `test_0D_*` tests; from the original port | P8-13 (`wave3_P8_unscent_decorator.py`) | leave; the docstring states what a 0-D input returns | `7f771b0`, `83ea3d8` |
| W3-32 | P8 (int F14) | `_validate_vectorized_target_output:144-152`, `batch_call:441` | control flow | A logger built with `noise_flag=True` at level 0 records SD 1 through `__call__` and NaN through `batch_call`. | Python-only defect, unreachable from `VBMC` (P8-14); not dated | P8-14 (`wave3_P8_logger.py`) | as proposed: `FunctionLogger` rejects a flag and a level that disagree | `5aa4e89` |
| W3-33 | P8 (cmp F1 to F3) | `_to_unit_interval` (`:539-548`), `_from_unit_interval` (`:551-556`), `_inverse_student4` (`:604-608`); `warpvars_vbmc.m:106-107`, `:256-257`, `:265-266`, `:457-459`, `:451` | formula/gradient | Three differences near a bound: a nudge where MATLAB returns ±Inf, `nextafter` against `eps(bound)` (one ulp at a power of two), and one grouping in the student4 inverse (one ulp). | nudge intentional (PR 89, `3e9d1c2`, 2022-08-28); clamp matched MATLAB until `3e9d1c2`; grouping never matched (`4ce90e2`) (P8-15a to c) | P8-15 (`wave3_P8_transforms_vs_matlab.py`, `wave3_P8_nudge_reachability.py`) | as proposed: keep all three; sheet entries | — (sheet, `9c29ce1`) |
| W3-34 | P8 (cmp F6) | `whitening.py:154-159`; `misc/warp_input_vbmc.m:52-56` | control flow | The mask zeroes `abs(corr) <= thresh` where MATLAB keeps `> thresh`; the two differ on NaN alone. | port discrepancy, unreachable (P8-16); `4aa4ca9` | P8-16 (`wave3_P8_corr_mask.py`) | as proposed: fix (one comparison) | `490daae` (helper `_drop_low_correlations`) |
| W3-35 | P8 (cmp F9) | `function_logger.py:663-707`; `misc/funlogger_vbmc.m:234-238` | formula/gradient | The pooling of a repeat falls back to a scale-invariant form when the precisions overflow; the ordinary path is MATLAB's, bit for bit. | intentional difference, missing from the sheet (P8-17); `6769a9a` (2026-09-16) | P8-17 (`wave3_P8_logger.py`) | as proposed: keep; sheet entry | — (sheet, `9c29ce1`) |
| W3-36 | P8 (cmp F10) | `whitening.py:198`; `misc/warp_input_vbmc.m:85-86` | formula/gradient | The plausible bounds after a warp use `np.quantile`, whose convention differs from MATLAB's. | intentional in kind, not recorded for this site (P8-18) | P8-18 (`wave3_P8_warp.py`) | as proposed: keep; the sheet entry names this site too | — (sheet, `9c29ce1`) |

Found during the fix pass (an unnumbered list in `wave3.md`):

| id | found by | finding | disposition / fix |
|---|---|---|---|
| W3-FP1 | agent C (fixing W3-28) | `test_vectorized_initial_design_matches_scalar` gave both arms an `f_vals` value, which construction now refuses for the noisy arm. | the cached value stays in the noiseless arm: `74414ce` |
| W3-FP2 | agent A | The second parameter of the output-dependent noise had infinite bounds: the defect of W3-5 and W3-7 in the unreachable branch. | `894a353` |
| W3-FP3 | agent A | The W3-2 window read the last entry of an empty array when the history holds no GP; the `gp_fit` oracle's stand-in history produces that case. | squashed into `14a01e2` |
| W3-FP4 | agent C | The docstring of `handle_0D_1D_input` named `kwarg` and `argpos`; the `f_vals` description did not say what it cannot carry. | `83ea3d8` |
| W3-FP5 | — | `pyvbmc/testing/whitening/` has no `__init__.py`; four tests draw from the unseeded global generator. | not acted on |
| W3-FP6 | branch smoke | `test_svbmc_filters.py::test_infinite_bounds_have_to_agree_too` built a half-bounded run, which W3-23 refuses; it needs Torch, so the local suite had skipped it. | the test now uses a run bounded on both sides (commit not cited; git `fbcc58e`); the PI kept the refusal; the Torch and PyMC tests joined the gates of a pass (739 and 107 passed) |
| W3-FP7 | second smoke | `test_vbmc_warp_branch.py::test_warp_refit_sieve_is_sized_at_the_new_component_count` failed with `assert 20 != 20`: on Ubuntu the run no longer pruned a component. | the fixture's options now force the pruning and the warp: `92eb2cc` |
| W3-FP8 | agent C | `scripts/wave3_P8_logger.py` stops at its first `add` without an SD on the fixed package. | noted; the script shows the code as it was verified |
| W3-FP9 | — | The fix agents' commits carried a `Claude-Session:` trailer. | removed from the unpushed commits; `AGENTS.md` says so |

**The independent check of the pass** (2026-09-21, `wave3.md`). Five fresh
read-only reviewers: the inputs of the hyperparameter fit; its policy, the
recorded chain and the option checks; slice P8; `wave3.md`, the user-facing
records and the re-baselined fixtures; the consequences outside the changed
lines. No defect in what a run computes; the gates ran on `095c29e`.

| finding | statement | disposition | fix |
|---|---|---|---|
| warm-up trim (W3-16) | The trim at the end of warm-up now keeps the earlier of equal values (`private/vbmc_warmup.m:123`). | PI ruling 2026-09-21: the stable order goes into all three sites | `2ff2dfe` |
| `determine_best_vp` ties (W3-16) | The earlier of two iterations with equal scores ranks first (`misc/best_vbmc.m:36`, `:40`). | PI ruling (both rankings) | `06bc8d5` |
| `get_hpd` on integers (W3-16) | The negation W3-16 introduced wraps around for unsigned zero and for the smallest signed value. | applied | `027e972` |
| option checks in `load` (W3-13) | `noise_shaping`, `gp_hyp_sampler` and `search_acq_fcn` were checked only at construction. | PI ruling: the checks run in `load` | `9b5213d` |
| `batch_call` at level 2 (W3-28) | It refused a cached value only after recording the rows before it; it now refuses before calling the target. | applied | `a775bb6` |
| evaluation time (W3-21) | A known time now replaces an unknown stored average, the other direction of W3-21. | applied; sheet entry | `0dfc282` |
| `f_vals` of NaN alone (W3-28) | It supplies no value and now passes the check. | applied | `68f7984` |
| `_gp_hyp` bound statements (W3-5, W3-8; the output scale) | Each statement now carries the other half of its pair over; the mean-constant test pins MATLAB's recommended values. | applied | `b00fe40` |
| W3-19 test | Now also runs on a bounded problem; `build_short` is defined once. | applied | `791c51d` |
| descriptions | The `gp_mean_fun` description names the three values; a comment explains the noise switches of an oracle state. | applied | `095c29e` |
| flaky ADAM test (from the gate rerun) | `test_minimize_adam_matyas_with_noise` failed 48 of 400 seeded calls. | PI: fix it before the push | `b731fac` |
| records | Corrected: four `CHANGELOG.md` items; entry 18 of `matlab_side_defects.md`, with a flag in `wave3_P5.md`; `wave3.md`'s counts, its sheet list, the W3-34 helper and the stopped script; the capture mode in `dev/README.md` and the fixture plan; `AGENTS.md`; 83 of 161 sheet citations; the `noise_shaping` entry. | corrected | `8a33203`, `99e787c` (git) |
| reviewer statement on W3-20 | A new route through the warp was claimed. | did not hold | — |
| stored oracle state at level 1 | No fixture holds one. | PI ruling: an item of `TODO.md` after the review (`dev/TODO.md`, "An oracle state at uncertainty level 1") | — |

Left as they are (`wave3.md`, "Left as they are"): the Cholesky test of W3-6
has no tolerance; `hyp_dict["logp"]` after a fit without samples; the order
of `min` and rounding in `_gp_hyp`; old files keep a `runcov` key, and a
level-1 run saved before the pass continues with its recorded blocks left
out of the sampler widths; the short test run, the whitening tests
without a generator of their own, the thinned stand-in histories.

### Wave 4: the acquisition functions (P3) and the noisy importance sampling (P4)

Read at `f873556`; verified on 2026-09-20 (`verification/wave4.md`, with
`wave4_P3.md` and `wave4_P4.md`). The first question of both slices, the
noise of a candidate point at uncertainty level 1, is answered on both
sides: it is MATLAB's and consistent with the GP's noise model
(`wave4.md`, "The first question"). One Opus agent made ten of the fixes
(`fixes/wave4_agent.md`).

| id | slice | where | category | finding | class; dating | evidence | disposition | fix |
|---|---|---|---|---|---|---|---|---|
| W4-1 | P4 | `vbmc/active_importance_sampling.py:240-247`; `private/activeimportancesampling_vbmc.m:206-208` | indexing/shape (P4c adds "with consequences for random draws") | The resampling weights of the MCMC step (IMIQR only) are a column, and their max is taken over the length-1 axis, so each chain's start is drawn uniformly. The guard raises `ValueError` if one sample has weight zero. | port discrepancy; never matched (Python: first commit `e38351ab`, 2022-05-23, #79; MATLAB `2a076c9`, 2020-06-16); no reason recorded | wave4.md part 1; `wave4_A1_isr_weights.py`, `wave4_A1b_isr_start_cost.py` | fix: max over the samples; guard only when no sample has weight. Moves the `acq_AcqFcnIMIQR` oracle alone. The cost is within Monte Carlo error. | `53ad16a`, `2dc98ce` (re-baseline) |
| W4-2 | P3 | `acquisition_functions/abstract_acq_fcn.py:279-285` (`_real2int`), called at `vbmc/active_sample.py:624`; `private/activesample_vbmc.m:325`, `misc/real2int_vbmc.m:7` | indexing/shape | `_real2int` indexes `X[:, integer_vars]` but receives the 1-D result of the local search. Every run with `integer_vars` and a search optimizer raises `IndexError` after the initial design. | port discrepancy; never worked (call `4949c826`, 2021-11-04, #30; function `9d1b6590`, 2021-08-11, #25; both in v1.0.4) | wave4.md; `wave4_A2_integer_vars_run.py` | fix, with a test through `active_sample` and a search optimizer | `dbfe895`; check: `315919a` |
| W4-3 | P3 | `acquisition_functions/utilities.py:17-22` (`string_to_acq`); MATLAB uses `str2func` | control flow (P3i F2); — (P3c minor obs 5) | The argument string is split on "," and "=", with trailing whitespace stripped only. A spaced keyword raises `TypeError`, a value holding "," raises `SyntaxError`, and a value holding "=" is silently dropped. | Python-only defect; `70325a34` (2022-06-02, #80); no reason recorded | `wave4_A3_string_to_acq.py`, `wave4_P3_9_minor_observations.py` | fix: parsed as Python parses a call; what cannot be parsed raises | `e5e6015`; check: `601555e` |
| W4-4 | P3 | `abstract_acq_fcn.py:281`; `misc/real2int_vbmc.m:7` | formula (rounding convention) | `_real2int` rounds with `np.around` (a half to even); MATLAB's `round` sends a half away from zero. A tie is reachable, against the report: the midpoint of a box with an even number of levels. | port discrepancy (P3-2); never matched (`9d1b6590`) | P3-2 (`wave4_P3_2_real2int_rounding.py`) | fix with W4-2; `test_real2int` asserts MATLAB's values | `3c088ab`, `70ce067` |
| W4-5 | P4 | `active_importance_sampling.py:470-474` (`fess`); `misc/fess_vbmc.m:4-12` | cross-module (P4c F3); control flow (P4i F4) | `fess(vp, gp, X)` with a number for `X`, its documented default, raises: the pair returned by `vp.sample` is used as the array. MATLAB's one caller of that form is not ported. | port discrepancy (P4-2), no caller; the scalar form never worked in any revision (`e38351ab`) | P4-2 (`wave4_P4_2_fess_scalar.py`) | fix | `dd3d1d3` |
| W4-6 | P4 | `active_importance_sampling.py:86-126`; `activeimportancesampling_vbmc.m:57-92`, `:81`, `:94-103` | control flow / cross-module (P4c F4); control flow (P4i F3) | The `mcmc_importance_sampling` branch hands `SliceSampler` a walker matrix, which it refuses. It runs one chain with `burn_in = 0` and keeps consecutive rows. Its weights are `ln_y` alone, as in MATLAB. | port discrepancy in parts (i) and (ii), shared by design in part (iii) (P4-3). The branch never ran in any revision: gpyreg's check `2bf7f10` (2021-06-08) predates the branch. The sheet's premise was false. | P4-3 (`wave4_P4_3_mcmc_hook.py`) | stricter interface: flag refused at construction and in `active_importance_sampling`; body removed; `active_importance_sampling_fess_thresh` registered inert; sheet entry rewritten | `1c23fda`; check: `44c1973` |
| W4-7 | P4 | `active_importance_sampling.py:396-398` (`active_sample_proposal_pdf`); `activeimportancesampling_vbmc.m:148`, `:335-337` | control flow | Raises `ValueError("Invalid value.")` where every component of the proposal has density zero. MATLAB's arithmetic gives NaN, which `:148` turns into a weight of `-Inf`. | port discrepancy, latent (P4-4); never matched (an `assert` at `e38351ab`, a `ValueError` since `d34fc154`, 2022-08-30) | P4-4 (`wave4_P4_4_proposal_pdf_zero.py`); gate `wave4_A7_proposal_pdf_bitwise.py` | fix: the point gets weight zero | `799852a` |
| W4-8 | P3 | `acq_fcn_viqr.py:136`, `acq_fcn_imiqr.py:27`; `acq/acqviqr_vbmc.m:4`, `acq/acqimiqr_vbmc.m:4` | defaults | `u = norm.ppf(quantile)` against MATLAB's literal `0.6745`. It is the whole difference in VIQR (1.7e-5) and IMIQR (8e-5); the best candidate is the same. | intentional difference, missing from the sheet (P3-1). The port had the literal until `70325a34`; the reason is in the body of PR 80. | P3-1 (`wave4_P3_1_u_constant.py`) | keep; sheet entry | — |
| W4-9 | P4 (also P3c F4) | `active_importance_sampling.py:329`, `:497-499` (`renormalize_weights`); MATLAB stores `lnw` raw | cross-module (state/caching) (P3c F4); formula (P4c F2, P4i F2) | One log-sum-exp over the whole array of log weights is subtracted. The readers are IMIQR and VIQR's `iqr_reduction`; everything downstream is unchanged by a constant. | port discrepancy without effect on any decision (P4-1a); `70325a34`, the file's second commit; no reason recorded. The docstring gave the constant as `log N_s`. | P4-1a (`wave4_P4_1_renormalize_shift.py`) | keep; sheet entry; docstring constant corrected. The independent check found this last part had not been carried out. | check: `76ed8e9` |
| W4-10 | P4 | `acq_fcn_imiqr.py:170-175`; `activeimportancesampling_vbmc.m:230` | formula | Each row of MCMC weights carries a factor `1/Z_s`. The report took the rows to weigh unevenly and blamed the joint normalization. | not a defect, and shared (P4-1b). Each row estimates a relative reduction: totals spread 17 nats, contributions sit within 0.05 nats. The verifier's "chain noise" explanation does not hold. | wave4.md (orchestrator); `wave4_A1c_isr_row_totals.py`; P4-1b | none; a line under the settled non-differences | — |
| W4-11 | P4 | `vbmc/active_sample.py:803-811`, gpyreg `gaussian_process.py:815-824`; `gplite/gplite_post.m:76-79`, `activesample_vbmc.m:481-487` | state/caching | The sheet said a noisy observation's noise enters the rank-one update on both sides. MATLAB sets `update1 = false` whenever `s2` is given and recomputes; PyVBMC extends by rank one. | intentional difference whose description of MATLAB does not hold (P4-5). Never matched on a noisy target (MATLAB `d9822e3`, 2019-09-17; PyVBMC `510a493`, 2026-09-19). Results equal to about 1e-15; after a Cholesky retry, 2.4e-9. | P4-5; `wave4_P4_5_rank_one_update.py`, `wave4_P4_5b_sn2_mult.py` | keep; sheet entry corrected | — |
| W4-12 | P4 (also P3c minor obs 4) | `active_sample.py:379-381`, `:414-417`; `activesample_vbmc.m:208-218` | random draws | PyVBMC draws the search candidates before the importance samples, MATLAB after; neither routine changes what the other reads. | port discrepancy of the random stream alone (P4-6; the P3 verifier calls it not a defect, P3-9d); since `e38351ab`, not `c4035ff8` as the report has it | P4-6 (`wave4_P4_6_call_order.py`) | keep; a line in the sheet's entry on randomness | — |
| W4-13 | P4, P3 | `acq_fcn_imiqr.py:264` (`is_log_full`), `acq_fcn_viqr.py:526`; `activeimportancesampling_vbmc.m:345-356` (`log_isbasefun`) | cross-module | The target of IMIQR's MCMC step predicts with `add_noise=True` and no `s2`, so at levels 1–2 only the constant term is added. Every other density uses the latent variance. | shared by design (P4-7). The MATLAB fixture matches the with-noise value to 1.4e-4. Always agreed (`is_log_full` from `c4035ff8`, 2022-07-27). | P4-7; `wave4_P4_7_noise_levels.py`, `wave4_P4_12_test_claims.py` | none; a settled line | — |
| W4-14 | P3 | `active_sample.py:351`; `activesample_vbmc.m:174` | formula | `sn2_new` is the mean over GP hyperparameter samples and enters each sample's formula. At level 1 the per-sample values spread from 1.36 to 3.70 around 1.91. | shared by design (P3-3; appendix §C.1); always agreed | P3-3 (`wave4_P3_3_sn2_new_sample_mean.py`) | none; a settled line. A noise per sample would change the algorithm. | — |
| W4-15 | P3, P4 | `active_sample.py:345-351`; `activesample_vbmc.m:172`, `gplite_pred.m:121`, `gplite_post.m:208` | cross-module | `sn2_new` leaves out the `sn2_mult` of a retried Cholesky factorization, which the prediction with noise and the GP update apply. | shared by design (P3-4); unreachable at the shipped noise floor | P3-4 (`wave4_P3_4_sn2_mult.py`) | none (in the settled line of W4-14) | — |
| W4-16 | P4 | `active_sample.py:336-341`; `activesample_vbmc.m:165` | formula | At a repeated input with unequal SDs (level 2), `S**2 * n_evals` is the harmonic mean of the variances. | shared defect, latent (P4-8); never differed | P4-8 (`wave4_P4_7_noise_levels.py`) | leave; in the shared list of `matlab_side_defects.md` | — |
| W4-17 | P4 | gpyreg `noise_functions.py:139-140`; `gplite/gplite_noisefun.m:114-115` | — (adjacent observation) | The noise multiplier is bounded to `[1e-3, 1e3]` on the variance. The report's upper SD limit does not follow; the lower limit (SD about 0.03 at level 1) holds on both sides. | shared by design (P4-9) | P4-9 (`wave4_P4_9_level1_bounds.py`) | none; the proposed sentence in the `uncertainty_handling` description is not written | — |
| W4-18 | P3 | `acq_fcn_viqr.py:127`, `acq_fcn_imiqr.py:21`; `acqviqr_vbmc.m:7-12`, `acqimiqr_vbmc.m:7-11` | state/caching | VIQR and IMIQR assign `acq_info` without calling the base constructor, so `compute_var_log_joint` is absent. | Python-only defect, latent (P3-5): every reader uses `.get`; since `dd0e17e7` (2022-11-23) | P3-5 (`wave4_P3_5_6_acq_info_and_quantile.py`) | fix | `4554135` |
| W4-19 | P3 | `acq_fcn_viqr.py:122-136`, `acq_fcn_imiqr.py` constructor | defaults | `quantile` is unchecked. VIQR with 0.5 or less, or 1, is all NaN or all `-realmax` and silently takes the first candidate; IMIQR raises an unnamed error. | Python-only defect (P3-6); `70325a34` | P3-6 (same script) | stricter interface: refuse a quantile outside (0.5, 1) | `0477a2f`; check: `c14a6b9` |
| W4-20 | P3 | `acq_fcn_viqr.py:418-431` (`_log_iqr_reduction`) | formula | A clamped `s_pred` stands beside an unclamped `tau2/(s_a + s_pred)`. | not a defect (P3-7): `tau2 <= f_s2_a*s2/(s2+sn2)`; the largest ratio on 409 600 entries is 0.87 | P3-7 (`wave4_P3_7_iqr_reduction_clamp.py`) | none | — |
| W4-21 | P3, P4 | sheet; `acq_fcn_viqr.py:120` | cross-module (sheet contradiction) (P3c F1); defaults (P3i F6, P4i F6); cross-module (documentation) (P4c F8) | The sheet said `AcqFcnVIQR` offers `loss="iqr"` alone. `iqr_reduction` is shipped, documented, tested and in the changelog. | the sheet's entry is wrong (P3-8); the removal is `fd9c7e8e` (2026-09-14) | P3-8; reading | entry corrected | — |
| W4-22 | P3 | `abstract_acq_fcn.py:178`; `acqwrapper_vbmc.m:47`, `:12-14`; `acqviqr_vbmc.m:107-108` | — (minor observations) | (a) `np.maximum(acq, -realmax)` lets a NaN through; (b) the log sums guard an all-`-inf` row; (c) the `vp.delta` quadrature branch is not ported. | (a) port discrepancy, latent, never matched (`9d1b6590`); (b) not a defect; (c) already on the sheet (P3-9a–c) | P3-9a–c (`wave4_P3_9_minor_observations.py`) | leave; a sheet line for (a) | — |
| W4-FP1 | P3 | `_real2int` rounding | — | `floor(abs(x) + 0.5)` rounds its own sum: the largest double below a half goes to 1, and an odd 53-bit integer goes to the even integer above it. | found by the orchestrator during the fix pass | wave4.md, "Found during the fix pass" | replaced by the truncated part plus a comparison of the exact fractional part with a half; those values are in the test | `70ce067` |
| W4-FP2 | P3 | test of `_real2int` | — | The probit transform maps the midpoint 4.5 of `[-0.5, 9.5]` to 7.5e-17 on that machine, not to the exact 0 that `wave4_P3.md` reports, and back to 4.5; the tie is still reached. | agent's report | wave4.md | the test transforms the midpoint instead of writing the literal | in the agent's W4-4 commit (`3c088ab` on the branch) |
| W4-FP3 | P4 | `vbmc.py: _validate_search_acq_fcn_option` | — | The check runs at construction only. A flagged acquisition that arrives through `load(new_options=)` is refused in `active_importance_sampling` instead. | — | wave4.md | not acted on in the pass; `load` runs the check since the check of the wave-3 pass | `9b5213d` |

**The independent check of the pass** (2026-09-21, `wave4.md`). Four fresh
read-only reviewers, one of whom ran tests: the commits of the
acquisitions, those of the importance sampling, `wave4.md` against its sources, and the user-facing and
cross-document records. No defect in the twelve commits.

| finding | statement | disposition | fix |
|---|---|---|---|
| C4-1 | `wave4.md`'s evidence for W4-7 was the VIQR oracle, which never reaches the changed function. | corrected: a bitwise comparison of the first importance-sampling step, before and after, over five seeds (`wave4_A7_proposal_pdf_bitwise.py`) | records |
| C4-2 | `wave4.md` said the weighted start ranks the best candidate no worse; the log has it 0.0043 worse, within noise. | corrected | records |
| C4-3 | The dating of `renormalize_weights`: it is from the file's second commit. | corrected | records |
| C4-4 | A disposition of W4-9 (the docstring's constant) had not been carried out. | carried out | `76ed8e9` |
| C4-5 | A changelog sentence on `mcmc_importance_sampling` held for some acquisitions only. | corrected | changelog, `e63a7f33` (git) |
| C4-6 | One reviewer showed `70ce067` is needed in a run: a coordinate saturated at the lower bound -0.5 comes back as the largest number below a half, negated, and `floor(abs(x)+0.5)` rounds it to -1. That point is outside the box, and its forward transform is NaN. | confirms the fix | `70ce067` |
| C4-7 | Code follow-ups. | the quantile taken as a real scalar of any type (or a one-element array) by one shared check | `c14a6b9` |
| C4-8 | | `search_acq_fcn` must be a list of acquisition objects or strings; its description says which strings are read | `44c1973` |
| C4-9 | | the state of the integer-variable search test seeded and built on the module's state helper | `315919a` |
| C4-10 | | three sentences of documentation: the parser's docstring (which name it looks up), the two unused fESS thresholds, and the comment over the draw of the chain's start | `601555e` |
| C4-11 | The sheet's stale line citations (worklog). | corrected | records |

Left, on the reviewers' weighing: a restored acquisition object keeps its
`acq_info` and quantile; only the `cmaes` branch is exercised with an
integer variable; `fess` and `AcqFcnVIQR.is_log_full` have no caller in the
package. The half-to-even rounding of `active_sample.py` and `get_hpd`
became W5-6.

### Wave 5: the variational posterior, entropies and statistics (P7), the priors (P9), and the internal track of P2

The P7 and P9 reviewers read `f873556`, the P2 reviewer `831acef`; verified
on 2026-09-21 (`verification/wave5.md`, with `wave5_P2.md`, `wave5_P7.md`
and `wave5_P9.md`). Rows W5-1 to W5-8 can change what a run computes; P7
comparison F9 (`fess` and the pair that `vp.sample` returns) is answered in
the first questions, fixed by W4-5. Three Opus agents made thirty commits,
twenty-six of them fixes (`fixes/wave5_agent_A.md`, `_B.md`, `_C.md`).

| id | slice | where | category | finding | class; dating | evidence | disposition | fix |
|---|---|---|---|---|---|---|---|---|
| W5-1 | P7 | `variational_posterior/variational_posterior.py:641-644` (`sample`, balanced); `vbmc_rnd.m:67-71` | formula (P7i F6); formula, "and a Python `sum`/`np.sum` trap" (P7c F3) | The builtin `sum` of a `(1, K)` array returns its row, so the remainder of a balanced draw is drawn ∝ `e_k + w_k(R − e_k)` instead of ∝ `e_k`. Expected counts are off by up to about one draw per component. | port discrepancy; never matched (`42a0deef`, 2021-03-14; MATLAB `a00a609`, 2018-08-29); no reason recorded | `wave5_A1_balanced_remainder.py`, `_A1b_gate_runs_corrected_sampler.py`, `_A1c_oracles_corrected_sampler.py` | fix (`np.sum`), with a test on weights for which the two laws differ; a moving fix, with no benchmark sweep | `23d962a` |
| W5-2 | P7 | `stats/kl_div_mvn.py:34-39`; `shared/mvnkl.m` | formula (P7i F8); — (P7c last minor obs) | Raw determinants leave the range of a double at moderate `D`: zero below an SD of about 8e-9 at `D = 20`, giving `sKL = inf`; infinite above about 5e7, giving NaN and then `sKL = 0`. | shared defect. Determinants from `fdb1e56d` (2021-04-22), moved by `1b31ffc0` (2021-06-09); the guard is Python's own, `fdb6e94f` (2022-12-19, PR 125). | `wave5_A2_kl_div_mvn_determinants.py` | fix: log determinants from `slogdet` | `a37945a` |
| W5-3 | P7 | `entropy/entmc_vbmc.py:217-218`, `:234` (and `entlb_vbmc`); `ent/entmc_vbmc.m:63-67` | formula | With a weight of exactly zero and separated components, both entropies compute `0 * (-inf)`, so `H` is NaN. This needs an `eta` gap of 745 or more, out of a run's reach. | shared defect, latent; not dated | `wave5_A3_entmc_zero_weight.py` | leave; a line in `matlab_side_defects.md` among the shared ones | — |
| W5-4 | P7 | `variational_posterior.py:1492-1493`, `:1499-1500` (`kl_div`), `:917-928` (`pdf`); `vbmc_kldiv.m:75-76`, `:82-83` | control flow (P7i F1, P7c F2); formula (P7i F10) | `q == 0 \| np.isinf(q)` parses as `q == (0 \| isinf(q))`: an infinity is never caught, and NaN is missed. The zero cap is shared. `y / exp(logJ)` overflows where the quotient is representable. | port discrepancy for the infinity: a regression of `4a071d8c` (2022-09-13, PR 96); it was correct in `fdb1e56d`. Never matched for NaN. The cap is shared by design; the division is a Python-only defect, latent. | `wave5_A4_kl_div_guards.py` | as proposed: guard `~np.isfinite` with a test; the density from the difference of the logs | `25a4bb4`; check: `bc0ab84` (dead `jacobian == 0` clause, R1-5) |
| W5-5 | P9 | `vbmc.py` (`_init_log_joint`; `Prior.support()` unread); `misc/funlogger_vbmc.m:121` | cross-module | A prior narrower than the hard bounds stops the run in the function logger with an unrelated message. A wider one gives the evidence of the prior restricted to the box, not renormalized. | not a defect of the computation; shared by design. The Python-only part is that the support is never read. Not dated. | `wave5_A5_prior_support_and_bounds.py` | as proposed: refused at construction, plus documentation. Later (PI): a slack of `1e-9*(ub−lb)`, none at an infinite bound. | `a613fe3`, `725e35e`; check: `42c3942`, `cc909c8`, `5c4fc87`, `851fd17` |
| W5-6 | P7 (also P2 sites) | `stats/get_hpd.py:38`; `active_sample.py:946`, `:964`, `:973`, `:981`, `:1026`, `:994`; five more sites in `gaussian_process_train.py` and `variational_optimization.py`; `misc/gethpd_vbmc.m:10` | defaults (P7i F11); defaults (numeric convention) (P7c F7) | Built-in `round` or `np.round` send a half to even where MATLAB's `round` goes away from zero, at twelve sites. The one tie reachable at shipped options is through the starting cache (the quarter shares of the sieve). | port discrepancy; never matched (`aa454736`, 2021-07-26; MATLAB `f62075b`) | `wave5_A6_get_hpd_rounding.py` | fix at all twelve sites. Later (PI): the sieve caps each source at what is left; the check at construction stays. | `8e2977a`; check: `94638fe`, `3b07758` |
| W5-7 | P2 | `active_sample.py:939`, `:1072`, `:383`, `:679-698`; `activesample_vbmc.m:555`, `:637`, `:219`, `:388` | state/caching | A cached starting point that wins the sieve keeps its stored value at the candidate as clipped and snapped, with no target call. | shared defect, latent; not dated. The clip is out of reach at `active_search_bound = 2`. | `wave5_A7_cached_value_moved_point.py` | fix taken by the orchestrator under "as proposed" (the value reused only at the cached point itself). The PI then ruled "W5-7 stays". | `052c442`; check: `7d5c29a` |
| W5-8 | P2 | `function_logger.py:685` (`_record`); `misc/funlogger_vbmc.m:220` | cross-module | An acquired point equal to a row switched off by the trim is pooled into that row and reaches neither the GP nor `y_max`. This is W3-20, reached by a second route, a grid. | shared defect, latent (= W3-20); not dated | `wave5_A8_acquired_point_on_dead_row.py` | leave, as W3-20 | — |
| W5-9 | P7 | `variational_posterior.py:1482`; `vbmc_kldiv.m:63` | indexing/shape | `kl_div(samples=…, gauss_flag=True)` takes `np.mean(samples)`, a scalar, for the mean vector. | port discrepancy (P7-1); never matched (`fdb1e56d`) | P7-1 (`wave5_P7_1_kldiv_samples_mean.py`) | as proposed (`axis=0`) | `7749ddb` |
| W5-10 | P7 | `variational_posterior.py:1078-1090` (`set_parameters`); `misc/rescale_params.m` has no such check | indexing/shape | The positivity check slices `theta[-check_idx:]` with a negative `check_idx`, so it reads the wrong entries. With `optimize_mu` off it checks nothing. | Python-only defect (P7-2); `8a4644bf` (2021-03-17) | P7-2 (`wave5_P7_2_set_parameters_check.py`) | as proposed | `dfc3334` |
| W5-11 | P7 | `variational_posterior.py:1269-1277` (`mode`) | indexing/shape | `vp.mode()`, the documented default call, raises `AxisError` for every one-dimensional posterior. | Python-only defect (P7-3a); `ba8116fa` (2022-11-03) | P7-3a (`wave5_P7_3_mode.py`) | as proposed | `1ea2f6d`; check: `a188c31` |
| W5-12 | P7 | same; `vbmc_mode.m:39-41` | formula (P7i F4); — (P7c minor obs 3) | The mode's box is offset by an absolute `sqrt(eps)`, and `x0` is clamped to the raw bounds rather than the shrunken ones. | the offset is shared by design; the clamp is a port discrepancy without effect (P7-3b). The clamp never matched (MATLAB `48c82e0`, 2021-03-27). | P7-3b | clamp to the shrunken pair; the offset left | `1ea2f6d` |
| W5-13 | P7 | `variational_posterior.py:1021`, `:1227-1228`, `:1296`; `misc/rescale_params.m:39-40`, `vbmc_mode.m:18`, `:53-55` | state/caching (P7i F5, F13); — (P7c minor obs 1) | The mode cache ignores `n_opts`. `get_parameters` rescales the posterior in place without clearing the cache. The mode is stored unconditionally. | the `n_opts` part and direct assignment are shared by design. The missing clearing and the unconditional store are port discrepancies (P7-3c, P7-5), with no numerical effect. | P7-3c, P7-5 (`wave5_P7_5_get_parameters_inplace.py`) | as proposed: `get_parameters` clears the cache; an explicit `n_opts` bypasses it. Later (PI): `mode(n_opts=k)` neither reads nor writes the store. | `eab21fa`; check: `a188c31` |
| W5-14 | P7 | `variational_posterior.py:1242-1298`; `vbmc_mode.m:21-47` | control flow (and random draws) | The mode search is not MATLAB's: `ceil(sqrt(K))` optimizations from 1e5 draws of the run's generator, so a `vp.mode()` call moves the run's stream. | intentional difference, missing from the sheet (P7-3d); `ba8116fa` (PR 115) | P7-3d | sheet entry; `mode` draws from a copy of the generator | `13cb245` |
| W5-15 | P7 | `sample` (`:612-615`, `:625`, `:690`), `moments` (`:1167-1170`); `vbmc_rnd.m`, `vbmc_moments.m:27` | indexing/shape | The index array is documented `N`-by-1 and returned `(N,)`, float dtype at `K = 1`. `moments` gives a 0-d covariance at `D = 1`. A float `N` is refused. | port discrepancy, cosmetic, for the index array (`42a0deef`, never matched); Python-only defects for the other two (P7-4a, b, d) | P7-4 (`wave5_P7_4_shapes_types.py`) | as proposed: `atleast_2d`; a whole float accepted; the index array flat and of integer dtype, docstring brought to it | `6c38beb`; check: `bc0ab84`; later `6d492a2`, `f9ab814` |
| W5-16 | P7 | `pdf` `:876-903`, `sample`; `vbmc_pdf.m:87-102`, `vbmc_rnd.m` | indexing/shape | `pdf` takes a negative `df`; `sample` raises NumPy's `shape < 0`. | shared by design (P7-4c) | P7-4c | stricter interface: `sample` refuses a negative `df` with the reason | `a5e66d2` |
| W5-17 | P7 | `stats/kde_1d.py:22-31`, `:247-250`, `:261`; `shared/kde1d.m:46-48` | formula | `kde_1d` is Botev's estimator independently implemented: nearest-grid-point binning, a Scott fallback, a floor of 0. MATLAB's estimate sits half a step low. Both sum to `n/(n-1)`. | intentional difference, missing from the sheet (P7-6a–d); `522a901b` (2021-05-26); the normalization is shared | P7-6 (`wave5_P7_6_kde1d.py`) | sheet entry; nothing to fix | — |
| W5-18 | P7 | `variational_posterior.py:787`, `:801` (`pdf`); `vbmc_pdf.m:36-38` | indexing/shape | `pdf(orig_flag=True)` on integer input writes transformed coordinates into an integer copy, so the density is evaluated at truncated points. | Python-only defect (P7-8); from the first port of `pdf` | P7-8 (`wave5_P7_8_pdf_integer.py`) | fix: input promoted to float (ruled as one commit with W5-31; made as two) | `d40e9fd` |
| W5-19 | P7 | `variational_posterior.py:791-795`, `:913`, `:915`; `vbmc_pdf.m:36-39` | control flow | `pdf` gives rows on or outside the original bounds density 0, a NaN row too. MATLAB warps every row and returns NaN or a complex value. | intentional difference, missing from the sheet (P7-9); since the first port | P7-9 (`wave5_P7_9_pdf_bounds.py`) | sheet entry, which says a NaN row gets zero | — |
| W5-20 | P7 | sheet; `variational_posterior.py:1379`, `:1383`, `:1406`; `shared/qtrapz.m:34` | cross-module (sheet accuracy) | The sheet's `qtrapz` entry claims the endpoint handling differs; both are the trapezoid rule at unit spacing. | the sheet's entry is wrong (P7-10) | P7-10 (`wave5_P7_10_qtrapz.py`) | entry corrected | — |
| W5-21 | P7 | constructor, `variational_posterior.py:135-137` | — (P7c minor obs 2) | A `(D, 1)` column `x0` is refused, because the `reshape` result is discarded. | Python-only defect (P7-11); `4c2d107c` (2021-03-13) | P7-11 (`wave5_P7_11_constructor_x0.py`) | fix: the column accepted | `b503ae6`; check: `bc0ab84` (x0 docstring, R1-10) |
| W5-22 | P7 | `entlb_vbmc.py:78`, `entmc_vbmc.py:239`; `ent/entlb_vbmc.m:45-47`, `ent/entmc_vbmc.m:96-101` | formula/gradient | At `K = 1` without the Jacobian the weight gradients are 0 and `H − 1`; the MC weight gradient keeps an extra term. | not a defect (P7-7, P7-13). Both conventions are MATLAB's. MATLAB's fix `1b72896` is present, to 1.8e-15. | `wave5_P7_7_entropy_K1.py`, `wave5_P7_13_entmc_weight_gradient.py` | none (recorded for the O2 third reader) | — |
| W5-23 | P2 | `active_sample.py:320-323`, `:408`; `activesample_vbmc.m:22-26`, `:153` | control flow | With `acq_hedge=True`, `idx_acq` is never assigned, and the first active-sampling step raises `UnboundLocalError`. | port discrepancy, partly shared (P2-1). MATLAB's default single acquisition fails alike. Never assigned in any revision (`4949c826`). | P2-1 (`wave5_P2_1_acq_hedge.py`) | as proposed: refused at construction (and in `load`); description marked; sheet amended | `06fd70e`; check: `bc0ab84` |
| W5-24 | P2 | `active_sample.py:432`, `:453`, `:467-468`, `:960`, `:1050-1057`, `:1069`; `activesample_vbmc.m:230-242`, `:567`, `:627-633` | control flow / state-caching | Search cache: (a) the deletion of the acquired point is dead code; (b) the acquired point is cached first; (c) training inputs come back from the cache unflagged; (d) starting-cache rows lose their index; (e) `search_cache_frac > 0.25` raises at the second step. | (a), (b), (d) shared by design (`4949c826`); (c) Python-only (`014b287`); (e) a Python-only guard (`2551469f`, never matched) (P2-2a–e) | P2-2 (`wave5_P2_2_search_cache.py`, `wave5_P2_11_search_cache_frac_second_step.py`) | as proposed: fractions checked at construction and the message repaired; training rows kept out of the cache; (a), (b), (d) left with a comment | `a25fc73`; check: `bc0ab84`, `7500dba`, `94638fe`, `5c4fc87`, `6dcd027` (`cache_frac`) |
| W5-25 | P2 | `active_sample.py:609-619`; `activesample_vbmc.m:291-296` | defaults / control flow | Nelder-Mead runs without bounds and without `search_max_fun_evals`, and `tol` also governs the step. The point runs away on an unbounded variable; the mask stops it on a bounded one. | port discrepancy in three parts (P2-3a–c), never matched (`4949c826`); part (d) shared by design | P2-3 (`wave5_P2_3_nelder_mead.py`) | value and code removed. `load` replaces a stored "Nelder-Mead" by "cmaes" at `D = 1` and refuses it at `D > 1`. | `4cc09cc`; check: `d66ddf2` |
| W5-26 | P2 | `active_sample.py:127-218`; `advanced_vbmc_options.ini:4`; `docsrc/source/faq.md:429-431`; `misc/initdesign_vbmc.m` | control flow / defaults | With `integer_vars` the initial design is evaluated off the grid. The FAQ says integers are unsupported while the option's description and the changelog imply support. | the design is shared by design (P2-4); the documentation is a Python-only discrepancy (P2-5) | P2-4 (`wave5_P2_4_integer_paths.py`), P2-5 | experimental, and documented as such in the FAQ, the description and the changelog; no refusal of an off-grid `x0` | `0fe371b`; check: `851fd17`, `7500dba` |
| W5-27 | P2 | `function_logger.py:685-763`, `active_sample.py:803-806`; `funlogger_vbmc.m:220-247` | — (first-question answer) | On a grid the search often returns a training input. On a noiseless target it is pooled, no point is added and the posterior is recomputed. | shared by design (P2-6); the update path is already on the sheet | P2-6 (`wave5_P2_5_noiseless_repeat.py`) | none in the code; a sentence of the W5-26 documentation | `0fe371b` |
| W5-28 | P2 | `active_sample.py:208-218` against `:693-708`; `initdesign_vbmc.m:52-58` | state/caching | The initial design's evaluations are outside the `fun_time` timer, and the timers are not stopped in a `finally`. | port discrepancy, diagnostic (P2-7a), never matched (`4949c826`); the consequence drawn for the `finally` does not follow (P2-7b) | P2-7 (`wave5_P2_6_fun_time.py`) | leave, as wave 1 did | — |
| W5-29 | P2 | `active_sample.py:268`/`:849`, `:1010`/`:1015`, `:1039-1040`, `:1072`/`:383`, `:818-832`; `activesample_vbmc.m:55-58`, `:593`, `:596`, `:618-619`, `:637`, `:495` | control flow (F8); formula (F9); indexing/shape (F10) | The `recompute_var_post` save/restore, two covariance normalizations, a literal 3, clip-before-snap, and the search-bound expansion ignoring the integer step. | not a defect of the port; each line is MATLAB's (P2-8, P2-9a,b, P2-10a,b) | `wave5_P2_7_recompute_var_post.py`, `_8_cov_and_box.py`, `_9_clip_then_snap.py` | none (a settled line) | — |
| W5-30 | P9 | `priors/scipy.py:69-70`, `product.py:66-67` | cross-module | `SciPy.support()` reads the standardized `.a`/`.b` and ignores `loc`/`scale`: `uniform(loc=2, scale=3)` reports `[0, 1]`. | Python-only defect (P9-1); `2da98b5f` (2023-02-08) | P9-1 (`wave5_P9_1_scipy_support.py`) | fix: `distribution.support()` | `24b07f4`; check: `cc909c8` |
| W5-31 | P9 | `trapezoidal.py:96`, `spline_trapezoidal.py:98`, `smooth_box.py:79`; `mtrapezlogpdf.m:49` and kin | indexing/shape (dtype) | `np.full_like` inherits the input dtype: integers truncate, and the trapezoids wrap (density 1, or `inf`); float32 in gives float32 out. | port discrepancy (P9-2); never matched | P9-2 (`wave5_P9_2_dtype_fill.py`) | fix: float64 in `Prior.log_pdf`, with a dtype test | `5c05678` |
| W5-32 | P9 | `priors/uniform_box.py:69-70`; `munifboxlogpdf.m:51` | control flow | `UniformBox` gives a row with a NaN coordinate the full density; the other families give `-inf`. | shared defect, latent (P9-3); never differed | P9-3 (`wave5_P9_3_nan_coordinate.py`) | as proposed: depart from MATLAB (test membership) | `da12c24` |
| W5-33 | P9 | four constructors; `munifboxlogpdf.m:44`, `msmoothboxlogpdf.m:26`, `:46` | control flow (argument checks) | NaN or infinite arguments pass every check. `UniformBox(lb, ub)` with infinite bounds builds with density zero. At a NaN pivot the sampler is uniform. | shared defect, latent (P9-4); never differed | P9-4 (`wave5_P9_4_degenerate_args.py`) | stricter interface: each constructor refuses a non-finite argument | `4a43731`; check: `3468704` |
| W5-34 | P9 | `priors/product.py:88` | cross-module | `Product._log_pdf` calls a `UserFunction` marginal with `keepdims=False`, which raises `TypeError`; a tolerant callable is called once on the whole column. | Python-only defect (P9-5). The density half dates from `2da98b5f`; the sample half worked at v1.0.4 and was kept by `d02c517e`. | P9-5 (`wave5_P9_5_product_userfunction.py`) | fix: row by row; the existing test extended | `fa3df99`; check: `a0bb6c2` |
| W5-35 | P9 | `prior.py:17-19`, `:50-52`; box-family docstrings; `convert_to_prior.py:28-35`; `spline_trapezoidal.py:101`; `vbmc.py:150-154` | defaults (documentation); defaults/contract (documentation); formula (comment only) | Wrong documented shapes, `sample_prior` named twice, a sign flipped in a comment, and a cut sentence in the `log_prior` docstring. | Python-only documentation defects; the comment is a cosmetic port discrepancy (P9-6); `2da98b5f` | P9-6 (`wave5_P9_6_documented_shapes.py`) | fix, one commit | `05e4934`; check: `02fc9e6` |
| W5-36 | P9 | `trapezoidal.py:104-108`, `spline_trapezoidal.py:105`, `:124` | control flow | `Trapezoidal` warns at `x == a`. The spline's `np.seterr` pair has no `finally`; the leak stays in the thread. | Python-only defect (P9-7) | P9-7 (`wave5_P9_7_seterr.py`) | fix: `np.errstate` in both | `a652cd8` |
| W5-37 | P9 | `vbmc.py:564-568`, `:3471-3480`; `convert_to_prior.py:50-51` | control flow | The PyMC guard leaves out `sample_prior`; given alone, it builds a prior without a density. | not a defect (P9-8): intended, pinned by `test_vbmc_binding.py:206-230`; guard `975e45a8` | P9-8 (`wave5_P9_8_sample_prior_guard.py`, 2 PyMC tests run) | none | — |
| W5-38 | P9 | `priors/tile_inputs.py:37-42`, `:56` | indexing/shape | `tile_inputs` never checks against `size`, so `UniformBox(np.zeros((2,2)), np.ones((2,2)), D=4)` builds. | Python-only defect, latent (P9-9) | P9-9 (`wave5_P9_9_tile_inputs.py`) | stricter interface: the check the docstring promises | `70b8f2d`; check: `b79a127` |
| W5-39 | P9 | `priors/scipy.py:22-30`, `:67-70` | control flow | A frozen univariate distribution with array parameters is taken as a 1-D prior: its sampler draws from each distribution and its density raises. | Python-only defect, latent (P9-10) | P9-10 (`wave5_P9_10_scipy_array_params.py`) | stricter interface: refused at construction | `474b273` |
| W5-40 | P9 | `trapezoidal.py:72-77`, `spline_trapezoidal.py:74-79`; `mtrapezlogpdf.m`, `msplinetrapezlogpdf.m` | control flow (argument checks on one side only) | The trapezoids refuse `u == v` and `v == b`, which MATLAB computes and normalizes correctly. The message tests use strict violations. | intentional difference, missing from the sheet (P9-11); `2da98b5f` | P9-11 (`wave5_P9_11_matlab_degenerate.py`) | keep the refusal; sheet entry | — |
| W5-FP1 | P2 | `load`; v1.0.4 `active_sample.py:356-360` | — | v1.0.4 wrote "Nelder-Mead" into every one-dimensional run's options, and `load` checks the stored options. | — | wave5.md | `load` replaces it by "cmaes" before `new_options`; an explicit "Nelder-Mead" is refused; `test_vbmc_optimize.py` no longer passes it | `4cc09cc` |
| W5-FP2 | P2 | sieve shares | — | A W5-6 tie is within reach of the shipped options: a starting cache leaving random points ≡ 2 mod 4 (agent C). No gate has such a state. | — | wave5.md | later R3-1/R3-5 | `94638fe`, `3b07758` |
| W5-FP3 | P2 | scripts `wave5_P2_2_*`, `wave5_P2_11_*` | — | These scripts use fractions that construction refuses since `a25fc73`; they are records of `831acef`. | — | wave5.md | noted | — |
| W5-FP4 | P7 | `pdf` | — | A density that truly overflows is still infinite after W5-4 (a tenth of the draws at SD 30); the guard keeps `kl_div` finite. `np.errstate` hides the overflow warning (agent A). | — | wave5.md | left (R1-7) | — |
| W5-FP5 | P7 | `mode` | — | `mode(n_opts=k)` stored its result, which a later `mode()` returned (agent A). | — | wave5.md | PI: neither read nor written | `a188c31` |
| W5-FP6 | P9 | `_rebuild_log_joint` | — | `load(new_options={"vectorized_target": …})` runs W5-5's check on a saved object (agent B). The FAQ's uniform list needs W5-30 and the slack. | — | wave5.md | slack; support read at every call | `42c3942`, `cc909c8` |
| W5-FP7 | P9 | `test_vbmc_init.py` | — | Two statements compared a pair with a pair; with their `assert` they unpack the value and the noise (agent B). | — | wave5.md | done | `614cdfd` |

**The independent check of the pass** (2026-09-21, `wave5.md`). Five fresh
read-only reviewers on `0bf7963` (R1 the variational posterior, R2 the
priors, R3 the active sampling and the options, R4 `wave5.md`, R5 the
user-facing texts and the records) returned 68 findings; two Opus agents made
thirteen commits (`fixes/wave5_check_agent_D.md`, `_E.md`), and a sixth
reviewer (R6) read the round on `f86d373`.

| finding | statement | disposition | fix |
|---|---|---|---|
| R2-1, R5-1 | The W5-5 check compared exactly and refused the FAQ's own `uniform(loc=low, scale=high−low)` list for about a quarter of decimal bounds. | PI: slack `1e-9*(ub−lb)`, none at an infinite bound | `42c3942` |
| R2-9 | A scalar support should be broadcast to the bounds (optional). | taken | `42c3942` |
| R2-2, R2-10 | `SciPy` and `Product` should read their support at every call, so a 1.0.4 pickle reports the right interval and `load(new_options=…)` no longer refuses it; `a` and `b` as read-only float64 (R2-10 optional). | fixed | `cc909c8` |
| R2-3 | A `UserFunction` marginal may return a float or a one-element array; more values raise with the marginal and the row named. | fixed | `a0bb6c2` |
| R2-5 | `tile_inputs` refused a row given with `squeeze=False`, which built in 1.0.4. | fixed; also in the changelog | `b79a127`, `09d0310` |
| R2-7, R5-17 | `check_finite` is public in name for a helper neither exported nor documented. | renamed `_check_finite` | `3468704` |
| R2-6, R5-16 | Docstrings that `05e4934` left wrong, the `Returns` header of `Prior.pdf` among them. | fixed | `02fc9e6` |
| R2-8 | `wave5.md`'s claim that the FAQ list "is taken because W5-30 came first" was false when written. | corrected in `wave5.md`; flag in the header of agent B's report | records (`b2370f3` adds the flag) |
| R2-4 | The FAQ on integer parameters; the `integer_vars` description should say a `prior=` must cover the half-integer bounds. | taken | `851fd17`, `7500dba` |
| R3-1, R3-5 | At W5-6's tie, three quarter-shares of two points round to three. The sieve raised `ValueError` mid-run: with the shipped fractions when two points were left, and with `search_cache_frac` 0.25 when the starting cache left 2 or 3 mod 4. | PI: each source capped at what is left; the check at construction stays. The guard is gone; draws are unchanged where no cap binds. | `94638fe`; test `3b07758` |
| R4-1 | W5-6 said the rounding fires in no run at the defaults, against the last section of `wave5.md`. | corrected in the row | `7411919` |
| R4 (15, incl. R4-11 = R1-13) | Corrections to `wave5.md`. | each in the row or the sentence it names | `7411919` |
| R3-2, R3-3, R5-7 | The FAQ on integer parameters. | texts | `851fd17` |
| R3-4 | A test that a cached point on the grid keeps its stored value, with no target call. | taken | `7d5c29a` |
| R3-6 | Optional items: the last `else` of the search chain names the option; the dead `optim_state["hedge"]` line; the `new_options` hint in two refusals; the comment above `np.delete`; the unseeded test. | taken | `d66ddf2`, `bc0ab84`, `7d5c29a` |
| R5-2 | "With one variable a bounded scalar search is used whatever the option holds" is false for `search_optimizer="none"`. The sentence came from the orchestrator's brief. | fixed in the description, the refusal, the changelog and the sheet; flag in agent C's report | `d66ddf2`, `f86d373`, `09d0310` |
| R5-18 | The helper in `test_gp_training_policy.py` (optional). | taken | `3b07758` |
| R1-1, R1-2, R5-14, R5-15 | `mode(n_opts=k)` and the store; the `Returns` section of `mode`; the docstrings of `mode` and `get_parameters`. | PI: neither reads nor writes the store | `a188c31` |
| R1-5, R1-8, R1-10 | The dead `jacobian == 0` clause; a 0-d `N` in `sample`; the `x0` docstring (all optional or should-fix). | taken | `bc0ab84` |
| R5-8 | The descriptions of the five fractions should give the range and the sum. | taken | `7500dba` |
| R5-20; R1-4, R5-6 | The API page of the priors; the README of the variational posterior. | texts | `851fd17` |
| R5-9, R5-10 | Corrections to `matlab_side_defects.md`. | corrected | `7411919` |
| R5-3, R5-4, R5-5, R5-12, R5-19, R1-9 | Changelog corrections. | corrected | `09d0310` |
| R5-11 | Sheet. | corrected (with the cap, the slack and entry 35) | `f86d373` |
| R1-3 (to_arviz) | The sample count of the ArviZ exports. | ruled to wait for slice N2; fixed after the merge because N2 has no pass ahead (`_whole_number` helper) | `f9ab814` |
| R1-3 (moments) | `moments` truncated a fractional count with `int()` (the part no ruling covered). | fixed on the PI's word | `6d492a2` |
| agent E note | `cache_frac` was unchecked; above one it overfilled the search set, and a negative one took all but that many rows. | checked in `[0, 1]` at construction and in `load` | `6dcd027` |
| R1-6, R1-7, R1-11, R1-12 | The 0/0 row of `pdf` is NaN; `errstate` hides an overflow warning; `mode()` returns the stored array; half a test recomputes the implementation. | left | — |
| R2-11, R2-12, R2-13 | A 0-d argument's message names a shape the caller did not pass; only `UniformBox` points an unbounded problem to another prior; the KS test may draw differently on another platform. | left | — |
| R5-13, R5-21 | The changelog does not name `SVBMC.sample(balance_flag=True)`; the `AGENTS.md` sentence on `__deepcopy__` does not say `mode()` copies the generator on purpose. | left | — |
| D notes | A support of the wrong length gives NumPy's broadcast error; new pickles lack `a`/`b` for 1.0.4 (PI: no action); `tile_inputs` has no `Returns` section. | left | — |
| R6-1 | The worklog's in-flight entry described a state the cherry-picks had ended. | replaced by the entry of the round | records (no hash; `b2370f3` rewrote the worklog) |
| R6-2 | Agent C's raw report states the false one-variable sentence twice. | flag in its header | records (`b2370f3`) |
| R6-3 | The sheet said without condition that the cache keeps the acquired point first. | qualified | records (git: `b66f371`) |
| R6-4 | The burn-in was given as `400/sqrt(N)`; it is a whole number, so its rounding meets no tie. | corrected; the changelog no longer lists it | records; git: `b55131b` (changelog) |
| R6-5 | R1-13 was carried under R4-11 alone. | named | records |
| R6-6, R6-9 | The notes of `_get_search_points`; the docstring of the support check. | taken | `5c4fc87` |
| R6-10 | The range of `D` in the argument on the other rounding sites. | taken | records |
| R6-7, R6-8, R6-11 | `a`/`b` render twice on the API page; the fractions' hint names `search_cache_frac` whichever fraction is at fault; changelog rounding (2e-11 is 1.67e-11, 8e-9 is 8.27e-9). | left | — |

Left, and listed in `wave5.md`, "Left as they are": the items marked so
above. W5-7 stays as ruled.

### Wave 6: gpyreg, the fit and posterior (G1) and the prediction, quadrature and components (G2)

Read at gpyreg `fdbafdf`, whose package code is that of the pin `9e70e6b`,
with PyVBMC at `8af5daac` as the caller; verified on 2026-09-21
(`verification/wave6.md`, with `wave6_G1.md` and `wave6_G2.md`). W6-1 to
W6-4 can change what a run computes. Three Opus agents made the pass on
worktrees of gpyreg and PyVBMC (`fixes/wave6_agent_A.md`, `_B.md`, `_C.md`).

| id | slice | where | category | finding | class; dating | evidence | disposition | fix |
|---|---|---|---|---|---|---|---|---|
| W6-1 | G2 | gpyreg `covariance_functions.py:451` (`:401` for rational quadratic), `mean_functions.py:488`, `:511-524`; MATLAB `gplite_covfun.m:126`, `gplite_meanfun.m:142`, `:220-230` | indexing/shape (G2 comparison F3, F4); defaults (cross-module consequence) (G2 internal F6) | The bound recommendations take the statistics of `X` over all `N*D` entries where gplite takes them per column. This covers the starting length scales and the negative-quadratic mean's hard box, plausible box and start, which come out one for all dimensions. | confirmed port discrepancy, unintended; never matched (the pooled forms are in `16c441d`, 2021-05-31, and no gpyreg revision had the axis) | wave6.md; `wave6_A1_pooled_recommendations.py`, `wave6_A1b_gp_fit_per_column.py`, `wave6_per_column_patch.py`, `wave6_A1c_gate_runs_per_column.py`; later `wave6_A1d_mean_bounds_probe.py`, `wave6_A1d_read.py`; verifier row G2-7 (`wave6_G2_3_bounds.py`, `wave6_G2_4_reach_bounds.py`) | Fix, with the benchmark sweep as the gate. Taken on the sweep. Reopened when the check (R5-1, R5-16) found the orchestrator's summary wrong. Kept after the sweep went to 10 seeds on four targets and after the probe of the mean's bounds (PI, 2026-09-22). | gpyreg `f76eca2`; oracles `3123135e` |
| W6-2 | G1 | gpyreg `slice_sample.py:562`; MATLAB `slicesamplebnd.m:362`, `:367-368`, `:371` | control flow | With an odd burn-in, MATLAB sums one term more than its divisor `floor(burn/2)`. Its "variance" can then be negative, and it keeps the old widths of every coordinate. gpyreg's window is consistent. | confirmed MATLAB-side defect; gpyreg is the consistent side, so an intentional difference in effect, missing from the sheet; the MATLAB lines predate gpyreg's history | wave6.md; `wave6_A2_burn_in_window.py` (parts 1-2) | No fix. A sheet entry, a line of the MATLAB-side list, and a test with an odd burn-in that pins the window (as proposed). | — (the window test is in gpyreg `8dfbae4`) |
| W6-3 | G1 | gpyreg `slice_sample.py:574` (MATLAB has no floor) | — (G1 comparison M6) | The burn-in variance estimate is floored at zero, and a zero width freezes a coordinate. At a burn-in of 2 or 3 the chain returns one point repeatedly, without a message. | Python-only defect; not dated | wave6.md; `wave6_A2_burn_in_window.py` (part 3); the reach through `gp_sample_thin = 1` was read, not run | Fix in gpyreg: a non-positive estimate keeps its width, and a window of fewer than two iterations adapts nothing (as proposed). | gpyreg `8dfbae4` |
| W6-4 | G1 | gpyreg `covariance_functions.py:453-454`, `noise_functions.py:131`, `gaussian_process.py:428`; MATLAB `gplite_covfun.m:130-131`, `gplite_noisefun.m:104`, `gplite_train.m:142` | control flow (G1 internal F9) | Equal targets give the bounds `(-inf, -inf)`, which L-BFGS-B answers with `KeyError`. A run without hard bounds stops at its first GP training when the log joint is constant over the initial design. | the pair is shared with MATLAB; that nothing refuses or repairs it is Python's to decide; not dated | wave6.md; `wave6_A3_constant_target.py` (also G2-20) | PI: a range of one for equal targets, as both sides do for a single target, with a warning; the run goes on. | gpyreg `c3e1901`; check: `fa2c70c` (R3-5) |
| W6-5 | G1 | gpyreg `gaussian_process.py:1441`, `:1457`; MATLAB `gplite_hypprior.m:36`, `gplite_nlZ.m:12` | formula/gradient (G1 internal F1); formula (G1 comparison F3) | The masks read `df == 0 \| ~np.isfinite(df)` as `df == (0 \| ...)`. A prior with `df = inf` or NaN is dropped, while its truncation constant is still subtracted. | confirmed port discrepancy (G1-1); never matched (`16c441d`, 2021-05-31) | wave6_G1.md G1-1; `wave6_G1_1_prior_masks.py` | Fix the parentheses, with a test of the log prior against independent densities (as proposed). | gpyreg `1e19aa9` |
| W6-6 | G1 | gpyreg `gaussian_process.py:1479`, `:1483`, `:1541-1551`, `:1579-1601`; `get_priors` `:466`; no MATLAB counterpart | indexing/shape (G1 internal F2) | The smooth-box normalization constant is not masked to its coordinates. With two coordinates on different sides of the box the log prior doubles, and with three it raises. `get_priors` raises on the same priors. | Python-only defects (G1-2, G1-32); the constants moved into the mask cache unsubscripted with `a2f8ddc` (2026-09-05) | G1-2 (`wave6_G1_2_smoothbox_C_mask.py`), G1-32 (`wave6_G1_9_api_minors.py`) | Fix both, with the fixture's smooth box on the length scales (as proposed). Check: the fixture's boxes over blocks (R2-6). | gpyreg `5a0ede8`; check: `ebc9dcd` |
| W6-7 | G1 and G2 (all four reviewers) | gpyreg `gaussian_process.py:2620-2622` (`__robust_cholesky`); MATLAB `gplite_rnd.m:97-99` | indexing/shape (all four reports) | The eigenvalue fallback flips single entries of the eigenvector matrix, not whole columns, so `T'T` is not the matrix. `sp.linalg.eig`, used where `eigh` is meant, need not return an orthogonal basis. | confirmed port discrepancy (G1-3, G1-3b, G2-3); never matched (`009007b`, 2021-07-12) | G1-3 (`wave6_G1_3_robust_cholesky.py`), G2-3 (`wave6_G2_5_robustchol.py`) | Fix: the flip per column (or none) and `eigh`, with a test on matrices the direct factorization refuses (as proposed). | gpyreg `d96a602` |
| W6-8 | G2 (listed under the G1 heading) | gpyreg `gaussian_process.py:2635`; MATLAB `gplite_rnd.m:110`, `:61` | control flow (G2 comparison F6) | When a negative eigenvalue survives the tolerance, the factor is all zeros and `random_function` silently returns the predictive mean. MATLAB fails on the array sizes (inferred). | the trigger is a confirmed shared defect, the silence Python-only (G2-4); `009007b` (2021-07-12) | G2-4 (`wave6_G2_5_robustchol.py`) | PI: both parts. Raise where no factor can be built, and count an eigenvalue of rounding size as zero, with a relative tolerance and a raise beyond it. Check: the band is measured against the prior variance (R1-1). Round: the low-noise covariance comes from a Cholesky factor (F1-1). | gpyreg `2682a49`; check: `97d2b1a`; round: `5db2255` |
| W6-9 | G1 | gpyreg `gaussian_process.py:885-899`, `:920-928`, `:2023`; MATLAB `gplite_post.m:233-237`, `gplite_pred.m:120` | control flow (G1 internal F4) | The rank-one update tests `sqrt_arg <= 0` only in the `L_chol` branch. The low-noise branch divides by `v_star`, which `predict` has clamped at zero, with no test. | confirmed shared defect (G1-4, G1-25); the other branch's guard is gpyreg's own; the rank-one update dates from `9b0ea12` (2021-06-18) | G1-4 (`wave6_G1_4_rank_one.py`) | Fix: the low-noise branch recomputes where the variance is at or below what the clamp can produce (as proposed). The PI left the guard as made (2026-09-22). | gpyreg `4267466`; check: `2d5796d` (R1-3, a test) |
| W6-10 | G1 | gpyreg `gaussian_process.py:1126`, `:1073-1075`, `:1221`, `:1345` (`fit`) | defaults (G1 internal F5) | `fit` reads `options["sampler"]` but documents `sampler_name`, so the documented name is ignored. Its `"laplace"` clause is dead. | Python-only defect (G1-5); `009007b` (2021-07-12) | G1-5 (`wave6_G1_5_fit_details.py`) | Fix: both names are read, the documented one first, and the clause goes (as proposed). | gpyreg `69b59f4` |
| W6-11 | G1 | gpyreg `gaussian_process.py:1674-1675`, `:1706-1707` | indexing/shape (G1 internal F6) | `log_likelihood` and `log_posterior` raise on the dictionary their docstrings offer: the `(1, hyp_N)` result is sliced along the first axis. | Python-only defect (G1-6); `b5e3ac4` (2021-06-24) | G1-6 (`wave6_G1_9_api_minors.py`) | Fix, with a test of the dictionary form (as proposed). | gpyreg `70a76d0` |
| W6-12 | G1 | gpyreg `gaussian_process.py:1243`, `:1259`, `:1262`; MATLAB `gplite_train.m:206-207` | state/caching (G1 internal F7, comparison F5) | `fit` writes its low-noise start through a view into the design, then takes the default sampler widths from the altered design. gplite copies the design and takes the widths first. | confirmed port discrepancy (G1-7); never matched (`dbc91d6`, 2021-06-24) | G1-7 (`wave6_G1_5_fit_details.py`) | Fix (`.copy()`). It moves the widths of fits at gpyreg's defaults, so gpyreg's suite is the gate (as proposed). | gpyreg `3f9d258` |
| W6-13 | G1 | gpyreg `gaussian_process.py:815-824` (`update`); MATLAB `gplite_post.m` (`update1` from the caller) | control flow (G1 internal F8) | `update` chooses the rank-one path from the argument shapes without checking for factors. It raises `TypeError` after `clean()` and after `update(..., compute_posterior=False)`. | confirmed port discrepancy (G1-8); the shape test dates from `9b0ea12` (2021-06-18), and `cec1f85` added `hyp is None` to it | G1-8 (`wave6_G1_4_rank_one.py`) | Fix: the rank-one path asks for the factors and falls through without them (as proposed). | gpyreg `e4a5aac` |
| W6-14 | G1 | gpyreg `slice_sample.py:186`, `:188`, `:213-220`, `:590-592`; MATLAB `slicesamplebnd.m:168`, `:176`, `:198`, `:377` | defaults (G1 internal F10) | `base_widths` is copied before an infinite width is replaced by 10. The infinity returns after the burn-in and the chain is NaN. | confirmed shared defect (G1-9); shared from the first Python version (`2bf7f10`, 2021-06-08) | G1-9 (`wave6_G1_6_slice_sampler.py`) | Fix: copy after the replacement (as proposed). | gpyreg `d07c849` |
| W6-15 | G1 | gpyreg `slice_sample.py:683-685`, `:926-930`; MATLAB `slicesamplebnd.m:506` (`psrf`, in no file of the repository) | control flow (G1 internal F11) | A frozen free parameter is recognized only if its constant has an exact mean: 0.3 gives `exit_flag = -1`, not -3. The autocorrelations are paired from lag 0, not lag 1. | Python-only defects (G1-10, G1-26); both from `cec1f85` (release 1.2.1) | G1-10, G1-26 (`wave6_G1_6_slice_sampler.py`, `wave6_G1_10_test_claims.py`) | Detect the frozen chain by its range; pair from lag 1 or document it (it was kept at lag 0 and documented); test over several constants (as proposed). | gpyreg `a29d0cf`; check: `f1415de` (R3-6, R3-8) |
| W6-16 | G1 | gpyreg `f_min_fill.py:193-214`, `:218-250`, `:130-132`, `:252-253` (`uuinv`); MATLAB `fminfill.m:133-136`, `:77`, `:169` | formula/gradient (G1 internal F12) | The docstring gives the two tails equal weight where the body weights them by length. "Half of all starting points" holds only for the whole design, with no prior, finite bounds and no fixed coordinate. An out-of-range `p` is marked in one of three paths. | confirmed shared defects of the wording (G1-11, 17, 19); the body is right; gpyreg's sentence came first (`96f1090`, 2022-01-22; MATLAB `1d1f20d`, 2022-01-30) | G1-11/17/19 (`wave6_G1_7_uuinv.py`) | Fix the three texts and add a line to the MATLAB-side list (as proposed). "Per coordinate" was corrected (R3-1); gpyreg's comment gained the fixed coordinates in the round's `2938581`, and the row of `wave6.md` after the check for substantial errors (R-2). | gpyreg `3983eb3`; check: `c36bada`; round: `2938581` |
| W6-17 | G1 | gpyreg `gaussian_process.py:1453`, `f_min_fill.py:119`; MATLAB `gplite_hypprior.m:35`, `fminfill.m:73`, `gplite_nlZ.m:13` | control flow (G1 comparison F4) | The no-prior mask is `~isfinite(mu) & ~isfinite(sigma)` where MATLAB has `\|`. MATLAB's flat prior, `sigma = Inf`, then gives a log posterior of `-inf` or NaN and a NaN design column. | intentional difference with a defect left in it, missing from the sheet (G1-12); never agreed (`27f8d66`, 2021-06-11) | G1-12 (`wave6_G1_1_prior_masks.py`) | Stricter: `set_priors` refuses a `sigma` that is not finite and positive and names `None`; sheet entry (as proposed). Check: a block coordinate whose location and `sigma` are both NaN has no prior (R4-1, R2-2). The remainder was closed by W7-16. | gpyreg `836af5d`; check: `c4a441e`; W7-16: `5499b59` |
| W6-18 | G1 | gpyreg `gaussian_process.py:2812-2819`; MATLAB `gplite_core.m:244` | indexing/shape (G1 comparison F6) | With a scalar total noise and more than one noise hyperparameter, the gradient loop takes the row `dsn2[i]` and raises. MATLAB's linear index returns a wrong entry. | confirmed shared defect (G1-13); both wrong from the start (`c7d3541`, 2021-06-01) | G1-13 (`wave6_G1_8_noise_grad.py`) | Fix (`dsn2[0, i]`), with a gradient test; a MATLAB-side list entry (as proposed). | gpyreg `ff1f14d` |
| W6-19 | G1 | gpyreg `f_min_fill.py:184-185`, `gaussian_process.py:1305-1321`; MATLAB `fminfill.m:104-110`, `gplite_train.m:276-296`, `gp_objfun` | control flow (G1 comparison F7) | Neither the design loop nor the optimizer loop catches an exception of the objective, where MATLAB does in both places. `np.argsort` is not stable where MATLAB's `sort` is. | confirmed port discrepancies (G1-14, G1-36); never agreed, from the first implementation (2021-05-31) | G1-14 (`wave6_G1_5_fit_details.py`) | PI: leave it loud, with a sheet entry. | — |
| W6-20 | G1 | gpyreg `update` (rank-one path) and its sheet entry; MATLAB `gplite_post.m` | — (G1 comparison, "departures", M14) | The sheet names one difference of state. A second is a new point whose noise is below 1e-6, which keeps `L_chol = True`. The sheet's figure of 1e-15 holds only with `L_chol` true. | the sheet entry is incomplete (G1-15, G1-38); the state bookkeeping dates from `cec1f85` (2026-09-14) | G1-15, G1-38 (`wave6_G1_4_rank_one.py`) | Amend the entry in the verifier's words (as proposed). | — |
| W6-21 | G1 | gpyreg `slice_sample.py:238`; MATLAB `slicesamplebnd.m:189` | — (G1 internal M-B) | The Metropolis option is read as `"metopolis_rnd"`, so the options never turn the step on. The flag is an `and` of the two options where MATLAB has an `or`. | Python-only defect (G1-18); `2ccea65` (2021-06-03); setting the three attributes directly did run the step (one test did, R3-3) | G1-18 (`wave6_G1_6_slice_sampler.py`) | PI: the step and its options removed. The removal was kept after R3-3 corrected the premise. | gpyreg `919742e` |
| W6-22 | G1 | gpyreg `set_priors`, `set_bounds`, `get_recommended_bounds`, `fit`, `update`, `slice_sample.py`, `f_min_fill.py`; MATLAB `gplite_train.m:250` | — (minor observations) | Unchecked inputs, and texts that disagree with the code: unknown names taken silently, a tuple of bounds raises, an inverted bound silently repaired, a negative `sigma`, `df_base` written into the GP's priors, `nll` starting at `-inf`, `gamma` overflow, NaN factors, a `thin` or `burn` that is not whole. | Python-only defects, latent (G1-20, 21, 22, 24, 27, 28, 29, 34); `abs(sigma)` and `UB = max(LB, UB)` are MATLAB's too; each item is dated in wave6_G1.md | `wave6_G1_9_api_minors.py` | Fix as one commit of input checks and one of texts (as proposed). | gpyreg `9b282da`, `836af5d`, `e0a057d`; check: `ad685ef` (R3-11) |
| W6-23 | G1 | PyVBMC `gaussian_process_train.py:178`; MATLAB `gptrain_vbmc.m:66` | — (G1 comparison M2) | `hyp_dict["logp"]` receives `res["log_priors"]`, which is zeros, where MATLAB stores the thinned log posterior. Nothing reads it. | Python-only defect of PyVBMC, without effect (G1-31); not dated | G1-31 (`wave6_G1_9_api_minors.py`) | Remove the key and its comment (as proposed). | `4d3c1515`; oracles `0a6682fc` |
| W6-24 | G1 | gpyreg `gaussian_process.py:1649`, `:1519-1523`, `fit`'s `hyp0`; MATLAB `gplite_train.m:98` | — (G1 comparison M5, M8) | These have neither a MATLAB counterpart nor a sheet entry: the smooth-box families, the renormalization of the prior over the bounds, the `-inf` prior of a fixed hyperparameter off its value, and `fit`'s default `hyp0`. | intentional differences, missing from the sheet (G1-33, G1-35); the families date from `27f8d66` and `b0c5cde` (2021-06); flagged later: the renormalization fails far in the upper tail (W7-15) | G1-33, G1-35 (`wave6_G1_9_api_minors.py`, `wave6_G1_5_fit_details.py`) | Sheet entries in the verifier's words (as proposed). | gpyreg `a077e70` (the docstring on the renormalization) |
| W6-25 | G1 | gpyreg `f_min_fill`, `fit` (`res.x`); MATLAB `gplite_train.m:298`, `gplite_post.m:250` | — (minor observations) | `f_min_fill` with no more evaluations than points returns the first `N`. `fit` does not clamp `res.x` one `eps` inside the bounds. MATLAB's `nll` search reads unwritten entries, and MATLAB's `gp.s2` falls shorter than `gp.X`. | not defects of the port (G1-23, G1-30, G1-37, G1-16): the first is MATLAB's own (`46b6f5e`), the second has no effect, the last two are MATLAB's and avoided | G1-23, G1-30 | None; the last two go to the MATLAB-side list (as proposed). | — |
| W6-26 | G2 | gpyreg `gaussian_process.py:1852` (`predict_full`) | indexing/shape (both G2 reports) | `predict_full(add_noise=True)` adds per-point noise as a column broadcast over the rows. The result is neither symmetric nor a covariance. | Python-only defect (G2-1); `3b39954` (2021-07-02) | G2-1 (`wave6_G2_1_predict_full_noise.py`) | Fix (`np.diag`), with a test on noise that varies by point (as proposed). | gpyreg `f803be0` |
| W6-27 | G2 | gpyreg `gaussian_process.py:2193-2196` (`quad`) | state/caching (G2 internal: with a formula consequence) | `quad` recomputes the factor's scale where `Posterior.sl` stores it. After a rank-one update with a point of lower noise the variance is clamped to `eps`. | Python-only defect (G2-2); introduced by `cec1f85` (release 1.2.1) | G2-2 (`wave6_G2_2_quad_scale.py`, `wave6_G2_9_rank_one_test_gap.py`) | Fix (use the stored `sl`), with one `quad(compute_var=True)` in the rank-one test (as proposed). | gpyreg `003d75d` |
| W6-28 | G2 | gpyreg `covariance_functions.py:414` | indexing/shape (both G2 reports) | `RationalQuadraticARD.get_bounds_info` writes the shape's plausible upper bound into index `D`, the output scale's slot. | Python-only defect (G2-5); `9237c82` (2022-11-04) | G2-5 (`wave6_G2_3_bounds.py`) | Fix, with W6-1's test of the recommendations (as proposed). | gpyreg `0ed8cfb` |
| W6-29 | G2 | gpyreg `isotropic_covariance_functions.py:233-246`; MATLAB `gplite_covfun.m:114-118` | control flow (dead computation) (G2 internal F5); formula (G2 comparison F10) | The isotropic bounds take the log of the mean width where MATLAB takes the mean of the log widths, and take `min` and `max` of a scalar. The start is W6-1's pooled SD. | confirmed port discrepancy (G2-6); never agreed (`20730b5`, 2023-04-12) | G2-6 (`wave6_G2_3_bounds.py`, `wave6_G2_10_report_examples.py`) | Fix to MATLAB's means of logs, with W6-1; the sheet entry completed (as proposed). | gpyreg `4ac63a1` |
| W6-30 | G2 | gpyreg `quad`; MATLAB `gplite_quad.m:66-67` | formula (G2 comparison F8) | gpyreg normalizes the variance by the minimum total noise. MATLAB uses the constant term alone and clamps to `eps` under per-point noise. | intentional difference, missing from the sheet: gpyreg's repair of a MATLAB-side defect (G2-8); since `cec1f85` (release 1.2.1) | G2-8 (`wave6_G2_2_quad_scale.py`) | Sheet entry and MATLAB-side list (as proposed). | — |
| W6-31 | G2 | gpyreg `predict(return_lpd=True)`; MATLAB `gplite_pred.m` | formula (G2 comparison F9) | With `separate_samples=False`, gpyreg returns the log density of the moment-matched Gaussian. MATLAB returns the matrix per sample. | intentional difference, missing from the sheet (G2-9); `9b3d46b` (2022-06-02) | G2-9 (`wave6_G2_0_predict_lpd.py`) | Sheet entry; the docstring names the moment matching (as proposed). | gpyreg `a077e70` (docstring) |
| W6-32 | G2 | gpyreg `covariance_functions.py:221`, `:289`; MATLAB `gplite_covfun.m:198`, `:218` | formula/gradient (G2 comparison F11) | The degree-1 Matern length-scale gradient is `inf * 0` on the diagonal, so the likelihood gradient is NaN and the kernel cannot be fitted. MATLAB has the repair as a commented-out line. | confirmed shared defect (G2-10); gpyreg `24592ac` (2021-05-26) | G2-10 (`wave6_G2_6_matern1_grad.py`) | Fix with MATLAB's own line; the test asserts finite values; MATLAB-side list (as proposed). | gpyreg `7c382c4` |
| W6-33 | G2 | gpyreg `gaussian_process.py:2761`, `:2770` (`__core_computation`) | — (the G2 verifier's own) | In the low-noise branch `pL` is computed before the `L is None` test. Ten failed factorizations therefore end in `TypeError`, not `LinAlgError`. | Python-only defect of a message (G2-11); not dated | G2-11 (`wave6_G2_7_minor.py`) | Fix the order (as proposed). | gpyreg `e542794` |
| W6-34 | G2 | gpyreg `quad`, `_convert_shapes`, `compute(compute_diag=True, compute_grad=True)`, `update(hyp=)`; MATLAB `gplite_quad.m:16-19`, `:26` | — (minor observations) | Unchecked inputs: `quad` without data, after `clean()`, with a 1-D `mu` or a foreign mean. Also `_convert_shapes`' types, the meaningless diagonal-and-gradient pair, and the width in `update(hyp=)`. | Python-only defects, latent (G2-19, 21, 24, 27, 30); each item is dated in wave6_G2.md | `wave6_G2_7_minor.py` (also `wave6_G2_8_tests.py`) | Fix as one commit of input checks, with `test_predict_lpd` and its twin repaired (as proposed). The refusal of a one-column `sigma` was withdrawn by the check for substantial errors. | gpyreg `b291d21`, `5ff9353`; that check: `286b595` |
| W6-35 | G2 | gpyreg docstrings of `predict_full` and `predict`, `AbstractKernel.compute`, `noise_functions.py:251`, `docsrc`; MATLAB `gplite_pred.m:120` | — (minor observations) | Texts that disagree with the code: the `add_noise` default, `predict`'s summary, `(N,)` for a `(N, 1)` diagonal, an undocumented nugget, no page for the isotropic kernels, a typo. `predict_full` does not clamp its diagonal. | Python-only, in the documentation (G2-13, 14, 15, 22, 23, 25, 26) | `wave6_G2_7_minor.py`, `wave6_G2_11_negative_diagonal.py` | Fix in one commit; `predict_full` documents its unclamped diagonal (as proposed). | gpyreg `97ac1c2`, `a077e70` |
| W6-36 | G2 | gpyreg `predict:2049`, `quad:2248`, `random_function`, the noise functions; MATLAB `gplite_pred.m:158`, `:160`, `gplite_quad.m:115`, `gplite_rnd.m:67`, `gplite_noisefun.m:186-198` | — (minor observations) | Not differences: `ddof=1` over the samples, the noise omissions, `s2_star` ignored, degenerate recommendations, an empty lpd, no Matern default. Every ported formula and gradient agrees with MATLAB. | not defects (G2-12, 16, 17, 18, 20, 28, 29, 31 to 37); MATLAB's conventions | G2-0 §E (`wave6_G2_0_predict_lpd.py`), `wave6_G2_7_minor.py` | None; two lines under the sheet's settled non-differences (as proposed). | — |
| W6-37 | outside the slices | PyVBMC `vbmc.py:3197-3208` (`VBMC.load(new_options=)`) | — | `load` checks the names and runs `_validate_option_values`, but not `update_defaults` or `_init_optim_state`. A construction-only option therefore takes no coherent effect and bypasses checks: `gp_mean_fun="nonsense"` is accepted. | Python-only defect; not dated (W2-7 and W2-8 did not touch `load`) | G1-X (`wave6_G1_X_load_new_options.py`) | Stricter: `load` refuses construction-only options; the checks of `gp_mean_fun` and `integer_vars` move into `_validate_option_values`; docstring and changelog (as proposed). Check: 1.0.4's `integer_vars` (R4-3) and a scan test (R4-5). Round: the run's mask wherever the reading differs (F2-3). | `e657eba8`; check: `edc42739`, `326c7676`; round: `b2d7c6e8`, `503562d7` |
| W6-38 | found during the fix pass (agent B, while making W6-4; fixed by agent A) | gpyreg `noise_functions.py`, `fit`'s clips of the plausible bounds; MATLAB `gplite_noisefun.m:105-106`, `gplite_train.m:157-158` | — | The noise's recommended plausible pair is inverted for targets whose SD is below 1e-3. The clips keep it inverted (unless the range is below 1e-6), and the design fails `uuinv`'s ordering assertion. | Python-only defect, latent: the inversion and the clips are gplite's, and only gpyreg's `uuinv` asserts the order; statement, verdict and reach corrected after R1-2 and R5-2; not dated | fixes/wave6_agent_A.md: `test_fit_with_targets_of_a_tiny_range` (seen to fail); the check's probe (no script under `verification/scripts/`) | Fix made as the last commit, for the PI to strike; kept (PI, 2026-09-22). | gpyreg `caacbc1`; check: `a4f7969`, `4994bbe` (R1-6) |
| W6-FP1 | G1 | gpyreg's GP tests (`test_fitting`) | — | `test_fitting`, in both GP test files, was an unseeded statistical test that failed 2 of 6 runs on the untouched revision | found by fix agent A (its note 2) | `wave6.md`, "Fix commits" | seeded in the pass (PI's ruling) | gpyreg `8dec795` |
| W6-FP2 | G2 | gpyreg `_convert_shapes` | — | its type annotation still names the narrower types | found by fix agent A (its note 5) | `wave6.md`, "Fix commits" | left as it stands | — |

**The independent check of the pass** (2026-09-22, `wave6.md`). Five fresh
read-only reviewers on gpyreg `dc2a930` and PyVBMC `a3d4a70d` (R1 the
posterior, the rank-one update, the prediction, `quad` and `random_function`;
R2 the priors, the bounds of `fit` and its interface; R3 the other modules,
W6-1 and the release notes; R4 PyVBMC's commits and their coupling with the
branch; R5 the records) returned 7, 13, 11, 9 and 30 findings. Fix agent D
made the gpyreg side (`fixes/wave6_check_agent_D.md`), the orchestrator the
PyVBMC side; the other findings were taken as the orchestrator proposed,
and those left for later were ruled on after the close (above, "Findings
ruled after the close").

| finding | statement | disposition | fix |
|---|---|---|---|
| R4-2 | No gate of the pass had run PyVBMC's suite against the branch. Run against `dc2a930`: 1 failed, 1986 passed, 58 skipped. | The suite now runs against the branch (gate 3); the account of gate 1 was corrected. | records |
| R4-1, R2-2 | `set_priors` refused the block prior that `_gp_hyp` writes for `noise_rectified_log_multiplier`: one coordinate has no prior, with location and `sigma` NaN. This came from W6-17's commit. | PI: such a coordinate has no prior; any other `sigma` that is not finite and positive is refused. | gpyreg `c4a441e` |
| R1-1 | `random_function` raised `LinAlgError` on every grid inside the data at small noise, because W6-8's band was measured against the largest eigenvalue. | PI: the band is measured against the prior variance at the test points. | gpyreg `97d2b1a` |
| R2-1 | `get_priors` returned `None` for a prior with NaN `df`, so `set_priors(get_priors())` dropped it. | PI: the prior is returned in a form `set_priors` reads back, and a NaN `df` is documented. | gpyreg `2e96c90` |
| R4-3 | `load` refused a file saved by 1.0.4 whose `integer_vars` has a form the check refuses. | PI: the stored value that fails the check gets the mask the run was made with. | `edc42739` |
| R5-1, R5-16 | The orchestrator's summary of the sweep, on which the PI had taken W6-1, was wrong. | PI reopened W6-1: 10 seeds on four targets, then the probe; W6-1 kept. | records ("Gates") |
| R1-2, R5-2, R5-14 | W6-38's mechanism, verdict and reach were wrong. | Corrected. | gpyreg `a4f7969` (comment and test docstring); `d93f03c` (release notes) |
| R3-3 | W6-21's premise was wrong: one test set the three attributes and ran the step. | The removal stands, with its record corrected. | records; gpyreg `d93f03c` |
| R3-1 | W6-16's claim on half of the design ("per coordinate"). | Corrected. | gpyreg `c36bada` |
| R2-7 | The margin of the seeded `test_fitting`. | gpyreg `8dec795`'s message stands, with its error flagged in "Fix commits". | records |
| R3-6, R3-8 | The comparison with BDA3 and Stan under W6-15 was overstated. | The docstring no longer makes it. | gpyreg `f1415de` |
| R5-3 to R5-8, R5-10 to R5-13, R5-26 | Citations and claims of the sheet, its history phrasing, and five entries it lacked. | PI: the sheet states the revisions of its gpyreg citations and states facts. | records |
| R4-4, R5-22 | The changelog's gpyreg requirement named a version not yet released. | Corrected. | records |
| R5-24, R5-30 | Entries 41 and 49 to 52 of `matlab_side_defects.md`. | Corrected. | records |
| R2-6 | The prior fixture put each smooth box on a single hyperparameter. | PI: the fixture's boxes go over blocks. | gpyreg `ebc9dcd` |
| R4-5 | The construction-only options of `load` could drift from the code. | PI: a scan test keeps them in step. | `326c7676` |
| R1-5 | An unsupported type of `s2`. | It raises `TypeError`. | gpyreg `43bf232` |
| R3-5 | Targets with a NaN, or all of one infinity, were reported as all equal. | Fixed. | gpyreg `fa2c70c` |
| R3-11 | An infinite `thin` or `burn` was accepted. | Refused with `ValueError`. | gpyreg `ad685ef` |
| R1-3 | The test of the low-noise duplicate did not fire the guard by construction. | Tests only. | gpyreg `2d5796d` |
| R1-6 | The tiny-range test did not assert on the plausible pair the design receives. | Tests only. | gpyreg `4994bbe` |
| R1-4, R2-4, R2-9, R2-11, R5-29 | `predict`'s pooled variance; the `Raises` sections of `update`, `get_recommended_bounds` and `fit`; three comments; the wording of `update`'s refusal. | Docs. | gpyreg `51e492f` |
| R2-5 | The comment in the test of the documented densities. | Docs. | gpyreg `7a48db8` |
| R3-9 | What the isotropic kernels take from each parent. | Docs. | gpyreg `39ef329` |
| R1-2, R1-3, R2-3, R3-2, R3-3, R3-4, R3-7, R5-9, R5-14, R5-15 | The release notes. | Made true point by point, with the Upgrading lines of the new refusals. | gpyreg `d93f03c` |
| agent D, found on the way | `set_priors` takes a NaN location beside a finite `sigma`. | `TODO.md`; later ruled as W7-16. | gpyreg `5499b59` (1.3.1); the TODO line was removed in `3fb370cd` |

**The independent check of the fix round** (2026-09-22, `wave6.md`). Three
fresh read-only reviewers (F1 agent D's commits and the release notes; F2
PyVBMC's `edc42739` and `326c7676`; F3 the records) returned 8, 8 and 27
findings. Fix agent E made the gpyreg side
(`fixes/wave6_check_round_agent_E.md`). The PI sent the small findings that
no record held to the same `TODO.md` item, where fix agent E's two are too:
the rounding of `predict` and `predict_full` in the low-noise
representation, and `fit` with a whole float `thin`.

| finding | statement | disposition | fix |
|---|---|---|---|
| F1-1 | `random_function` still raised in the low-noise representation, inside and outside the data. A default fit on noiseless targets ends there. | PI: the covariance of a low-noise posterior is formed from a Cholesky factor. | gpyreg `5db2255` (release notes `dd12db3`) |
| F3-3 | The reading of the sweep said the fix was better on about as many seeds as worse, where it is worse on 35 of 59 by the ELBO error, and it left out losses. | Corrected where it stands. | records (`858742f1`) |
| F3-2 | The probe paragraph said the mean's log scale almost never left its box; it does in up to 9.3% of vectors. | Corrected. | records |
| F3-1 | W6-16's correction was not carried to the MATLAB-side table and entry 48. | Corrected. | records |
| F1-2 to F1-4, F1-8 | The release notes, plus the other changed exception types that agent E found. | As proposed. | gpyreg `dd12db3`; F1-8 also `ec0f085` |
| F1-5, F1-7 | Docstrings and comments. | As proposed. | gpyreg `2938581`; F1-5 also `ec0f085` |
| F2-3 | `load` and the stored `integer_vars`. | PI: the run's mask wherever the reading differs, not only where the check refuses. | `b2d7c6e8` |
| F2-6 to F2-8 | The `Raises` of `load`, the test of the all-zero form, the `logp` lines. | As proposed. | `b2d7c6e8` |
| F2-1, F2-2, F2-4, F2-5 | The scan test and the comment of the tuple. | As proposed. | `503562d7` |
| F3-4 to F3-27 | The records. | As proposed; the probe's numbers are printed by `wave6_A1d_read.py`. | records |
| ruling | The changelog names its gpyreg requirement with `pyproject.toml`'s minimum after the release, and the sheet's header moves with the pin. | Done. | `4196649a`, `2dc2a6c3` |

**The check for substantial errors of the last round** (2026-09-23,
`wave6.md`). Three fresh read-only reviewers (G agent E's commits and the
release notes; P PyVBMC's `b2d7c6e8` and `503562d7`; R the records) found
three errors, all corrected; it left nothing.

| finding | statement | disposition | fix |
|---|---|---|---|
| G | `quad`'s width check (W6-34, `5ff9353`) refused a `(N, 1)` `sigma` in more than one dimension, which 1.2.1 broadcast and `gplite_quad.m` broadcasts. The release notes also misstated what 1.2.1 did. | Fixed by the orchestrator, with a test seen to fail; gpyreg's suite 320 passed. | gpyreg `286b595` |
| R-1 | The plan's pickup point left out the parallel branch `dev-port-review-w1check`. | Corrected. | records (git: `ee24dfd9`) |
| R-2 | Row W6-16 gave the condition for half of the design without gpyreg's fixed coordinates. | Corrected. | records (git: `ee24dfd9`) |

### Wave 7: the third readers O1 to O4

Read in a worktree frozen at `a65b96f4`, with gpyreg at `v1.3.0`; verified
on 2026-09-23 (`verification/wave7.md`, with `wave7_O1.md` to `wave7_O4.md`).
Every answer to the first questions held, O2's apart from the tail case of
W7-3 (`wave7.md`, "The first questions"). Four Opus agents made the fixes (`fixes/wave7_agent_A.md` to
`_D.md`, with agent D's later rounds).

| id | slice | where | category | finding | class; dating | evidence | disposition | fix |
|---|---|---|---|---|---|---|---|---|
| W7-1 | O3 | `whitening.py:435-472` (`warp_gp_and_vp`), `vbmc.py:3905`; MATLAB `misc/warp_gpandvp_vbmc.m:38` | control flow | `warp_gp_and_vp` raises for a `ZeroMean`, which `gp_mean_fun="zero"` allows. Nothing catches the error, and the instance is left half-warped. MATLAB's cases are off by one mean code. | zero mean: confirmed shared defect; constant mean: confirmed MATLAB-side defect; Python branches since `4aa4ca97` (2022-03-03), MATLAB `case 0` since `2a076c9` (2020-06-16) | wave7_O3.md O3-2; `wave7_O3_F2_zero_mean_warp.py` | Fix (a): warp the length scales alone for a zero mean; a warp test on a bounded state for the constant and zero means; MATLAB-side list (as proposed). | `58c4be10` |
| W7-2 | O3 | `whitening.py:219`, `vbmc.py:1571`, `:1586`, `:1941`, `:3252-3253` | state/caching | At a kept warp the deep copy of `optim_state` unlinks `hyp_dict`. The iteration's record holds the pre-warp `hyp_dict`, so `load(file, iteration=k)` resumes the GP fit from pre-warp hyperparameters. | confirmed Python-only defect; dates from the resume feature, `a5f6effb` (2022-10-27) | O3-6; `wave7_O3_F6_hyp_dict_at_warp.py`; the run of the close: `wave7_A_w7_2_resume_effect.py` | Fix: relink after the warp (as proposed). Kept as ruled even though old files keep their record (PI, 2026-09-23). Check: the link moved to the end of the warp block. | `017bda47`; check: `3dd58381` |
| W7-3 | O2 | `variational_posterior.py:919-928`, `:994-1000` (`_pdf`), `mode`; MATLAB `vbmc_pdf.m:58-66`, `:107-110`, `vbmc_mode.m` | formula/gradient | `_pdf` sums in the linear domain and logs afterwards. Below about -745 `log_pdf` is `-inf` and its gradient NaN, and `vp.mode()` returns its start unrefined on narrow posteriors, printing only a NumPy warning. | log density and gradient: confirmed shared defect; `mode`: confirmed on the Python side (MATLAB would need a run); always matched MATLAB (`c87019b9`, 2021-03-16) | O2-1; `wave7_O2_F1_pdf_tail_underflow.py`, `_mode_narrow.py`, `_mode_bfgs_trace.py`, `_mode_lse_counterfactual.py`, `_mode_start_error.py`, `_mode_warning_origin.py` | Fix: log-sum-exp under `log_flag` where the linear sum is below the smallest normal; tests; MATLAB-side list; records corrected (as proposed). The ruling's premise that nothing moves failed for `vp_pdf`, whose reference was replaced (PI, 2026-09-23). Check: the proposal of the active importance sampling keeps the log density as held. | `0ccda687`; oracles `83a592c6`; check: `d8c3990c` |
| W7-4 | O2 | `handle_0D_1D_input`, `variational_posterior.py:877`; MATLAB `vbmc_pdf.m:41` | — (O2 §1) | For `D = 1` a flat array of `N > 1` points is read as one point. `pdf` raises, except in the transformed space with a finite `df`, where it returns one meaningless number. | confirmed Python-only defect (interface); the 1-D reading dates from `98472e9a` (2021-03-26) | O2-2; `wave7_O2_D1_input_shape.py` | Fix, stricter: refuse an `x` whose width is not `vp.D` (as proposed). | `78365e55` |
| W7-5 | O3 | `whitening.py:333-347`; MATLAB `warp_input_vbmc.m:143-148` | state/caching | After a warp the search box is the min/max of 1000 draws of the old box. It falls short of the exact image more as `D` grows (median extent 0.590 at `D = 20`). | confirmed shared defect, of no consequence; always MATLAB's scheme (Python `4aa4ca97`, 2022-03-03) | O3-5; `wave7_O3_F5_search_box_after_warp.py`, `wave7_O3_F5_two_warps.py` | Leave it in both; MATLAB-side list, among the items left as they are (as proposed). | — |
| W7-6 | O3 | the bounded transforms of `parameter_transformer.py`; MATLAB `warpvars_vbmc.m:106-108`, `:256-258`, `:442-459` | formula/gradient | Every bounded map goes through `(x - a)/(b - a)`, so `b - x` is resolved only to `(b - a) eps/2`, a loss where `abs(b) < b - a`. This answers the wave-0 question: MATLAB shares it. | confirmed shared defect (precision); Python `4ce90e25` (2022-07-28), MATLAB 2018-2020 | O3-4; `wave7_O3_F3_F4_precision.py` | Leave it in both; MATLAB-side list; the plan's wave-0 question corrected (as proposed). | — |
| W7-7 | O3 | `parameter_transformer.py:636-643` (`_student4`); MATLAB `warpvars_vbmc.m:265-269` | formula/gradient | `_student4` forms `q - 1` by cancellation near `z = 1/2`: an error in `t` up to 1.55e-8, and 0 within about 6e-9 of the midpoint. The Torch export is free of the cancellation. | confirmed shared defect (precision); the two never differed (Python `4ce90e25`, 2022-07-28; MATLAB `9da4afd`, 2020-05-01) | O3-3; `wave7_O3_F3_F4_precision.py`, `wave7_O3_F3_round_trip_fine.py` | Leave it in both (not the default, negligible); MATLAB-side list (as proposed). | — |
| W7-8 | O1 | `variational_optimization.py:1228`, `variational_posterior.py:1237-1240` (`set_parameters`), `:615`; MATLAB `misc/negelcbo_vbmc.m:33-48` | state/caching | With `sigma` frozen and `lambd` optimized, `set_parameters` rescales the frozen `sigma` on every call, which ruins the optimization (0.78 to 34648). The mirror case gives two values of the objective at one θ. | confirmed port discrepancy (MATLAB right); regression of `abf5c3ec` (2021-10-31, a shape fix) | O1-1; `wave7_O1_F1_frozen_sigma.py` (with `wave7_O1_common.py`) | Fix: rescale only when both are optimized; finite-difference and repeated-call tests (as proposed). | `047a75ee` |
| W7-9 | O1 | `variational_optimization.py:1211-1212` (`_neg_elcbo`); MATLAB `negelcbo_vbmc.m:10`, `:16`, `gplogjoint.m:25-29` | defaults | `compute_var=None` computes the variance only for a nonzero `beta`, which is undocumented. The reviewer's statement that MATLAB's default computes it is wrong: MATLAB's call stops. | documentation only; the MATLAB default is a dormant MATLAB-side defect; the Python line dates from `9267816c` (2021-08-25), never matched | O1-2; `wave7_O1_F2_F3_defaults_docstrings.py` | Docstring, sheet entry, MATLAB-side entry 56 (as proposed). | `fa10366c`; check: `a087961d` |
| W7-10 | O1 | the docstrings of `_initialize_full_elcbo` and `_neg_elcbo` | indexing/shape | The docstrings misdescribe `D`, `Ns` (twice), `entropy_alpha` and the type of `compute_var`. | documentation only; since `9267816c` (2021-08-25) | O1-3; the same script | Docstring edits (as proposed). | `58d68a3d` |
| W7-11 | O2 | `papers/acerbi2018variational_appendix.md:88`, `:113` | — (O2 §2) | The transcription has `(σ_k λ)^2` where the derivative gives `(σ_l λ)^2`, and a factor `1/K^2` at line 113. The code and MATLAB are right. The published PDF was not checked. | documentation only; not dated | O2-3; reading | A note beside each line (as proposed). | `25787c80` (git) |
| W7-12 | O3 | the sheet entry; MATLAB `warpvars_vbmc.m` action `'g'`, `vbmc_pdf.m:119` | cross-module | The sheet says `'g'` returns the gradient of the log Jacobian. It returns derivatives of the inverse map, its one caller is dead, and no PyVBMC path needs that gradient. | documentation only (plus one dead MATLAB line); not dated (the entry's premise came from the plan's O3 row, `0e4e27a5`) | O3-1; `wave7_O3_F1_log_jacobian_gradient.py` | Replace the entry; the dead line goes into the MATLAB-side list (as proposed). | — |
| W7-13 | O3 | the sheet entry on the probit default | defaults | The sheet's reason cites `AGENTS.md`, which only ever said "probit by default". The default comes from `6cee9bb5` (2022-11-24). | documentation only; not dated | O3-7; `git log -S`, `git show 6cee9bb5` | Correct the "Why" (as proposed). | — |
| W7-14 | O4 | gpyreg `gaussian_process.py:1716-1720` (`__compute_log_priors`); MATLAB `gplite_hypprior.m` | formula/gradient | A NaN is written into the prior's gradient for every coordinate with equal bounds and survives where no family branch overwrites it. It reaches `log_posterior(compute_grad=True)`, and it stops L-BFGS-B when another coordinate has a prior. | confirmed Python-only defect; the sheet entry does not hold as written; never agreed (gpyreg `6754f01`, 2021-06-22) | O4-1; `wave7_O4_F1_fixed_bound_gradient.py`, `_fixed_with_smoothbox.py`, `_tiny_range_gpyreg.py`, `_pyvbmc_reach.py` | Fix in gpyreg (0 where no branch sets it), a test assertion, the sheet entry replaced, a patch release (as proposed). | gpyreg `e5b7238` |
| W7-15 | O4 | gpyreg `gaussian_process.py:1592-1607`, `f_min_fill.py:300`, `:337` | formula/gradient (the value only) | The prior's mass inside its bounds is `cdf(ub) - cdf(lb)`, which is 0 with both bounds far in the upper tail. `log_posterior` is then `+inf` everywhere, and a fit returns a poor point. | confirmed Python-only defect in the listed renormalization; the sheet and W6-24 fail in that regime; `64dc49d` (2021-06-29) | O4-2; `wave7_O4_F2_prior_mass_upper_tail.py` | Fix: the survival function in the upper tail, bit-identical for PyVBMC's priors; a test; the sheet entry qualified; W6-24 flagged (as proposed). | gpyreg `5ae1c44`; check: `c81063c` (a test) |
| W7-16 | O4 | gpyreg `set_priors`; MATLAB `gplite_hypprior.m:35` | defaults | `set_priors` validates `sigma` alone. A non-finite location or smooth-box end is accepted, and the log posterior is NaN or `-inf`. MATLAB reads a non-finite location as no prior. | confirmed Python-only defect (input validation); never agreed (the location was never validated, since `6754f01`/`b0c5cde`) | O4-3; `wave7_O4_F3_nonfinite_prior_location.py` | Fix, stricter: refuse a non-finite location. This closes the `TODO.md` NaN-location item and W6-17's remainder; the sheet entry on `None` is corrected (as proposed). | gpyreg `5499b59` |
| W7-17 | outside the slices (O4 verifier §9) | gpyreg's recommended length-scale bounds; PyVBMC `active_sample.py:127-185`; MATLAB `gplite_covfun.m:105`, `:121-124` | — | An input column of zero width makes the length-scale bounds `(-inf, -inf)`, and `fit` ends with `KeyError`. PyVBMC reaches it when at least `fun_eval_start` starting points share one coordinate. | candidate: measured on a PyVBMC-built GP, the path from `x0` read from the code and confirmed by the capped run of the close; not dated | `wave7_O4_F1_pyvbmc_reach.py`; the close: `wave7_A_w7_17_shared_coordinate_run.py` | The capped run first, then a ruling together with the `TODO.md` item on a single training point. After the run: refusal at construction (2026-09-23). The TODO item stays for triage. | `8921845f` |
| W7-FP1 | fix round (agent D) | gpyreg `f_min_fill.py`, the design of `fit` | — | The space-filling design maps its draws through `cdf(lb) + (cdf(ub) - cdf(lb)) S`, which is `+inf` for almost every draw with both bounds far in a prior's upper tail; a fit with hyperparameter samples then raises. | a candidate of the kind of W7-15; PyVBMC does not reach it | `wave7.md`, "Found during the fix round" | fixed in 1.3.1 (PI, 2026-09-23) | gpyreg `0cf27e1` |
| W7-FP2 | fix round (agent D) | gpyreg `set_priors` | — | `set_priors` takes an inverted smooth box (`a > b`), whose log prior is then a wrong finite value or NaN. | a candidate of the kind of W7-16; PyVBMC does not reach it | same | fixed in 1.3.1: refused, and a box of zero width taken | gpyreg `201b52b` |
| W7-FP3 | fix round | the docstrings of `_gp_log_joint`, `warp_gp_and_vp`, `log_pdf` | — | `_gp_log_joint`'s `compute_var` takes 2; `warp_gp_and_vp` returns an array, not a dictionary; `log_pdf` returns the log density. | documentation | same | corrected | `72a39732` |
| W7-FP4 | fix round | `optimize_vp` with one scale not optimized | — | It returns scales that MATLAB's `rescale_params` would rescale, with the same density. | a difference of form | same | recorded in the sheet's entry on `set_parameters` | — |
| W7-FP5 | fix round | `_vb_init`, type 2; `misc/vbinit_vbmc.m:32` | — | The type-2 starting points overwrite a `sigma` that is not optimized, as MATLAB's do. | shared | same | left as it is | — |
| W7-FP6 | fix round | `_pdf` | — | Where `nf w / sigma^D` overflows, at component scales near 1e-16 in 20 dimensions, `_pdf` gives `inf` or NaN, outside the ruling of W7-3. | — | same | left as it is | — |
| W7-FP7 | fix round | the warp block of `optimize` | — | An exception inside the warp block other than the one W7-1 removes still leaves the instance half-warped. | — | same | left as it is | — |

**The independent check of the pass** (2026-09-23, `wave7.md`). Five fresh
read-only reviewers: the warps and the resume; the density and the entropy;
the variational objective and the scales; gpyreg's commits and release notes;
the records. None found a defect in the code of a fix.

| finding | statement | disposition | fix |
|---|---|---|---|
| S1 (density and entropy) | After W7-3, a point outside every box of the proposal where the density underflows got a finite, very large log weight instead of `-inf`. | PI: the proposal takes the log density as it is held, its value before W7-3 to the bit (120,000 points). | `d8c3990c` |
| O1, O2 (warps) | With `skip_active_sampling_after_warmup`, an undone warp at the end of warm-up left the pre-warp `hyp_dict` in the record, and the undo branch had no test. | PI: the link moved to the end of the warp block, and a second short run checks an undone warp. | `3dd58381` |
| S3 (density and entropy) | `eaac0349` kept the two asserted fixed-draw checks of the Monte Carlo entropy, where the test note asked for their replacement. | PI kept them (2026-09-23). | — |
| gpyreg (S2, S3) | No test pinned the design's unchanged path at a lower bound exactly at the prior's centre. | Fixed in 1.3.1. | gpyreg `6ffff06` |
| gpyreg (S2, S3) | The Upgrading point on the location refusal gave `None` also for one coordinate of a block, and the refusal of an inverted box had no Upgrading point. | Fixed in 1.3.1. | gpyreg `1f959de` |
| gpyreg (beyond the pass) | The design of a coordinate fixed by equal bounds that has a prior lands 1 to 2 ulp off its bound, so the log prior is `-inf` at every design point but those of `x0`. It changes 144 of 2,304 PyVBMC-built GPs. | Fixed in 1.3.1. | gpyreg `4eb8789` |
| (round) | The smooth-box families in the test of the mass's unchanged path. | — | gpyreg `c81063c` |
| Text | `_neg_elcbo`'s Raises section, and a test docstring for the condition of MATLAB's default variance. | — | `a087961d` |
| Text | Corrected in the records: the sheet's two gpyreg entries (wrong even for 1.3.0) rewritten; `wave7.md`, the plan, the changelog's account of 1.0.4 for W7-4, the fixtures' reason for `vp_pdf`, the MATLAB-side list, three sheet entries, the reports' headers, the modernization note on `vp.pdf`. | — | `265348e7` |

Left as they are: past about 37.7 scales the survival function underflows
too, and the prior's mass and the design fail as the lower tail always did;
a GP pickled by gpyreg 1.3.0 keeps its stored constants until a refit.

**The two runs of the close** (`wave7.md`). W7-17: a capped run from ten,
and from twelve, starting points that share one coordinate stops at its
first GP fit with L-BFGS-B's `KeyError`, which the refusal at construction
(`8921845f`) now prevents. W7-2: a resume at a kept warp reproduces the
uninterrupted run exactly with the record the fix writes, and departs from
it with the record of before the fix.

## Intentional differences found, and the sheet entries withdrawn

The known-differences sheet had 67 entries when the comparison reviewers of
wave 1 received it and has 134 at the close. Each wave brought it to the
state of the code; its entries are consolidated into the porting log
`pyvbmc/vbmc/README.md`, the catalogue that is kept current.

**Entries withdrawn or rewritten because they did not hold.**

| Wave | Entry as it stood | What was wrong | Now |
|---|---|---|---|
| 1 | "Noise shaping is an unported stub" | it said the option only flipped a noise-function flag; the option is refused since `bb6ab65` | "Noise shaping is not ported, and `noise_shaping=True` is rejected" |
| 1 | "`vp.stats["J_sjk"]` is pruned on both component axes" | its account of MATLAB was false (P6 cmp sheet note 1) | the entry, corrected |
| 1, check | "`search_optimizer` takes different values, and `D == 1` is a scalar search"; "The initial design does not cluster surplus starting points" | the first gave `cma.fmin` running at `D = 1` as its reason; the second said a surplus point needed no target call, which holds only with `f_vals` | both entries, corrected |
| 2 | "The sieve asks for candidates in proportion to the current `K`"; "Posterior tempering (`vbmc_power`, `vptrain2real`) is not ported" | the first misread the warp branch (W2-16), the second the reach of `temperature` (W2-22) | both entries, corrected |
| 4 | "The EIG acquisition and the experimental VIQR losses were removed" | `loss="iqr_reduction"` is shipped (W4-21) | "The EIG acquisition and two experimental VIQR losses were removed" |
| 4 | "The rank-one GP update is taken for a fresh observation, noisy or not" | MATLAB recomputes whenever a noise variance is given (W4-11) | the entry, corrected |
| 4 | "The MCMC branch of importance sampling is a retained but dormant hook" | the branch had never run (W4-6) | "The MCMC refinement of the variational importance samples is not ported" |
| 5 | "`qtrapz.m` is replaced by SciPy's trapezoid rule" | the two rules are the same formula (W5-20) | the entry, corrected |
| 5 | "The acquisition-portfolio hedge (`acqhedge_vbmc.m`) is not ported" | it said PyVBMC picks one acquisition at random, while `acq_hedge=True` raised (W5-23) | the entry, corrected |
| 6 | "The rank-one GP update is taken for a fresh observation, noisy or not"; "Isotropic kernels and the rational-quadratic kernel are Python-only" | the first's 1e-15 held only with `L_chol` true (W6-20); the second missed MATLAB's bound branch of `seiso` (W6-29) | both entries, amended |
| 6 | "NumPy quantile and standard-deviation conventions in GP bound recommendations" | what differed for the statistics was the axis, not the normalization (W6-1) | "NumPy's quantile convention in the GP mean-function bound recommendations" |
| 7 | "The gradient of the log Jacobian is not ported" | MATLAB's `'g'` action returns the derivatives of the inverse map (W7-12) | "MATLAB's `'g'` action of `warpvars_vbmc.m` is not ported" |
| 7 | "The hyperprior is renormalized to the bounds, and a fixed hyperparameter gets a prior" | a fixed coordinate's gradient was NaN, and the mass vanished in the upper tail (W7-14, W7-15) | "The hyperprior is renormalized to the bounds, and a hyperparameter with equal bounds is held at its value" |
| 7 | "The default bounded transform is probit, not logit"; "'No prior' is expressed by `None`, not by an infinite scale"; "The mode search starts from draws of the posterior" | the first's reason cited `AGENTS.md`, which held none (W7-13); the second took a non-finite location beside a finite `sigma` (W7-16); the third named L-BFGS-B for `orig_flag=False`, which uses BFGS | the three entries, corrected |
| close | seven entries, listed in the sheet's header | found wrong in part by the independent check of the porting log | corrected in the porting log and flagged in the sheet |

**Entries added**, by wave, under their titles in the sheet (the porting log
names them by the same subjects):

- Wave 1 (`583ef86`): declared options that nothing reads; MATLAB's
  `samples` output; `search_optimizer` values and the one-dimensional
  search; an acquired starting point leaving the cache; the rank-one GP
  update; the sieve's candidate count; GP smoothing (`Bandwidth`); the exact
  entropy of a one-component posterior; the deterministic-entropy optimizer;
  the soft bounds of the training inputs; the probit default; and seven
  settled non-differences. Its check (`aed685ff`, `35606cb`): a NaN score
  ranking last; the checks of `warp_cov_reg` and `hpd_frac`; the schedule of
  space-filling points without span; the training counts after every
  evaluation.
- Wave 2 (`7f9a26c`): `results["problem_type"]`; the empty warm-up window;
  the true-posterior diagnostic on a copy of the generator; the separate
  search GP; the log form of the search acquisition; `integer_vars` as a
  mask or indices; two defects of MATLAB's input checks that PyVBMC does not
  reproduce;
  `display`; `warp_input` with the current transform; the closing display
  line; the final boost with fixed means; and seven settled
  non-differences.
- Wave 3 (`9c29ce1`): the lower bounds from the HPD subset; slice sampling as
  the only hyperparameter sampler; the evaluation times not on the GP; the
  warp keeping its covariance where the correlation threshold would leave it
  indefinite; one finite bound
  refused; points within rounding of a bound; the pooling fallback for
  extreme SDs; a cached value needing its SD; `scale` and the noise flag
  checked; and, from the check, the evaluation time of a repeat.
- Wave 4: the quantile argument of VIQR and IMIQR; the normalized importance
  log weights; a NaN acquisition value; and three settled non-differences
  (the candidate's noise, the target of IMIQR's MCMC step, IMIQR's row
  weights).
- Wave 5: the stored value of a cached point; the checked search fractions;
  the mode search; `vp.pdf` outside the bounds; `kde_1d`; the interface of
  `vp.sample`; the argument checks of the priors and the check against the
  hard bounds; a NaN coordinate in a box prior; and three settled
  non-differences.
- Wave 6: the window of the burn-in statistics; the variance of `GP.quad`;
  the pooled log predictive density; the smooth-box hyperpriors; the
  renormalized hyperprior; "no prior" as `None`; an exception of the
  objective ending the fit; the default start of `fit`; and, after its
  checks, equal targets, `random_function`'s eigenvalues, an infinite
  sampler width, the degree-1 Matern gradient, the removed Metropolis step
  and the scalar noise gradient; two settled non-differences.
- Wave 7: the default variance of `_neg_elcbo`; the log density where it
  underflows; `pdf` refusing a width other than `D`; `set_parameters` with
  one scale frozen; the warp of a zero-mean GP; starting points without
  spread refused; one settled non-difference (three properties of the warps
  and transforms shared with MATLAB).

## Test notes acted on

Every fix that changes code carries a test written against the contract
(the MATLAB lines, the docstring or the ruling), seen to fail on the code
before it except where a per-wave ledger or a fix report says otherwise;
fixes of documentation carry none. The reviewers' notes on tests that
mirrored the implementation or could not see a defect were acted on as
follows; from wave 3 on, each per-wave ledger has a section on them.

- Waves 0 and 1 kept no separate list of test notes: the reports' notes were
  acted on through the tests of the fixes (for instance `181a63b`, which
  replaced a test of P2 that passed without seeing its defect), M F2's note
  by the scan test of `INERT_OPTIONS` (`a63b17a`), which recomputes the set
  of options that nothing reads, and the check's by `e71eab2`, `3bc079f`,
  `c469ba5`, `893d0ba`. Outside the notes, two flaky tests were made
  deterministic (`e48fade`, `39ad28a`), and the CI matrix led to `99fdb2f`
  (mock targets that Python 3.10 resolves differently) and `8e1df32` (a
  tolerance set to one machine).
- Wave 2 kept no separate list: the minimum-iteration test that asserted the
  defect (`567444f`, with `2d3f876` from the check), the guard test of the
  inert options (`82c624c`), and `ce59fbe` from the check. Outside the
  notes, the CI matrix led to `e3ccd0e` (a test that re-saved the static
  VBMC pickle crashed Python 3.11).
- Wave 3: `test_gp_hyp` on the transformed bounds (`23a69a9`); the sampler
  and schedule tests (`bee1866`, `3b8b26c`, `175629a`); the first GP at
  uncertainty level 1 (`095c82c`); `train_gp` called twice (`bf081c2`);
  warps on a logger with evaluations (`9ebaa48`, `791c51d`); `scale`
  (`dea6b29`); `add` at level 2 (`fbe25b6`); a repeat with an unknown time
  (`490ea14`); the specification test of W3-25's note (`b74c65c`); a flaky
  Adam test (`b731fac`).
- Wave 4: `test_real2int` asserting MATLAB's values (`3c088ab`); the first
  value test of IMIQR's MCMC step (`53ad16a`); `fess` with a count
  (`dd3d1d3`); the `acq_info` tests (`4554135`); the quantile range
  (`0477a2f`); `sn2_new` at level 1 with a repeated row, the first test below
  the GP fit at that level, with the dead lines of `test_acq_log_f` removed
  (`9d5d682`).
- Wave 5: tests that see the defects the old tests were blind to
  (`23d962a`, `7749ddb`, `dfc3334`, `1ea2f6d`); a Kolmogorov-Smirnov test of
  each box sampler (`48518c8`); twelve comparisons in `test_vbmc_init.py`
  that lacked their `assert` (`614cdfd`); the oracle harness's copy of
  `active_sample` documented (`a093f2e`).
- Wave 6 (gpyreg): the bound recommendations (`f76eca2`, `0ed8cfb`,
  `4ac63a1`); `test_predict_lpd` and its twin (`5ff9353`); seeded tests
  (`3faeaa9`, `4415940`, `8dec795`); the low-noise rank-one branch
  (`4267466`, `2d5796d`); `quad` after a rank-one update (`003d75d`); the
  prior fixture's smooth boxes over blocks (`ebc9dcd`); `__robust_cholesky`
  (`d96a602`, `2682a49`, `5db2255`); frozen chains (`a29d0cf`); odd and small
  burn-ins (`8dfbae4`).
- Wave 7: `_gp_log_joint` with the zero and constant means and the other
  gaps of slice O1 (`fee60c43`, `3dfa93eb`, `de876410`, `91122ff0`,
  `9f18f491`); the exact check of the Monte Carlo entropy, with the two
  asserted fixed-draw checks kept (`eaac0349`); a budget that splits one
  component's samples and agreement across the gradient flags (`e060c768`);
  the linear density's gradient without Torch (`24669ad9`); the upper
  saturation of probit and Student-t(4) (`005256f3`); the large-`N`
  transform tests on distinct rows (`601efaee`); a warp test on a bounded state (`58c4be10`); in
  gpyreg, `e5b7238`, `14736b5`, `dcaea44`, `8be6ee4`.

The notes left are in the per-wave ledgers' sections on test notes;
`wave4.md` gives its reasons, while wave 5's note on a reference for
`test_kde1d.py` and wave 6's on the rank-one tests with one sample and
`sn2_mult = 1` have no recorded outcome.

## Defects on the MATLAB side

`experiments/port_review_20260919/matlab_side_defects.md` holds what the
review found wrong in MATLAB VBMC, from a reading of its source and, where
the entry says so, a Python transcription; nothing was run in MATLAB.
Entries 1 to 14 come from waves 1 and 2 (entry 9 from a plan that predates
the review), 15 to 20 from wave 3, 21 to 24 from
wave 4, 25 to 40 from wave 5 (37 to 40 shared defects that PyVBMC no longer
shares), 41 to 53 from wave 6 (the GP layer, `gplite/`), 54 and 55 from the
check of the wave-1 pass, 56 to 61 from wave 7, and 62 from the rulings
after the close. The one that changes
what a MATLAB user draws is entry 1, `shared/msmoothboxrnd.m` for `D > 1`.
The unnumbered sections list the questionable points shared by both
implementations and left as they are, and the state that MATLAB writes or
declares and never reads.

## Gates, CI and merges

Every fix pass passed its gates before it was merged; each is recorded in
the plan's worklog and, from wave 3 on, in its per-wave ledger's section
"Gates".

| Wave | Gates recorded | CI | Into `dev-next` |
|---|---|---|---|
| 0 | the tests of each module touched (calibration 86, S-VBMC with Torch 226, PyMC 107); for the chunk-independent entropy, the exact oracle check 11 of 11 | with wave 2 | merge `57cbfb4` (2026-09-20) |
| 1 | the tests of every module touched; the exact oracle check 11 of 11 after the re-baseline of `dd89374`; the evidence was not kept (`wave1.md`) | with wave 2 | `57cbfb4` |
| 2 | the four seeded runs, made for this pass, bit for bit at every batch that had to leave trajectories alone; the exact oracle check 11 of 11 at every batch; the default suite 1540 passed and 58 skipped after W2-30; the S-VBMC tests with Torch, 227 passed | full matrix, nine cells: green on `e3ccd0e` (the third run), and on `ae4e651` after W2-30 | `57cbfb4` |
| 2, check | the tests of every module touched, 413; the oracle tests, 143; the exact oracle check 11 of 11; the rest with the wave-5 round | with wave 5 (`63a0808`) | `03a8d46` |
| 3 | exact 11 of 11 after the re-baselines; suite 1608 passed; Torch 739, PyMC 107; eight benchmark targets, 32 seeded pairs; the check: suite 1699 passed, Torch 785, PyMC 107 | green on `92eb2cc`; the check on `f0be99b` | `82aee90`; the check `5acd382` |
| 4 | exact 11 of 11 after the re-baseline; seeded runs bit for bit; suite 1687 passed; Torch 777, PyMC 107 | green on `11fb766` | `9d9c01b` |
| 5 | seeded runs bit for bit except what W5-1 moves; exact 11 of 11 with nothing re-baselined; the check round: suite 1961 passed, Torch 1009, PyMC 107 | green on `63a0808`; the ArviZ count on `0c67e30` | `03a8d46`; `1563053` |
| 6 | gates 1 to 4 against gpyreg's branch with `PYTHONPATH` naming it: gpyreg's suite (293 passed at gate 1 to 319 at gate 4, and 320 on `286b595`), PyVBMC's suite (1992 passed at gate 4), exact 11 of 11 after the re-baselines, the seeded runs bit for bit after W6-1; the benchmark sweep on eight targets | green against the pin at `a4c2cc0` and at `v1.3.0` (`647a886a`) | fast-forward |
| 1, check | exact 11 of 11; suite 2073 passed; Torch 841, PyMC 110; the noisy seeded runs moved in 40 of 92 arrays; the noisy benchmark sweep | green on `0d09e63d` | merge `e86bbb1c` into `dev-port-review`, then fast-forward |
| 7 | exact 11 of 11 after the re-baseline of `83a592c6`; seeded runs bit for bit; suite 2143 passed; extras 973; the same against gpyreg's `w7-fixes`, with gpyreg's suite 386 passed; W7-17's refusal: suite 2145 passed | the branch smoke on `acdbfd01` and the full matrix on `6ef6a084`; both on `3fb370cd` (gpyreg 1.3.1) and on `6e1082d8` | fast-forward |
| after the close | on `a7dd32d5` and on `b24a9238`, the heads before and after the fixes of what the rulings found: exact 11 of 11 with nothing re-baselined, the seeded runs bit for bit, suite 2227 and 2274 passed, extras 1066 and 1068 (the calibration tests with them); gpyreg's suite on `w6-leftovers`, 478 passed before the round that dropped the removal of `np.abs` and 525 at `dedf90b`; the same PyVBMC gates against `dedf90b`, every one as on the installed gpyreg. After the check of the rulings: on `03c6401e`, the head of the round's first fix agents, alone and against gpyreg `107c829`, exact 11 of 11, the seeded runs bit for bit, suite 2358 passed, extras 1071, gpyreg's suite 669 passed; on the final heads, `68f47251` alone and against gpyreg `38e8ada`, exact 11 of 11 with nothing re-baselined, the seeded runs bit for bit (92 arrays), suite 2376 passed and 58 skipped, extras 1073 passed, the same against the branch as alone, and gpyreg's suite 696 passed; after the release, on `95377395` against the installed gpyreg 1.3.2, the same four gates with the same results | the branch smoke and the full matrix (nine jobs) green on `d729b276`, with the pin at `v1.3.2` | fast-forward |
