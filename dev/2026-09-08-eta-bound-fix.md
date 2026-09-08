# Production eta-bound correction

Status: implementation, focused checks, bounded replays and independent
review complete; one accuracy-fence flag retained (2026-09-08).
Branch: `dev-eta-bound-fix`, based on `c4c692c`.

## PI decision

After the [equal-iteration experiment](2026-09-08-eta-equal-budget.md), the
PI selected A: remove eta-bound loss and gradient, and stop `_neg_elcbo`
from mutating caller theta. Preserve stable private softmax calculations,
the parameter layout, location/scale bounds, the separate capped small-weight
penalty, pruning, and all optimizer settings. The selected final-boost policy
remains zero boost-only weight penalty with the joint 0.1 guard.

The stopping-rule question is future research and improvement, explicitly
outside this latent-fix campaign and not a release blocker. The two saved
noisy states improved their surrogate scores with additional iterations;
that does not establish better true-posterior accuracy or justify a global
400-iteration requirement. No stopping-rule experiment or modification is
scheduled as part of this fix.

## Verification scope

Focused deterministic/common-draw Monte Carlo finite differences cover the
full bound-augmented objective, common eta shifts across historical bounds,
caller theta/bounds preservation and the retained weight penalty. Check the
existing exact numerical fixtures; do not change unrelated references.

The bounded paired trajectory gate is Normal D5, Cigar D4 and Rosenbrock D2
noise 1, each seed 0: three pre-change and three post-change runs. Pre-change
code is pinned at `c4c692c` in
`dev/scripts/runs/latent_fixes/eta_A_before_worktree/`. Actual PyVBMC imports
were verified there; gpyreg imports resolve to the sibling checkout. Both
sides use the same final-boost default and original reference environment,
with OMP/OpenBLAS/MKL single-threaded. Original reference and experiment
artifacts remain untouched. Compare initial designs, pre-boost trajectories,
returned metrics and the existing accuracy fences separately.

Outputs: `dev/scripts/runs/latent_fixes/eta_A_20260908/before/` and `after/`.
The final 870-run benchmark remains the later integrated population gate.

Historical comparison drivers intentionally pin the pre-fix numerical
source. Reproduce those experiments from `c4c692c` (the equal-budget report's
recorded code and source hashes), or their earlier documented checkpoints;
do not relax their source guard to run a different treatment under an old
experiment label.

## Results

The production change copies eta before max-shifting and evaluates the
existing soft-bound routine against private floating-point bound arrays with
the eta entries set to infinities. The original packing/folding of mu and
scale gradients is unchanged. A new weight-only regression initially caught
integer upper-bound arrays from `get_bounds`; explicitly floating private
copies fix that edge case without changing caller arrays. The old eta-only
loss expectation is updated to zero; no MATLAB NPZ fixture was changed.

**64 focused tests passed without reruns (16.15 s)**, covering variational
optimization, full-objective finite differences, deterministic/common-seed MC
eta-shift invariance, input/bound nonmutation, weight-only/fixed-weight paths,
the retained capped-weight penalty, and final boosting. **All 11 numerical
fixtures remain exact**, with no rebaseline. Independent Sol core review
passes with no remaining findings.

The before and after replay groups took 3.691 and 3.244 minutes, respectively.
All six runs converged with finite metrics and satisfied the existing usability
criterion (evidence error < 1, gsKL < 1, MMTV < 0.2). Initial designs are exact
across each pair. Main-loop ELBO paths first differ at iterations 9, 9 and 8
for Normal, Cigar and noisy Rosenbrock (zero-based); these are expected moving
trajectories, not a trajectory-neutral refactor.

| Configuration, seed 0 | Evidence error before -> A | gsKL before -> A | MMTV before -> A | Evaluations before -> A |
|---|---|---|---|---|
| Normal D5 | 0.001041 -> 0.018957 | 0.00006074 -> 0.00049074 | 0.007579 -> 0.007748 | 70 -> 70 |
| Cigar D4 | 0.007534 -> 0.004922 | 0.00090686 -> 0.00082544 | 0.016562 -> 0.009079 | 130 -> 140 |
| Rosenbrock D2, noise 1 | 0.026465 -> 0.007515 | 0.052013 -> 0.010131 | 0.049749 -> 0.019986 | 130 -> 125 |

**The accuracy gate is not all green:** the before group has zero flags;
the after group returns exit code 1 for Normal D5 gsKL 0.000490743 exceeding
its fixed Q3+3IQR fence 0.000429099. The excess is about 0.00006164. Evidence
error and MMTV remain inside their fences; Cigar and noisy Rosenbrock pass
all fences and improve all three final accuracy metrics in these seeds.
No threshold is loosened, run repeated to obtain a pass, or flagged result
discarded. Retain the Normal flag for integrated population assessment; this
three-pair gate does not establish overall superiority or equivalence.

To separate the loop from boost, reconstruct the selected pre-boost VPs from
the recorded parameters and per-iteration transformers, using the existing
fixed-seed benchmark metrics. This diagnostic-only check took 4.057 s with
no optimizer fits or target evaluations. The existing `boost_comparison`
helper from the pinned campaign checkout is reused. It also reconstructs
the final metrics and moments and matches their recorded values at
rtol=1e-9, atol=1e-11. The final-transformer identity itself is absent from
the trace, so this is a validated diagnostic reconstruction under the
best-iteration-transformer assumption, not proof of exact transformer state.

| Configuration | Pre-boost evidence error before -> A | Pre-boost gsKL before -> A | Pre-boost MMTV before -> A |
|---|---|---|---|
| Normal D5 | 0.019090 -> 0.006539 | 0.00061264 -> 0.00027878 | 0.008595 -> 0.008768 |
| Cigar D4 | 0.011810 -> 0.004733 | 0.00154485 -> 0.00146898 | 0.015222 -> 0.009591 |
| Rosenbrock D2, noise 1 | 0.095274 -> 0.002146 | 0.161594 -> 0.011000 | 0.079925 -> 0.022420 |

Normal's pre-boost evidence error and gsKL improve with A. Its new boost
worsens these metrics (gsKL 0.00027878 -> 0.00049074), whereas the previous
boost improved them. Both runs use the selected joint 0.1 guard and both
return K=50. Their GP/VP endpoints and optimization randomness differ, so
this is an interaction with final refinement, not an isolated comparison of
boost policies. A guard based on surrogate scores does not guarantee true
posterior improvement. The chosen boost policy remains unchanged for final
population testing.

## Evidence and next step

[eta_bound_fix_20260908.json](experiments/eta_bound_fix_20260908.json) records
both replay reports, all six sidecars, pre-boost scores/quality, file hashes,
validation and actual import/source provenance. Root verifies JSON/NPZ
counts, initial designs, ELBO agreement horizons, finite/usable final metrics,
and the exact after-source hash. The numerical source was measured uncommitted
on base `c4c692c`; its normalized SHA-256 is
`8461ae568d197db1668e1be16656b2c3e879ff2037aeff3f115b672a7530e262`.
The report's `(dirty)` label and this hash distinguish it from the clean
pre-change checkout. No full pytest suite or population rerun was performed;
those remain final-integration gates after the Phase 7 dependency repair.
Independent Sol report review verified the embedded reports, all six
sidecars, all 12 trace/sidecar hashes and both metric tables; no findings remain.

The selected correction is implemented. Next is the upstream gpyreg
step-out bracket repair, followed by integrated validation. Stopping-rule
research stays deferred and is not a prerequisite for either step.

## CI discovery follow-up

The first pushed commit, `e2639a1`, failed CI run 34240597505 because bare
pytest also collected `dev/scripts/test_eta_bound_variants.py`. Those
historical experiment checks intentionally require the pre-fix source hash;
the production correction changes that hash. The earlier focused checks
did not exercise default discovery.

The repair adds `testpaths = ["pyvbmc/testing"]` to pytest configuration.
Default and explicit package collection now match all 1,082 locally available
test cases exactly. Explicit historical collection still finds 35 checks;
reproduction uses checkout `c4c692c`, which includes those checks and retains
the pinned numerical source. No experiment guard, numerical implementation,
accuracy fence or fixture is changed.

The first local full-suite attempt reached 602 passed and 35 skipped before
a sandbox permission error creating pytest's temporary directory. The retry
uses a fresh workspace-local temporary directory and passes: **1,049 passed,
35 skipped, no reruns, 320.64 s**, using bare pytest with CI's rerun/stop
flags and one BLAS thread. Independent Sol review passed. Repair `96a8e3a`
was pushed and [replacement CI](https://github.com/acerbilab/pyvbmc/actions/runs/34246862361)
passed on Ubuntu/Python 3.12: **1,122 passed, 60 skipped, no reruns,
574.40 s** (whole job 10m41s). The noisy half-normal end-to-end test passed.
Next integration step is merging into `dev-next` and checking its CI.

## Integration and deferred upstream issue

`dev-next` was fast-forwarded to `89ac5f0` and pushed. Its
[integration CI](https://github.com/acerbilab/pyvbmc/actions/runs/34248221965)
passed: 1,122 passed, 60 skipped, no reruns, 541.28 s.

The PI subsequently deferred the optional gpyreg step-out repair to
[gpyreg #44](https://github.com/acerbilab/gpyreg/issues/44). It is not used
by current PyVBMC GP training and is no longer a release prerequisite.
This supersedes the upstream-repair-first sequencing above; keep the
dependency pin unchanged. Full-matrix and final population gates remain.
