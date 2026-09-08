# Main-loop eta-bound comparison

Status: bounded experiment and independent reviews complete, 2026-09-08.
Branch: `dev-eta-bound-comparison`, based on `03650a2`; its numerical code
passed CI 138 at `b0d3437`. The [live plan](plans/latent-bug-fixes.md) owns
the checklist. The paired boost experiment remains parked. No production
penalty choice or whole-trajectory campaign is part of this pickup.

## Treatments and mathematical contract

All arms retain the separate capped small-weight penalty, pruning, and
location/scale bounds. A removes only the eta-bound loss/gradient and the
objective's mutation of caller theta. B uses MATLAB's raw-eta soft bounds
with consistent derivatives and private stable softmax arithmetic. C runs
the existing objective, including its mutation and supplied gradient.

For B, `a=log(0.5*tol_weight)`, `b=0`, and `s=(b-a)*tol_con_loss`.
Each eta contributes
`0.5*((a-eta)_+/s)^2 + 0.5*((eta-b)_+/s)^2`, with derivative
`(eta-a)/s^2` below a, zero inside the interval, and `(eta-b)/s^2` above b.
This is the MATLAB implementation contract inspected at pinned commit
`396d649c3490f1459828ac85f552482869edf41c`, as recorded in Phase 6.
The paper appendix A.3 describes softmax, its Jacobian and optimization;
it does not specify this raw-eta penalty there.

C evaluates bounds after subtracting max(eta) in place but supplies the
unadjusted bound gradient. Away from maximum ties, the correct derivative
of that relative-eta loss is `g - one_hot(argmax(eta))*sum(g)`, where g is
the componentwise bound derivative at the shifted eta. Tests must exercise
active lower/upper bounds and common shifts, including fixed-draw MC entropy.

`vp.get_parameters()` returns log normalized weights, not the stored
max-shifted `vp.eta`. B therefore can penalize an actual optimizer start
whose stored relative eta is inside the interval. Input diagnostics must
observe theta before C mutates it.

## Bounded design

The [manifest](experiments/eta_bound_20260908.json) identifies eight authentic
saved GP/VP states and hashes every source file. Five are tracked numerical
snapshots; three are local authentic pre-boost captures with original GP
factors and main-loop options. They are historical states shared across arms,
not exact continuations of their historical optimizations. No reference
snapshot or population is regenerated.

| State | K | GP samples | Role |
|---|---:|---:|---|
| Normal D2 warmup | 2 | 8 | Fixed-weight negative control |
| Normal D2 single GP | 5 | 1 | Later fit, single-posterior regime |
| Cigar D4 | 14 | 7 | Correlated late fit |
| Correlated D5 | 17 | 8 | Warped late fit |
| Halfnormal D2 | 9 | 10 | Bounded coordinates |
| Rosenbrock D2 noise 1, seed 0 | 19 | 7 | One initial raw-eta lower violation |
| Rosenbrock D2 noise 3, seed 7 | 21 | 6 | No initial bound violation |
| Rosenbrock D2 noise 3, seed 17 | 14 | 6 | Three initial raw-eta lower violations |

Use fixed input K with the main loop's incremental sieve budget (5K
candidates under these options), one selected optimization start, and the
unchanged 400/600/700-iteration stochastic budgets. Generate the C sieve once
per state/replicate, then give identical selected candidates and the same
post-sieve RNG state to independent copies for all three arms. This isolates
local optimization conditional on common C-selected starts; it does not test
arm-specific candidate ranking or K growth. Retain every input, candidate,
option set and RNG state. Do not rely on VP deepcopy to isolate generators.

The cap is eight states x three arms x three paired optimizer RNG replicates
= 72 local `optimize_vp` calls. Time the first three-arm comparison before
continuing; its results count toward the cap. No GP fitting, new target calls,
final boosts or full VBMC trajectories are requested. Warmup and fixed weights
remain as captured; do not silently enable weight optimization for the control.

## Evaluation and interpretation

Rescore pre/post VPs with the same unpenalized ELBO computation, an independent
diagnostic RNG reset per candidate, and approximately 100,000 total entropy
draws. Report ELBO, its GP uncertainty estimate, and ELCBO at beta 5.
The current objective sets entropy variance to zero: reported SD does not
include Monte Carlo entropy error. The approximately 100,000 paired draws
reduce that error but do not remove it or constitute its uncertainty estimate.
The diagnostic stream must not advance optimization. Different component
counts after pruning limit draw alignment and must be reported. Compare
paired differences within each state/replicate, not differently penalized
training objectives or historical scores from another GP.
Here `pre` denotes the source VP, before the shared sieve; the actual selected
optimizer start is retained separately in the candidate pool. These two
starting references must not be conflated when interpreting improvement.

Record natural bound violations, eta offsets/ranges, supplied/correct penalty
gradients versus the ELBO gradient, objective progress, stopping information,
weights/pruning and runtime. Controlled bound violations in mathematical tests
are separate from natural frequency in the experiment.

Three optimizer seeds per state measure local randomness; they are not three
independent benchmark targets. These selected states are diagnostic coverage,
not a random population sample. This experiment cannot establish statistical
equivalence, downstream acquisition effects or overall posterior accuracy.
Any promising treatment still needs explicitly scoped whole trajectories and
discussion with the PI before a production change. If C wins reproducibly,
inspect mutation, gradient and stopping effects rather than approving the
inconsistent objective/gradient pair from its score alone.

## Verification and evidence

The initial 31 variant tests pass in 2.04 seconds. They cover isolated
regularizer and full real-GP/entropy finite differences for A/B, eta shifts
-2/0/+2 with active bounds, actual deterministic and paired MC entropy,
caller-theta preservation, C's exact production result/RNG/mutation,
small-weight penalty invariance, and K=1/fixed-weight paths. The experiment
pins normalized-LF production source, so line endings do not change the
scientific source identity. No production file is edited.

Artifacts are under `dev/scripts/runs/latent_fixes/eta_bound_20260908/`.
`before.json` records actual imports, versions and original source hashes;
`variant_tests.log` records the mathematical gate. PyVBMC imports from this
checkout, gpyreg from the clean sibling at
`a2f8ddce867f502e29717959cf0ff3529f598618`. All numerical work uses the
original `.venv` and single-threaded BLAS, one process at a time.

Independent review requested explicit noninterference and fidelity checks for
the observational helper. The expanded 35-test gate passes in 1.93 seconds
(`variant_tests_final.log`), including unchanged theta/bounds/VP arrays/RNG
and agreement between diagnostic losses/gradients and the actual three-arm
objective penalties. Follow-up variant review has no remaining findings.

The eight-state input preflight and two repeated instrumentation/restoration
cycles per arm pass without optimizer fits. They verify identical inputs and
candidate pools, objective/result/theta/RNG parity, and restoration of cached
module globals. Before the timed work, runner checks corrected observational
mutation, instrumentation isolation, transformer hashing, and diagnostic
scoring/draw accounting. No failed optimizer runs were needed for these fixes.

## Measured results

All 24 paired comparisons completed: eight states, three optimizer RNG
replicates, three arms, exactly 72 local fits. The three-arm pilot took
0.470816 s and was reused; the remaining 23 comparisons took 21.651237 s.
Total invocation work was **22.122053 s**, including preparation, fitting,
diagnostics, scoring and capture writes, excluding interpreter startup.
The saved comparison directory occupies 49,740,587 bytes. These timings are
for one local incremental refit per state, not whole VBMC runs or boosts.

The [complete compact results](experiments/eta_bound_20260908_results.json)
contain all 24 paired pre/A/B/C ELBO, GP SD and beta-5 ELCBO values, differences,
pruning, iteration counts, penalty activation and draw-alignment diagnostics.
Raw `comparison/pairs/<state>/replicate_<n>/` artifacts retain exact inputs,
candidate pools, VPs, RNG states and objective traces. Their config records
the measured uncommitted experiment source hashes and base HEAD `03650a2`.
Do not mistake a later experiment commit for the measured production base.

| Arm | Fits | Fit time (s) | Optimizer objective calls | Fits with active eta penalty | Components pruned |
|---|---:|---:|---:|---:|---:|
| A: no eta bounds, no theta mutation | 24 | 3.128 | 1,180 | 0 | 7 |
| B: raw eta bounds, no theta mutation | 24 | 4.578 | 2,580 | 7 | 7 |
| C: current production behavior | 24 | 3.139 | 1,220 | 0 | 7 |

C's relative-eta penalty never activates in this sample. Consequently its
incorrect active-bound gradient is demonstrated by controlled mathematical
tests, but is **not naturally exercised by these fits**. A's recorded raw-bound
distances are counterfactual; its penalty is disabled. B has 1,721 active
optimizer calls across seven fits (Correlated D5 replicate 0 and all six
noise-1/seed-0 and noise-3/seed-17 fits). No arm has an upper-bound violation.

The fixed-weight warmup control is exact across arms. Across the five
noiseless states A and C agree within approximately 2e-9 ELBO / 1.2e-7 ELCBO.
B has one larger difference: Correlated D5 replicate 0 gains 0.00002239 ELBO
and 0.00002265 ELCBO against C. This is a small local score difference;
Monte Carlo entropy uncertainty has not been estimated.

The consequential differences are in noisy Rosenbrock D2. Each triple below
is optimizer replicate 0, 1, 2; all deltas use the shared unpenalized scoring
rule, and positive values favor the named arm over C.

| Saved state | Comparison | Delta ELBO | Delta ELCBO (beta 5) |
|---|---|---|---|
| Noise 1, source seed 0 | A - C | -0.000777, -0.000846, -0.000763 | +0.004012, -0.007445, +0.003688 |
| Noise 1, source seed 0 | B - C | +0.012973, +0.019084, +0.017610 | +0.014411, +0.010173, +0.022797 |
| Noise 3, source seed 17 | A - C | within 2e-9 of zero | within 4e-8 of zero |
| Noise 3, source seed 17 | B - C | +0.103479, +0.078882, +0.073168 | +0.088250, +0.051817, +0.044682 |
| Noise 3, source seed 7 | A = B; either minus C | +0.189329, +0.088180, +0.381403 | -0.254886, -0.854698, +1.220840 |

B's gains on the first two states coincide with substantially more optimizer
iterations: 160/180/180 versus A/C's 40 on noise 1; all 400 available iterations
versus A/C's 40/60/100 on noise 3 seed 17. The extra raw-eta penalty changes
the stopping behavior as well as the objective. These results do not isolate
a useful weight restriction from a benefit of simply optimizing longer.

Noise 3 seed 7 has no active eta penalty in any arm, and A equals B exactly.
Their difference from C therefore concerns removal of caller-theta mutation
and its numerical/optimizer consequences. It cannot be evidence for removing
an active penalty. ELBO improves in all three replicates but ELCBO worsens in
two because the reported GP uncertainty also changes. The traces retain the
different stopping times (A/B: 40/60/40; C: 80/40/60) for further diagnosis.

Five paired comparisons have unequal output K across A/B/C; six have unequal
K when the source `pre` VP is included. Their entropy draws therefore do not
have identical shapes. All score variances are finite and nonnegative, but
reported SD is GP-only; no posterior-accuracy or population-equivalence claim
follows from these selected states and three optimizer RNG replicates.

As a rough entropy Monte Carlo noise indicator, rescoring the unchanged source
VP across the three diagnostic seeds gives ELBO ranges of 0.00427 nats for
noise 1 seed 0, 0.00892 for noise 3 seed 17, and 0.00487 for noise 3 seed 7.
These ranges are not uncertainty estimates for the paired arm differences.

## Decision and next scoped experiment

No production treatment is selected. The local evidence gives a reason to
investigate B's longer optimization, but does not establish that its raw-eta
penalty is desirable. Raw eta also depends on an arbitrary common offset;
consistent differentiation does not by itself justify that restriction.

The next useful local comparison is to hold iteration counts/stopping fixed
on the two substantially bound-active noisy states, comparing A and B from
these same saved starts and retaining fixed-step checkpoints. Also vary common
raw-eta offsets representing the same posterior, and rescore checkpoints with
several independent diagnostic seeds. This tests whether B's score gains
survive equal optimization effort, arbitrary eta offsets and scoring noise.
First use the retained traces to specify this ablation, and explicitly scope
any additional fits; none are launched or scheduled by this pickup.
Natural active-relative-bound cases and whole-trajectory consequences remain
separate coverage gaps before a production decision. The parked boost campaign,
Q1/Q4 decisions and PyTorch feasibility decisions are unchanged.

The artifact audit verifies 24 unique completions, 72 arms/traces, zero failures,
matching input/candidate/post-sieve RNG hashes, consistent pruning counts and
exact agreement of compact scores with raw scores. A resume validation reports
24 skipped-complete and zero new fits (`resume_validation.log`, 0.222 s).
The output config deliberately pins source and HEAD: after committing, use a
new output directory for a newly authorized experiment instead of overwriting
this completed evidence or bypassing its config check.

Independent Sol implementation/artifact and fresh-context scientific reviews
are complete with no remaining findings. The 35-test mathematical gate,
instrumentation checks, complete artifact audit, resume check and repository
pre-commit hooks pass. Production files are unchanged; a full production test
suite or whole-trajectory campaign was not rerun for these dev-only additions.
