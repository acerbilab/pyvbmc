# Equal-iteration eta-bound comparison

Status: complete; artifact and independent scientific reviews passed
(2026-09-08).
Branch: `dev-eta-equal-budget`. Follow-up to the
[initial comparison](2026-09-08-eta-bound-comparison.md).

Subsequent PI decision (2026-09-08): select A for production; explicitly
defer stopping-rule research outside the fix campaign. Implementation and
verification are recorded in [the fix note](2026-09-08-eta-bound-fix.md).

## Question and scope

The initial experiment's raw-eta-bound treatment B often optimized for
160-400 iterations while treatment A stopped after 40. Compare the same
effort to distinguish a benefit of the bounds from longer optimization.
A removes eta-bound regularization; B uses consistent raw-eta bounds.
Both preserve the separate small-weight penalty, location/scale bounds,
learning-rate schedule and stochastic entropy settings. Both avoid mutating
the optimizer's parameter vector inside the objective.

Reuse saved sieve-selected starts for Rosenbrock D2 noise1 seed0 and noise3
seed17, three paired optimizer RNG replicates each. Run both arms for exactly
400 Adam iterations with early stopping disabled: 12 fits in total. Save
40/100/200/400 checkpoints from each continuous run, averaging the latest
20 parameter iterates as Adam does on return. Construct fixed-K posteriors
before midpoint selection and pruning. These checkpoints isolate the local
optimizer and are not full production `optimize_vp` returns.

Rescore checkpoints with ten independent diagnostic streams, reset identically
for each A/B pair, and approximately 100,000 entropy draws per evaluation.
Keep optimization and diagnostic RNGs separate. Compare unpenalized ELBO,
GP-based SD and ELCBO at beta5; reported SD excludes entropy Monte Carlo error.
Repeated scoring measures diagnostic uncertainty for a fixed candidate pair;
three optimizer replicates do not provide population inference.

The original input captures, reference population and production code remain
unchanged. No GP fitting, target evaluations, final boosts, common-eta-offset
fits or whole trajectories are part of this experiment. If B remains
meaningfully better, the next decision is an equivalent-posterior eta-offset
check before choosing a production treatment.

## Preflight

All six saved input files are available under
`dev/scripts/runs/latent_fixes/eta_bound_20260908/comparison/pairs/`.
Actual imports resolve to this PyVBMC checkout and the sibling gpyreg.
The private variants' pinned numerical source hash remains
`118ca558acc7034368d5f8a1c31c8d40c3a73eabda25b3251cebfd27ace0280e`.
All 35 existing mathematical/nonmutation variant tests pass (1.75 s).
Root runs one sequential numerical process; Sol implements and independently
reviews the experiment.

## Results

All 12 fits completed with exactly 400 objective calls each (4,800 total),
48 checkpoint posteriors and 480 diagnostic scores. The first pair took
5.503 s including interpreter startup and 80 scores. The remaining five pairs
took 19.757 s including startup and verified reuse of the pilot pair. Total
successful invocation time was 25.261 s; the Adam work itself took 5.277 s.

**The earlier sizable B advantage disappears when iteration counts match.**
Both arms continue improving their unpenalized surrogate scores with longer
optimization. The eta-bound penalty contributes only small differences in
these states, with inconsistent ELCBO direction.

At 400 iterations, the table gives B minus A after averaging ten paired
diagnostic scores. Positive means a higher surrogate score with B. Replicates
are the three saved optimizer RNG starts, not independent problem instances.

| State | Optimizer replicate | Delta ELBO | Delta ELCBO (beta 5) |
|---|---|---|---|
| Noise 1, seed 0 | 0 | -0.0001733 | +0.0030849 |
| Noise 1, seed 0 | 1 | +0.0004132 | +0.0009537 |
| Noise 1, seed 0 | 2 | +0.0003037 | -0.0009873 |
| Noise 3, seed 17 | 0 | +0.0001949 | -0.0007232 |
| Noise 3, seed 17 | 1 | +0.0004771 | -0.0002326 |
| Noise 3, seed 17 | 2 | +0.0002439 | -0.0003501 |

The earlier B-versus-C ELBO gains were approximately 0.013-0.019 for noise 1
and 0.073-0.103 for noise 3. At equal effort, absolute B-versus-A ELBO
differences across all checkpoints are at most 0.00105; at 400 iterations
they are all below 0.0005. These remaining ELBO effects are practically tiny.
ELCBO differences change sign with state, replicate and checkpoint; no
consistent practical score advantage for raw-eta bounds emerges.

Extending A alone from 40 to 400 iterations raises its mean diagnostic ELBO
by 0.0437-0.0451 on noise 1 and 0.0865-0.0943 on noise 3. Its ELCBO gains
are 0.0488-0.0507 and 0.0746-0.0759, respectively. B has almost the same
gains. These ranges are across optimizer replicates; scoring at distinct
checkpoints uses distinct diagnostic seeds, so they are not common-draw
uncertainty intervals for the temporal gains.

Repeated scoring resolves the small A/B differences: they should not all
be dismissed as Monte Carlo noise. But resolving a tiny difference does not
make it useful. Diagnostic variability is measured conditional on fixed
candidates; it does not capture between-problem uncertainty or GP error.
Complete per-checkpoint paired means, MC standard deviations/standard errors,
ranges and absolute arm scores are retained in
[checkpoints.csv](experiments/eta_equal_budget_20260908/checkpoints.csv).

## Interpretation and next decision

The evidence supports **A: remove eta-bound loss/gradient and avoid mutating
caller theta**, retaining the separate small-weight penalty and other bounds.
The apparent substantial case for B was largely longer optimization, rather
than a useful restriction of raw eta in these states. These checkpoints also
remove midpoint selection/pruning differences from the comparison; they
should not be described as exact replacements for the earlier returned VPs.

There is no compelling B advantage here that calls for the conditional
common-eta-offset experiment. That experiment was not run. A production
treatment still requires PI selection and targeted trajectory verification.
The independent stopping-rule question is whether selected noisy fits stop
too early; these two states do not justify disabling early stopping globally
or imposing 400 iterations on every fit. Higher surrogate scores also do
not establish better true-posterior accuracy. Full integrated benchmark
validation remains necessary after the accepted corrections.

## Verification and reproduction

The source/input/output hashes match; all six starts match candidate 0 after
the production `get_parameters()` normalization sequence. The saved peek-only
theta metadata differs by at most 8.88e-16 on noise 3 because production calls
`get_parameters()` twice. Reproducing those two calls matches actual Adam x0
exactly. A/B starts, schedules and optimizer RNG consumption match exactly.
All checkpoints equal the latest-20 trajectory mean and reconstruct the
retained VP parameters exactly; K stays 19 for noise 1 and 14 for noise 3.
Each A/B diagnostic pair has matching RNG states and draw counts.

Most decisively, **all 12 new objective trajectories exactly match their
historical prefixes**, through the original stopping steps: noise 1 A
40/40/40 and B 160/180/180; noise 3 A 40/60/100 and B 400/400/400.
Thus disabling early stopping preserves the previous path through its stop,
including the entire previous 400-step B paths for noise 3.

Independent Sol static review passes. Root independently audits every raw
capture, source/artifact hash, budget, checkpoint, paired score difference,
RNG alignment and historical objective prefix. The 35 variant tests passed;
production files and original experiment artifacts are unchanged.
Final independent Sol review reproduced all six reports and all 24 checkpoint
summaries/CSV rows exactly, including paired MC variability and gain ranges;
no substantive findings remain. Repository hooks pass. A final resume check
verified six completed pairs and skipped all six with zero new fits.

One resume attempt stopped before computation when a late assertion edit
changed the driver hash after the pilot. The edit was reverted exactly to
the pilot source, verified by SHA-256, and the completed pilot was reused.
No extra optimizer fit or discarded numerical run resulted.

Raw trajectories, checkpoint VPs and scores are under
`dev/scripts/runs/latent_fixes/eta_equal_budget_20260908/`.
[results.json](experiments/eta_equal_budget_20260908/results.json) retains
all compact pair reports, the input/import/source configuration, summaries
and audit evidence. Checkpoint VP statistics inherited from their templates
are not the diagnostic scores; use the separately retained score records.
The configuration pins measured source files and pre-commit HEAD, not a later
commit containing the results. Use a fresh output directory to reproduce
after a source/HEAD change.

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.venv/Scripts/python.exe -u dev/scripts/eta_equal_budget.py --out dev/scripts/runs/latent_fixes/eta_equal_budget_20260908 --limit-pairs 1
.venv/Scripts/python.exe -u dev/scripts/eta_equal_budget.py --out dev/scripts/runs/latent_fixes/eta_equal_budget_20260908
```
