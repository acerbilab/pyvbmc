# PyMC setup budget and reuse of setup evaluations

This preapproval investigation informs the
[adapter plan](../plans/pymc-target-adapter.md). The package and the historical
`dev/scripts/pymc_feasibility.py` prototype are unchanged. The separate runner
is `dev/scripts/pymc_setup_probe.py`; raw outputs are inventoried in
`dev/scripts/runs/LOCAL.md`.

## Part A: initial findings, recorded before Part B

The eleven specified models were exercised with the adapter-coordinate log
density, its joint compiled gradient, and its exact Hessian. A search call
costs one unit and supplies one reusable density observation when finite; a
Hessian costs D units and supplies no additional observations. Rejected line
search trials count. Reference searches allow 2000 calls, with `ftol=1e-14`
and `gtol=1e-8`; the candidate cap is `19 + 4D` calls.

The cap meets the 0.3 marginal-SD centre criterion on all eight models with
positive-definite reference curvature. Only the badly conditioned regression
uses the full cap: 39 calls against 75 for the reference, a maximum marginal
shift of 0.253 SD and a density gap of 0.0321 nat. Its Hessian is constant,
so its widths agree exactly. The bounded model has a ridge and no invertible
Laplace covariance; centered eight-schools has an unbounded density as its
scale collapses. Neither admits the stated Hessian-distance yardstick.

Among tested stopping rules, an accepted-step improvement of at most 0.001
nat is the loosest absolute threshold that passes the centre criterion on
all eight models. Relative tolerance `1e-7` also passes; `1e-5` stops too early
on the varying-intercept model. A projected-gradient tolerance of 0.1 passes
uncapped, but the badly conditioned regression needs 70 calls. Absolute
improvement is invariant to additive log-density constants and is the
provisional choice for Part B, with the cap enforced by the objective wrapper
and `ftol=1e-14`, `gtol=1e-8` as numerical termination safeguards.

All three tested prior intervals flag only `tau_log__` in centered
eight-schools, for both capped and reference searches. Use the widest,
0.5%–99.5%, as a provisional location guard. This is evidence from the named
models, not a general guarantee: sufficiently informative data can put a
valid posterior outside a prior quantile interval. The custom likelihood Op
raises `NotImplementedError` during gradient construction and uses one
density finiteness check with prior widths.

There is a concrete counterexample to unconditional relocation. With
`x ~ Normal(0,1)` and one observation `10 ~ Normal(x,0.1)`, the posterior is
Normal with mean `1000/101 = 9.90099` and SD `1/sqrt(101) = 0.09950`.
Its mode lies far outside even the widest tested prior interval (about
`[-2.576,2.576]`). Moving that valid mode to the prior midpoint and using
prior widths loses the likelihood's location information. The quantile test
is therefore useful as a diagnostic, but **unconditional relocation is not
supported as a general default**. A candidate revision is to require unusable
curvature as well as an outlying location before relocating, and otherwise
report the location flag. This preserves the observed centered-eight-schools
fallback and leaves a well-defined shifted mode intact. The PI should settle
that behavior before implementation.

The probe requires a positive-definite precision matrix before using its
inverse for marginal variances. Positive diagonal entries of an indefinite
inverse do not establish a valid covariance. The bounded ridge uses prior
widths; centered eight-schools uses curvature and location fallbacks.

Part B uses the capped 0.001-nat setup path and compares discard, in-box
`f_vals`, and all finite setup observations seeded directly into the logger.
Its total caps are 100 units for the D=3 vector model and D=5 regression,
150 for non-centered eight-schools (D=10), including setup's Hessian charge.
Successful arms must spend exactly the same cap. Stability is recorded but
does not terminate an arm early; only the final acquisition batch is
shortened to avoid overshooting. `min_fun_evals` equals the remaining
fresh-call budget and `min_iter=0` allows termination when that budget
is reached. Five seeds per model and arm were run.
NUTS references use four sequential chains, 2000 tuning and 4000 retained
draws per chain, dense mass adaptation and target acceptance 0.95; acceptance
requires rank R-hat <1.01, bulk and tail ESS >=1000, and no divergences.

## Status

Part A, the three NUTS references and all 45 reuse fits are complete.
The focused follow-up is also complete: 20 additional attempts, 18 returned
fits and two numerical failures, compared with ten saved baseline fits.
The findings have been discussed with the PI and incorporated into the
consolidated plan. The PI approved implementation on 2026-09-16; the plan's
live checklist tracks execution. The historical experiment results below
are unchanged by subsequent design-review decisions.

## Setup measurements

The selected rule gives the following setups. The reference is a tight local
search from the same initial point, not a proof of a global optimum.

| Model | D | Search calls | Setup units | Largest centre shift / marginal SD |
|---|---:|---:|---:|---:|
| scalar | 1 | 3 | 4 | 0 |
| positive | 1 | 6 | 7 | 0.00010 |
| vector regression | 3 | 11 | 14 | 0.00006 |
| bounded ridge | 3 | 7 | 10 | undefined |
| one-sided intervals | 2 | 6 | 8 | 0.0107 |
| centered eight-schools | 10 | 59 | 70 | undefined |
| non-centered eight-schools | 10 | 14 | 24 | 0.0100 |
| badly conditioned regression | 5 | 27 | 32 | 0.2537 |
| correlated logistic regression | 20 | 20 | 40 | 0.0662 |
| non-centered varying intercept | 20 | 22 | 42 | 0.0298 |
| custom Op without gradient | 1 | 0 | 1 | undefined |

The badly conditioned regression has five normal coefficients, 80 observations
with known noise SD 0.7, and columns with means/scales `(1, constant)`,
`(2, 0.3)`, `(20, 3)`, `(200, 30)`, `(0.02, 0.003)`. Its posterior and
evidence are analytic. Logistic regression has 200 observations, 20
coefficients and correlated covariates (population correlation 0.85).
The varying-intercept model has 18 groups with ten observations each and
two population parameters. Eight-schools uses the conventional eight
observations and their known SDs, a `Normal(0,5)` population mean and
`HalfCauchy(5)` scale. The runner fixes each simulated dataset separately
from the inference seed.

The selected box widths are 0.9876–1.0035 times the reference marginal SDs
over the eight usable-curvature models. Compilation is a separate cost:
on Python 3.12.6, PyMC 6.3.2, PyTensor 3.3.1 and its numba linker, building
the derivative functions takes 0.3–2.8 seconds per model, the first search
including gradient JIT 0.7–4.4 seconds, and the first Hessian call including
JIT 3.1–22.1 seconds. Warm capped searches take 0.5–9.4 milliseconds on
these cheap models. The evaluation-unit budget does not imply negligible
construction latency. The experiment's reference searches and diagnostic
Hessians are additional measurement work; they are not part of the proposed
setup algorithm's budget.

A supplemental check compiles just the one-off Hessian with
`mode="FAST_COMPILE"`, leaving the repeatedly called target and gradient
on the default linker. Compilation plus first evaluation takes 0.071–0.911
seconds across the ten differentiable models, and the matrices agree
with the default-linker results to a maximum scaled difference of
`3.8e-16`. This is a promising implementation option for construction
latency. It was measured after the reuse campaign and did not change its
setups. The timing is a local sequential measurement, not a benchmark of
all PyTensor graphs or a fresh-process cold-cache comparison.

## NUTS references

All references meet the prespecified diagnostic gate. The seed is 7341;
each reference has 16,000 retained draws. Diagnostics cover the adapter
coordinates, including log scales. Split-chain comparisons use the first
two chains against the last two to indicate reference Monte Carlo variation.

| Model | Maximum R-hat | Minimum bulk ESS | Minimum tail ESS | Divergences | Fit seconds |
|---|---:|---:|---:|---:|---:|
| vector | 1.00058 | 12,717 | 9,799 | 0 | 31.1 |
| non-centered eight-schools | 1.00093 | 10,905 | 8,084 | 0 | 40.5 |
| badly conditioned regression | 1.00091 | 23,514 | 10,816 | 0 | 46.8 |

The badly conditioned regression reference also agrees with its analytic
posterior: maximum marginal mean discrepancy 0.0070 SD, marginal SD ratios
0.9928–1.0078.

Eight-schools also admits an independent evidence check. Conditional on
`tau`, integrating all school effects and the population mean gives
`y ~ Normal(0, diag(observation_sd**2 + tau**2) + 25 * ones(8,8))`.
Integrating this density against the `HalfCauchy(5)` prior gives
`ln Z = -31.311347352292785`. Quadrature tolerances `1e-10` and `1e-12`
give identical displayed values; the tighter integration reports relative
error `9.4e-13`. This applies to both centered and non-centered coordinates.
The summary command computes the reference and adds its errors to the
eight-schools rows after fitting.

## Evaluation-reuse protocol and a cached-output defect

Every arm pays for the same setup: 14, 24 or 32 units for the three models.
The remaining budget is fresh density calls. Exact duplicate finite setup
rows are deduplicated before reuse; nonfinite trials are charged but cannot
be supplied as observations. The three arms are:

- `discard`: ordinary initial design, including a fresh evaluation of `x0`.
- `f_vals`: setup points strictly inside the plausible box, highest value
  first; `fun_eval_start=max(D,10,n_reused)`. Cached points replace initial
  design slots; extra points raise the initial design size. The box is
  asserted unchanged by the constructor.
- `logger`: all finite unique setup points added in transformed coordinates
  through `FunctionLogger.add`, plus the ordinary initial design (`x0` supplied
  from the setup and the usual `max(D,10)-1` uniform draws). The plausible
  box and `x0` are independent of these extra observations.

The setup is computed once per fixed model and reused across the paired
experiments; each fit is charged its full setup cost as if constructed
independently. The five seeds change VBMC randomness, not the data or the
setup path. Arm order rotates with seed. The point sets are:

| Model | Total units | Setup units | Fresh calls in every arm | Finite setup points | Points inside box | Worst setup log joint below best |
|---|---:|---:|---:|---:|---:|---:|
| vector | 100 | 14 | 86 | 11 | 7 | 30,999 nats |
| non-centered eight-schools | 150 | 24 | 126 | 14 | 13 | 1,607 nats |
| badly conditioned regression | 100 | 32 | 68 | 27 | 15 | 3,416,188 nats |

The solver's `func_count` retains its existing meaning of fresh calls;
`max_fun_evals` is reduced externally by the setup charge. This measures
reuse with unchanged internal counter semantics. Redefining `func_count`
to include supplied observations would also change solver decisions that
depend on it and is not validated by this experiment. A future interface
should distinguish fresh calls, reused observations and charged setup work.

The `f_vals` arm makes three uniform initial draws for the vector model and
none for the other two, where supplied points already exceed ten. The
logger arm makes nine uniform initial draws for every model. Consequently,
the comparison changes both point retention and initial coverage; it cannot
attribute an arm difference to either factor separately. An interface that
seeds arbitrary training points could also support filtered points with a
full design, a combination this three-arm experiment does not measure.

Both reuse routes encountered an existing presentation defect in
`VBMC.optimize()`: the cached display format expects nine fields, but its
final-boost row passes eight, putting `"finalize"` into a numeric formatter.
It raises `ValueError: Unknown format code 'g' for object of type 'str'`
after inference, even with `display="off"`. The first failed attempts are
retained separately. The measured comparison patches only
`_setup_logging_display_format` to return an empty string in every arm;
the package source is untouched. Repeating the completed discard case with
this bypass gives bit-identical target-call and posterior-draw arrays; the
completed `f_vals` target calls are also bit-identical. These comparisons
are recorded in `logging_bypass_check.json`. A shipped reuse route must fix
this final-row defect.

The logger emulation inserts the starting point through the initial cache
and every other setup point directly, so each stored observation is counted
once. An assertion checks `n_evals == 1` on the complete initial design.
An earlier emulation inserted the starting point twice; its logger runs are
excluded and preserved separately. Completed discard and `f_vals` runs are
carried forward with their original source hashes. Only the `b_final`
comparison is used for conclusions.

Posterior scores use 20,000 VBMC draws in adapter coordinates: a symmetric
KL between the moment-matched Gaussians, mean and covariance discrepancies,
and the mean absolute difference between marginal quantiles from 1% to 99%,
normalized by NUTS marginal SD (the reported marginal W1 approximation).
These complement each other; the Gaussian score alone cannot assess
non-Gaussian shape. ELBO errors use the analytic or quadrature evidence
where available. Five paired seeds assess the three arms; they do not
establish a general initialization policy. Reported run times cover
optimization, posterior sampling/scoring and saving the posterior draws;
they exclude common model construction and setup compilation.

## Reuse results

All 45 fits returned, and all spent exactly their assigned total budget.
Returning a result does not imply convergence. Only the vector model
reached stability: five of five discard and `f_vals` runs, four of five
logger runs. Neither harder model reached stability in any arm. The
[complete tables](../experiments/pymc_setup_probe/summary.md) retain every
seed, including the extreme results.

The posterior-distance column below is the median mean marginal quantile
distance, in reference SD units; smaller is better. Its range matters as
much as its median. ELBO errors are absolute differences from independent
evidence calculations, in nats.

| Model | Arm | Posterior distance, median [range] | Median absolute ELBO error | Median run seconds |
|---|---|---:|---:|---:|
| vector | discard | 0.0113 [0.0100, 0.0194] | 0.00365 | 45.2 |
| vector | in-box `f_vals` | 0.0117 [0.0110, 0.0150] | 0.00333 | 44.8 |
| vector | all-points logger | 0.0143 [0.00958, 0.0150] | 0.00350 | 51.8 |
| non-centered eight-schools | discard | 0.232 [0.142, 40.7] | 0.945 | 160.6 |
| non-centered eight-schools | in-box `f_vals` | 0.439 [0.126, 1.66] | 0.756 | 171.7 |
| non-centered eight-schools | all-points logger | 0.130 [0.121, 0.237] | 0.576 | 165.2 |
| badly conditioned regression | discard | 0.432 [0.407, 1.22] | 3.52 | 21.0 |
| badly conditioned regression | in-box `f_vals` | 0.422 [0.404, 0.434] | 3.42 | 24.8 |
| badly conditioned regression | all-points logger | 0.376 [0.171, 0.391] | 2.95 | 29.1 |

On the vector model, all arms are accurate and their small differences do
not establish a practical reuse benefit. The NUTS split-chain marginal
distance is 0.0283; this is a sampling-variation indicator from smaller
reference subsets, not an error bar on each arm's score. Logger seeding
adds about seven seconds to the median fit.

On eight-schools, all-points seeding has the best median posterior score
and the narrowest error range. Discard seed 3 returns an unusable ELBO
about `5.0e32` nats above the reference and posterior distance 40.7.
Filtered `f_vals` seed 2 returns an ELBO 5.16 nats above the reference and
distance 1.66; seeds 3 and 4 also overestimate evidence by 1.35 and 0.756
nats. Logger errors range from -0.873 to -0.184 nat. These are observations
of unstable short-budget inference, not trustworthy evidence estimates.
The NUTS split-chain distance is 0.0157, far smaller than the median error
of any arm. No algorithmic remedy is investigated here.

On the badly conditioned Gaussian regression, logger seeding improves the
posterior-distance score in every paired seed, at a median eight-second
overhead versus discard. Filtered reuse also avoids discard's worst seed,
but its median posterior error changes little. All three arms still have
substantial covariance error at 100 total units; knowing the exact target
is Gaussian does not make this budget sufficient. The NUTS split-chain
distance is 0.0207. The logger's worst setup point is over three million
nats below its best; retaining it did not prevent improvement here, but
this is insufficient evidence for accepting every distant trial on every
model.

## Design implications for discussion

1. Retain the proposed `20 + 5D` setup cap and provisionally use the
   0.001-nat accepted-step stopping rule. The eight usable-curvature
   models meet the stated centre criterion. This tests local setup
   accuracy on the named models, not global mode discovery.
2. Require positive-definite curvature for a Laplace covariance. Treat
   prior-quantile location as a diagnostic; it cannot independently
   justify relocation. Combining it with unusable curvature preserves
   the observed centered-hierarchy fallback, but remains a heuristic
   requiring an explicit policy and user-visible diagnostics.
3. Prefer an interface that supplies evaluated training points
   independently of `x0`, plausible bounds and the ordinary initial
   design. This supports the most promising tested arm and avoids the
   existing `f_vals` route's coupling between reuse and design coverage.
   The interface and any target convenience method remain PI decisions.
   The focused follow-up below tests density-based pruning separately
   from uniform coverage and supports retaining all finite setup points
   initially, with the ordinary initial design and later warmup pruning.
4. Report setup cost, fresh evaluations and supplied observations
   separately. All arms paid for setup here; Hessian work does not create
   density observations. Preserve existing internal fresh-call counters
   unless a separate algorithmic change is explicitly validated.
5. Fix the cached final-display defect in any shipped reuse route and
   test single insertion of the starting observation. Keep package
   implementation, additional campaigns and GP remedies outside this
   preapproval investigation.

The evidence supports a direction for the reuse interface, with clearer
benefit on the hard cases than on the easy case. Five seeds on three
models, at deliberately short budgets and one fixed setup per model, do
not settle a universal point-filtering policy or performance at normal
convergence budgets.

## Validation and reproducibility

The runner's `check` command passed for all ten differentiable models:
finite-difference gradients have maximum scaled error `4.9e-9`, and
finite-difference Hessians have maximum scaled error `5.8e-10`. Every
selected start and plausible box survives the VBMC constructor unchanged,
without a bound-adjustment warning. Derivative arrays are float64.
The no-gradient model's fallback is recorded in Part A.

Recomputed cached log densities agree within `rtol=1e-12, atol=1e-8`.
Their largest scaled discrepancy is `3.6e-15`. An initial absolute-only
check failed on the centered hierarchy: one extreme search trial has
log joint about `-3.07e18`, and joint value/gradient versus density-only
compilation differs by 1024, exactly two ULPs there. That diagnosis is
retained in `cache_scale_check.json`; the final check uses both relative
and absolute tolerances. No Part B model has this discrepancy.

The shifted-normal counterexample was executed as well as derived:
three search calls reach 9.90099 with positive-definite precision, then
the unconditional location rule moves the start to 0.02990. Its sampled
prior location interval is `[-2.456, 2.529]`. This confirms that the
policy discards a valid mode; checking more prior draws would not fix it.

All 45 unique model/seed/arm combinations pass setup-plus-fresh-budget
and initial-row accounting checks, and all three NUTS references pass
their diagnostic gate. The historical prototype matches its committed
content. AST comparisons confirm that the numerical model, setup,
reference and reuse functions in the final runner match their executed
source snapshots. No package source changed, so a package-wide numerical
regression run was not needed for this developer experiment.

Compact evidence, source hashes and validation records are under
`dev/experiments/pymc_setup_probe/`; raw arrays, complete paths, logs and
excluded provisional runs are listed in `dev/scripts/runs/LOCAL.md`.
All numerical work ran serially with BLAS limited to one thread. The
implementation plan, TODO and experiment index point to these results;
The PI approved Phase 0 and package implementation on 2026-09-16.

## Focused follow-up: filtering and initial coverage

The PI requested a focused comparison before choosing the reuse default.
Code inspection identified an additional coupling in the original `f_vals`
arm: `VariationalPosterior.__init__` initializes its component means from
multiple `x0` rows. Logger seeding uses a single `x0`. Therefore the original
`f_vals` versus logger comparison changes initial component locations as
well as retained observations and uniform coverage.

The PI clarified that relative log density is the relevant filtering
criterion. The follow-up uses the normal warmup pruning rule at setup:
retain `max(y) - y < 10D`, with the best `min(D+1, N)` observations retained
if needed. Here `y` is the adapter log joint, the same quantity stored as
the logger's `y_orig`; it includes Jacobians of retained PyMC transforms.
The false-alarm warmup branch's looser `100(D+2)` threshold and the multiplier
for previous trimming events are not applied to this first setup pruning.
VBMC's ordinary later warmup behavior is unchanged.

The initial follow-up used the box filter. In eight-schools its retained
points and their order are exactly identical to the density rule, so all
ten schools cases can be reused, including failures. In the regression,
box membership keeps 15 points while the density rule keeps 24: the box
discards ten observations only 7–16 nats below the best and keeps one
90.8 nats below it. This confirms that geometric membership is an unsuitable
proxy for relative density. Regression runs use the density rule.

The follow-up uses the logger interface and the same single `x0` throughout:

| Arm | Retained setup rows, schools / regression | Uniform initial draws |
|---|---:|---:|
| All points, full design (existing) | 14 / 27 | 9 |
| Density-filtered points, full design (new) | 13 / 24 | 9 |
| Density-filtered points, shortened design (new) | 13 / 24 | 0 |

The two hard models use their existing datasets, setups, NUTS references,
five seeds and total budgets of 150 and 100 units. Every arm retains the
full setup charge, regardless of how many observations it filters out.
The new arms differ only in the number of uniform initial draws. Their
comparison isolates coverage within the filtered policy. Comparing the
filtered/full arm with the existing all-points/full arm isolates filtering
with full coverage. It does not estimate a filtering-by-coverage interaction
because the all-points/shortened cell is absent.

Twenty attempts are complete under the runner's `coverage` command, with
the equivalent schools cases carried forward. The unchanged
fit implementation accepts one optional initial
design-count override; all other inference settings match the original
logger arm. The easy vector model is omitted because all original arms were
already accurate. The follow-up targets the two cases that motivated the
interface recommendation, and its conclusions have that scope.

### Follow-up results

Eighteen new fits returned and spent their full caps. Two density-filtered,
full-design schools fits failed during GP hyperparameter sampling with
`LinAlgError: Singular matrix for L Cholesky decomposition`, at 123/150 and
143/150 units (seeds 2 and 3). No failed fit is assigned a posterior score,
and the summaries of returned fits must be read alongside the failure counts.
None of the returned hard-model fits, including the saved baselines,
reached stability. The new attempts consumed 19.6 minutes of measured run
time, excluding common setup and analysis.

| Model | Setup observations / coverage | Returned | Median marginal distance [range] | Median absolute ELBO error |
|---|---|---:|---:|---:|
| eight-schools | all / full | 5/5 | 0.130 [0.121, 0.237] | 0.576 |
| eight-schools | density-filtered / full | 3/5 | 0.138 [0.0986, 0.183] | 0.701 |
| eight-schools | density-filtered / shortened | 5/5 | 0.192 [0.145, 0.269] | 0.772 |
| regression | all / full | 5/5 | 0.376 [0.171, 0.391] | 2.95 |
| regression | density-filtered / full | 5/5 | 0.397 [0.377, 0.421] | 3.10 |
| regression | density-filtered / shortened | 5/5 | 0.413 [0.396, 0.546] | 3.56 |

**Pruning at fixed full coverage:** filtering improves marginal distance in
one of the three returned schools pairs, and one of five regression pairs.
It also introduces the two schools numerical failures. This experiment
provides no consistent benefit for pruning setup observations before the
first GP fit, even with a density-based rule.

**Coverage at fixed density filtering:** full coverage improves marginal
distance in two of the three returned schools pairs and four of five
regression pairs. Schools has two additional full-design failures, so its
successful-fit median does not establish a reliability advantage. In the
regression, shortening coverage produces two severe joint posterior errors:
Gaussian symmetric KL 314 and 98 (seeds 2 and 4), versus 3.11 and 3.55 for
the corresponding full designs. Their ELBOs overestimate evidence by
4.00 and 3.91 nats. Marginal distances alone understate those joint errors.

Relative density is the appropriate pruning criterion, but pruning at setup
is a different operation from VBMC's later warmup pruning. The latter runs
after additional observations and GP fitting. All arms retain that ordinary
behavior; this comparison tests adding an earlier pruning step. It does not
test removing or changing normal warmup pruning, and it does not establish
that extreme points are universally useful. Threshold sensitivity and the
filtering-by-coverage interaction remain unmeasured.

**Recommendation for discussion:** seed all finite unique setup observations
independently of `x0` and the plausible box, preserve the ordinary uniform
initial design, and let existing warmup pruning operate at its normal stage.
There is no measured reason to add a default setup-pruning step. Keep setup
cost, fresh calls and reused observations separately visible. The numerical
failures are recorded outcomes; no GP algorithmic remedy was attempted.

The [follow-up tables](../experiments/pymc_setup_probe/coverage/summary.md)
contain every paired difference and individual result. The summary command
checks all assigned caps, exact spending for returned fits, initial-row
counts, identical `x0` and bounds, and bit-identical initial random points
and their density values between the full-design policies. Constructor-only
checks confirm identical initial variational parameters and RNG states for
the two filtered arms at all ten model/seed combinations. Source comparisons
confirm the fit implementation changes only through the explicit initial
design-count override. Carried schools cases retain their original source
hashes and exact retained-row equivalence record.
