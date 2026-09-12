# Realistic benchmark targets from benchflow

Execution plan for extending the benchmark target suite
(`dev/scripts/benchmark_targets.py`) with real-data targets, the follow-up
task recorded in the
[roadmap](modernization-roadmap.md#benchmark-coverage-and-hpc-support).
Opened 2026-09-11 on `dev-next`. The candidates were surveyed in the
lab-private [benchflow repository](https://github.com/acerbilab/benchflow),
cloned as a sibling checkout (`../benchflow`) at `5920788`; the decisions
below are the PI's, taken the same day on the survey and on the smoke runs
recorded at the end of this file.

## Decisions (PI, 2026-09-11)

- **Targets.** Two families, three targets: the Bayesian timing model
  (D = 5) and the multisensory causal-inference model on two subjects
  (D = 6 each). Both are problems of the 2020 noisy-VBMC paper
  (`papers/acerbi2020variational_*.md`, Section 4.1 and Table S2), and
  both are pure NumPy/SciPy with a small data file. The Goris neuronal
  model, the real problem of all three VBMC papers, stays out for now: its
  likelihood runs only through the MATLAB engine, a port would need
  MATLAB to validate, and its evaluation cost would dominate every
  campaign. Benchflow keeps its data and MCMC truths for a later item.
- **Data format.** The benchflow `.mat` files are not copied. A one-off
  export script reads them from the sibling checkout and writes one plain
  `.npz` per family (float64 arrays, loaded with `allow_pickle=False`),
  with a README recording the source files, the benchflow commit, the
  subject indices and the column meanings, in the style of the fixture
  directories' `FIXTURES.md`.
- **Prior.** Spline-trapezoidal on every parameter, with the pivots at
  the plausible bounds and the outer edges at the hard bounds, the prior
  PyVBMC's documentation recommends. The 2020 paper used uniform priors;
  the change is deliberate (better-posed posteriors, no mass piled
  against hard bounds) and does not change difficulty in any measurable
  way (see the evidence section). The density is implemented inside the
  targets module. It is the same formula as `pyvbmc.priors.SplineTrapezoidal`,
  but the target must not depend on the package under test.
- **Plausible boxes.** The paper's, verbatim from Table S2, including the
  timing model's lower plausible bound of 0.05 on the motor Weber fraction
  although that parameter's posterior mass lies below it (median 0.031).
  Benchflow lowered the bound to 0.02. The paper's value is kept on
  purpose, as the imperfect box a modeller would set, and with the spline
  prior it also shapes the prior's taper over that parameter. The code
  carries a comment saying so.
- **Configurations.** Noiseless: multisensory subject 1 only, the harder
  of the three posteriors. Noisy: all three, with emulated Gaussian noise
  through the existing `noise_sd` wrapper at the 2020 paper's noise levels
  (the IBS noise at the MAP): 2.2 for timing, 1.3 for both multisensory
  subjects, at the paper budget of 50 (D + 2) evaluations like every other
  noisy entry. The logistic-regression stand-in stays in the suite.
- **Ground truths.** Regenerated for all three targets on the deterministic
  likelihoods, in one overnight session: MCMC samples for the marginals
  and moments, and the log normalizing constant by Geyer's reverse
  logistic regression (which, with one normalized proposal, is the optimal
  bridge sampling estimator) against a variational Gaussian mixture fitted
  to the samples plus a broad defensive component, the method of the 2018
  paper's neuronal truths. scikit-learn's `BayesianGaussianMixture`
  provides the mixture and joins the `dev` extra; the estimator code takes
  any mixture with sampling and a log-density, so a Python port of the
  lab's `vbgmm` could replace it later without touching the estimator.
- **Public tests.** Which of these targets and data move into
  `pyvbmc/testing` is decided at the pre-release documentation review.
  The `.npz` files and READMEs make that a copy plus `MANIFEST.in` lines.

## Targets

### `timing` (D = 5)

Bayesian time-interval reproduction model of Acerbi, Wolpert and
Vijayakumar (2012), fitted to one subject of Experiment 3 (uniform interval
distribution, subject 2): 1512 trials over six intervals from 0.6 to 0.975 s,
responses discretized at 0.02 s. Parameters, in order: sensory Weber fraction
`w_s`, motor Weber fraction `w_m`, prior mean `mu_p` (s), prior SD `sigma_p`
(s), lapse rate `lambda`. The likelihood integrates the Bayesian observer's
response distribution numerically per interval (grids of 101 by 401 points),
about 40 to 50 ms per evaluation. Benchflow's Python port (`benchflow/tasks/timing.py`)
is the source; its value at `(0.15, 0.15, 0.7875, 0.225, 0.035)`,
−4586.122592352263, is checked against the original MATLAB code in
benchflow's tests and pins our port.

| bound | w_s | w_m | mu_p | sigma_p | lambda |
| --- | --- | --- | --- | --- | --- |
| LB | 0.01 | 0.01 | 0.3 | 0.0375 | 0.01 |
| PLB | 0.05 | 0.05 | 0.6 | 0.075 | 0.02 |
| PUB | 0.25 | 0.25 | 0.975 | 0.375 | 0.05 |
| UB | 0.5 | 0.5 | 1.95 | 0.75 | 0.2 |

Data exported from `timing.mat` (layout in `dev/scripts/data/README.md`):
the 0-based interval index of each trial, the responses, the six interval
values, the bin size and the four bound vectors. The file also holds the
paper's MCMC truth for the uniform prior (lnZ −3859.868, mean, covariance,
marginals on 8192-point grids); it is exported for comparison, not used as
the truth (see the evidence section).

### `multisensory_s1`, `multisensory_s2` (D = 6)

Visuo-vestibular unity-judgment model of Acerbi, Dokka, Angelaki and Ma
(2018): the "Fixed" causal-inference rule (same source if the noisy
measurements differ by less than `kappa`) with a lapse, three visual
coherence levels. Subjects 1 and 2 of the 2020 paper are benchflow's
subject indices 0 and 1 (1069 and 857 trials). Parameters in the paper's
order: `sigma_vest`, `sigma_vis` at low, medium and high coherence,
`kappa`, `lambda` (benchflow orders them differently). The likelihood is
analytic and vectorized over parameter rows, 0.6 ms per evaluation and
10 ms per 100 rows. Source: `benchflow/tasks/multisensory_6D.py`. Pin: the
log joint under the spline prior at benchflow's stored mode for subject 1,
−503.4863062430452 (benchflow's pivots equal the paper's, so the value
applies as is).

| bound | sigma_vest | sigma_vis (3) | kappa | lambda |
| --- | --- | --- | --- | --- |
| LB | 0.5 | 0.5 | 0.25 | 0.005 |
| PLB | 1 | 1 | 1 | 0.01 |
| PUB | 40 | 40 | 45 | 0.2 |
| UB | 80 | 80 | 180 | 0.5 |

Data to export from `acerbidokka2018_data.mat`: per subject and coherence
level, the vestibular direction, the visual direction and the response
(columns 3 to 5 of benchflow's arrays; the first two columns are trial ids
and a constant).

### Suite entries

Labels follow the existing scheme. Golden suite additions:
`multisensory_s1_D6`, `timing_D5_noise2.2`, `multisensory_s1_D6_noise1.3`,
`multisensory_s2_D6_noise1.3`. Whether the profile suite takes one of them
is an implementation choice.

## Ground-truth generation

`dev/scripts/make_benchmark_truths.py` (its docstring is the usage
reference) generates the truths; its `--check` mode reloads a target's
files, verifies the stored moments against the stored weighted draws,
re-evaluates the target at a hundred stored draws against their stored
log densities, and applies the gates below. The pinned density values are
checked by `benchmark_targets.py --check`. Per target:

1. **Space.** PyVBMC's `ParameterTransformer` (probit, as VBMC uses) maps
   the box to unbounded coordinates; the target there is the log joint
   plus the log-Jacobian, which leaves the normalizing constant unchanged
   and makes the posterior closest to a Gaussian mixture.
2. **MAP, Laplace, whitening.** A multi-start search for the MAP in those
   coordinates, a finite-difference Hessian there, the Laplace estimate of
   lnZ from it, and the Cholesky factor of its inverse as the whitening of
   the sampling coordinates, so the strong correlations (0.9 in timing,
   0.8 in multisensory subject 1) do not stall coordinate-wise moves.
3. **Samples.** gpyreg's slice sampler in the whitened coordinates. Four
   chains from dispersed starts around the MAP, burn-in discarded,
   thinned; effective sample sizes and split R-hat recorded; the chains'
   log densities kept for the estimator. Each chain is saved to
   `data/truths/chains/` (gitignored) in chunks as it runs; a completed
   chain is reused by a rerun with the same settings, MAP and whitening,
   an interrupted one is resampled, so the granularity of a resume is one
   chain.
4. **Proposal.** scikit-learn's `BayesianGaussianMixture` fitted on the
   first half of the draws (components below weight 0.001 dropped), plus
   a broad component of weight 0.05 with the draws' mean and four times
   their covariance. Exact sampling and log-density implemented locally;
   the mixture is used as a density over the unwhitened coordinates, so
   the estimate is the target's own constant.
5. **lnZ.** Geyer's estimator between the second half of the chains and
   20 000 proposal draws (the only new likelihood evaluations), standard
   error from the second half's effective sample size (Frühwirth-Schnatter
   2004). Recorded alongside: defensive importance sampling from the same
   proposal draws with its effective sample size, and the Laplace
   estimate (the 2018 paper found it within about one point).
6. **Stored truth.** The population is the importance-weighted proposal
   draws, not the chains (PI, 2026-09-12): with the proposal fitted this
   closely, importance sampling reaches effective sample sizes of 17 000
   to 18 000 per 20 000 draws on every target, while the timing chains
   reached 267 to 320 on three dimensions at a thousand times the cost
   per effective draw. One `.npz` per target with the proposal draws in
   the original space, their normalized log weights, their log densities,
   the weighted mean and covariance, the weights' effective sample size,
   lnZ and its standard error; a JSON sidecar with the settings, the
   chains' moments, effective sample sizes and R-hat, the cross-check
   values, seeds, the MAP, the proposal summary, timings and the
   generating commit. Both files are tracked (about 1 MB per target;
   every tracked file also enters the sdist). The `Problem.sampler` hook
   resamples the population with its weights; `ln_Z`, `true_mean`,
   `true_cov` and the effective sample size come from the file.

Gates, checked by the generator: the weights' effective sample size at
least a quarter of the proposal draws; Geyer's and the importance-sampling
estimates within three combined standard errors; the chains' split R-hat
and effective sample sizes reported, with a warning above 1.01 or below
400 (they no longer decide anything).
Benchflow's stored constant for multisensory subject 1 (−502.479) is
reported but is not a gate: three estimators sharing no code (Geyer's and
importance sampling in the transformed space, and a defensive importance
sampler in the original space with an effective sample size above 10^5)
agree on −502.19 ± 0.01, while the log joint at benchflow's stored mode
is reproduced to 2e-10, so the stored constant is off by about 0.29 (the
evidence section has the numbers).

For timing, the new moments are also compared with the paper's stored
uniform-prior truth. The smoke runs below suggest that truth's covariance
is inflated (the file records an IBS setup, so its MCMC may have run on the
noisy likelihood); the regenerated truth settles whether that is so or
VBMC is under-dispersed on this target.

Cost, from the dry runs of 2026-09-11: a stored timing draw costs about
0.33 s per unit of thinning (about seven evaluations per sweep), a
multisensory draw about 8.5 ms. The overnight settings: the defaults for
the two multisensory subjects (50 000 draws, thin 5, four chains, 20 000
proposal draws: about 45 minutes each) and 24 000 draws at thin 2 for
timing (5 to 6 hours, plus a minute of MAP search and 15 minutes of
proposal evaluations). One night sequentially on any machine; truth
generation has no replay-baseline constraint, so the cluster could run the
targets and chains in parallel.

## Reference population

The golden reference for the new configurations is generated by the
extension pattern of the
[population plan](final-population-benchmark.md): campaigns of seeds 0–29
per configuration on the benchmark machine with the campaign settings,
then joined to the reference with their manifests, README and the even/odd
null check. `population_run.py` pins the source checkout and refuses a
`benchmark_targets.py` or a `dev/scripts/data/` (the archives and the
truths) that differs from the prepared candidate, and refuses an unclean
checkout, so these campaigns run from a new frozen checkout that contains
the targets and the committed truths and whose `pyvbmc/` numerics equal
the frozen treatment `68a43db`; the oracle `--check --exact` and
`golden_replay.py` reporting `identical` on the existing configurations
certify that before the campaign. `golden_trace.py run` refuses a
configuration whose target has no truth, so a population cannot be
recorded with NaN metrics.

Done 2026-09-12: the campaign ran from a detached worktree at `fc50ee1`
(gpyreg `a2f8ddc`) after both gates passed from it (11 of 11 oracles
bit-exact; the five default replays identical against the `68a43db`
traces), 10:11 to 16:17 local, 6 h 05 min, and was joined as
`reference_990_20260912` with `dev/scripts/reference_join.py`; the record
with manifests, validation, null check and final replay is
[realdata_extension_20260912](../golden/realdata_extension_20260912/README.md),
the results are in the evidence section below. The noiseless runs stop on
stability at a median of 162 evaluations and the noisy ones at 205 to 215,
inside their budgets; a timing run takes about 4 minutes, of which 13
seconds are target evaluations.

## Work breakdown

Code work runs on `dev-benchmark-targets` (branched 2026-09-11 from
`dev-next` at `f5764a5`).

- [x] Export script and the two `.npz` data files with README (benchflow
  `5920788`): `dev/scripts/export_benchflow_data.py`, `dev/scripts/data/`.
- [x] Spline-trapezoidal log-density and the three targets in
  `benchmark_targets.py`, with the pinned-value checks in `--check`, the
  plausible-box comment, and the suite entries; `--list` and `--smoke`
  pass (timing pin bit-exact against the MATLAB-derived value, the
  multisensory log joint within 2e-10 of benchflow's mode value, a
  regression pin for subject 2; `--only` filters `--smoke` too and
  rejects an empty selection). The shipped test that pins the golden
  suite lists the four new labels; `golden_trace.py run` refuses targets
  without a truth; `population_run.py` pins `dev/scripts/data/` as well.
- [x] scikit-learn in the `dev` extra; AGENTS.md's extras sentence and
  `dev/README.md` updated.
- [x] Truth generator with `--check`: written and reviewed, reproduces the
  logreg and halfnormal constants in quick mode. Overnight generation ran
  2026-09-11 23:56 to 2026-09-12 08:08 on the development machine
  (launcher and log in the ignored `dev/scripts/runs/truths_20260911/`),
  then the populations were regenerated as importance-weighted proposal
  draws with the chains reused (08:22 to 08:37). Both `--check` passes
  green; results and the timing comparison in the evidence section;
  truths committed under `dev/scripts/data/truths/`.
- [x] Reference campaigns for the four configurations from a certified
  frozen checkout; join to the reference (2026-09-12, see the reference
  population section and the evidence below).
- [x] Record the results and the pickup in the roadmap and `TODO.md`.

Local-only artifacts on the development machine (gitignored): the chain
files under `dev/scripts/data/truths/chains/`, which let a rerun of the
generator with the same settings redo only the proposal stage (a fresh
clone resamples the chains, about 7 hours), and the launcher, log and the
superseded first-format truth files under `dev/scripts/runs/truths_20260911/`.
The change to the importance-weighted population (loader, generator
output contract, gates) was tested by both `--check` passes and the smoke
of two configurations, but not by a second independent review round.

## Evidence: the overnight generation of 2026-09-11/12

Settings: the two multisensory subjects with the defaults (50 000 draws,
thin 5, four chains, 20 000 proposal draws), timing at 24 000 draws, thin
2; seed 0; started 23:56, finished 08:08 on the development machine.

| target | Geyer lnZ | importance sampling | Laplace | chains' max R-hat | chains' min ESS | IS ESS of 20 000 | wall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multisensory_s1 | −502.1859 ± 0.0027 | −502.1861 ± 0.0023 | −502.815 | 1.0008 | 8160 | 18 100 | 42 min |
| multisensory_s2 | −444.4889 ± 0.0023 | −444.4899 ± 0.0027 | −444.969 | 1.0002 | 23 826 | 17 400 | 33 min |
| timing | −3861.3093 ± 0.0027 | −3861.3096 ± 0.0023 | −3861.437 | 1.0159 | 267 | 18 100 | 6 h 57 min |

The two estimators agree within a fraction of a standard error on every
target. The timing chains cost 0.78 s per stored draw and mixed poorly on
`w_s`, `w_m` and `sigma_p` (the three parameters with correlations of
0.9), which is why the stored population is the importance-weighted one.
Benchflow's constant for subject 1 is confirmed 0.293 below the estimate,
at more than a hundred standard errors.

The stored populations (regenerated from the same chains and proposal
draws on 2026-09-12 at 08:22 to 08:37, chains reused) agree with the
chains' own moments: largest mean discrepancy 2.5, 3.0 and 0.65 chain
standard errors for subject 1, subject 2 and timing, SD ratios within
1.2 %, 0.9 % and 2.1 %. No proposal draw fell where the target density is
zero. Both `--check` passes are green.

Timing against the paper's stored uniform-prior truth: the marginal SDs
of `w_s`, `mu_p` and `sigma_p` are reproduced within 1 %, 4 % and 3 %
(ratios 0.99, 0.96, 0.97); `w_m` is 15 % narrower and `lambda` 37 % wider,
the two parameters the spline prior's taper acts on, and the means move
by 0.5 SD on `w_m` and 1.5 SD on `lambda` for the same reason. So the
paper's truth was not inflated. The VBMC posteriors of the smoke runs,
with SDs of 0.0051 to 0.0054 on `w_s` and 0.0091 to 0.0096 on `sigma_p`
against 0.0075 and 0.0126 here, are about 30 % under-dispersed on those
two parameters, which accounts for their steady gsKL of 0.23 and ELBO gap
of 0.2. That is a finding about VBMC on this target, to be examined on
the reference population, not a defect of either truth.

## Evidence: the reference campaign of 2026-09-12

Seeds 0–29 of the four configurations, campaign settings of the population
plan, one process on the benchmark machine; all 120 runs complete with
finite metrics. "Usable" is evidence error < 1, gsKL < 1 and MMTV < 0.2;
the medians are over the 30 seeds.

| configuration | converged | usable | ELBO − lnZ | gsKL | MMTV | evaluations | optimizer time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multisensory_s1_D6 | 30/30 | 28/30 | −0.39 | 0.51 | 0.11 | 162 | 1.35 min |
| multisensory_s1_D6_noise1.3 | 30/30 | 5/30 | −0.37 | 2.41 | 0.22 | 205 | 3.09 min |
| multisensory_s2_D6_noise1.3 | 30/30 | 22/30 | −0.37 | 0.69 | 0.15 | 208 | 3.32 min |
| timing_D5_noise2.2 | 29/30 | 28/30 | +0.08 | 0.39 | 0.14 | 215 | 4.10 min |

The noisy runs terminate on the reliability index rather than spending
their budgets (one timing run reached its 350). Noisy multisensory
subject 1 is the least accurate configuration of the whole reference in
posterior shape while its evidence error is ordinary; it and the two
other noisy configurations are what the deferred noisy-acquisition work
is assessed on.

Posterior widths, as the ratio of VBMC's marginal SD to the truth's
(median over seeds; quartiles for the two parameters the smoke runs had
flagged):

| configuration | parameters | SD ratio |
| --- | --- | --- |
| timing | w_s, w_m, mu_p, sigma_p, lambda | 0.68 [0.64, 0.78], 0.76, 0.85, 0.72 [0.66, 0.79], 0.91 |
| multisensory_s1 | sigma_vest, sigma_vis (3), kappa, lambda | 0.65, 0.81, 0.83, 0.90, 0.94, 0.98 |
| multisensory_s1 noisy | same | 0.38, 0.66, 0.63, 0.88, 0.94, 0.93 |
| multisensory_s2 noisy | same | 0.76, 0.73, 0.92, 0.95, 0.81, 0.83 |

So the 30 % under-dispersion of `w_s` and `sigma_p` seen in the smoke
runs holds across the 30 seeds, with the ELBO at the true log evidence;
on the multisensory subjects the ELBO sits about 0.4 below lnZ, and the
noise narrows subject 1's vestibular-noise marginal to 0.38 of its true
SD while shifting its mean by 0.6 SD. These are properties of the current
defaults on these targets, recorded in the reference for later work, not
defects of the truths (whose two estimators agree within a fraction of a
standard error on every target).

## Evidence: smoke runs of 2026-09-11

Runs made through benchflow's own task classes before any port, default
options, paper budget, one VBMC run per seed, on the development machine.
ELBO error is |ELBO − lnZ|; gsKL uses the stored moments for timing and,
for multisensory, moments from a 5000-draw slice-sampling chain (effective
sizes 500 to 2500 per dimension), so those values are approximate.

| target and prior | seed | ELBO error | gsKL | evaluations | reliability index |
| --- | --- | --- | --- | --- | --- |
| timing, uniform, paper box | 0, 1, 2, 3 | 0.20, 0.21, 0.22, 0.23 | 0.23, 0.25, 0.25, 0.21 | 190, 130, 155, 160 | 0.05–0.15 |
| multisensory S1, spline | 0, 1, 2, 3 | 0.17, 0.03, 0.51, 0.13 | 0.64, 0.29, 1.20, 0.11 | 155, 180, 200, 165 | 0.12–0.29 |
| multisensory S2, spline | 0, 1, 2, 3 | n/a | 0.26, 0.26, 0.23, 0.21 | 150, 155, 145, 150 | 0.26–0.75 |

Subject 1 is the hardest posterior (gsKL varying tenfold across seeds,
one seed above the usability threshold): broad, skewed visual-noise
marginals leaning on their lower bound and correlations of 0.8 between the
visual and vestibular noise. Subject 2's posterior is different (visual
noise of 14 and 23 degrees at medium and high coherence, lapse rate 0.12,
correlations at most 0.56), steadier in gsKL but with a higher reliability
index at termination. Timing is consistently usable. No ELBO error for
subject 2: benchflow's class stores subject 1's lnZ for every subject.

Other observations from the same session:

- The normalizing constant of multisensory subject 1 under the spline
  prior, from the generator's quick run (2000 draws) and from a separate
  original-space estimate: Geyer −502.194 ± 0.025; importance sampling
  from the same proposal draws −502.158 ± 0.035; defensive importance
  sampling in the original space (half a t(4) around the MAP, half uniform
  over the box, 2·10^6 draws, effective sample size 137 000)
  −502.186 ± 0.003; Laplace −502.815. Benchflow's stored −502.479 is 0.29
  below all three. Benchflow's README records that its constants were
  estimated by Geyer's reverse logistic regression on emcee draws, so the
  discrepancy points at those draws (its reference draws for the timing
  model show the same under-dispersion). Under the uniform prior the
  constant is −504.14 (the stored population reweighted), so the old
  value is not that either. Benchflow stored the constant for every
  subject until 2026-09-12; the correction is its pull request 1.
- Uniform against spline prior on subject 1, seed 0: means within 0.2 and
  SDs within 0.1 of each other in every dimension. The prior choice does
  not change difficulty.
- Timing with benchflow's lowered plausible bound (0.02), seed 0: ELBO
  error 0.23, gsKL 0.29, 165 evaluations, against 0.20, 0.23 and 190 with
  the paper's box.
- Timing's stored uniform-prior SDs are 25 to 35 % larger than both the
  VBMC posteriors' and benchflow's spline-prior samples' for the four
  main parameters (for example 0.0075 against 0.0054 and 0.0055 for
  `w_s`), which would account for the constant ELBO error of 0.2 and gsKL
  of 0.23 across seeds. Under the spline prior with benchflow's pivots the
  lapse-rate marginal moves off its hard bound (mean 0.0113 to 0.0131);
  the other parameters do not move.

## Rejected candidates

- Goris neuronal model (D = 7): MATLAB engine; deferred, see Decisions.
- Acerbi et al. (2015) bisensory model (D = 12 or 21): the MATLAB model
  code is not in benchflow at all.
- Lotka-Volterra from posteriordb (D = 8): needs BridgeStan and a
  posteriordb checkout; a SciPy reimplementation would need the Stan model
  and the data re-entered. The only non-neuroscience option, kept in mind.
- Diamonds from posteriordb (D = 26): beyond the range VBMC is for.
- Morphology (NEURON) and the synthetic funnel, bimodal ring and
  multi-banana: heavy dependency or not real-data targets.
- Ricker, aDDM, rodent 2AFC and g-and-k from the 2020 paper: no code in
  benchflow.
