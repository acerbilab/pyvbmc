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
about 48 ms per evaluation. Benchflow's Python port (`benchflow/tasks/timing.py`)
is the source; its value at `(0.15, 0.15, 0.7875, 0.225, 0.035)`,
−4586.122592352263, is checked against the original MATLAB code in
benchflow's tests and pins our port.

| bound | w_s | w_m | mu_p | sigma_p | lambda |
| --- | --- | --- | --- | --- | --- |
| LB | 0.01 | 0.01 | 0.3 | 0.0375 | 0.01 |
| PLB | 0.05 | 0.05 | 0.6 | 0.075 | 0.02 |
| PUB | 0.25 | 0.25 | 0.975 | 0.375 | 0.05 |
| UB | 0.5 | 0.5 | 1.95 | 0.75 | 0.2 |

Data to export from `timing.mat`: the trial matrix (1512 by 6; column 3 is
the stimulus index), the responses, the six stimulus values, the bin size.
The file also holds the paper's MCMC truth for the uniform prior (lnZ
−3859.868, mean, covariance, marginals on 8192-point grids); it is exported
for comparison, not used as the truth (see the evidence section).

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

A new script beside `make_oracle_fixtures.py`, with a `--check` mode that
verifies the pinned density values and the stored files' internal
consistency. Per target:

1. **Space.** PyVBMC's `ParameterTransformer` maps the box to unbounded
   coordinates; the target there is the log joint plus the log-Jacobian,
   which leaves the normalizing constant unchanged and makes the posterior
   closest to a Gaussian mixture.
2. **Samples.** gpyreg's slice sampler in those coordinates, whitened by
   the covariance of a pilot run so the strong correlations (0.9 in timing,
   0.8 in multisensory subject 1) do not stall coordinate-wise moves. Four
   chains from dispersed starts, burn-in discarded, thinned to 50 000
   stored draws in total; effective sample sizes and split R-hat recorded.
   The chains' log-densities are kept for the estimator.
3. **Proposal.** `BayesianGaussianMixture` fitted on the first half of the
   draws, plus a broad component of weight 0.05 with the draws' mean and
   an inflated covariance. Exact sampling and log-density implemented
   locally.
4. **lnZ.** Geyer's estimator between the second half of the chain and
   20 000 proposal draws (the only new likelihood evaluations: 16 minutes
   for timing, seconds for multisensory), standard error from the chain's
   effective sample size (Frühwirth-Schnatter 2004). Cross-checks recorded
   alongside: defensive importance sampling from the same proposal draws
   with its effective sample size, and a Laplace approximation at the MAP
   from a numerical Hessian (the 2018 paper found it within about one
   point). For multisensory subject 1 the estimate must reproduce
   benchflow's −502.479, which used the same prior and pivots.
5. **Stored truth.** One `.npz` per target with the samples, mean,
   covariance and lnZ, and a JSON sidecar with the standard errors, the
   cross-check values, effective sample sizes, R-hat, seeds, chain lengths
   and the generating commit. The `Problem.sampler` hook resamples from the
   stored draws; `ln_Z`, `true_mean` and `true_cov` come from the file.
   MMTV is a marginal metric, so 50 000 draws are ample.

For timing, the new moments are also compared with the paper's stored
uniform-prior truth. The smoke runs below suggest that truth's covariance
is inflated (the file records an IBS setup, so its MCMC may have run on the
noisy likelihood); the regenerated truth settles whether that is so or
VBMC is under-dispersed on this target.

Cost: about 3 to 6 hours of slice sampling for timing, 30 minutes per
multisensory subject, plus the proposal evaluations. One night sequentially
on any machine; truth generation has no replay-baseline constraint, so the
cluster can run the targets and chains in parallel.

## Reference population

The golden reference for the new configurations is generated by the
extension pattern of the
[population plan](final-population-benchmark.md): campaigns of seeds 0–29
per configuration on the benchmark machine with the campaign settings,
then joined to the reference with their manifests, README and the even/odd
null check. `population_run.py` pins the source checkout and refuses a
`benchmark_targets.py` that differs from the prepared candidate, so these
campaigns run from a new frozen checkout that contains the targets and
whose `pyvbmc/` numerics equal the frozen treatment `68a43db`; the oracle
`--check --exact` and `golden_replay.py` reporting `identical` on the
existing configurations certify that before the campaign. Noiseless runs
stop on stability at 130 to 200 evaluations; noisy runs spend the full
budget, 2 to 3 minutes each for timing. Four configurations at 30 seeds
are 4 to 6 hours.

## Work breakdown

- [ ] Export script and the two `.npz` data files with README (benchflow
  `5920788`).
- [ ] Spline-trapezoidal log-density and the three targets in
  `benchmark_targets.py`, with the pinned-value checks in `--check`, the
  plausible-box comment, and the suite entries; `--list` and `--smoke`
  pass.
- [ ] scikit-learn in the `dev` extra; AGENTS.md's extras sentence and
  `dev/README.md` updated.
- [ ] Truth generator with `--check`; overnight generation; validation
  gates: subject-1 lnZ reproduces benchflow's value, the two estimators
  agree within their standard errors, R-hat below 1.01, timing comparison
  with the stored uniform-prior truth written up.
- [ ] Reference campaigns for the four configurations from a certified
  frozen checkout; join to the reference.
- [ ] Record the results and the pickup in the roadmap and `TODO.md`.

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
