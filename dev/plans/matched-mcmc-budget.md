# Evaluations that MCMC needs to match PyVBMC

Created 2026-09-30 on `dev-next`. Status: **not run**. Everything this plan
references is on `dev-next` from the commit that adds it, apart from the
video of step 7, which is on the branch `feat-3d-animation`.

## Purpose

Measure, for a given target, how many target evaluations a good black-box,
gradient-free MCMC sampler needs to reach the posterior accuracy that
PyVBMC reaches. The number supports claims of the form "PyVBMC gets the
posterior from about a hundred evaluations, where MCMC would need N". Such
a claim holds for one target and one setup, so the tool takes any eligible
benchmark target and reports that target's number.

The first use is the narrated video on the branch `feat-3d-animation`
(`docsrc/source/_static/vbmc3d/STORYBOARD.md`, scenes 2 and 12). It plays
one PyVBMC run of a two-dimensional banana and says that MCMC would need
tens of thousands of evaluations. Step 7 replaces that number with this
plan's.

The comparison concerns the posterior. MCMC gives no estimate of the model
evidence, which PyVBMC returns from the same evaluations. A claim that
cites the number says so.

## Where the work happens

- Start from a full clone of the repository (with its tags) at `dev-next`,
  at the commit that adds this plan or later.
- Work on the branch `feat-matched-mcmc-budget`, cut from there, and open a
  PR into `dev-next`.
- Steps 1 to 6 run on a cloud or cluster machine: one machine with several
  cores is enough, and no Slurm is needed. Step 7 runs on the developer's
  machine, where the video is rendered.
- Hand back the PR, the headline line of the report (sampler, metric,
  `N*`, interval, number of replicates), the commit it ran at, the versions
  of step 1 and the wall time of each step.
- The work is done when the report and its summaries are committed, this
  plan's worklog records the runs and its status line is updated. Step 7
  is done separately.

## Definitions

**Targets.** A target is eligible when it is noiseless and its truth is
analytic: an exact `sampler` of the generative process and known
`true_mean` and `true_cov`. That excludes `logreg` and the real-data
targets, whose samplers resample stored draws (`sampler_n_eff` is set).
The `check` subcommand prints the eligible targets of
`dev/scripts/benchmark_targets.py`. PyVBMC receives
`make_problem(name, D, seed=s).vbmc_args()`, and the samplers receive the
same log density, bounds and plausible box.

**The `exporter` configuration.** The run that
`dev/scripts/export_animation_trace.py --target banana` makes, the one the
video plays. PyVBMC runs through that script's own `run(seed, 200)` after
the assignment `export_animation_trace.LOBE_W = 0.0`. The module sets 0.22
when it is imported and only its `main()` changes it, so without the
assignment the target has a lobe between the arms. The setup is the
exporter's: `x0 = (-3, 5.5)`, `plb = (-5, -3.5)`, `pub = (5, 7.5)`, no
bounds, `display` off. The 200 evaluations are PyVBMC's default at `D = 2`,
`50 (D + 2)`. The truth and the samplers' log density come from the
benchmark's `banana` at `D = 2` (`sig1 = 2`, `b = 0.5`), the same density to
rounding (see "Checks"). The runs of this plan are on this configuration;
other targets run on request.

**Accuracy.** Two metrics of the VBMC papers, as `benchmark_targets.py`
computes them.

- For PyVBMC, `metrics(problem, vp, elbo)`: gsKL from the posterior's exact
  moments, and MMTV from `VariationalPosterior.mtv` against 10^5 exact
  draws. The report also gives PyVBMC's evidence error (`elbo_err`).
- For a sampler, `sample_metrics(problem, states, {}, reference=ref)`, with
  `ref = problem.sampler(DIAG_TV_SAMPLES, np.random.default_rng(DIAG_SEED + 1))`
  drawn once. These are the exact draws that `metrics` compares against.
- On the banana the true covariance is diagonal, so gsKL cannot see the
  ridge (the target's `notes` say so), while MMTV sees the skewed marginal
  of the second coordinate. The report gives both side by side.
- An error that cannot be computed is `+inf`. That covers fewer than
  `D + 1` states, a singular covariance, a marginal with a single value and
  a run that raised.

**The reference.** PyVBMC on seeds 0 to 99, every run kept. A run that
raises counts with error `+inf`. The reference accuracy of a metric is its
median over the seeds. The reference budget is the median of `func_count`,
which includes the evaluations trimmed at the end of warm-up. A run chosen
for a figure is never the reference, because choosing it flatters PyVBMC.
The video plays seed 42, the best run of a seed sweep, so seed 42 is not
the reference.

**The samplers.** Each is given what PyVBMC is given: the log density as a
function (`-inf` outside the bounds), the bounds, the starting point `x0`
and the plausible box. No information from the truth or from a
pilot's samples passes into a run, apart from the scalar settings of "The
pilot".

1. **Slice sampling** with `gpyreg.slice_sample.SliceSampler`, the sampler
   PyVBMC itself uses. It is built as
   `SliceSampler(f, x0.ravel(), widths=c * (pub - plb).ravel(), LB=lb.ravel(), UB=ub.ravel(), options={"display": "off", "diagnostics": False, "step_out": s, "adaptive": a})`
   and run with one call `sample(n, burn=B)` per replicate, where `B` is
   the number of adaptation sweeps (0 when `a` is false).
   - `burn` is always given, because its default, `round(N / 3)`, makes the
     run depend on the length requested.
   - `sample` returns only the states after burn-in, and evaluates its
     starting point once at the start of every call, hence the single call.
   - The wrapper around `f` logs every evaluation. Each returned state is
     matched to the index of its last occurrence in the log (see "Checks").
   - The `B` adaptation sweeps count toward the budget, and their states
     are never used.
2. **emcee**'s `EnsembleSampler` with the stretch move and `W` walkers.
3. **zeus**'s `EnsembleSampler` (ensemble slice sampling) with `W` walkers.
4. **Random-walk Metropolis** with a Gaussian proposal of standard
   deviations `c * (pub - plb)`. The report shows its curve for comparison.
   It does not enter the headline.

The ensembles start with their walkers drawn uniformly in the plausible
box, and the other two samplers start at `x0`. The log density passed to
the ensembles is not vectorized, so each call evaluates one point. Every
evaluated point counts, including each walker's initial evaluation and the
stepping out and shrinking of the slice samplers.

**The pilot.** Seeds 1000 to 1009, disjoint from the replicates, after the
reference. The grid:

| Sampler | Settings |
|---|---|
| slice | `c` in {0.1, 0.3, 1}; `step_out` in {false, true}; `adaptive` false (`B = 0`) or true with `B = 20` |
| emcee, zeus | `W` in {4, 8, 16, 32} |
| random-walk Metropolis | `c` in {0.05, 0.1, 0.2, 0.4} |

For each sampler and metric, the runs use the setting with the smallest
`N*` on the pilot seeds, computed as below against the reference. The
pilot's evaluations are not counted, and the report lists the grid with
each setting's pilot `N*`.

**Budgets and replicates.** 100 replicates per sampler and chosen setting,
with seeds 0 to 99.
- Each replicate runs in its own process, which first calls
  `np.random.seed(seed)` and `random.seed(seed)`. zeus draws from NumPy's
  global stream and Python's `random`, and emcee copies NumPy's global state
  when it is built. The slice sampler and random-walk Metropolis get
  `np.random.default_rng(seed)`.
- The walkers of replicate `r` are drawn from a generator of their own,
  `np.random.default_rng(np.random.SeedSequence(r).spawn(1)[0])`. For a general target, replicate `r`
  starts at `make_problem(name, D, seed=r).x0`, the start of PyVBMC's
  seed `r`; for `exporter`, every replicate starts at the exporter's `x0`.
- One long run per replicate goes up to `N_max = 3 * 10^5` evaluations. The
  settings are fixed for the run, so a prefix of it is a run of that
  budget.
- Budgets are 30 points evenly spaced in `log N` from 100 to `N_max`.
- The estimate at budget `N` uses the states whose evaluation index is at
  most `N`. Burn-in discards the states whose evaluation index is at most
  `N / 2`, the first half of the evaluations. An ensemble's states are
  taken by whole steps: a step is kept when its last evaluation falls in
  the kept half, and the initial positions are not states.
- Each replicate also records its errors with 10% and 25% burn-in, so that
  the burn-in rule can change after the runs without rerunning them.

**The matched budget.** For each sampler and metric the report gives the
median error over replicates at each budget, and `N*`. That is the
smallest budget at which the median falls to PyVBMC's median error, by
linear interpolation in `log N` and `log error` between the two grid points
around the crossing. A median of `+inf` counts as above the reference.
`N*` below 100 is reported as "< 100". When the median at `N_max` is still
above the reference, that sampler's runs go on to `10 N_max`. If it is
still above there, the report gives `N* > 10 N_max` as a bound.

The interval comes from 1000 bootstrap resamples. Each resample draws
PyVBMC's seeds and each sampler's replicates with replacement, paired by
seed for a general target, where seed `r` shares `x0` on both sides. It
then recomputes the medians and `N*`, counting a resample without a
crossing as `+inf`. The interval is the 5th to 95th percentile.

**The headline** is the smallest `N*` over the three black-box samplers
(slice sampling, emcee, zeus) and the two metrics. It is the value most
favourable to MCMC. Its interval takes the minimum over those six
combinations inside each resample, so that the choice of the minimum is
part of the interval. A claim quotes the headline with its sampler, its
metric, the number of replicates and its interval, and calls it a median.
The report also shows the curve of exact independent draws, with `N / 2`
draws at budget `N`. It bounds what any sampler can reach under this
burn-in rule.

## Deliverables

- `dev/scripts/matched_mcmc_budget.py`, whose subcommands take
  `--config exporter`, or `--target NAME --D D` for a general target:
  - `check`: the checks listed under "Before the runs", and the list of
    eligible targets.
  - `pyvbmc --seeds 0-99 --out DIR`: one JSON per seed with `func_count`,
    `gskl`, `mmtv`, `elbo`, `elbo_err` and whether the run raised.
  - `pilot --sampler NAME --out DIR`: the grid on seeds 1000 to 1009.
  - `mcmc --sampler NAME --setting ... --seeds 0-99 --n-max N --out DIR`:
    one JSON per replicate with the errors at every budget for the three
    burn-in fractions, and the replicate's evaluation count.
  - `report DIR`: the curves, `N*`, the intervals, the figures, and the
    tables of the report as Markdown.
  - Each subcommand resumes: a seed whose output exists is skipped.
- Everything that `report` reads, committed under
  `dev/experiments/matched_mcmc_budget_<date>/`. That covers the per-seed
  PyVBMC JSON, the per-replicate error curves, the pilot's grid and its
  results, and the figures in `figures/`, so that the report reruns from a
  fresh checkout. The chains themselves are not kept.
- `dev/results/<date>-matched-mcmc-budget.md`. It holds the headline and
  the table of `N*` by sampler and metric with intervals. It also gives the
  reference (medians and quartiles of the metrics and of the evaluations),
  the error curves with the curve of exact draws, the settings chosen and
  tried, the platform and the versions. It states that MCMC gives no
  evidence estimate and that gsKL cannot see the banana's ridge.
- Entries in `dev/README.md` for the script (the "Scripts" list) and the
  report (next to this plan's entry), and this plan's worklog.

## Steps

1. **Environment.** Clone the repository in full, with its tags; a shallow
   clone holds no tags and builds PyVBMC as version 0.1.dev1 (AGENTS.md).
   Check out `dev-next` and create `feat-matched-mcmc-budget`. Then:

   ```console
   python -m venv .venv && . .venv/bin/activate      # Python 3.12
   pip install -e ".[dev]"                           # takes gpyreg >= 1.4.0 from PyPI
   pip install emcee zeus-mcmc                       # for this work only; not PyVBMC dependencies
   git describe --tags --always --dirty
   python -c "import sys, platform; from importlib.metadata import version as v; print(sys.version, platform.platform(), *(f'{p} {v(p)}' for p in ('pyvbmc', 'gpyreg', 'numpy', 'scipy', 'emcee', 'zeus-mcmc')))"
   ```

   If zeus fails to import, or fails in the checks with the environment's
   NumPy, record why in the worklog and go on without it. The headline is
   then taken over slice sampling and emcee.
2. **The script.** Write `dev/scripts/matched_mcmc_budget.py` and run
   `python -u dev/scripts/matched_mcmc_budget.py check --config exporter`
   until it passes.
3. **The reference.**
   `python -u dev/scripts/matched_mcmc_budget.py pyvbmc --config exporter --seeds 0-99 --out dev/scripts/runs/matched_mcmc_budget_<date>/pyvbmc`,
   with the seeds split over parallel processes.
4. **The pilot**, `pilot --config exporter --sampler NAME` for each of the
   four samplers.
5. **The runs**, `mcmc --config exporter --sampler NAME --setting ...` for
   each sampler and each setting that the pilot chose, the seeds split over
   parallel processes. Each process runs with BLAS single-threaded:
   `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
6. **The report.** Run `report`, read it against the checks under "After
   the runs", and write `dev/results/<date>-matched-mcmc-budget.md` from its
   tables. Then copy the summaries into `dev/experiments/`, add the
   `dev/README.md` entries, update the worklog and the status line, commit,
   and open the PR into `dev-next`.
7. **The video** (on the developer's machine, on the branch
   `feat-3d-animation` once it has taken in `dev-next` with this work; the
   files are under `docsrc/source/_static/vbmc3d/`).
   - Rewrite lines `m2`, `m3`, `f2` and `f3` of `narration.json` to match
     the headline, with the rows of `STORYBOARD.md` that quote them and its
     "Claims to check". `m3` and `f3` speak of weeks, which at 3 minutes per
     evaluation needs at least 6,720 evaluations.
   - Add a footnote marked with an asterisk to the MCMC scene and to the
     payoff of `film.html`, along the lines of "\*Median evaluations that
     SAMPLER, a black-box MCMC sampler, needs to match the METRIC of a
     typical PyVBMC run on this posterior (100 runs each). MCMC gives no
     evidence estimate.", with the headline's sampler and metric in place
     of the capitals.
   - The opening draws its chain with `scripts/export_intro.py`, which
     implements only random-walk Metropolis. Extend it to run the headline
     sampler with the report's settings up to the matched budget. A slice
     sampler takes several evaluations per state, and an ensemble moves
     many walkers at once, so which states the opening draws is settled
     with the owner.
   - `film.html` counts the chain's states as its evaluations (`CH.n`, in
     the readout and in the payoff). With any sampler but random-walk
     Metropolis, the count comes from the evaluations instead, which
     `trace_intro.js` then records.
   - Voice, export the events, score and render again (`README.md` of the
     folder, "The film").

## Checks

**Before the runs**, in `check`:
- The exporter's `log_density_vec` with `LOBE_W = 0` and the benchmark
  banana's `log_density_vec` agree to a relative 1e-12 at 10^4 points drawn
  uniformly in the exporter's plausible box.
- For each sampler, on a short run, the wrapper's count equals the
  sampler's own. That is `func_count` for the slice sampler (which counts
  the calls inside the bounds), `ncall` for zeus, `W * (steps + 1)` for
  emcee and `steps + 1` for random-walk Metropolis.
- Every state that the slice sampler returns is found in its evaluation
  log.
- Exact draws give errors that fall with the number of draws, for both
  metrics; the check prints their values at 10^5 draws, the floor of the
  metrics.
- On the `exporter` configuration, for seeds 0 to 4 on the executing
  machine, the tool's `func_count` equals the evaluations that
  `python dev/scripts/export_animation_trace.py --target banana --sweep 0:5`
  prints. The tool also evaluates the exporter's `gskl(vp)` right after
  each run, which must equal the sweep's value to the digits printed. The
  exporter's gsKL uses Monte Carlo moments from 10^6 draws, so it differs
  slightly from the exact-moment gsKL of `metrics`, which the report uses.
  A PyVBMC run reproduces only on the platform that made it, which is why
  both sides of this check run on the executing machine.

**After the runs**, in the report:
- Each sampler's median error at its largest budget is below the reference,
  or the report gives the bound.
- The reference: medians and quartiles, and the number of runs that raised.

## Compute

The banana costs microseconds per evaluation, so the samplers' and
PyVBMC's own overhead set the cost.
- The reference is 100 PyVBMC runs of about a minute each.
- The runs are about 4 samplers x 100 replicates x 3 * 10^5 = 1.2 * 10^8
  evaluations. Most of them run in Python loops, for an hour or two of one
  core, plus the pilot.
- The metrics are about 12,000 evaluations of MMTV on up to 1.5 * 10^5
  states. The kernel density estimate of the reference draws is computed
  once and reused.

Every run is independent, so processes run in parallel. For a target that
costs milliseconds per evaluation, such as `timing`, these budgets would
take CPU-months. Such a target needs budgets and replicates chosen for it
before the runs.

## Open decisions

Defaults that the PI may change before the runs:

- The headline takes the metric that gives the smaller `N*`. The
  alternative is MMTV alone, since gsKL cannot see the banana's ridge.
- The headline takes the best of the three black-box samplers. The
  alternative is slice sampling alone.
- The reference is PyVBMC's median over 100 seeds.
- Burn-in is the first half of each budget. The errors with 10% and 25%
  burn-in are recorded as well.

## Worklog

Nothing has run yet.
