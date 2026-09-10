# Noisy-target acquisitions, 2026-09-08 to 2026-09-10: evidence

Machine-readable evidence behind
[the summary note](../../2026-09-08-noisy-acquisitions.md) and its two
reports, the
[acquisition experiments](../../results/2026-09-08-noisy-acquisition-experiments.md)
and the
[search analysis](../../results/2026-09-09-acquisition-search-analysis.md).
Everything was measured in the Claude Code container (four cores, one
BLAS thread per process, several runs sharing the cores unless a file
says otherwise) on the branch `claude/pyvbmc-noisy-acq-funcs-9kwdny`;
each JSON records the commit it ran at under `meta.git`. gpyreg 1.1.0.

## `arms/`: the end-to-end arms

One directory per arm, one JSON per run (`<config>_seed<k>.json`): the
per-run sidecar that `dev/scripts/golden_trace.py run --out` writes, with
the requested and effective options, the run's final metrics (`final`:
`elbo`, `elbo_err`, `gskl`, `mmtv`, `rmse`, `func_count`, `final_N`,
`final_K`, `min_Ns_gp`, `wall_s`, `peak_rss_mb`, ...) and the environment
(`meta`). The `.npz` iteration traces are not committed. The options
that define each arm are in `requested_options`; in short:

| arm | configurations | seeds | options beyond the defaults |
|---|---|---|---|
| `baseline_here` | rosenbrock_D2_noise1, rosenbrock_D2_noise3 | 0–9 | none |
| `refits_off` | same | 0–9 | `active_sample_gp_update = False`, `active_sample_vp_update = False` |
| `gp_off_vp_on` | same | 0–9 | `active_sample_gp_update = False` |
| `gp_map_inloop` | same | 0–9 | `ns_gp_max_active = 0` |
| `var_reduction` | same | 0–9 | `search_acq_fcn = [AcqFcnVIQR(loss="var_reduction")]` |
| `ns_search_2048` | same | 0–9 | `ns_search = 2048` |
| `eig` | same | 0–9 | `search_acq_fcn = [AcqFcnEIG()]` |
| `eig_components` | same | 0–9 | `search_acq_fcn = [AcqFcnEIG(components=True)]` |
| `repeat3` | same | 0–9 | `max_repeated_observations = 3` |
| `combo` | same | 0–9 | `var_reduction` and `max_repeated_observations = 3` |
| `hard_baseline` | logreg_D5_noise3, student_D8_noise3 | 0–5 | none |
| `hard_var_reduction` | same | 0–5 | as `var_reduction` |
| `hard_repeat3` | same | 0–5 | as `repeat3` |
| `hard_combo` | same | 0–5 | as `combo` |

`python summarize_arms.py` (in this directory) regenerates the arm
tables of the experiments report from these files: per configuration,
medians over seeds, the usable fraction (`elbo_err < 1`, `gskl < 1`,
`mmtv < 0.2`), and the paired columns against the baseline arm on the
same seeds (median log-ratio of gsKL, wins, median wall-time ratio).

## `ops/`: operation-level timers

`<config>.json` for seed 0 of the five configurations of the report's
"Operation-level timers" section (rosenbrock_D2_noise1,
rosenbrock_D2_noise3, logreg_D5_noise3, student_D8_noise3,
lumpy_D10_noise3): the run summary and the timer totals by call site
(GP fits by site, slice sampler and space-filling design inside them,
both VP optimizations, the sieve call against the CMA-ES calls, GP
prediction inside each, and the VIQR core by operation). The
instrumented core performs the original operations in the original
order, so the trajectories are the default ones. Four of the runs shared
the four cores; the 10-D run ran alone.

## `profile/`: the cProfile run

`summary.json` and `profile.txt` of the `dev/scripts/profile_run.py` run
of rosenbrock_D2_noise1 (seed 0, defaults) behind the report's first
table; the binary `.prof` is not committed.

## `search/`: the acquisition-search benchmarks on saved states

The printed outputs of the search analysis, one file per saved mid-run
state (the states themselves, pickled `VBMC` objects of 20–190 MB, are
not committed; the report names their configuration, iteration, N and
Ns):

- `bench_<state>.txt`: the sieve, CMA-ES (production settings and
  variants) and L-BFGS-B compared on the same state, with evaluation
  counts and times ("The sieve and CMA-ES on saved states").
- `sieve_size_<state>.txt`: the sieve-size sweep with and without a
  local optimizer after it.
- `landscape_<state>.txt`: the basin census (distinct optima reached
  from many starts) and the Monte Carlo overfitting check (values
  refined on the 100-sample set judged on an independent set).
- `refine_na_<state>.txt`: refinement on importance sets of 100, 400 and
  1600 samples, judged on an independent 6400-sample set, five draws.
- `pipeline_<state>.txt`: the cost and judged gain of the pipelines A–F.

`<state>` is `rosenbrock_D2_noise3`, `logreg_D5_noise3`,
`student_D8_noise3` (N = 200), `student_D8_noise3_late` (N = 325),
`banana_D6_noise2`, `lumpy_D4_noise1` or `lumpy_D10_noise3`.
