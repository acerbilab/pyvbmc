# PyMC setup and evaluation-reuse evidence

Preapproval experiments for the [adapter plan](../../plans/pymc-target-adapter.md),
run from `dev-next` at `741d635` with gpyreg 1.2.1 (`9e70e6b`). The
[report](../../results/2026-09-16-pymc-setup-probe.md) explains the methods,
findings and unresolved design choices. The runner is
[`pymc_setup_probe.py`](../../scripts/pymc_setup_probe.py); the historical
feasibility prototype and the package remain unchanged.

- `part_a.json`: eleven-model setup comparison, reference and capped search
  results, stopping-rule candidates, prior-location flags, and timings.
  Undefined curvature quantities are JSON nulls.
- `setups/`: the selected 0.001-nat stopping rule's starting points, boxes,
  finite evaluated points and values, charged costs and fallback flags.
- `references.json`: four-chain NUTS settings, diagnostic gates and
  split-chain comparisons for the three reuse models.
- `part_b.json`: 45 paired cases, with setup and fresh-call accounting,
  initial-design audits, inference results, posterior distances, stability
  and wall times. Eight-schools evidence errors are added during summarization
  from one-dimensional quadrature after Gaussian marginalization.
- `summary.json` and `summary.md`: arm summaries and paired marginal-distance
  differences. Failed attempts are counted separately from completed fits.
- `checks.json`: independent finite-difference checks of derivative order,
  cached-value checks, constructor-bound checks and the shifted-normal
  counterexample to unconditional prior-quantile relocation; also timings
  and agreement for a Python-linked one-off exact Hessian.
- `cache_scale_check.json`: the two-ULP discrepancy on an extreme centered
  hierarchy trial that requires relative as well as absolute cache tolerance.
- `logging_bypass_check.json`: bit-identical arrays with the probe's
  display-format bypass on the completed comparison cases.
- `numerical_source_check.json`: AST equality of numerical functions
  between the 45-case summary's runner and the executed stage snapshots.
- `sources.json`: summary-runner provenance and hashes of the raw inputs.
  Per-stage reports record their executed source hashes, and carried-forward
  cases retain their individual source hashes.

`coverage/` records the focused follow-up on the two hard models: 20
additional attempts with density-filtered/full and density-filtered/shortened
designs, compared with ten saved all-points/full baselines. Filtering follows
normal warmup pruning (`10D` nats and a `D+1` minimum) at setup; later warmup
behavior is unchanged. Its `summary.md` and `summary.json` show paired
differences, numerical failures and control checks; `comparison.json`
contains all 30 outcomes. Eighteen new attempts returned and two failed.
Schools cases computed under the initial box criterion are valid density-rule
cases because the retained observations and their order match exactly;
`carried_cases.json` preserves that provenance. `initialization_controls.json`
also records the small initial-component differences introduced by the
original multi-row `f_vals` route. The summary independently verifies matching
initial VP and RNG states for the controlled arms.

Detailed search paths, posterior draws, calls, logs, executed source snapshots
and provisional runs excluded from the analysis are gitignored. Their
locations and commands are in `dev/scripts/runs/LOCAL.md` when available.
The final reuse directory is `b_final`; earlier logger runs with a duplicate
starting observation are excluded. Completed discard and `f_vals` runs from
that earlier pass were unaffected and are retained with their provenance.

Use the PyMC environment recorded in the local inventory. Run `a`,
`references`, `b` and `check` sequentially, followed by `summarize`; the
runner's module docstring gives their arguments. Compilation and the long
NUTS references are measurement overhead. The paired inference budgets charge
each model's candidate setup as if constructed independently, plus fresh
target calls. No package implementation has been approved by this experiment.
