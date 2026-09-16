# PyVBMC 1.5 teaching material

Status: complete and integrated into `dev-next`, 2026-09-16.

Finish the worked model-to-posterior-export tutorial and review existing
teaching material against the settled 1.5 interfaces. The scope and release
constraints are in [TODO](../TODO.md);
the interface contracts are in [Stage 3](stage3-pipeline-features.md).
Work starts from `dev-next` at `3138a73`, on `dev-teaching-material`.
Astra owns teaching design and integration; Sol provides a read-only
consistency review. Keep at most one heavy process running.

## Checklist

- [x] Read the handoff, overview, quickstart, export API pages, Stage 3
  contracts, Examples 7 and 8, and the local environment index.
- [x] Review teaching coverage and choose the model and notebook layout.
- [x] Author a compact Example 9 and add links from the guides and index.
- [x] Correct confirmed inconsistencies in existing teaching material.
- [x] Execute new or changed notebook code and retain outputs; regenerate
  scripts through `examples/scripts/Makefile`.
- [x] Run the pinned formatting hooks, build Sphinx, and inspect rendered
  pages and figures.
- [x] Complete independent verification with the doublecheck skill and
  record outcomes and any remaining release follow-ups.

## Teaching design (approved 2026-09-16)

Use a psychometric response curve with binomial observations. Infer threshold
and positive slope; fix guessing and lapse rates (PI decision). Explain the
response convention and threshold as the midpoint between the asymptotes.
Show the data, adapt a vectorized Torch likelihood, supply a normalized
bounded prior, and fit once with a fixed seed.
Demonstrate Torch posterior samples in original coordinates through a
prediction plot, briefly show differentiable density evaluation, and export
named posterior draws for ArviZ summaries. Include an equivalent short JAX
adapter with x64 enabled and verify its likelihood agrees on fixed inputs.
Use one full Torch fit and a short equivalent JAX likelihood, checking batches
and a single-row input without duplicating the inference run. Prediction
intervals describe uncertainty in response probability, distinguished from
the variability of future observed response proportions.

Teach initial-design batching explicitly: subsequent evaluations have shape
`(1, D)`. Keep framework execution and host conversion visible. Explain
the separate random streams for VBMC and Torch samples, and that ArviZ
summaries describe an approximation rather than validate its accuracy.
Use Example 8 for the fuller PyMC and NUTS comparison workflow.

## Acceptance and prerequisites

The notebook runs from a fresh kernel with declared optional dependencies,
retains its outputs, and is reachable in the rendered documentation. Both
framework adapters execute; exported densities and sample shapes are checked
against the NumPy posterior. Figures and explanatory text agree with actual
outputs. Existing examples and API pages use the settled public interfaces.
Generated scripts match the notebooks and repository formatting.

Execution needs Python >=3.12, Torch, JAX, current ArviZ, a Jupyter kernel,
and Sphinx. Local environments are indexed in the ignored
`dev/scripts/runs/LOCAL.md`; historical run pools are unnecessary.
No numerical core changes, S-VBMC headline changes, GP remedies, or final
release benchmark are in scope. Example 7's bias explanation is unchanged
unless the separate release gate selects shrinkage.

## Initial inspection

The quickstart already documents both adapter contracts and both exports.
Example 7 covers stacking and the current headline; Example 8 covers a
three-parameter PyMC regression, named exports, prediction and a NUTS
comparison. The new notebook should connect the manual framework adapter
to a fitted posterior and downstream calculations in one compact workflow.

Confirmed follow-ups:

- The quickstart's cached-initial-value paragraph points to the lower-level
  logger method. Add the public `VBMC(precomputed_evaluations=...)` route
  and link to the existing reuse guide, distinguishing likelihood values
  from the logger's log-joint values.
- The overview's remaining-release list still describes reference promotion
  as pending. Reconcile it with the completed promotion record and TODO's
  separate final release gate.
- Add purpose and optional-dependency guidance to the examples landing page;
  its current glob list offers no overview of Examples 7, 8 and the new one.
- Clarify that Torch/JAX integration uses user-written target wrappers; the
  packaged model adapter is `PyMCTarget`. JAX coverage is an input workflow;
  the existing posterior exports are Torch and ArviZ.
- Consider explicit summary display in Example 8: its generated script has
  bare ArviZ summary expressions, which are displayed only in a notebook.

Sol's static consistency review found no blocker to Example 9 and confirmed
that Example 7's headline explanation matches the implementation. The docs
and wheel discover notebooks automatically; add `9` to the script Makefile's
example list. This inspection is not the final independent verification.

At initial inspection, the core Python 3.12.6 environment had notebook and Sphinx tools but no
Torch, JAX or ArviZ. The earlier Stage 3 environment contains Torch 2.7.0
CPU, JAX 0.11.1 and ArviZ 1.3.0; its GPyReg 1.2.0 must be replaced or
resolved to the current 1.2.1 sibling before execution.

## Execution and verification

- Created and executed `examples/pyvbmc_example_9_torch_jax.ipynb`: eight
  code cells, two figures, no cell errors. Fixed guessing rate 0.5 and upper
  error rate 0.02; threshold is the 0.74 response-probability point.
- One seeded Torch fit converged after 70 target evaluations, with ELBO
  -28.003805 and reported SD 0.000872. Midpoint quadrature on 200-by-200
  and 400-by-400 grids agreed on log evidence to 2.1e-7 nats; the finer
  grid gave -27.997890. Torch sample means were [0.6281, 2.9002] against
  quadrature [0.6250, 2.9068]; sample SDs were [0.1439, 0.7062] against
  [0.1426, 0.7322]. This is a teaching-example adequacy check, not a
  benchmark or a new accuracy guarantee.
- Torch/JAX likelihoods agree on batch and single-row inputs at 1e-10;
  SciPy independently confirms the normalized binomial likelihood.
  Exported Torch and NumPy log densities agree at 1e-10; gradients are
  finite, samples are float64 and strictly within the bounds, and ArviZ
  dimensions, names and original-coordinate metadata pass checks.
- Regenerated the companion script through `examples/scripts/Makefile`
  using GNU Make 4.4.1. Its syntax tree matches the concatenated notebook
  code. The complete generated script exits 0; its noninteractive run
  emits the expected Matplotlib warnings for `plt.show()`.
- All five repository-pinned pre-commit hooks pass on the changed files.
- A fresh strict Sphinx build passes. Only existing Examples 1–7 duplicate
  labels and Example 2's Plotly MIME warning category are suppressed, as
  in the preceding integration checks. Seven rendered pages, 492 internal
  links/anchors and both notebook figures pass static checks.
- Visually inspected the rendered model equations, prediction plot and
  interpretation, gradient output, ArviZ summary, JAX section and examples
  index. The in-app browser had no backend; an isolated local headless Chrome
  session supplied the final screenshots. Early command-line captures had
  incomplete rendering and were replaced with captures taken after MathJax
  and fonts loaded.
- Examples 7 and 8 remain unchanged. The plain-script treatment of bare
  notebook display expressions in Example 8 follows the existing notebook
  export convention; it does not contradict its interfaces or require a
  teaching correction in this task.

Local execution uses Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1,
Torch 2.7.0 CPU, JAX 0.11.1, ArviZ 1.3.0 and the current GPyReg 1.2.1
sibling source. The Stage 3 interpreter resolves the current checkout,
sibling GPyReg and core environment packages through the temporary kernel's
`PYTHONPATH`; the optional packages come from its own environment. The
initial environment probe lacked `threadpoolctl`; resolving current core
dependencies corrected it before the successful execution. BLAS was
single-threaded and heavy processes ran sequentially. Logs, numerical checks,
execution helpers and screenshots are under the ignored
`dev/scripts/runs/teaching_20260916/`, indexed in `runs/LOCAL.md`.

## Independent review and completion

A fresh-context Sol review through the doublecheck skill checked the model
mathematics, teaching prose, settled interfaces, notebook outputs, script
parity, numerical checks, rendered sections and final documentation records.
Its cleanup findings were resolved: temporary Sphinx notebook copies were
removed, the completed pickup was retired from TODO, and the overview,
roadmap and local artifact index were updated. The final verdict reports
no remaining must-fix or should-fix findings. The reviewer used static
analysis; all execution and builds stayed in the main thread.

Final pinned hooks and `git diff --check` pass. The notebook and generated
script were executed on the recorded Windows/Python 3.12 environment;
cross-platform optional-dependency validation remains part of release
checks. The full numerical suite and CI matrix were not rerun for these
documentation-only changes. No core source, Example 7 headline explanation,
or Example 8 code changed. Final release benchmarking and any shrinkage
headline selection remain separate release work.

The task checklist is retained here as the durable design and verification
record, renamed from its transient `TASK_` filename. All in-scope work is
complete.

## Delivery

Commit, push and integration into `dev-next` were authorized on 2026-09-16.
The development-branch CI path filter excludes this documentation/example
change. The numerical source and CI configuration match the nine-job-tested
`9cc6882` baseline. Delivery uses the completed local notebook, script,
documentation, formatting and independent-review checks above; it does not
dispatch the documentation-publishing workflow.

- [x] Commit and push the teaching feature branch.
- [x] Verify the pushed revision and merge it into `dev-next`, then push
  the integration and confirm a clean, synchronized checkout.

Feature commit `1be3111` passed all five commit hooks and was pushed to
`origin/dev-teaching-material`. GitHub Actions reported no run for that
branch, consistent with the path filters. Remote `dev-next` was still at
the starting `3138a73`; integration fast-forwarded it to `1be3111` and
pushed successfully. The integrated tree equals the verified feature tree,
and the checkout was clean and synchronized with `origin/dev-next` before
this documentation-only delivery record was added.
