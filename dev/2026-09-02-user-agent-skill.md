# 2026-09-02 — A PyVBMC skill for users' coding agents

**Status:** proposal refreshed on 2026-09-10 against the implemented 1.5 API.
The proposed skill covers RNG control, vectorized initial evaluation,
posterior exports, machine-local calibration and runtime tips alongside the
core inference workflow.

## Purpose

Help a user's coding agent turn a model into a sensible PyVBMC analysis:
choose the target and bounds, run within the intended evaluation budget,
check the fitted posterior, and explain what the results support. When the
user brings an existing script or failed run, enter at the relevant step and
preserve sound existing choices.

The useful content is the judgment between API calls. An MLE or MAP estimate
can give PyVBMC a good starting point; a precise estimate of a lower bound can
still be far below the true evidence. Connect each recommendation to its
purpose and link to the relevant example or explanation. Introductory advice
should be understandable without knowledge of Gaussian processes or Bayesian
quadrature.

The [baygent-skills project](https://github.com/Learning-Bayesian-Statistics/baygent-skills)
provides a useful precedent: workflow instructions, references and examples
in skill folders. Use that structure, while deriving scientific guidance from
PyVBMC's own documentation and reviewed tips.

## Proposed first version

Maintain one source folder, `skills/pyvbmc/`, in this repository. Use the
[Agent Skills format](https://agentskills.io/specification): `SKILL.md` with
`name: pyvbmc` and a description explaining when to use it, plus resources
loaded as needed. Keep the main instructions comfortably below 500 lines.
Avoid assuming a particular agent's tools or configuration paths.

Deliver:

- A short `SKILL.md` covering applicability, the workflow below, common
  failure modes and links to further guidance.
- Focused references for setup and priors, noisy targets, validation and
  evidence, and optional 1.5 integrations. Adapt relevant FAQ material to
  Python names; a complete FAQ port is not needed for the first version.
- One small runnable deterministic example with a known posterior and log
  evidence, with instructions for adapting it to a user's model. Keep the
  model definition and inference steps visible.
- Installation and compatibility instructions linked from the README and
  user documentation, with focused validation of the delivered artifacts.

Executable helpers are optional. A results summarizer may help interpret
existing outputs. Start setup checks in the workflow and example; add a
standalone validator only if it provides value beyond PyVBMC's own checks.
Any helper that probes a user's model must state its evaluation count and
stay within the user's intended budget. Reading results needs no new model
evaluations.

## The workflow to teach

### 1. Establish the inference problem

Check suitability: continuous parameters, typically up to roughly 10–20
dimensions, and an evaluable, potentially expensive or noisy log density.
Explain alternatives when relevant; an optimization-only request may call
for PyBADS. Do not silently round integer parameters or invent a likelihood
for a simulator.

Identify parameter order, data, prior, support and desired outputs
(posterior estimates, model evidence, or both). Inspect the installed PyVBMC
version before using a 1.5-specific feature. Core PyVBMC requires Python 3.10+;
export dependencies are needed only for their respective workflows.

### 2. Construct the target and starting region

- Include the prior exactly once: pass a log joint directly, or pass a
  log-likelihood with `prior=` or `log_prior=`. Explain the chosen convention.
  For model evidence, retain likelihood normalization constants and use a
  normalized proper prior: constants irrelevant to posterior shape still
  affect evidence.
- The default target accepts one parameter vector in original coordinates
  and returns one finite real scalar. Diagnose non-finite values where PyVBMC
  can evaluate the target. Check numerical stability, support and
  parameterization instead of substituting arbitrary finite penalties.
- Hard bounds encode support; finite plausible bounds guide the initial
  search without restricting the posterior. Use `LB < PLB < PUB < UB`.
  Without better information, each prior's **16th and 84th percentiles** are
  a useful starting choice, roughly mean minus and plus one SD for a Gaussian
  prior. Prefer a better-informed region when available; see the
  [plausible-bounds FAQ](https://github.com/acerbilab/vbmc/wiki#how-do-i-choose-plb-and-pub).
- Choose `x0` near plausible high density. An MLE or MAP estimate found with
  [PyBADS](https://acerbilab.github.io/pybads/) can provide a good starting
  point for PyVBMC. Optimization is an optional aid, not a required first fit.
- For simulated log-likelihoods, set `specify_target_noise=True` and return
  `(log_density, noise_sd)` with a finite, positive SD. It describes noise in
  that log-density evaluation, not observation noise, posterior uncertainty,
  or a variance.
  Approximate unbiasedness and normality assumptions concern the
  **log-likelihood estimates**; a useful estimate of their SD is also needed.
  Discuss noise levels as practical guidance rather than universal thresholds;
  see the [noisy-target FAQ](https://github.com/acerbilab/vbmc/wiki#noisy-target-function)
  and [Python tutorial](../examples/pyvbmc_example_6_noisy_likelihoods.ipynb).

### 3. Run and preserve the analysis

Set a budget appropriate to model cost, construct `VBMC(...)` and call
`optimize()`. Probes, preliminary optimization and repeated fits belong in
the overall cost estimate.

Explicit seeds are optional, useful when reproducibility is needed. Document
`seed=` and generator support in a reference without making seed management
part of every example or the repeated-run advice. PyVBMC's generator does not
control randomness inside a user's stochastic target. Do not reset that
target's seed on every evaluation to disguise its noise as determinism.

Offer `pyvbmc.calibrate()` when machine-specific performance tuning is useful:
it takes tens of seconds, evaluates no user model, and compatible saved
settings are reused automatically. It is optional performance tuning, not
statistical validation of a posterior. Link to the
[calibration guide](../docsrc/source/api/functions/calibrate.rst).

Save expensive runs with `vbmc.save(...)`. To extend a run, use
`VBMC.load(..., new_options={"max_fun_evals": N})` and call `optimize()`;
`N` is the larger **total** evaluation budget. See the
[diagnostics and saving tutorial](../examples/pyvbmc_example_3_diagnostics_and_saving.ipynb).

### 4. Validate and interpret

Read the termination message and diagnostics, plot the posterior, and assess
agreement across runs. Recommend **3–4 runs with different starting points**,
comparing posterior plots and estimated evidence, as in the
[validation tutorial](../examples/pyvbmc_example_4_validation.ipynb).
Agreement increases confidence but cannot prove that every run captured the
true posterior. A small demonstration budget exercises the interface; it
does not establish convergence.

Use the actual Python outputs:

| Output | What the skill should explain |
| --- | --- |
| `results["success_flag"]`, `results["message"]` | Whether PyVBMC reports successful termination and why it stopped. Success still needs posterior validation. |
| `results["elbo"]`, `results["elbo_sd"]` | Estimated evidence lower bound and uncertainty in estimating that bound. The fitted posterior introduces a separate, generally unknown gap between the bound and the true log evidence. Estimating the bound precisely does not close that gap. |
| `results["r_index"]` | A convergence diagnostic to interpret alongside termination and stability, not an accuracy certificate. |
| `results["components"]` or `vp.K` | The mixture component count, not a quality score by itself. |
| `vbmc.iteration_history["sKL"]` | Changes between successive variational posteriors; a history diagnostic, not a `results["sKL"]` field or a comparison between independent fits. |

Troubleshooting should connect symptoms to useful checks: non-finite values
to support and numerics; budget exhaustion to setup, noise and stability
before extending the run; persistent late ELBO oscillations to fit diagnostics;
disagreement across runs to missed posterior structure or local solutions.
An occasional ELBO decrease alone is expected as PyVBMC updates its estimate.
Do not mechanically loosen stopping tolerances until a run reports success.

## Optional 1.5 workflows

Make these discoverable from `SKILL.md` and explain them in a reference when
needed. Use the [quickstart](../docsrc/source/quickstart.rst) for interfaces.

| Feature | Guidance to preserve |
| --- | --- |
| `vectorized_target=True` | Batches missing initial-design evaluations. Later calls contain one point with shape `(1, D)`. NumPy inputs are `(N, D)`; outputs are `(N,)` or `(N, 1)`, including when `N=1`. Noisy outputs are a pair of arrays. Separate priors keep their scalar interface. |
| Torch/JAX targets | Adapt the existing model to NumPy inputs and outputs and preserve float64 where supported. The solver remains NumPy/SciPy; initial batching does not move inference onto a GPU. |
| `vp.to_torch()` | An independent distribution snapshot, by default CPU float64 in original coordinates. Exported samples use torch's RNG. Install the optional `torch` extra when needed. |
| `vp.to_arviz()` | Independent draws from the variational approximation in a one-chain DataTree, advancing `vp.rng`. Requires the `arviz` extra and Python 3.12+. MCMC convergence diagnostics on these draws do not validate the approximation. |
| Runtime tips | Leave defaults unless the user wants different output. `show_tips=False` disables tips; `display="off"` also suppresses the calibration reminder and ordinary optimization output. |
| S-VBMC | Link to the [S-VBMC tutorial](https://github.com/acerbilab/svbmc#how-to-use-s-vbmc) for combining independent runs on the same model and data without new model evaluations. |

## Distribution and maintenance

Retain the goal of shipping the skill with the library so users can obtain
guidance for their installed release. The implementation plan must settle
how the single source under `skills/pyvbmc/` enters both wheel and sdist, and
how users copy it to a supported agent's skill location. Test built artifacts,
not just a checkout.

A dedicated installer, such as `python -m pyvbmc install-skill`, is one
possible addition. First establish whether documented folder copying suffices.
If an installer is justified, use an explicit destination
and defined replacement behavior rather than guessing the active agent or
overwriting a user's edited skill. Keep general agent management out of scope.

Bundling does **not** guarantee that an installed copy stays current. Record
the supported PyVBMC release in the skill, explain updating, and check
compatibility before applying version-specific advice. A copied folder must
work outside this checkout: bundle its focused references and use public
documentation links for further reading. Relative source links in this
proposal are for maintainers, not links to copy unchanged into the skill.

Keep options guidance tied to the maintained
[options reference](../docsrc/source/api/options/vbmc_options.rst). If an
offline options list is needed, generate it from the `.ini` files instead of
maintaining another default-value table. Keep advanced option changes out of
the basic workflow.

## Acceptance checks for implementation

- Validate frontmatter and local resource links against the
  [format specification](https://agentskills.io/specification). Check that
  the installed folder is self-contained and version information is accurate.
- Check examples and named outputs against the public API. Exercise the small
  example with a bounded budget; reuse existing inference tests for algorithm
  correctness instead of adding several full fits.
- Review representative requests: a first fit, a supplied prior, a noisy
  likelihood, non-convergence, conflicting repeated fits, small ELBO SD,
  save/resume, and a vectorized target. Each should lead to a concrete next
  action with its reason and a useful supporting link.
- Check optional integrations with their dependencies installed. Basic skill
  use and the deterministic example must work with core PyVBMC alone.
- Verify delivery from wheel and sdist, working references and a documented
  update path. Add the README/docs entry when installation instructions work.

## Sources

Use the [1.5 overview](2026-09-06-pyvbmc-1.5-overview.md),
[quickstart](../docsrc/source/quickstart.rst),
[VBMC API](../docsrc/source/api/classes/vbmc.rst),
[prior documentation](../docsrc/source/api/classes/priors.rst), examples and
reviewed [runtime-tip catalog](../pyvbmc/vbmc/_tip_catalog.py) together.
The [VBMC FAQ](https://github.com/acerbilab/vbmc/wiki) supplies supporting
explanations; translate MATLAB interfaces and check scientific wording against
the current Python implementation.
