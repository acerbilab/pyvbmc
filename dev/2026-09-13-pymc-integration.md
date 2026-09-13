# 2026-09-13 — A scoped PyMC integration for PyVBMC 1.5

**Status:** proposal based on inspection of the 2022 PyMC integration PR,
the current PyVBMC source and current PyMC/ArviZ documentation. The next
technical step would be a bounded feasibility prototype. No adapter has
been implemented or executed; effort estimates below are provisional.

## Purpose

Let users define a model in PyMC, fit it with ordinary PyVBMC, and return
posterior samples with the model's variable structure intact. This would
extend the ecosystem integration in 1.5 while keeping initialization and
inference decisions visible to the user.

There is a useful split between connecting the two libraries and automating
the inference workflow. A target adapter, structured posterior export and
a worked example are plausible additions to 1.5. Automatic initialization,
bounds suggestions and orchestration can be developed separately.

## What PR #73 proposed, and what 1.5 covers

[PR #73, “pymc integration”](https://github.com/acerbilab/pyvbmc/pull/73),
opened by `aloctavodia` on 2022-03-25, proposed a `pymc_fit` wrapper that
accepted a PyMC model, compiled its joint log density, obtained initial
points and bounds, ran VBMC and returned posterior samples through PyMC's
ArviZ conversion. Its target helper could also compile prior and likelihood
separately. The example notebook demonstrated a vector-valued parameter,
a positive scale parameter and a regression with a deterministic mean.

The inspected PR head was `f91d0956097d9191625c244b79bf1a6d9d1a1fcc`.
The code uses Aesara-era PyMC interfaces and an older VBMC constructor and
return signature. It is useful as a statement of intent and a set of
examples; its implementation requires substantial updating.

PyVBMC at `a53b903` supplies
[`VariationalPosterior.to_arviz()`](../pyvbmc/variational_posterior/variational_posterior.py),
which exports independent draws to an ArviZ DataTree with a single
`posterior` group. Users can name parameters, but each parameter becomes a
separate scalar variable. The exporter has no PyMC model, variable-shape
mapping or mechanism for evaluating model-defined deterministic quantities.
The Torch/JAX examples likewise ask users to supply a target function;
there is no PyMC model adapter.

Thus 1.5 covers the basic ArviZ output capability of the PR, while native
PyMC integration remains outstanding.

## Bounds and the original discussion

The PI's replies in the PR already establish the relevant distinction:
hard bounds belong to the model, while plausible bounds guide inference
and can be suggested from priors. The prototype used minima and maxima of
prior samples as hard bounds. That truncates the prior and can change the
model between runs, so that construction should not be carried forward.

An adapter must preserve the model's support. If it offers prior-based
bounds suggestions, those should concern **plausible bounds**. Quantiles
of simple fixed priors are a relatively small problem; hierarchical priors,
dependent constraints and transformed coordinates need further design.
Users can supply plausible bounds explicitly in the first version.

The PI also preferred exposing a target-function handle that users could
pass to ordinary VBMC. The PR author explained the complementary expectation
of PyMC users: define a model and let an inference wrapper handle setup.
Both workflows are useful, but the target adapter can be delivered before
the automatic wrapper.

## Proposed split

| Capability | Relative effort | Proposed place in 1.5 |
| --- | --- | --- |
| Export vector/matrix parameters with names, shapes and coordinates | Small extension of the existing ArviZ export | Include; useful beyond PyMC |
| Convert a PyMC model into a callable target, retaining the parameter mapping | Medium; the main integration work | Aim to include, subject to a bounded feasibility check and an explicit supported model scope |
| Recover deterministic quantities and run posterior predictions | Small once the parameter mapping works | Demonstrate with PyMC's own functions |
| Suggest plausible bounds from priors | Small for simple fixed priors; more involved for general models | Keep separate from the initial adapter |
| Automatically choose initialization, run multiple fits and return everything | Larger, with substantial policy decisions | Defer |

These estimates describe relative implementation difficulty, not measured
development time. Supporting an optional PyMC dependency also entails a
tested version range, documentation and CI coverage.

The intended first workflow is:

1. Define a supported continuous model in PyMC.
2. Obtain its target function and parameter mapping through the adapter.
3. Run ordinary `VBMC`, supplying starting points and plausible bounds explicitly.
4. Return samples with the model's variable structure and use PyMC/ArviZ
   for further analysis.

The existing scalar ArviZ export should retain its behavior. The structured
extension would group and reshape the same posterior draws using explicit
metadata. ArviZ's
[`from_dict`](https://python.arviz.org/en/stable/api/generated/arviz.from_dict.html)
already accepts variable dimensions and coordinates.

## Where the adapter needs judgment

Compiling a log density is supported by PyMC's
[`Model.compile_logp()`](https://www.pymc.io/projects/docs/en/stable/api/model/generated/classmethods/pymc.model.core.Model.compile_logp.html).
The harder part is making the coordinate convention, Jacobians and parameter
reconstruction consistent. A posterior over `log(sigma)` must be converted
correctly before presenting it as a posterior over `sigma`; density
normalization must also remain correct for model-evidence estimation.

Current PyMC provides public operations for
[converting unconstrained samples to model variables](https://www.pymc.io/projects/docs/en/stable/api/model/generated/pymc.model.transform_values.constrain_values.html)
and [removing model value transforms](https://www.pymc.io/projects/docs/en/stable/api/model/generated/pymc.model.transform.remove_value_transforms.html).
These provide possible implementation routes. The choice of inference
coordinates, supported constraints and handling of distributions whose
transforms change dimension remains to be established by the prototype.

Once samples have the correct names, shapes and coordinates, PyMC can
[compute deterministic quantities](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.compute_deterministics.html)
and [generate posterior predictions](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html).
The integration should demonstrate those existing operations. Predictive
simulation entails additional model computation and should be an explicit
user step.

## Feasibility check

A small prototype should cover an ordinary scalar model, a positive
parameter, and a vector parameter with a deterministic quantity. Those
cases exercise target compilation, transformations, reconstruction and
the return path into PyMC. Check pointwise log densities, including
normalization and Jacobians, against known expressions; verify sample
shapes, names and coordinates, and demonstrate deterministic and predictive
calculations. Reject unsupported models clearly rather than accepting a
misinterpreted parameterization.

The prototype should establish how much of this can be supported through
current public APIs and what maintenance burden the chosen PyMC version
range introduces. Its result determines the adapter scope for 1.5; the
structured ArviZ export is independently useful if the adapter proves larger.

Static design work can proceed during the active golden campaign.
Compilation and execution checks should wait until they can respect the
repository's one-heavy-computation constraint. The proposal requires no
change to ordinary VBMC inference defaults.
