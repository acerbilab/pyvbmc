# Stage 3: pipeline features

Created: 2026-09-06. Status: **implemented and locally verified; pre-night integration in progress**.
Branch: `dev-next-stage3`, based on `dev-next` at `4d91a5e`.
This file owns the Stage 3 implementation design, decisions, gates and live
tracker for maintainers. The roadmap owns release sequencing; this plan
implements its pickup 11 and the pipeline features of devlog sections 8/10.

## Summary

Stage 3 makes PyVBMC easier to use within an existing modelling workflow:
connect a model, fit it, and use the posterior in downstream tools. The PI
confirmed both workflows below as part of the 1.5 modernization (2026-09-06).
This framing also organizes the user documentation and eventual release
presentation; it does not change the agreed feature scope.

1. **Bring a model into PyVBMC.** Document short torch/JAX likelihood adapters
   and add the opt-in vectorized target interface, batching the initial
   design. Success means a user can connect an existing framework model,
   evaluate the initial design in one batch, and continue through the same
   target interface for later single-point evaluations. This reduces adapter
   work and can accelerate initialization for models that benefit from batches.
2. **Use the fitted posterior downstream.** Export an independent torch
   distribution for differentiable density evaluation and samples in current
   ArviZ DataTree format for analysis and plotting. Success means a user can
   take a fitted VP into either tool with one documented export call, with
   explicit coordinate, dtype and random-state behavior.

Both workflows ship together in 1.5. The numerical solver and default target
path keep their existing behavior; no torch or ArviZ objects are retained
on a VBMC/VP. The implementation steps and gates below support these two
user outcomes.

## Scope and prerequisites

In scope: FunctionLogger, the initial-design call site, the likelihood/prior
wrapper, two VP export methods, optional dependencies, CI and user docs.
The explicit target contract, lazy extras, default float64 state, separate
worktree and merge timing are already decided by roadmap pickup 11.

Out of scope: Stage 4, batched acquisition point selection, process pools,
new sampling algorithms, latent bug fixes assigned to roadmap pickup 9,
and the form of S-VBMC integration (pickup 10). Avoid changing VP attributes
that S-VBMC's pickles rely on; its full compatibility campaign remains pickup 10.

### Worktree and environment

- Worktree: `C:/Users/luigi/Documents/GitHub/pyvbmc-stage3`.
- Interpreter: `.venv/Scripts/python.exe` inside that worktree.
- During implementation, the original checkout stayed on `dev-next` and
  retained its own venv. The revised integration below advances that checkout
  only after CI; its numerical dependency versions remain the reference ones.
  Commands explicitly name their intended checkout and interpreter.
- The new venv has the reference venv's pinned development dependencies
  (local manifest `.venv/baseline-requirements.txt`), an editable PyVBMC
  install from this worktree, and the released gpyreg 1.1.0 wheel.
  Verify its source against the reference's editable sibling gpyreg before
  claiming any numerical identity.
- `dev/scripts/runs` is a Windows junction to the original checkout's run
  directory. Treat reference traces as read-only. Every new gate supplies a
  distinct `--out` under `runs/stage3_*`; never overwrite a reference.
- No full-suite or replay runs during a reference night. Check that a night
  is not running before heavy gates; one heavy process, single-threaded BLAS.
  Long campaigns start only on the PI's word. This plan does not start one.
- Revised by the PI after implementation: merge Stage 3 into `dev-next`
  before the nights once CI passes; require exact oracles, an identical replay
  and a green full suite after integration. Freeze that code for both nights;
  preserve existing traces and their provenance, and defer latent bug fixes.

## Existing code and constraints verified by reading

- `pyvbmc/function_logger/function_logger.py`: `__call__` handles one
  transformed point, `add` records supplied values, `_record` owns
  duplicate aggregation, Jacobian corrections, growth, indices and timing.
  The default constructor signature is used positionally by tests/oracles.
- `pyvbmc/vbmc/active_sample.py`, initial `gp is None` branch: constructs
  the complete design and cached-value vector, transforms it, then interleaves
  `__call__` and `add` in design order. Later point choices depend on the
  preceding GP update and must stay sequential.
- `VBMC._init_log_joint` adds prior and likelihood directly. Built-in
  priors normally return columns; adding one to a likelihood vector would
  broadcast to an N-by-N matrix. `UserFunction.log_pdf` retains the public
  contract of one 1-D point and a scalar result.
- `VBMC.load` can replace options after restoring the historical logger.
  Old options/loggers lack the new flag. Saved wrappers also need
  checking when a load-time override changes vectorization.
- VP mixture arrays are `w (1,K)`, `mu (D,K)`, `sigma (1,K)`,
  `lambd (D,1)`. `sample` defaults to original coordinates and uses
  `vp.rng`; `__deepcopy__` intentionally shares that RNG.
- `ParameterTransformer.inverse` undoes scaling, then rotation, then
  centering/bounds. Both finite bounds, unbounded coordinates, mixed
  coordinates, logit/probit/student4, and whitening all need coverage.
  Its ordinary rotation is orthogonal and scale positive. Numerical
  defects in the NumPy transformer remain pickup 9, not export refactors.
- No current export or parameter-name metadata exists. VP.stats may be None.
  New methods must work on manually built VPs, without an optimize run.
- The repository supports Python >=3.10. Current ArviZ uses DataTree and
  released ArviZ/arviz-base 1.3.0 require Python >=3.12 (verified PyPI metadata); this affects the optional
  feature, not the core's Python floor.

## Approved API and implementation

### 1. Vectorized target and initial design

Add `vectorized_target = False` with a descriptive comment to
`pyvbmc/vbmc/option_configs/basic_vbmc_options.ini`. Validate it as a
boolean. Add an optional final `vectorized_target=False` argument to
FunctionLogger and pass it by keyword from VBMC.

The settled target contract when enabled is:

- Input is always a NumPy array (N,D), N >= 1, in original coordinates,
  including later single-point evaluations as (1,D).
- Deterministic/unknown-noise output is (N,), with (N,1) accepted and squeezed.
  Scalar output is invalid even for N=1.
- User-provided noise returns a pair of those arrays, the second containing
  finite positive SDs. Reject an (N,2) ndarray explicitly: return a pair.
- Validate real, finite values and the entire returned shape before recording.
  Preserve existing exception context and give row/shape context for batches.
- Do not probe for batch support, retry as scalar, change the initial design,
  or modify the draws used to select it.

Approved public logger method:
`batch_call(x, f_vals=None)`, returning `(values, SDs, indices)`.
Input `x` is the full nonempty (N,D) design in transformed coordinates.
`f_vals`, if supplied, is a length-N vector of already evaluated
original-space log-joint values, with NaN indicating rows to evaluate.
Outputs are length-N values/indices; SDs are length N for noisy loggers
and None otherwise. Values/indices follow the existing single-row return
semantics, including duplicate aggregation.

Evaluate the missing M rows in a single target call. Validate supplied and
returned data before recording; distribute measured target wall time equally
over those M rows. Then walk all N rows in original order: cached rows go
through `add`; evaluated rows increment `func_count` and use `_record`.
Thus `func_count` rises by M, `cache_count` by N-M. If M=0, make no target
call and record through `add` only. Cached values already include any prior.

Reusing `_record` preserves current duplicate and total-time semantics,
including its existing limitation that total time only accumulates new rows.
Do not repair that unrelated behavior during Stage 3. Validate before mutation
for normal user errors; no promise of transactional recovery from MemoryError.
Require vectorized mode for `batch_call`; an explicit error is preferable
to silently sending a matrix to a scalar target.

The enabled `__call__` delegates to a one-row batch and returns today's
scalar/None/index tuple. The disabled path retains its current arithmetic
and target invocation. Read the flag with a backwards-compatible default
for old pickles. Initial `active_sample` calls `batch_call(Xs, ys)` only
when enabled; keep its disabled loop intact. Later evaluations stay sequential.

Normalize vectorized likelihood output before combining it with a prior,
using one private shared shape validator so the wrapper and logger agree.
Evaluate the prior row by row through its existing scalar contract; this
also supports custom Prior subclasses without requiring new keyword args.
Coerce each prior result to a scalar, then add two length-N vectors.
Normalize and validate noisy pairs before unpacking, including the N=2 trap.
The likelihood remains one batch; inexpensive prior evaluation is sequential.

On load, absent flag means False. Synchronize the restored logger and options.
A change to vectorization via `new_options` must either rebuild a compatible
wrapper from the saved original likelihood/prior or fail explicitly before
optimization if the saved object lacks the necessary provenance; never leave
a logger flag and closure using different contracts. Verify existing saved
scalar runs and same-mode vectorized save/load without new optimize runs.

### 2. Torch posterior export

Approved signature:
`vp.to_torch(orig_flag=True, *, dtype=None, device=None)`.

- Default dtype is explicitly torch.float64, default device explicitly CPU.
  Support explicit float32/float64 and device selection; reject integer,
  complex, float16 and bfloat16 exports initially. Never change torch's global
  default dtype/device. Float32 is an exported copy, not solver state.
- Return a snapshot: copy the VP parameter and transformer arrays into tensors.
  No zero-copy aliases, cached torch object, or tensor attribute on the VP.
  No draws during conversion; sampling the result uses torch's RNG.
- Base distribution:
  `MixtureSameFamily(Categorical(w.ravel()),
  Independent(Normal(mu.T, sigma.T * lambd.T), 1))`.
  Its event shape is (D,), with arbitrary leading sample dimensions.
- `orig_flag=False` returns the internal Gaussian mixture.
  Otherwise wrap it in a TransformedDistribution with a private vector-event
  Transform implementing the inverse parameter map and its log Jacobian.
  Put torch subclasses in `pyvbmc/variational_posterior/_torch.py`,
  imported only inside the public method.
- Support unbounded, mixed, all three bounded transforms and whitening.
  Validate scale/rotation as the states the core actually produces; reject
  nonorthogonal custom rotations and nonpositive scales with a clear error
  instead of copying latent NumPy assumptions into an incorrect distribution.
- Express transforms in torch operations so `log_prob(x)` differentiates
  with respect to input x. No gradient path back into the NumPy VP and no
  promise of reparameterized mixture sampling (`rsample`).
- Use stable probit inverse primitives (investigate `torch.special.ndtri`,
  not `erfinv(2p-1)` in tiny tails), stable logit Jacobians, and a student4
  implementation with finite derivatives at its center. Verify these before
  completing the original-space export. Inactive branches must not inject NaN
  gradients into mixed-coordinate batches.
- Approved support behavior: finite strict interior of bounded dimensions;
  exact/outside bounds are rejected by distribution validation. Support
  validation is part of the returned distribution, not an interior-only
  documentation assumption. Do not promise NumPy's zero/-inf sentinel there.
  Keep sampling strictly inside representable bounds using a documented
  sample-only endpoint correction if saturation requires it; do not put a
  hard clamp into the analytic bijector/Jacobian and claim exact invertibility.
  Test this boundary behavior separately from ordinary density agreement.

Verification: compare log_prob against VP.log_pdf on representable interiors
in both spaces, all transform types and nontrivial whitening; autograd/finite
differences in each case, centers and tails, D=1 and K=1, arbitrary sample
shapes and empty sample shape, copied state independence, support/sampling,
explicit dtype/device and default behavior under changed torch globals.
Assert NumPy and torch RNG states unchanged by conversion and torch RNG
controls subsequent samples. Test old saved VPs and check the original VP
with the existing dtype canary after conversion.

### 3. ArviZ export

Approved signature:
`vp.to_arviz(n_samples=1000, *, var_names=None, orig_flag=True)`.

Agreed format (PI, 2026-09-06): use the current
ArviZ API and return an xarray.DataTree via lazily imported
`arviz_base.from_dict({"posterior": posterior}, ...)`.
One posterior group, one chain, n_samples draws of each scalar parameter.
Default names `x_0` through `x_{D-1}`; custom names must be nonempty,
unique strings of length D and must not conflict with sample dimensions.
Do not populate likelihood, observations, diagnostics or ELBO from a
standalone VP.

Draw once with `vp.sample(n_samples, orig_flag=orig_flag, balance_flag=False)`.
Like sample(), this explicitly advances vp.rng; say so in the docstring.
Reject invalid sample counts/names and check the dependency before drawing.
Use the actual samples unchanged, per-variable shape (1,n_samples), and set
sample dimensions explicitly so user ArviZ configuration cannot change them.
Mark metadata as independent draws from a variational approximation.
MCMC convergence diagnostics on these draws do not assess approximation
quality; the example should use posterior summaries/plots.

No synthetic multiple-chain option or new seed mechanism. Users can seed
the VP as today. The export is a convenience around the existing sampler,
not another independent posterior sampler.

Verification: return type/group/dimensions, custom/default names and invalid
inputs, exact exported values against a reference sample from a separately
seeded VP, original-space bounds, transformed space, K=1, and intended RNG
advancement. No optimize run required.

### 4. Extras, CI and framework adapters

Add `pyvbmc[torch]` and `pyvbmc[arviz]` without adding either to base or
test dependencies. Lazy import errors identify the extra. Move the existing module-level `corner` import into `VP.plot`: corner
currently imports ArviZ when available, so lazy-loading only the new export
helpers is insufficient. Check fresh-process imports both without extras and
with both installed: importing pyvbmc and using NumPy methods must leave torch,
arviz and arviz_base unloaded. Retain plotting coverage with/without ArviZ and
cover actual missing-dependency errors in tests that do not skip themselves.

Torch floor >=2.7 (also the S-VBMC roadmap's floor), verified by
running the chosen APIs at that minimum. Approved ArviZ current API:
`arviz>=1.0; python_version >= '3.12'` in the extra and an explicit method
error on older Python naming the requirement. Check released-package
metadata/resolution at the selected minimum before locking this in; a marker must be paired with
prominent install docs so pip's no-op on Python 3.10/3.11 is not misleading.
Declare arviz-base directly as well if implementation imports it directly,
at a compatible tested minimum. Keep the core Python requirement >=3.10.

In `.github/workflows/test-matrix.yml`, the Ubuntu newest-Python leg
(currently 3.12) installs torch from the official CPU wheel index first,
then this checkout with `[test,torch,arviz]`. Other legs remain base/test.
Use a condition derived from the passed matrix's newest version, or document
and test a centralized explicit version; do not silently stop testing extras
when the matrix advances. Ensure both smoke and full-matrix callers enable it.
Optional numerical test modules skip when the extra is absent.
Check installation of declared minimum versions separately during development.

Document scalar and vectorized torch/JAX likelihood adapters in
`docsrc/source/quickstart.rst`: explicit float64, host NumPy return arrays,
torch device transfer plus detach/cpu before NumPy, JAX x64 enabled by the
user before model construction, and correctly shaped noisy return pairs.
No generic adapter API, automatic target probing or JAX dependency.
Explain one initial batch followed by (1,D) calls, separately supplied
scalar priors, and that framework execution must finish before returning
the host array for meaningful target timing. Run small adapter snippets
against available frameworks as an explicit verification task; JAX can use
a disposable optional environment rather than expanding the core CI matrix.

## Documentation

Update existing authoritative docs:
- `docsrc/source/quickstart.rst`: batching, framework adapters and exports.
- `docsrc/source/installation.rst` and README installation: extras,
  CPU torch install sequence, optional Python floors and conda package names.
- `docsrc/source/api/classes/variational_posterior.rst` and
  `.../function_logger.rst`: new public method references.
- Hand-written method pages under `docsrc/source/api/` for batch_call,
  to_torch and to_arviz, linked through the documentation toctree reachable
  from `index.rst`, as AGENTS.md requires.
- VBMC/FunctionLogger/VP docstrings and the basic option's comment.
- This plan for implementation status and actual verification evidence;
  `dev/README.md` plan index, roadmap pickup 11 and Stage 3 status.
  Do not rewrite the original checkout's TODO while it tracks the nights.
  Reconcile its records at merge.
No new notebook is necessary; if one is later chosen, edit the notebook and
regenerate its script, never hand-edit generated scripts.

## Live implementation checklist

- [x] Plan approved and reread; two workflows, current DataTree, core Python >=3.10 retained.
- [x] Vectorized logger, prior wrapper, initial design and save/load integration with focused tests.
- [x] Torch distribution helper and numerical/export tests; independent review endpoint finding fixed and tested.
- [x] ArviZ export, public VP methods, optional extras and CI wiring implemented; minimum-version export tests passed.
- [x] Framework adapters and user/API/install documentation implemented; all seven new quickstart code blocks executed successfully.
- [x] Focused verification, examples, packaging and docs build.
- [x] Independent implementation review and fixes: export and batching findings fixed; focused regressions pass.
- [x] Exact oracles, identical replay and full suite passed outside reference nights.
- [x] Final records and conventional feature commit on dev-next-stage3 (`4ee612d`); verification record committed alongside it.
- [~] Revised integration gate: merge before reference nights after CI, then repeat required integration checks (live checklist below).

Primary agent owns this tracker. Implementation work is separated by file ownership;
all test commands are centralized and run one process at a time.

## Implementation sequence and gates

1. Approve this implementation plan; the current DataTree format is agreed.
2. Implement vectorized logger, wrapper and initial-design integration with
   focused unit tests, legacy-save checks and accompanying docs.
3. Add optional dependency plumbing and private torch export, starting with
   internal mixture tests, then transform/support/gradient tests.
4. Add ArviZ export and its focused tests.
5. Complete adapter examples, API docs, install instructions and CI wiring.
6. Independent implementation review; fix confirmed findings. When no reference
   night is running, run the final local gates below, one process at a time.
7. Commit conventional commits on Stage 3; push only within the user's
   authorized workflow. The revised PI decision below moves integration before
   the nights, after CI and with repeated integrated verification.

Gates (checklist entries record completed verification):

- [x] Focused FunctionLogger tests: output shapes, noisy N=2 rejection,
  single-point input shape, mixed cached rows and duplicates, growth, row
  order, function/cache counts, mock-clock timing and validation failures.
- [x] Initial-design equivalence at fixed seed, scalar vs vectorized target:
  X/X_orig/y/y_orig/S/n_evals/counts and post-call RNG state, including
  partially/all provided values, D=1 and noisy targets. Call active_sample
  directly; do not add full optimize tests.
- [x] Separate built-in/custom prior integration, column/vector likelihood,
  noise pair, cached log-joint values not given a prior twice.
- [x] Save/load compatibility: existing static files, saved enabled loggers,
  restored options and mode overrides. Run save/load modules in their
  required order; do not regenerate static fixtures.
- [x] Torch export matrix above, minimum API version, missing-extra behavior.
- [x] ArviZ export matrix above, declared Python floor and missing-extra behavior.
- [x] Execute small torch/JAX adapter examples; record versions and outcomes,
  distinguishing an unavailable optional framework from a passing check.
- [x] Build/inspect wheel and sdist for helper modules and docs/fixtures as
  applicable; import exports from an installed wheel outside the source tree.
- [x] Sphinx build and changed-file formatting; distinguish pre-existing warnings.
- [x] Single-threaded exact oracles:
  `python dev/scripts/make_oracle_fixtures.py --check --exact`.
  No reference re-baseline for Stage 3.
- [x] Golden replay with defaults for baseline/sidecars/configs and a unique
  Stage 3 output directory: require `identical`, not merely unflagged.
- [x] Full local suite: `python -m pytest --reruns=5 -x -vv`.
- [ ] After merge at the permitted time, replay identical and full suite green;
  Ubuntu extras leg plus the release's full CI matrix green before main PR.

Before local numerical gates:
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` set in the test
process environment before importing NumPy. No baseline generation, profile
campaign or population run is part of this plan. If identity breaks, identify
and repair the default-path change rather than relaxing/replacing references.
Exports are new numerical implementations; their own tolerance-based
cross-backend checks are separate from the unchanged NumPy identity gate.

## Approved implementation decisions

1. **Batch and cached rows enter one logger method** so evaluation is one call
   and recording retains input order. Rejected recording all missing rows
   before cached rows: changes indices and duplicate semantics.
2. **Separate priors retain their scalar contract**. Rejected requiring
   vectorized priors or guessing from shape: unnecessary compatibility change
   and ambiguous broadcasting. The expensive likelihood still batches.
3. **Torch export is an independent CPU/float64 snapshot by default**, with
   explicit float32/device options and original coordinates. Rejected sharing
   NumPy storage or using torch global defaults: mutable exports or silent
   precision/device changes. No trainable NumPy VP or reparameterized sampling.
4. **Torch support validation rejects exact/outside bounds**. Rejected silently
   returning NaN from inverse transforms or widening support. Sample-only
   endpoint correction is distinct from the analytic density map.
5. **ArviZ export uses one chain and scalar parameter names**, draws from the
   existing VP RNG and adds no invented sampler diagnostics. Rejected
   synthetic chains and redundant seed/sampling APIs.

## ArviZ decision and Python-floor discussion

**Current DataTree support approved** (PI, 2026-09-06). The export uses the
current ArviZ format, with no legacy InferenceData compatibility branch.
The devlog's InferenceData wording predates ArviZ 1.0.

**Core Python floor retained at >=3.10** (PI, 2026-09-06). Stage 3 requires
Python >=3.12 only for the optional current-ArviZ export. Defer reconsidering
the core floor to Stage 4, alongside the numerical-backend and dependency
choices. If Stage 4 does not proceed, the supported-version policy can still
be reviewed independently; it does not require undertaking a backend port.
Keep the optional dependency markers, explicit method error on older Python,
and installation documentation described above. These decisions settle the
format and Python scope; the full implementation plan was approved on 2026-09-06.

## Evidence and review tracker

- [x] Read roadmap/TODO and AGENTS.md; inspected logger, initial design,
  prior wrapper, posterior, transformer, docs and CI.
- [x] Two read-only Sol explorations (batch integration; exports).
- [x] Created isolated worktree/venv, installed editable Stage 3 and matching
  development dependency versions, linked traces; pip check reports no
  broken requirements.
- [x] Import probe resolves PyVBMC to this worktree and Python to its venv; gpyreg 1.1.0 Python sources match the reference sibling (line-ending normalization; generated version metadata excluded).
- [x] Independent Astra review using doublecheck: one P2 finding, the existing
  corner import eagerly loads ArviZ when installed. Confirmed against source;
  added lazy corner import at VP.plot, fresh-process checks with extras
  installed, and plotting coverage. No other consequential findings reported.
- [x] PI approval: 2026-09-06, "ok use the task skill to implement this".
- Implementation began after explicit approval. Verification results below
  supersede the planning-time unchecked gates only where named.
- Focused exports on torch 2.7.0+cpu, arviz/arviz-base 1.0.0:
  `pytest test_vp_arviz.py test_vp_export_dependencies.py test_vp_torch.py -x -q`
  (all paths under testing/variational_posterior): 77 passed, 1 skipped (CUDA),
  10.11 s. Includes actual DataTree, lazy/missing imports, RNG, torch numerical
  values and input gradients. This was the first focused pass; later gates are recorded below.

## Upstream references checked 2026-09-06

- Torch distribution construction and transforms:
  https://docs.pytorch.org/docs/2.14/distributions.html
- ArviZ 1.0 removal of InferenceData:
  https://github.com/arviz-devs/arviz/issues/2548
- Current nested-dictionary to DataTree API:
  https://python.arviz.org/projects/base/en/stable/api/generated/arviz_base.from_dict.html
- ArviZ's Python floor and dependencies (published 1.3.0 metadata checked):
  https://pypi.org/pypi/arviz/json
  https://pypi.org/pypi/arviz-base/json
- JAX explicit x64 setting:
  https://docs.jax.dev/en/latest/default_dtypes.html

### Implementation verification (2026-09-06)

- Expanded exports and existing plot on current ArviZ 1.3.0: 92 passed,
  1 CUDA-only skip, 8.98 s.
- Export review found division underflow for strict-interior endpoints.
  Logit now uses log distances; probit/student4 protect only underflowed
  strict-interior tail probabilities. Torch regression pass after correction:
  75 passed, 1 CUDA-only skip, 3.25 s, float32/float64 and both tails included.
- Initial batching integration pass: 31 passed, 95 deselected, 1.81 s.
  Independent review then identified missing whole-batch coordinate validation;
  the correction passed 36 focused tests (95 deselected), 1.82 s.
- Executed seven actual quickstart blocks plus scalar/vector/noisy shape,
  dtype, finiteness and framework-agreement assertions: passed on torch
  2.7.0+cpu, JAX 0.11.1 and ArviZ 1.3.0, using a seeded standalone VP.
  No new optimize run was used for examples.
- Initial Sphinx pass succeeded with four missing-example warnings because
  that direct invocation omitted the Makefile's notebook-copy preparation.
  A complete build with that preparation follows; not counted as a clean gate.

- Complete Sphinx build with the Makefile notebook-copy preparation succeeded;
  one existing notebook-2 Plotly MIME warning, no new documentation warnings.
- Changed Python files formatted with repository-pinned Black 23.3.0 and
  isort 5.12.0; git diff --check passed.
- Exact numerical oracle gate: 8/8 fixtures passed without rebaselining.

- Golden replay with defaults: all five seed-0 configurations identical,
  including initial design, live points and every ELBO iteration; 3.0 min.
  Output: `dev/scripts/runs/stage3_replay_20260906`, reference
  `golden/item7_20260906`; no reference files changed.
- All five pre-commit hooks passed on changed files. CI export-selection
  script exercised for every full-matrix leg and the smoke caller: passed.

- Wheel/sdist built and inspected; wheel-installed exports passed outside
  the source tree (torch 2.7.0+cpu, ArviZ/base 1.3.0). Metadata retains core
  >=3.10 and the >=3.12 ArviZ markers. Five changed core modules parse with
  Python 3.10 grammar. Removed an incidental UTF-8 BOM from the new helper.
- Packaging observations: current wheel includes testing files/fixtures,
  contrary to the existing AGENTS description; packaging controls are unchanged
  from HEAD, confirming this is pre-existing. docsrc is pruned by MANIFEST.
  Existing build warnings concern the license-table metadata and SCM version
  already being set. Packaging policy changes remain outside Stage 3.

- Full local suite with both exports installed: **957 passed, 34 skipped,
  2 xfailed, 3 warnings in 299.37 s**, no reruns. The 34 skips are 15
  oracle combinations, 18 inapplicable D=1 mixed-transform cases, and the
  CUDA-only export check;
  two existing float32/float16 boundary-cast xfails remain. Three warnings
  are the existing VP gradient division warnings in oracle cases.
- Fresh-process plotting with ArviZ imports blocked also passed, covering
  corner's base-only path after the lazy-import change.

## Completion and integration handoff

The approved two-workflow implementation is complete on `dev-next-stage3`,
feature commit `4ee612d`. Core Python remains >=3.10; only current-ArviZ
export requires >=3.12. All local implementation gates passed. Independent
batching and export reviews found one consequential edge case each; both
were fixed and covered by passing regressions. The numerical references
were unchanged. The task tracker remains here as the durable design and
verification record.

At implementation completion, the original `dev-next` checkout was clean
and unchanged, and the branch had not been pushed or merged. The subsequent
PI decision below supersedes that handoff: pre-night integration and remote
CI are now in progress. The local verification above remains the feature
branch evidence; integrated results are recorded separately below.

Local detailed logs are `.venv/stage3-{pytest,oracles,replay}.log` and
`.venv/stage3-sphinx{,-warnings}.log`; artifacts and wheel smoke evidence
are under `.venv/stage3-*`. These are ignored local verification artifacts;
this plan records the durable outcomes needed from a fresh checkout.


## Revised integration sequence (PI, 2026-09-06)

The PI approved moving Stage 3 integration before the reference-extension
nights. This supersedes the earlier post-night merge restriction in this
plan and roadmap pickup 11. Preserve default `vectorized_target=False`,
require CI and integrated numerical checks, then freeze that code for both
nights. Existing Stage 2 traces retain their original provenance; new
sidecars record the integrated Stage 3 code. Do not land bug fixes before
the reference extension is finished. The PI must authorize the start; that instruction may cover both batches
as one sequential chain, with phase 2 gated on phase 1 and its checks.

- [x] Push Stage 3 and pass CI: smoke 34043031387 and all nine jobs of full matrix 34043071150 passed on 285cd74.
- [x] Update roadmap/TODO/reference instructions to the approved sequence.
- [~] Merge into clean dev-next and check its environment against the reference.
- [ ] Integrated exact oracles, identical replay and full local suite.
- [ ] Independent integration review; commit/push records and freeze the tested code.
- [ ] Later: start the reference campaign on explicit instruction, which may authorize both batches as one chain.

Integration preflight: original dev-next is clean at `4d91a5e`, fast-forward
merge is available, and no Python jobs were running. Its own editable
installation resolves to that checkout and the reference sibling gpyreg.
NumPy 2.5.2, SciPy 1.18.1, gpyreg 1.1.0 and cma 4.4.4 match the reference
environment; pip check is clean. Torch and ArviZ are absent, so integrated
local testing will also cover the base installation.

Operational clarification (PI, same session): light laptop use is compatible
with these reference runs. The population gate uses `elbo_err`, `gskl`,
`mmtv` and `func_count`; wall time is recorded but not gated. Mixed-use
timings must not be treated as an idle-machine speed benchmark. A start
around 20:00 with both batches chained is feasible (roughly 12-13 hours
plus overhead), on power and without sleep or competing heavy computation.
This discussion does not itself launch or schedule the campaign.

Independent Sol integration review: corrected the sidecar provenance
(280 records at `18a236c`, dirty false; documentation-only descendant of
numerical code `bdaf322`), residual authorization wording and CI status.
The future chain explicitly gates counts/pairs and compares, without the
legacy shell script's masked compare failures; it passes batching false
explicitly in JSON options. No historical sidecar or trace was modified.
