# Stage 4: bounded PyTorch variational-step feasibility

Created: 2026-09-08
Status: COMPLETE; independent review passed. PI backend decision pending.
User approval: 2026-09-08, "sounds good - go ahead"; deliver prototype results.
Planning branch: `dev-stage4-torch-feasibility`, created from `dev-next` at
`bf43c20327f5684e6a22e801f4e6f3964a23cd98`.
Production numerical source matches `feadf30`; no solver code changed here.

## Live execution checklist

- [x] Phase 1: frozen inputs, provenance and CPU/CUDA preflight.
  All nine workload inputs rebuilt; CPU/CUDA float64 environment smoke passed.
- [x] Phase 2: Sol numerical core and estimator tests.
  Explicit developer suite: 22 passed, including a CUDA smoke; initial
  CPU/CUDA comparison covers 48 objective/variance evaluations across all
  eight committed snapshots at the existing oracle tolerances.
- [x] Phase 3: Sol complete-fit adapter and optimizer tests.
  Nine focused checks pass, including exact NumPy adapter/control agreement
  and final boost. A matched warmup 100-update replay has maximum absolute
  objective/gradient differences of 3.11e-15 / 3.09e-14.
- [x] Integration: main-thread numerical gates and focused regression tests.
  74 focused checks plus four artifact-reader checks passed; all 28
  corrected CPU/CUDA kernel records pass.
- [x] Phase 4: bounded sequential CPU/CUDA measurements and diagnostics.
  72 traced fits plus 24 lean controls completed; auxiliary recovery retained.
- [x] Phase 5: independent Sol review, reconciled evidence and recommendation.
  Final numerical and measurement reviews report no remaining must-fix.
- [x] Record results, limitations and remaining release work; no full port.

## Purpose and authority

This plan owns the executable experiment design and, after execution, its
evidence and decision record for maintainers considering the Stage 4 port.
The [roadmap](modernization-roadmap.md), Stage 4, owns release-level status.
Read the [modernization discussion](../2026-09-02-modernization-discussion.md),
[1.5 overview](../2026-09-06-pyvbmc-1.5-overview.md),
[S-VBMC compatibility record](../results/2026-09-08-svbmc-compatibility.md) and
[parked integration proposal](../2026-09-08-ecosystem-integration.md) first.

Measure one complete variational fit against the modernized NumPy baseline:
GP integrals, entropy, parameter handling, optimizer, candidate scoring and
selection, including final-boost sizes. This is one `optimize_vp` operation,
not one Adam update or a full `VBMC.optimize()` trajectory. Preserve the
existing method, gradient estimators, regularizers and stopping policy.

Numerical reliability, float64 values and gradients, acceptable CPU performance,
installation requirements and substantial implementation simplification are
joint decision inputs. Report CPU and GPU independently. Around 1.2x runtime
is not an agreed cutoff; concern begins well below the clearly unacceptable
3x example. GPU gains cannot compensate for unacceptable CPU performance.
Missing GPU measurements leave feasibility open. Stage speed is not measured
end-to-end solver speed. Only an explicit evidence-based PI decision can
authorize the full port, followed by its own implementation/validation plan.

## Scope and implementation boundary

- Prototype modules and explicit developer tests under `dev/scripts/` only;
  raw outputs under ignored `dev/scripts/runs/stage4_torch/`. No production
  backend switch, public API, dependency/Python-floor change or GP vendoring.
- Freeze a fitted NumPy GP and convert its data, hyperparameters, `alpha`,
  `L`, `L_chol` and `sW` once per fit. Keep them constant during optimization.
  The tensor objective must not call gpyreg or rebuild/transfer these factors
  on every evaluation. Account for extraction, conversion and transfers.
- Include `_sieve`/`_vb_init` candidate construction and ranking, bounds,
  the selected starts, optimizer, midpoint/end scoring, selection, pruning,
  and returned VP/statistics. Host orchestration may reuse current Python
  logic through an isolated developer adapter, with all numerical objective
  calls dispatched to the selected backend. Validate that dispatch covers
  the sieve, fine scoring and pruning as well as Adam. Do not patch installed
  modules globally or introduce production dispatch infrastructure.
- For boost workloads also reproduce the `final_boost` option preparation
  and acceptance wrapper; retain pre-boost, candidate and returned VPs/scores.
  Ordinary fits preserve the input transformer identity. The current
  `final_boost` wrapper first deep-copies its input, so a boost returns that
  copied transformer; the prototype records and matches this existing
  behavior rather than silently repairing it inside the experiment.
- GP fitting, Cholesky retry implementation, acquisition, target evaluation,
  transform porting, save-format migration, S-VBMC integration, new posterior
  families, variance gradients, stopping-rule research and autotuning remain
  outside this experiment. Fixed-factor solves test only part of GP reliability;
  they cannot establish the reliability or speed of a future tensor GP fit.
- No full trajectories, final 870-case benchmark, HPC job, watcher or reattachment.
  One heavy computation at a time, owned by the main thread. Sol reviewers
  inspect statically and never run competing suites/builds/benchmarks.

## Phase 1: freeze inputs and preflight

Executor: Astra orchestrates (`gpt-6-astra`, high); Sol implements
(`gpt-5.6-sol`, high; xhigh if needed within repository policy).

1. Add `dev/scripts/torch_vi_fixtures.py` and a manifest under
   `dev/experiments/stage4_torch/`. Reuse `pyvbmc/testing/oracles/_state.py`
   loaders/rebuilders. Record source hashes, full PyVBMC/gpyreg revisions,
   fixture hashes, effective options, parameter flags and shapes, seeds and
   factor representation. Validate imports against the chosen checkouts.
   The CI gpyreg pin is `a2f8ddce867f502e29717959cf0ff3529f598618`.
2. Recompute the baseline from current NumPy source using the stored *inputs*;
   old stored outputs and option snapshots are not a substitute for current
   semantics. Load current option defaults and explicitly retain documented
   regime overrides (noise, warmup, entropy switch, fixed parameters).
   Publish the resolved options. Never regenerate/rebaseline existing oracles.
3. Separate immutable source states from per-arm working copies and private
   random streams. Capture candidate recipes and random draws, not just seeds:
   NumPy and Torch RNGs need not agree. Stream/replay entropy draws in bounded
   blocks with phase/start/evaluation IDs, shapes and hashes. Verify a NumPy
   replay against its uninstrumented seeded execution, including draw order,
   candidate types, pruning choices and final parameters. An adapter mismatch
   blocks comparison; investigate before timing.
4. Verify environment metadata without changing the reference `.venv`:
   Python, NumPy/SciPy/BLAS, Torch version/build, effective thread counts,
   OS, CPU model, GPU model/memory/driver/runtime and module paths. Use an
   isolated environment for any required Torch installation, with reproducible
   CPU and CUDA install commands and package/download sizes recorded.
   Do not reuse historical S-VBMC source or diagnostics as a prerequisite.
5. Verify actual float64 tensor execution, backward and triangular solves on
   CPU and CUDA, and sufficient memory before scheduling measurements.
   Planning inspection found Python 3.12.6 and an RTX 4060 Laptop GPU,
   8,188 MiB, driver 576.80 (`nvidia-smi`). The compatibility record's Torch
   2.14.0+cpu overlay proves neither CUDA compatibility nor GPU feasibility.
   Verify the installed build/driver combination; do not upgrade drivers as
   part of this prototype. If local CUDA cannot work, produce a portable
   fixture/runner bundle and leave GPU evidence pending an available machine.

Verification: reproducible manifest and NumPy adapter parity; explicit dtype
checks on raw state and outputs before serialization/casting; no caller state
mutation or shared RNG leakage between arms; CPU/CUDA smoke status recorded.

## Phase 2: implement the numerical slice and estimator gates

Executor: Sol implementation; Astra resolves mathematical/design conflicts.
Proposed files: `dev/scripts/torch_vi_core.py` and
`dev/scripts/test_torch_vi_core.py` (explicit pytest discovery only).

1. Implement pure parameter decoding from theta with the existing Fortran
   ordering of mu, log-sigma/log-lambd blocks, softmax weights and active flags.
   Cover fixed means, weights, sigma and lambd, D=1 and K=1. Match
   `VariationalPosterior.get_parameters/set_parameters`, including scale gauge
   normalization and frozen-block behavior, without mutating the caller.
   Check gauge shifts and common eta shifts against current values/gradients.
   Do not insert a stop-gradient through normalization merely because an old
   note suggests it: establish which factors cancel for each active mask.
   Current hand gradients use the unprojected physical log-gradient convention;
   choose the formulation that matches it. Single-width-active masks are
   robustness probes, not ordinary solver configurations: compare from fresh
   baseline copies and diagnose any stateful normalization separately.
2. Port `_gp_log_joint` forward formulas, with autograd for G. Include
   the production SE-ARD/NegativeQuadratic model and a focused ZeroMean
   kernel check; validate model/layout identifiers and reject other layouts.
   Include
   per-hyperparameter and averaged results, `I_sk`, `J_sjk`, `varG`, `var_ss`,
   single-sample scalar behavior, both `L_chol` representations, current
   lower-triangle mirroring and variance clamps. Keep cancellation-resistant
   direct differences and the current two-solve contraction; do not replace
   them with a Gram expansion/inverse for speed. Full variance is value-only;
   preserve errors for unsupported variance gradients and `compute_var == 2`.
3. Implement deterministic entropy lower bound and antithetic MC entropy.
   All parameters, constants, optimizer state and samples are explicit float64
   on the selected device; no global default-dtype mutation, autocast or
   lower-precision fallback. Use bounded entropy chunks and account for
   autograd saved tensors, not just forward temporary sizes. Fine scoring
   runs without a gradient graph.
   Preserve the current exp/sum/log mixture arithmetic; log-sum-exp and
   alternative distance formulas remain separate work. Classify nonfinite
   stress outputs against NumPy rather than silently changing the formula.
4. **MC estimator gate:** current `entmc_vbmc` uses pathwise gradients for
   mu/sigma/lambd while omitting direct density score terms, but retains the
   explicit weight derivative. Naive autograd of finite-sample H differs.
   Use a value-preserving autograd construction that freezes density locations
   and scales while retaining reparameterized sample paths and both weight
   dependences. Verify every gradient block against the current estimator on
   identical epsilon, in raw and optimizer coordinates. Document the detach
   semantics and their maintenance cost. Ordinary AD can be a diagnostic;
   it cannot silently replace the production estimator.
5. Assemble `_neg_elcbo` and soft bounds with current semantics: optimization
   beta=0, mu/log-scale penalties, no eta bound penalty or theta mutation,
   existing small-weight penalty (0.1 main loop; zero guarded boost), and
   unpenalized fine scoring with variance. Match threshold derivatives at
   exact ties explicitly; avoid differentiating inactive infinite-bound
   arithmetic into NaNs. Keep unsupported combinations explicit.

Verification uses all eight committed main oracle snapshots (not the three
GP-fit-history fixtures). Compare raw values/gradients on identical inputs,
plus focused synthetic stress cases: D=1, unequal D/K, frozen blocks,
overlapping/separated components, extreme width/weight ratios, soft-bound
violations/ties, near-coincident GP points, tiny noise and cancellation.
Exercise both factor representations through consistent GP states; never
flip `L_chol` without constructing the matching representation.

For deterministic quantities use central finite differences and existing
`pyvbmc/testing/_check_grad.py` conventions, independent of NumPy gradients.
For the MC surrogate, freeze its detached anchors across perturbations for
finite differences; ordinary finite differences of sampled H test a different
estimator. Also retain analytical K=1/high-sample statistical sanity checks
following `pyvbmc/testing/entropy/test_entmc_vbmc.py`. A disagreement among
AD, baseline and finite differences must be classified before proceeding.

Use the existing oracle comparison definitions in `_oracles.py`: GP-free
outputs at 1e-10; solve-dependent values/gradients at rtol 1e-6, atol 1e-10;
variance outputs at rtol 1e-3, atol 1e-8. Report actual absolute/relative and
blockwise errors, not only pass/fail. Tight same-factor well-conditioned
checks should additionally target rtol/atol 1e-10. These are numerical gates,
not timing criteria. Keep separate raw cancellation and clamped results;
finite nonnegative output after clamping alone is insufficient evidence.
Investigate any failure with residual/conditioning checks; no automatic
tolerance widening, reference updates or new jitter to force a pass.

## Phase 3: complete variational fit

Executor: Sol. Proposed files: `dev/scripts/torch_vi_step.py` and
`dev/scripts/test_torch_vi_step.py`.

1. Implement the complete boundary described above and compare its NumPy
   adapter to unmodified `optimize_vp`. Reuse Python candidate recipes and
   selection policy where possible; any isolated copied orchestration must
   carry a source-hash guard and a parity check. Do not time only preselected
   candidates and label that the complete step.
2. Implement tensor Adam matching `pyvbmc/vbmc/minimize_adam.py`: beta values,
   sqrt(machine-epsilon) fudge factor, step-size decay, clipping, 20-iteration
   stopping batches after at least 40 iterations, slope/covariance test,
   displacement test, trace indexing and final 20-iterate averaging.
   Preserve the pre-update objective/post-update parameter association used
   for midpoint scoring. A stock `torch.optim.Adam` is not the baseline.
   Preserve the stopping math; transfer only the required summaries at
   decision points and count these synchronizations in timings.
3. The deterministic-entropy route (`K=1`/entropy switch) retains the same
   SciPy `minimize` call, tolerance and success handling via a Torch objective
   adapter. Record the effective method selected by SciPy; do not assume it
   is L-BFGS-B or substitute Torch LBFGS. Count all host/device crossings in
   this route and label it a hybrid optimizer in the simplicity assessment.
4. Retain main-loop starts and budgets from effective caller options. Boost
   uses `K_new=max(K,50)` by default, one slow start, its actual enlarged
   sieve, no pruning, entropy-alpha=0, boosted sampling and up to 10,000 Adam
   iterations with existing early stopping. Preserve midpoint/end scoring,
   unpenalized statistics and the current joint ELBO/ELCBO(beta=5) 0.1 guard.
5. Provide two complementary modes: a 100-update Adam replay from identical
   selected starts with early stopping disabled, and a complete production-
   policy fit with ordinary stopping/selection. Fixed-work diagnostics isolate
   numerical differences; they do not replace complete-fit timing. Use the
   same epsilon at each matched event; when branches diverge, record that
   event and each subsequent stream, without forcing identical decisions.
6. Retain per-start update counts, objective/gradient errors, stopping reasons,
   midpoint/end candidates, pruning decisions, I/J shapes and final scores.
   Re-score both backends' candidates through NumPy with five independent
   common scoring streams, using the same fine sample counts. Here candidates
   means retained midpoint/end/selected/pruned outcomes, not the entire sieve.
   Report paired
   score differences and their MC spread, scale/weight validity and analytic
   transformed-space posterior mean/covariance via
   `moments(orig_flag=False, cov_flag=True)`. Do not call the default
   original-space `moments()` (one million RNG draws); original-space moment
   estimation is outside this pass. Do not infer statistical equivalence
   from three fits. Branch
   divergence alone is not a defect, and a material quality loss is not
   dismissed as rounding. Explain it before a feasibility recommendation.

Verification: adapter dispatch/mutation/RNG checks; exact deterministic Adam
update tests with supplied gradients; current early-stop and selection tests
adapted to tensor execution; same-input kernel gates on the actual candidates
around a divergence; successful complete fits include statistics and final
fine scoring, not just an optimizer return.

## Phase 4: bounded measurement matrix

Executor: Sol builds `dev/scripts/torch_vi_benchmark.py`; Astra runs one
heavy process at a time. Each invocation requires explicit cases, device and
output directory, fails on invalid provenance, and retains failures/timeouts.

The eight committed snapshots cover (D,N,K,Ns): warmup (2,20,2,8), K1
(2,20,1,8), single-sample (2,65,5,1), bounded (2,70,9,10), warped
(5,100,17,8), noisy (2,25,2,8), large-K cigar (4,115,14,7) and boosted cigar
(4,115,50,7). All get kernel tests. The following eight complete workloads
are a selected matrix, not a Cartesian sweep:

| Workload | Input and effective regime |
| --- | --- |
| Warmup | Committed warmup; fixed weights, ordinary MC Adam |
| Deterministic | Committed K1; deterministic entropy and SciPy optimizer |
| Bounded | Committed halfnormal; current transformed-space fit |
| Warped | Committed corr D5; ordinary full fit with shared transformer |
| Noisy | Committed noisy Rosenbrock; fixed noisy GP, current main-loop fit |
| Boost, sampled GP | Committed cigar large-K input grown to K=50, Ns=7 |
| Medium synthetic | D=10, N=250, K=25, Ns=5; current full-refit main-loop budgets (recompute flag, two starts) |
| Boost, single GP | D=15, N=750, input K=25 grown to K=50, Ns=1 |

Synthetic states use documented deterministic recipes based on existing GP/VP
test constructors and valid current options; no new target optimization.
Label them as shape/reliability probes, not authentic trajectory evidence.
Add one D=20, K=60, N=500, Ns=1 kernel-only shape to check K>50 handling.
Do not substitute large synthetic speedups for ordinary CPU workloads.
Bounded/warped inputs exercise transformed-space variational arithmetic and
transformer identity at return; they are not evidence for a tensor transformer.

Sample counts are per component after ceiling/even-antithetic rounding:
main objective `ceil(100*K**(2/3)/K)`, boost objective
`ceil(200*K**(2/3)/K)`, fine scoring 4096 per component. At K=50 that means
28/56 objective samples and 204,800 fine samples in total. Retain these
actual counts; reducing them to make GPU memory/timing look better changes
the experiment. The fast sieve uses deterministic entropy by default.

Bound the first measurement pass to three private seeds (1701,1702,1703)
per workload/backend: 24 complete fits each for NumPy CPU, Torch CPU and
Torch CUDA, at most 72 total. Additionally one 100-update replay per
applicable workload/backend, five fine re-scores per candidate, and five
short timing blocks per kernel shape with at most one second per block.
Warm kernels three calls per shape/device; no unrecorded full-fit warmups.
Use a five-minute wall cap per complete fit and a 45-minute aggregate cap per
backend for the measurement pass, including diagnostics. Timeouts are
incomplete evidence, not successful shorter fits. Stop before scheduling
more work when a cap is exhausted; Astra reports the gap and proposes a
bounded follow-up rather than silently expanding the run. No automatic
campaign retries or population launch. Test/debug runs stay focused.

Timing protocol:

1. Set OMP/OPENBLAS/MKL threads to 1 before importing numerical libraries;
   set Torch intra/inter-op threads explicitly in a fresh process. Record
   effective settings. Compare both CPU backends on the same machine/build
   conditions, alternating arm order across seeds. Keep a short unchanged
   NumPy control before/after blocks to identify throttling/background load.
2. Measure import/process startup, GP rebuild/extraction, CPU conversion,
   CUDA context/setup, initial input transfer, random-draw generation/staging,
   warmup and output transfer separately. Report a complete first-call wall
   and complete repeated-fit wall from equivalent host inputs to host result,
   plus resident objective/gradient timings. Include bounds/candidate building,
   optimizer/stopping syncs, fine scoring/pruning and required transfers in
   complete-fit totals. Do not compare Torch resident timing to NumPy total.
   Collect cold first-call measurements in fresh subprocesses; warm CUDA
   preflight state must not erase context initialization from that figure.
3. CUDA wall timing synchronizes the selected device immediately before and
   after the measured region; optional CUDA events are supplemental. Include
   host orchestration and transfers in the host-to-host wall. CPU-only Torch
   or unsynchronized enqueue time is never reported as GPU execution.
   See [PyTorch CUDA synchronization](https://docs.pytorch.org/docs/stable/cuda)
   and [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html).
4. Correctness comparisons use common NumPy draws. Primary complete timings
   include their generation/transfers for all arms. Also time the cost of
   Torch-native float64 RNG separately, without substituting unmatched draws
   in the correctness gate or claiming a native-RNG complete fit was measured.
   Report any resident-input figure as a conditional opportunity only.
5. Report raw repeats, median/range, per-stage and whole-step ratios,
   iteration counts, peak process memory and CUDA allocated/reserved memory.
   Keep diagnostic rescoring and artifact I/O outside fit timings, but record
   their walls. Mark uncontrolled/repeated observations; do not discard slow
   or unsuccessful seeds. A stage-share runtime projection must be labelled
   a projection using an identified profile source, not an end-to-end result.

## Phase 5: review and explicit decision

Executor: fresh Sol independent review (high/xhigh, static only); Astra
reconciles evidence and presents the decision to the PI.

- Review estimator mathematics, current-source parity, dtype/device handling,
  complete-step coverage, timing boundaries and all failures. Run relevant
  focused package tests and explicit developer tests in the main thread;
  do not run historical eta tests against current source or the full suite
  merely for an isolated prototype. Verify production source/fixtures unchanged.
- Quantify maintainability: forward/gradient-specific lines, detach/surrogate
  rules, duplicated orchestration, host bridges, chunking/memory logic,
  device-specific branches and extra dependencies. Count the copied prototype
  scaffolding separately, but do not hide support burden. Show whether a
  change to a current objective term becomes easier to implement and verify.
- Verify installation facts at execution time using installed metadata and
  [official PyTorch installation guidance](https://docs.pytorch.org/get-started/locally/).
  Record the exact CPU/CUDA wheel sources, download/installed footprint,
  Python/OS availability for the current 3.10-3.12 support matrix, and pip/
  conda user steps. Distinguish tested installations from documented routes
  and unavailable combinations. A changed Python floor or hard dependency
  remains a later design decision, not an experiment side effect.
- Present values/gradient gates, reliability limits, CPU and synchronized
  GPU performance, first-use/transfer costs, memory, simplicity and install
  friction together. Outcome choices: evidence supports planning a full port;
  a specified bounded follow-up is needed; or retain NumPy for this release.
  A recommendation does not authorize implementation. No numerical runtime
  cutoff or automatic GPU-overrides-CPU rule is introduced.

## Decisions made by this plan

1. Keep fitted GP state constant and transfer it once: enough to measure the
   complete variational stage without expanding into a GP-training port.
   Rejected: porting gpyreg now (broader evidence, but exceeds feasibility scope).
2. Start with eager tensor kernels, the current Adam policy and a SciPy bridge
   for deterministic entropy. Rejected: stock Torch optimizers or compilation
   as the primary comparison (would change behavior or mix compiler/setup
   questions into the first evidence). Compilation/custom kernels require a
   separately bounded follow-up if eager evidence motivates one.
3. Preserve the MC gradient estimator with explicit autograd semantics.
   Rejected: naive sampled-objective AD (simpler, but changes the estimator).
4. Reuse committed inputs plus clearly labelled synthetic larger shapes.
   Rejected: generating fresh full trajectories or relying on historical local
   boost captures (cost/scope and non-reproducible clean-checkout prerequisites).

## Documentation and remaining work

Update this plan in place with the approved checklist, reproducible commands,
manifest location, measured results, independent review and PI decision. It
uniquely owns this experiment; a separate session report/devlog is unnecessary.
Link it from `dev/README.md`, the roadmap Stage 4 and the current `dev/TODO.md`
pickup. Prototype scripts/tests and recipes are tracked; raw arrays/logs are
ignored, with identifying hashes and compact evidence in the manifest/plan.
No public `.rst` is needed until public API work is authorized.

S-VBMC compatibility is complete; integration remains parked (roadmap pickup
10). Preserve the user-facing `skills/pyvbmc/SKILL.md`, FAQ/reference/helper
and packaging follow-up after the API settles, and machine-local calibration
(roadmap pickup 12; API/cache/budget/release placement still open). Neither is
implemented here. The final 870-case population remains after release changes.

## Planning review and approval history

- Planning inspection only: branch creation, source/document reads, fixture
  metadata and `nvidia-smi`; no optimization, installation, benchmark or tests.
- Independent Sol plan doublecheck complete (2026-09-08): no must-fix
  findings. Fixed its one should-fix finding by specifying analytic
  transformed-space moments, avoiding the default million-draw diagnostic
  and its unbudgeted RNG consumption. Source/scope/decision gates and
  companion records checked; numerical feasibility remains unmeasured.
- Document verification: `git diff --check`, direct untracked-plan
  whitespace/EOF checks and local-link validation pass. Production source
  and numerical fixtures remain unchanged; no numerical tests were needed
  for this planning-only change.
- Execution prerequisite closed: isolated CUDA 12.6/Torch 2.14.0 environment
  validated and used for float64 diagnostics and synchronized complete fits.
- No user design question blocks this plan. Approval would authorize only
  the bounded prototype and measurements above (now approved). The full port needs a second,
  explicit decision after the evidence is reviewed.

## Execution evidence

Review-driven adjustment (2026-09-09): the first campaign retains the planned
72 production-policy fits as **instrumented prototype measurements** and
correctness/branch evidence. Independent review quantified tracing overhead
(the first warmup NumPy adapter is 31.8% above the bare production control)
and identified diagnostic-only host transfers in the Torch arm. Those walls
cannot establish clean backend performance. Complete the same bounded
prototype by adding at most **24 trace-disabled control fits**, one seed
(1701) for each of the eight workloads and three arms, with no algorithm or
kernel tuning. This changes the original fit-count ceiling to 96 including
the retained instrumented fits; it does not increase the original aggregate
45-minute wall budget per backend. Count all control work against remaining
time, retain every outcome, and make no automatic retry. The change repairs
measurement validity within the authorized prototype; it does not authorize
compilation, a full port, full trajectories or the 870-case benchmark.

Also preserve and correct two diagnostic defects without repeating primary
fits: standalone NumPy gradients need the same refreshed `eta` as
`_neg_elcbo`, and comparing production/fixed-100 outcomes with different
pruned K must report the shape difference instead of subtracting unequal
parameter arrays. Original artifacts remain unchanged; corrected kernel-only
checks and any recovered auxiliary diagnostics must identify their sources.

Environment preparation (2026-09-08): isolated Torch 2.14.0+cu126 installed
from `https://download.pytorch.org/whl/cu126` into ignored
`dev/scripts/runs/stage4_torch/deps_cuda`; reference `.venv` untouched.
Install log/report retained there (`install_cuda.log`, `install_cuda.json`).
The 2,602.8 MB wheel and dependencies installed in approximately 5m20s;
overlay footprint 4,350,399,261 bytes. Existing 2.14.0+cpu overlay verified
separately: 634,416,193 bytes. CPU and CUDA float64/autograd/triangular-solve
smokes pass (gradient absolute error 0, solve residual 1.11e-16), both with
one thread. Driver 576.80 remains unchanged. Actual module paths and build
configuration are in `environment_cpu.json`, `environment_cuda.json` and
`environment_cpu_only.json`; the installed PyVBMC version metadata is older
than the checkout, so source hashes/revision govern experiment provenance.
These smokes establish environment viability only, not prototype accuracy.


### Complete-fit performance

The performance comparison uses the 24 **trace-disabled controls**, with
three common-input objective warmups followed by one complete fit per
case/arm (seed 1701). Each worker is a fresh process, with one CPU thread.
These are single observations, not timing distributions. The original
72 instrumented fits retain three seeds of correctness/branch and timing
variability evidence; their walls are not substituted for clean timings.
The clean controls reuse the same numerical kernels and production policy.

Seconds below include state copying, candidate construction/sieve, GP
conversion once, optimization, required entropy draws/transfers, stopping,
fine scoring, pruning and final output synchronization. CUDA uses a host
wall bracket synchronized before and after the fit. Imports, CUDA context,
the three warmup calls and prototype source validation are recorded
separately. This measures one variational stage with frozen GP factors,
not a full VBMC trajectory or a tensor GP-training implementation.

| Workload | NumPy CPU s | Torch CPU s | Torch CUDA s | CPU / NumPy | CUDA / NumPy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmup | 0.015414 | 0.089094 | 0.35979 | 5.78x | 23.34x |
| Deterministic K=1 | 0.0050537 | 0.020764 | 0.092748 | 4.11x | 18.35x |
| Bounded | 0.12105 | 0.48044 | 1.7953 | 3.97x | 14.83x |
| Warped | 0.23485 | 0.49772 | 1.3103 | 2.12x | 5.58x |
| Noisy fixed GP | 0.017494 | 0.10946 | 0.32183 | 6.26x | 18.40x |
| Sampled-GP boost to K=50 | 1.5080 | 3.9318 | 5.0588 | 2.61x | 3.35x |
| Synthetic D=10,N=250,K=25,Ns=5 | 4.7297 | 11.763 | 20.937 | 2.49x | 4.43x |
| Synthetic D=15,N=750,K=50,Ns=1 boost | 8.3773 | 29.446 | 40.528 | 3.51x | 4.84x |

This eager prototype does not supply acceptable CPU performance evidence:
all eight controls are slower, from 2.12x to 6.26x. CUDA is also slower on
this RTX 4060 Laptop GPU, from 3.35x to 23.34x. This conclusion does not use
1.2x as a cutoff and does not trade CPU acceptance against GPU performance.
The single-machine, single-clean-observation design limits precise ratio
estimation and generalization to other hardware. It does not support an
end-to-end solver slowdown estimate.

Peak process working sets, sampled after each clean fit and before auxiliary
work, span 135-201 MiB for NumPy, 295-438 MiB for CPU-only Torch, and
1099-1162 MiB for CUDA Torch. These process high-water marks include imports,
fixture reconstruction and warmup, not just incremental fit storage. Maximum
CUDA allocated memory across the clean controls is 136.6 MiB. The original
instrumented campaign's Windows RSS field was unavailable; it is not filled
retroactively from these controls.

One-time fixed-GP conversion in the clean fits took 0.34-1.87 ms on Torch
CPU and 0.86-2.36 ms on CUDA. Fresh-worker CUDA context probes took
77-100 ms, outside the warmed fit. Resident objective-plus-gradient micro-
measurements also remain slower in representative cases: at the committed
K=50 state, CPU-only Torch took 11.95 ms versus its paired NumPy 5.24 ms;
CUDA took 15.96 ms versus its paired NumPy 3.36 ms (five single-call blocks
after three warmups). These use fixed resident parameters/draws and the
snapshot's objective entropy budget, not full final-fine scoring. The two
NumPy observations differ because they are separate worker measurements.
No cross-worker pooling or extrapolation to complete fits is implied.

Torch-native RNG was measured separately without changing the common NumPy
draws in any correctness gate or fit. For one D=15, K=50 raw-half draw of
shape (50,28,15), the CUDA median was 0.259 ms over five synchronized draws.
This is not a native-RNG complete-fit measurement or evidence that changing
the random stream would improve the complete step. All stage and draw
measurements remain auxiliary to the synchronized whole-fit comparison.

### Simplicity and installation assessment

Autograd removes separate algebraic GP-integral and deterministic-entropy
backward implementations. A new ordinary differentiable objective term can
be added to the forward graph and checked against finite differences without
maintaining a second hand-derived gradient. That is a real development
benefit, but preserving the current method is not a naive forward-only port:
MC entropy needs detached density anchors with both weight dependences kept,
and width normalization needs a detached gauge to reproduce the existing
parameter-gradient convention. Recomputing the MC anchors during a finite
difference test checks a different estimator. Single-width optimization also
exposes the existing stateful NumPy setter behavior; the pure tensor decoder
is compared against a fresh template, and does not silently reproduce drift
from repeatedly mutating a partially fixed-width VP.

The numerical core is 916 physical lines (837 nonblank/non-comment lines):
151 for GP integrals, 101 for MC entropy, 22 for its deterministic lower
bound, 71 for parameter decoding, plus validation, GP/VP conversion, bounds
and objective assembly. It contains 15 explicit `detach` sites and no NumPy
bridges. The developer step adapter is 1,928 physical lines and six NumPy
bridges, including both instrumented and lean paths; fixture/benchmark
scaffolding contributes another 2,636 lines. These are counts of the frozen measured control source, not an
estimate of the eventual production diff. Source cloning, AST guards,
Tensor Adam, SciPy bridging, chunked MC memory handling and transfer-aware
stopping are substantial prototype support costs. No net production
simplification has been demonstrated by this experiment.

The tested CPU-only overlay occupies 0.63 GB; CUDA occupies 4.35 GB and its
2.60 GB Torch wheel took about 5m20s to install with dependencies on this
connection. The CPU-only import took 1.45 s; a fresh CUDA import after the
initial installation cache effects took 1.58 s, with 0.158 s for the first
tiny CUDA float64 smoke. These are measured setup costs, separate from fit
walls. [Installation evidence](../experiments/stage4_torch/installation.md)
records exact builds, pip commands, tested versus available Python/OS
combinations, the untested conda route and packaging limitations. Only
Windows/Python 3.12 was exercised. No core dependency or Python floor changed.


### Numerical checks and diagnostic repairs

All 74 focused tests passed (17.00 s): the developer core/adapter/fixture/
reliability suites plus existing VP and variational-objective finite-
difference tests. This includes CPU/CUDA float64 gates, analytic and
high-sample K=1 entropy, coincident components, raw physical-parameter MC
gradients, independent GP/objective finite differences, parameter masks,
Adam stopping and the production midpoint convention. No full solver suite,
new full `VBMC.optimize()` trajectory or historical eta experiment was run.

All 28 corrected kernel-only records pass: eight committed oracle input
states and three larger synthetic probes on CPU and CUDA, plus warped
seed-1701/1702/1703 checks on each Torch arm. The tolerances retain the
existing oracle lower-quartile scale floor: GP/objective values and gradients
use rtol 1e-6 / atol 1e-10; entropy uses 1e-10 / 1e-12; cancellation-sensitive
variance uses 1e-3 / 1e-8. Original and returned floating arrays must be
float64. The full objective includes bounds/weight penalties. These are
fixed-input checks, not evidence about a future GP optimizer or retry ladder.

The inverse-factor stress test uses near-coincident training rows (minimum
separation 2.83e-8), small effective noise (7.30e-7 / 7.75e-7), and systems
with condition numbers approximately 3.98e6 / 4.11e6. Inverse residuals are
2.61e-10 / 3.30e-10. CPU and CUDA raw pairwise integral and pre-clamp variance
comparisons pass, as do the finite/float64 checks. Cholesky-form fixed-factor
states are covered by the committed snapshot gates. This exercises both
factor representations and cancellation; GP fitting, retry policy and all
possible singular systems remain untested.

All 24 trace-disabled controls match their same-arm, same-seed instrumented
outputs at the declared tolerances, including posterior arrays, retained
statistics, K, budgets, iteration/stopping decisions, pruning and boost
acceptance. Removing tracing therefore preserved these measured outcomes.
Across NumPy/Torch, the 48 complete-fit pairs have no recorded structural
branch differences, but **18 match and 30 fail the strict returned-state
comparison**. This is not a claim of exact complete-state parity.

Every strict mismatch includes stored `eta`. The pure tensor decoder does
not reproduce NumPy's `_neg_elcbo` mutation of that auxiliary field, and the
NumPy parameter setter changes weights without updating eta. The report
retains raw and max-centered eta comparisons: centering does not eliminate
the discrepancy. In the synthetic boost seed 1701, Torch's returned eta
still implies uniform weights while returned `w` does not (maximum weight
difference 0.07696); the returned physical weights themselves agree with
NumPy to about 3e-16. NumPy's own stored eta can also lag the chosen midpoint.
Thus this is an auxiliary-state consistency limitation, not merely a logit
shift or shape convention. A production backend would need to resolve and
validate these state semantics, including downstream posterior methods;
the prototype is not a drop-in returned-state replacement.

Four pairs also accumulate physical-parameter differences above 1e-8:
sampled-GP boost seed 1703 (maximum mean-coordinate difference 0.00284 on
CPU / 0.00774 on CUDA), and synthetic medium seed 1702 (0.000145 / 0.000120).
Their absolute returned ELBO differences are 2.06e-6 / 4.72e-6 and
3.38e-8 / 8.81e-8 respectively. No stopping, K, pruning or boost-acceptance
branch differs. These long-fit differences are retained; no tolerance was
relaxed or trajectory rerun to remove them.

There are 40 valid matched-input fixed-100 cross-arm comparisons, with
maximum objective/gradient absolute differences 6.86e-9 / 3.30e-9. Six K=1
pairs use SciPy and do not have Adam fixed-100 replays. Two synthetic-boost
seed-1703 pairs are excluded because the NumPy diagnostic captured a K=25
start instead of the actual K=50 start (440 versus 865 parameters); equal
entropy draws and gradient shapes alone do not prove matching inputs. The
recovered warped comparisons are included only after checking their saved
starts and draws. Five-stream final rescoring is available for all 48
cross-arm pairs after auxiliary recovery, with matching K/sample dimensions;
maximum paired ELBO difference is 5.18e-6. This bounded local-stage quality
evidence does not establish full-trajectory or population equivalence.

The first 72 complete production-policy fits all finished. Three warped
seed-1703 runs saved their primary results but lost auxiliary reporting when
fixed-100 and production-policy pruning returned different K (17 versus 16).
Shape-safe reporting now records that legitimate difference. Auxiliary-only
recovery rebuilt the six retained candidates from saved arrays, verified
reconstructed theta exactly, and reran only the fixed-100 replay and five
common scoring streams. Original result/raw/failure hashes were unchanged.
Two recovery workers initially rejected the valid one-dimensional Torch eta
representation before numerical work; after accepting either stored eta
shape without changing its values, those two auxiliary recoveries completed
in a separate directory. All failure artifacts are retained.

Four earlier standalone warped gradient records failed because the NumPy
diagnostic called `set_parameters` without refreshing eta before standalone
GP/entropy gradients. The production `_neg_elcbo` refreshes eta, and its
complete-objective gates had passed. Corrected same-input kernel-only
records are separate evidence; the earlier failed records remain failed.
The kernel-only coordinator also needed a null-safe summary for absent
`complete_run`; its first numerical record had already passed and was not
rerun. These reporting/reference repairs changed no production formula.

The primary campaign was instrumented with RNG/parameter hashes, candidate
copies and full trace transfers. Independent review identified their timing
cost, so performance uses the added 24 lean controls, not an estimated
subtraction. Raw `sync_count` counts blocking transfer/decision points and
explicit synchronization brackets; it is not a literal count of
`cuda.synchronize()` calls. Nested transfer timers may include enqueue or
overlapping intervals; they are diagnostic and cannot be added as a stage
partition. The synchronized complete-fit wall is the performance measure.


### Evidence and reproduction

The [timing figure](../experiments/stage4_torch/complete-fit-timings.png)
and its [SVG](../experiments/stage4_torch/complete-fit-timings.svg) show the
clean controls. Compact evidence lives under
`dev/experiments/stage4_torch/`: `results.json`, `timings.csv`, `tables.md`,
`kernel-diagnostics.json`, `recovery-lineage.json`, `environment.json` and
`installation.md`. `measured-source.json` and `control-source.json` identify
the two frozen measured source archives; `campaign-ledger.json` records the
original launch order. `execution-budget.json` totals observed worker walls
and conservatively charges an additional 1200 seconds to every arm for
preflight, validation, reporting and coordinator overhead. Charged totals
are 24.91 / 29.58 / 33.06 minutes for NumPy / Torch CPU / CUDA respectively,
all below 45 minutes. This reserve is an accounting allowance, not a measured
backend runtime. No primary fit was retried or timed out.

Raw JSON/NPZ/logs and the two source ZIPs remain ignored under
`dev/scripts/runs/stage4_torch/{campaign,controls,diagnostics}`. Auxiliary
attempts remain in `aux_recovery`; `aux_recovery_completed` combines the
successful original NumPy recovery with the corrected two Torch recoveries.
The lineage record identifies that composition. Raw arrays are not committed;
the tracked input recipes, fixture/source hashes and compact comparisons
allow a clean checkout to rebuild and rerun the experiment without historic
local captures. Measurements vary by hardware and load.

For a fresh reproduction directory, select the CPU or CUDA overlay as in
the installation record, set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` to `1`, and include `dev/scripts` and the repository root
on `PYTHONPATH`. Use the ordinary environment without overlays for NumPy.
For each of the three arms, the bounded original campaign command is:

```console
python dev/scripts/torch_vi_benchmark.py --out NEW_CAMPAIGN --cases all_complete --backend BACKEND --device DEVICE
```

The reportable tracing-free lane is a separate directory and explicit mode:

```console
python dev/scripts/torch_vi_benchmark.py --out NEW_CONTROLS --cases all_complete --seeds 1701 --backend BACKEND --device DEVICE --trace-disabled --skip-auxiliary --warm-kernels --prior-backend-wall SECONDS_ALREADY_CHARGED
python dev/scripts/torch_vi_benchmark.py --out NEW_DIAGNOSTICS --cases all_kernels --seeds 1701 --backend torch --device DEVICE --kernel-only --prior-backend-wall SECONDS_ALREADY_CHARGED
python dev/scripts/torch_vi_benchmark.py --out NEW_DIAGNOSTICS --cases warped --seeds 1701,1702,1703 --backend torch --device DEVICE --kernel-only --prior-backend-wall SECONDS_ALREADY_CHARGED
```

Here `BACKEND` is `numpy` or `torch`, and `DEVICE` is `cpu` or `cuda`.
The historical launch order rotated arms by seed for traced runs and by
workload for controls; independent commands above show each arm's contents,
not an assertion that all arms may run concurrently. Carry previously spent
time into later lanes; never reset the 45-minute allowance by changing the
output directory. The fixed inputs and the original three seeds are part of
the contract; no new seeds or automatic retries are implied by these commands.

Generate the compact report and figure from retained artifacts:

```console
python dev/scripts/torch_vi_report.py --campaign dev/scripts/runs/stage4_torch/campaign --controls dev/scripts/runs/stage4_torch/controls --diagnostics dev/scripts/runs/stage4_torch/diagnostics --recovery dev/scripts/runs/stage4_torch/aux_recovery_completed --out dev/experiments/stage4_torch
python dev/scripts/torch_vi_plot.py --controls dev/scripts/runs/stage4_torch/controls --out dev/experiments/stage4_torch
```

### Recommendation and explicit decision

**Recommendation: retain the modernized NumPy numerical core for this
release.** The eager prototype demonstrates useful float64 value/gradient
parity, but has returned auxiliary-state differences and some accumulated
fit drift. It does not demonstrate acceptable CPU performance or net
implementation simplicity. The tested GPU also loses at complete-fit level,
including final-boost sizes. Installation footprint and the remaining
estimator/host-bridge complexity reinforce the lack of a case for a full
port from this evidence.

PI decision: **pending**. This recommendation does not implement a backend
or release decision. A further investigation, if desired, needs a separately
bounded purpose and acceptance check (for example, reducing eager CPU
objective overhead while preserving these estimators and rerunning the
complete-step CPU gate). Compilation, custom kernels, native Torch RNG or a
GP-training port were not tested and are not assumed to erase the observed
costs. No additional experiment or full port is launched automatically.

S-VBMC compatibility remains complete and integration parked. The user-facing
`skills/pyvbmc/SKILL.md`, FAQ/reference/helpers and packaging follow-up remain
recorded for after the API settles. Machine-local calibration remains an
open follow-up, including API, cache, budget and release placement. The final
870-case population remains deferred until the final release changes and
explicit backend decision are settled.


### Final verification (2026-09-09)

Two independent Sol reviewers completed the final numerical and measurement
reviews after the report repairs. Neither found a remaining must-fix;
the measurement reviewer also found no remaining should-fix. They verified
the clean/traced parity, matched-start and entropy-stream gates, all recovered
rescoring comparisons, state/drift limitations, original-failure quarantine,
source/launch-order hashes, synchronization and cost boundaries, and the
scope of the recommendation. The agent-credit interruption affected only
final review availability; those critical reviewers were restarted after
credits were restored. No fit was repeated because of that interruption.

Validation: 74 focused prototype/package-gradient checks passed, followed by
four artifact-reader regressions (logit-gauge versus stale relative weights,
and recovered common-stream shape validation). Python syntax/format checks,
document whitespace and local links pass. The timing PNG was visually
checked. All 72 original result hashes still match the frozen campaign
ledger. Production code, numerical fixtures, requirements and CI files have
no diff. The work remains on `dev-stage4-torch-feasibility`. No full port or
final population campaign was performed.

Commit preparation subsequently applied the repository hooks: formatting and
unused standard-library import cleanup only, verified by AST comparison.
The recorded source manifests identify the frozen measured archives before
that cleanup. The generated SVG received whitespace cleanup; its drawing
content is unchanged. Local Git attributes preserve the experiment artifacts'
exact bytes, including the timing CSV's line endings and recorded checksums.
