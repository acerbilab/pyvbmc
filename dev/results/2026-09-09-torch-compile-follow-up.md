# Compiled Torch: bounded variational-fit follow-up

Status: complete on `dev-torch-compile-prototype`, from `e0ca4e6`.
The user authorized execution on 9 September 2026. The
[scope](../plans/stage4-torch-feasibility.md#compiled-torch-follow-up-2026-09-09)
reuses the Stage 4 fixed-GP prototype. NumPy/SciPy remains the selected solver
for PyVBMC 1.5; this experiment does not authorize a full port.

Compilation changes the GPU result substantially: on the two boost cases,
median paired warm speedups are 3.86x/6.51x over eager CUDA and 1.28x/1.54x over
NumPy. Compiled CPU remains roughly 2-4x the NumPy runtime by per-arm medians, with no
consistent improvement over eager CPU on the largest case. Cold compiled fits
cost 23-444 seconds on CPU and 12-122 seconds on CUDA. All fixed-input kernel
gates pass, but a sampled-boost seed shows endpoint parameter drift. This
supports retaining the standing NumPy/SciPy decision for 1.5.


The earlier study measured eager Torch. Its complete-fit slowdown is valid
evidence about that implementation, but does not settle whether compiling
the tensor objective and its backward pass changes the comparison. Here the
optimizer, entropy estimator and NumPy random draws remain the same. The
compiled boundary is the existing `_neg_elcbo_forward` for ordinary
optimization and sieve calls. Full-variance/per-component fine scoring stays
eager and remains in fit timings, together with preparation, parameter
handling, optimizer and transfers. The adapter counts these deliberate eager
calls separately and propagates compilation errors instead of hiding them.

This boundary avoids unrolling the fine-scoring entropy's memory-bounded
Python loop into a huge graph: D=15/K=50/4096 entropy samples per component produces
2,400 chunks at the existing 65,536-element workspace setting. This is a
bounded compilation experiment, not evidence for fully compiled fine scoring.

## Toolchain and setup

The existing environments contain Torch 2.14.0+cpu and 2.14.0+cu126 on
Windows/Python 3.12.6. The machine has an Intel Core Ultra 7 155H and an
RTX 4060 Laptop GPU. Runs use one BLAS/Torch thread and one compiler worker;
all heavy work runs serially.

MSVC 14.44.35207 and Windows SDK 10.0.26100.0 were already installed. Their
installation cost is not measured. The probe activates them only in its own
process, following the compiler requirement in
[PyTorch's Windows CPU instructions](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html).

The original CUDA overlay cannot compile: Inductor reports `TritonMissing`.
An isolated `triton-windows==3.8.0.post28` wheel was installed, matching the
[Triton Windows compatibility table](https://github.com/triton-lang/triton-windows).
The wheel download is reported as 52.8 MB; installation took 7.328 s and
initial installed files occupy 157,813,678 bytes. These costs are additional
to the existing Torch installations, whose costs are in the Stage 4 report.
No project dependency or production package was changed.

Because this experiment uses a `pip --target` overlay, Triton's default
lookup under Python's `platlib` misses the bundled CUDA toolkit. Setting the
process-local `CUDA_PATH` to the overlay's `triton/backends/nvidia` resolves
the missing `cuda.lib` failure; no separate CUDA toolkit was installed.

| Float64 toolchain smoke | CPU | CUDA with Triton |
| --- | ---: | ---: |
| Cold forward + backward, seconds | 44.224 | 7.025 |
| Value absolute error | 0 | 0 |
| Maximum gradient absolute error | 5.55e-17 | 4.86e-17 |

These are toolchain checks on a small synthetic reduction, **not VBMC
performance results**. Each uses a fresh Inductor/Triton disk cache and reports
`CompiledFunctionBackward`. Compiler activation, imports and device creation
precede the reported compile wall; the JSON also retains process totals.
The original CPU probe first failed because the harness loaded setuptools
before adding its overlay; that harness failure is retained too.

The [setup archive](../experiments/stage4_torch_compile/setup/manifest.json)
preserves the install record, failed attempts and successful checks. Larger
raw runs and compiler caches remain under ignored
`dev/scripts/runs/torch_compile/` and `dev/scripts/runs/tcc/`.

The first measured-harness pilot also exposed an MSVC generated-file error
with a deeply nested cache path. A retry using short unique cache directories
under `dev/scripts/runs/tcc/` passed. A later coordinator checkpoint hit a
Windows `PermissionError` after that worker had completed; bounded atomic-write
retries were added before the main campaign. The toolchain launcher was also
hardened to refuse reused caches and terminate compiler descendants on timeout.
Those launcher/cache checks postdate the successful smokes; their numerical
probe and compiler-activation logic are unchanged, but the smokes do not carry
a byte-level source manifest. Their JSON/log hashes are retained separately.

## Complete-fit measurements

The four cases reuse Stage 4 fixtures: a committed D=2/K=1 state using
SciPy and deterministic entropy; a committed warped D=5 state; a committed
D=4 sampled-GP state boosted from K=14 to 50; and a synthetic fixed-GP
D=15/N=750 state boosted from K=25 to 50 with one hyperparameter sample.
The last case tests a larger shape, not a full trajectory on a real target.

Each workload/backend runs in a fresh process with a fresh compiler cache.
The first complete fit (seed 1701) is cold. Three subsequent complete fits
(seeds 1701, 1702 and 1703) rebuild the same input state and reuse the process
and cache. Backend order rotates between workloads. All arms use the existing
production stopping policy, without tracing; no objective warmup precedes the
cold fit. This is three warm repetitions per case, not a population study.

The primary fit wall includes state copying, GP extraction/transfer, sieve,
parameter handling, optimizer, required NumPy draws/transfers, fine scoring,
pruning and final GPU synchronization. Imports, compiler setup, CUDA context
creation and fixture reconstruction are reported separately. The external API
wall also includes the prototype's source-provenance check; it is retained but
excluded from the primary fit wall, because it would dominate the smallest fit.

Completed 20 workers / 80 fits in 34.2 minutes. The 24 subsequent
compiled fits reused their graphs/kernels; no warm fit was classified as
recompiled. The other 36 warm fits used NumPy or eager Torch. All times
below are seconds. Each cell gives the median and observed minimum-maximum
over the three warm seeds.

| Workload | NumPy CPU | Eager CPU | Compiled CPU | Eager CUDA | Compiled CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| D2/K1 | 0.00434 [0.0043-0.0046] | 0.0188 [0.0186-0.0249] | 0.00867 [0.0081-0.0108] | 0.0756 [0.069-0.076] | 0.0236 [0.023-0.0249] |
| Warped D5 | 0.274 [0.221-0.278] | 0.722 [0.576-0.776] | 0.574 [0.417-0.575] | 1.16 [0.779-1.18] | 0.406 [0.386-0.585] |
| Sampled-GP boost D4/K50 | 1.48 [1.48-4.41] | 3.94 [3.63-13] | 3.16 [3.06-11] | 4.48 [4.2-15.2] | 1.19 [1.16-2.84] |
| Synthetic boost D15/N750/K50 | 7.9 [7.87-8.52] | 24.7 [24.3-26.4] | 30.4 [21.3-31.8] | 34.1 [30-39.3] | 5.13 [5.12-5.24] |

Paired speedups divide the reference fit time by compiled time for each seed,
then take the median. They are not ratios of the preceding medians.

| Workload | CPU vs eager CPU | CPU vs NumPy | CUDA vs eager CUDA | CUDA vs NumPy |
| --- | ---: | ---: | ---: | ---: |
| D2/K1 | 2.15x | 0.531x | 3.22x | 0.189x |
| Warped D5 | 1.35x | 0.484x | 2.85x | 0.675x |
| Sampled-GP boost D4/K50 | 1.19x | 0.468x | 3.86x | 1.28x |
| Synthetic boost D15/N750/K50 | 0.814x | 0.26x | 6.51x | 1.54x |

Cold first fits have fresh caches and include lazy compilation. The recorded
cold totals also include the measured imports, compiler/MSVC setup, CUDA context,
fixture rebuild and external first-fit API wall; they exclude unrecorded process
startup and environment/manifest work outside the API call.

| Workload | CPU cold fit | CPU recorded cold total | CUDA cold fit | CUDA recorded cold total |
| --- | ---: | ---: | ---: | ---: |
| D2/K1 | 23 | 27.7 | 12 | 17.8 |
| Warped D5 | 85.6 | 90.8 | 46.8 | 52.8 |
| Sampled-GP boost D4/K50 | 64.5 | 69.7 | 106 | 112 |
| Synthetic boost D15/N750/K50 | 444 | 449 | 122 | 128 |

With the observed paired median savings and **unchanged shapes/process/cache**,
compiled CUDA needs an estimated 345 additional warm fits for sampled boost
or 43 for the synthetic large boost to recover its recorded cold disadvantage
against NumPy. Against eager CUDA those estimates are 32 and 4 fits. They are
arithmetic amortization estimates, not measured full-solver break-even points.
None of the compiled CPU cases has a positive median paired warm saving over
NumPy, so this experiment supplies no CPU break-even estimate.

Required transfer/setup costs stay inside the fit wall. For example, warm
seed 1701 records 0.088 s of required transfers in the 1.160 s sampled-boost
CUDA fit, and 0.861 s in the 5.244 s synthetic boost fit. GP extraction/transfer
costs 0.00129 s and 0.00335 s respectively. These internal timers can overlap
and blocking transfers can include queued work; they are not additive device
kernel profiles. All eight compiled workers together left 522,925,315 bytes
of compiler cache after fits **and diagnostics**, recorded in
[cache footprints](../experiments/stage4_torch_compile/cache-footprints.json);
that is separate from installation size.

## Numerical interpretation and evidence corrections

Fixed-input diagnostics compare GP integrals, both entropy modes, the complete
objective and their gradients in float64 against NumPy and matching-device eager
Torch. They retain the Stage 4 oracle comparison: the relative-error denominator
has a lower-quartile reference-magnitude floor, with the original per-field
relative and absolute tolerances. A preliminary report used ordinary
`allclose` and incorrectly flagged the warped deterministic gradient (maximum
absolute difference 3.51e-10); reproducing the original gate resolves that
reporting discrepancy. The NumPy reference arrays from the compared workers
are bit-identical. No tolerance was relaxed and no fit was rerun for this.

The raw harness also incorrectly required the returned boost VP to retain
transformer object identity. Production `VBMC.final_boost` deep-copies its VP, and
the earlier Stage 4 evidence already records false identity on valid NumPy and
Torch boost returns; the outer VBMC caller restores the live identity. The
reporter narrowly reclassifies this sole failure for completed boost runs,
retaining the original `gate_failed` statuses and every other check. Serialized
artifacts do not contain transformer arrays, so this correction establishes the
expected identity contract, not a new measured transformer-value comparison.

Endpoint physical arrays use rtol=1e-8/atol=1e-10. Five common NumPy random
streams rescore each warm endpoint with 4096 antithetic entropy samples per
component (2048 independent standard-normal draws and their negatives).
Matching K/sample shapes receive identical random arrays;
rescore tolerances remain rtol=1e-6/atol=1e-10. These checks expose trajectory
and small-variance sensitivity that a fixed-input kernel gate can miss. Exact
stopping/iteration, pruning/boost decisions and RNG comparisons are retained,
as are normalized eta and final objective statistics. Passing kernel checks
alone is not evidence that complete fits are numerically interchangeable.

All 24 compiled warm fits match NumPy exactly in iteration counts, stopping
reasons, final K, pruning/boost acceptance decisions and final RNG state. The
physical arrays (`mu`, `sigma`, `lambd`, `w`) pass their declared tolerance in
22/24 fits: seed 1703 of sampled boost differs on both compiled devices. The
same 22/24 count holds against matching-device eager Torch. This is parameter
drift along otherwise matching control decisions, not a different stopping
budget.

| Compiled vs NumPy | Physical endpoints passing | Max absolute ELBO rescore difference | Max absolute ELBO-SD difference | Max absolute ELCBO (beta=5) difference |
| --- | ---: | ---: | ---: | ---: |
| D2/K1 CPU | 3/3 | 3.33e-16 | 3.75e-17 | 1.46e-16 |
| D2/K1 CUDA | 3/3 | 0 | 0 | 0 |
| Warped D5 CPU | 3/3 | 2.90e-9 | 5.16e-8 | 2.59e-7 |
| Warped D5 CUDA | 3/3 | 2.74e-9 | 3.90e-8 | 1.97e-7 |
| Sampled boost CPU | 2/3 | 3.41e-6 | 4.52e-7 | 1.29e-6 |
| Sampled boost CUDA | 2/3 | 3.38e-6 | 4.40e-7 | 1.18e-6 |
| Synthetic boost CPU | 3/3 | 1.42e-14 | 6.51e-19 | 1.42e-14 |
| Synthetic boost CUDA | 3/3 | 1.78e-14 | 5.42e-19 | 1.78e-14 |

Each rescore maximum spans three endpoints and five common streams. The
sampled-boost seed-1703 maximum `mu` differences from NumPy are 0.00519 (CPU)
and 0.00541 (CUDA). Against matching-device eager Torch they are 0.00680 and
0.0132; maximum rescored ELBO differences are 4.85e-6 and 8.92e-6. Warped
ELBO-SD/ELCBO checks and some sampled-boost rescores exceed the strict original
rescore tolerance. Small absolute errors do not make those checks passes.

Normalized auxiliary `eta` agrees with NumPy only in the K=1 case (6/24 fits).
This repeats the earlier prototype's state-contract limitation: the tensor
decoder does not reproduce NumPy's `_neg_elcbo` mutation of eta, and NumPy's
setter changes physical weights without refreshing it. Removing a common logit
offset does not fix stale relative logits. The report preserves these
mismatches separately from physical weights and does not claim a drop-in
state-compatible solver. See the original
[Stage 4 numerical findings](../plans/stage4-torch-feasibility.md#numerical-checks-and-diagnostic-repairs).


## Limits and decision

The experiment uses static shapes (`dynamic=False`), full graphs for the chosen
boundary (`fullgraph=True`), default Inductor compilation and one compiler
worker. Full-variance/per-component scoring stays eager by construction;
compilation failures do not trigger a silent fallback. Both CPU and CUDA probes
and campaign diagnostics record `CompiledFunctionBackward` and finite float64
gradients. This is compilation of the backward computation as well as forward.

Warm results assume a reusable process and compiler cache. A full VBMC run grows
its GP training set and changes mixture/entropy shapes, so it can incur further
compilation. This bounded fixed-GP experiment does not estimate that cost or
measure end-to-end `VBMC.optimize()`. No GP-training port, custom kernel, native
Torch random stream, S-VBMC change or broad compiler tuning is included.

CPU acceptance remains a separate requirement; GPU speedups cannot satisfy it.
There is no agreed 1.2x runtime cutoff. The standing decision is to retain
NumPy/SciPy for PyVBMC 1.5. Any full port requires an explicit evidence-based
PI decision. S-VBMC integration remains parked, and the final 870-case benchmark
was not launched. The user-facing skill and calibration follow-ups remain
recorded in the roadmap.

Initial execution caps were ten minutes per toolchain smoke, thirty minutes
per workload/backend worker and three hours total campaign wall. Unsupported
paths and timeouts are evidence, not timings for successful fits.

## Reproducibility and verification

The [artifact README](../experiments/stage4_torch_compile/README.md) links the
complete JSON/NPZ evidence, setup attempts and reproduction commands. The
[generated comparison](../experiments/stage4_torch_compile/comparison/comparison.md)
retains full timings; its JSON includes paired state, RNG, control, kernel,
rescore and amortization checks. All 20 workers and their original statuses
are archived with byte hashes. The exact seven measured script files are in
the source ZIP; they matched the live files when the archive was copied.

After measurement, only the runner's `_fit_gate` identity expectation changed;
an AST comparison confirms that scope, and all 80 archived fit summaries pass
the corrected predicate without rerunning numerical work. Three focused adapter
lifecycle tests pass. The reporter validates archived arrays by shape, dtype
and SHA-256 and reads the copied artifact tree without requiring ignored raw
runs. Production numerical code and the existing Stage 4 numerical prototype
remain unchanged.

Independent Sol reviews checked the compiler boundary, reporting arithmetic,
artifact portability and numerical interpretation. The final physical-state,
control/RNG and rescore table was independently verified against the archived
comparison. Repository formatting hooks and local documentation links pass.
