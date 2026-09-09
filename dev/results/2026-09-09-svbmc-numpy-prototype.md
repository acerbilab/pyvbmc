# Optimized NumPy prototype of S-VBMC weight optimization

The optimized NumPy prototype matches upstream numerically and runs 2.47x
faster in the primary complete-fit comparison. Giving Torch the same
preparation improvements narrows that to a median paired speedup of 1.06x
(1.15x in aggregate). This supports NumPy as a viable S-VBMC implementation;
much of the original speedup comes from shared preparation improvements,
and the remaining backend advantage is modest and workload-dependent.

This experiment asks whether S-VBMC's small Torch-dependent numerical core
can be implemented efficiently in NumPy while preserving its sampled
objective and optimization behavior. It is separate from the completed
PyVBMC variational-step feasibility study. S-VBMC keeps the input component
parameters and GP integrals fixed and optimizes component weights, or one
weight per input posterior. The production integration remains parked.

## Scope and implementation

The pinned S-VBMC source is
`13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01` (version 0.1.1), unchanged
from the completed compatibility check. The experiment uses all thirty
shipped posteriors in GMM, GMM_noisy and Ring. Every group contains ten D=2,
K=50 posteriors, giving 500 components in the stacked mixture. With the
default twenty entropy samples per component, each iteration constructs a
10,000 by 500 component log-density matrix.
The adapted developer-only numerical core retains the complete upstream
BSD-3-Clause copyright notice and license.

The [NumPy prototype](../scripts/svbmc_numpy_core.py) makes three substantial
implementation changes:

- It reuses the transforms and Jacobians shared by a posterior's components.
  Original-space densities still retain each posterior's own transform.
- It evaluates diagonal-Gaussian log densities directly in a reusable row
  block, with a default 32 MiB budget for its largest three-dimensional
  work array. The complete log-density matrix occupies another 40 MB.
- It evaluates entropy and its analytic derivative with one reusable
  exponential workspace and vectorized reductions. The derivative includes
  both the outer stratum weights and the mixture-density derivative;
  posterior-only gradients aggregate the component-logit derivatives.

The generic SciPy density wrapper was removed only after a pilot profile
identified it as the dominant preparation cost. The reference retains the
upstream computation. This is optimized NumPy, with explicit float64
arithmetic, rather than a transcription of the Torch component loops.

Initialization, Adam defaults, five-decimal loss comparison, stopping after
five successive non-improvements, and best-iterate selection are preserved.
Every optimization iteration draws fresh samples in the upstream order.
The first-draw correction caches used for debiased ELBO reporting are also
preserved. There is no GP refit, component-parameter optimization, or change
to the underlying finite-sample entropy estimator.

## Comparison design

The primary comparison runs the unchanged upstream `optimize()` against
the NumPy prototype, for both weight modes and seeds 1701, 1702 and 1703:
18 pairs, or 36 clean fits. Both use the defaults of twenty samples per
component, learning rate 0.1, and at most 500 steps. Twelve separate fits
record iteration traces for seed 1701 across the six configurations.

Preparation and fixed-matrix kernels are measured separately. The latter
include an upstream-style Torch loop and a vectorized Torch reduction,
alongside NumPy's analytic gradient. The Torch controls receive resident
float64 tensors; conversion is timed separately. These local kernel controls
are distinguished from the unchanged upstream implementation, which is
called directly for the independent gradient gate and the primary fits.

The kernel results motivated a further bounded attribution control: give
Torch the same optimized preparation as NumPy, vectorize the entropy
reduction, and retain upstream's ELBO and complete optimizer. Eighteen fresh
pairs compare this control with NumPy. The control is kept separate from
the primary measured sources and artifacts.

All fits are serial on an Intel Core Ultra 7 155H Windows laptop, with one
BLAS thread and one Torch intra/inter-op thread. The environment is Python
3.12.6, NumPy 2.5.2, SciPy 1.18.1 and Torch 2.14.0+cpu, with gpyreg pinned
at `a2f8ddce867f502e29717959cf0ff3529f598618`. Backend order alternates
between paired cells. Imports, warmup and loading/construction are recorded
separately from optimization wall time. Conversion needed on each Torch
iteration is included in full-fit timing. Trace and matrix-capture overhead
are excluded from the clean fits.

The upstream optimizer starts from NumPy float64 weights but normally casts
its returned weights to Torch's default float32 before widening them again.
This experiment explicitly sets Torch's default to float64. It tests
float64 arithmetic and behavior separately from that existing output policy;
it is not a claim that the ordinary mixed-dtype path returns identical bits.

## Primary complete-fit results

All eighteen clean pairs pass the declared output tolerances and finish
with identical random-stream states. The largest returned-weight difference
is 5.9e-16; the largest best-ELBO difference is 3.6e-15. These are comparisons
with the actual unchanged upstream optimizer, not the local Torch kernel
controls.

Times below are mean optimization wall times over the three paired seeds;
the speedup is upstream time divided by NumPy time. Loading, construction,
imports and warmup are excluded, while per-iteration preparation and Torch
conversion are included.

| Group / weight mode | Upstream Torch | Optimized NumPy | NumPy speedup |
| --- | ---: | ---: | ---: |
| GMM / all components | 13.28 s | 5.22 s | 2.54x |
| GMM / posteriors only | 16.82 s | 7.10 s | 2.37x |
| GMM_noisy / all components | 15.49 s | 6.30 s | 2.46x |
| GMM_noisy / posteriors only | 15.37 s | 6.11 s | 2.52x |
| Ring / all components | 17.37 s | 6.82 s | 2.55x |
| Ring / posteriors only | 17.59 s | 7.28 s | 2.42x |

Across all eighteen pairs, aggregate optimization time is 287.75 s upstream
and 116.50 s in NumPy, a 2.47x speedup. Individual paired speedups range
from 2.08x to 2.68x. The runs take 10-23 objective evaluations; matching
random-stream states rules out different draw counts as the explanation.

## Attribution and numerical checks

Preparation is bit-identical to upstream on all three checked density
matrices and Jacobian vectors, including the post-draw RNG state. Median
preparation times over three repetitions are 747, 775 and 705 ms upstream
versus 253, 285 and 304 ms in NumPy for GMM, GMM_noisy and Ring respectively.
Much of the full-fit gain therefore comes from preparation improvements
that can also benefit the Torch implementation.

The fixed-matrix timings confirm that a backend-only conclusion would be
too broad. Median value-and-gradient times over ten calls are:

| Group / weight mode | NumPy analytic | Upstream-style Torch loop | Vectorized Torch control |
| --- | ---: | ---: | ---: |
| GMM / all components | 75.5 ms | 155.7 ms | 68.4 ms |
| GMM / posteriors only | 80.2 ms | 155.5 ms | 64.0 ms |
| GMM_noisy / all components | 77.4 ms | 154.1 ms | 60.8 ms |
| GMM_noisy / posteriors only | 79.6 ms | 175.2 ms | 59.8 ms |
| Ring / all components | 109.3 ms | 294.3 ms | 228.2 ms |
| Ring / posteriors only | 150.3 ms | 430.0 ms | 243.0 ms |

Torch columns sum the forward and backward intervals for each call,
but exclude conversion and Adam. NumPy includes the analytic gradient.
The matrix is fixed within these microbenchmarks, whereas complete fits
always resample. Torch matrix conversion costs another 4.4-5.3 ms in the
preparation measurements. Full-fit timings include this cost.

The independent fixed-input gate calls unchanged upstream `stacked_ELBO`
and differentiates it at deliberately unnormalized weights, checking the
upstream distinction between normalized entropy weights and raw weights in
the expected log joint. The maximum direct-weight gradient difference is
5.8e-15. Both logit modes pass their moderate/extreme-logit comparisons;
ten focused NumPy tests also pass, including finite differences and the
local D=1/single-draw correction. Ten supplied Adam updates match real
Torch Adam to 5.6e-17 in the parameter trajectory.

All six separate diagnostic pairs pass, covering 105 matched iterations.
Step counts, stopping, best iteration, every rounded loss and the
per-iteration random-stream states agree. The maximum per-iteration weight
difference is 8.3e-16. No numerical failures required rescoring.

In the primary process, combined imports took 2.72 s and the recorded kernel
warmup took 3.96 s. Loading and constructing each ten-posterior stack took
2.8-14.2 ms. The Adam parity check and provenance checks also precede the
timed fits; their elapsed times were not separately recorded, so those
startup numbers are not a total launch-latency measurement.

## Complete-fit control with shared optimized preparation

The [control](../scripts/svbmc_numpy_control.py) changes only upstream's
`stacked_entropy`: it uses the same optimized NumPy preparation, converts
the matrix to float64 Torch, and uses the vectorized entropy reduction.
The expected-log-joint calculation, correction caches, Adam loop, stopping,
best-state policy and final result handling remain upstream's. Conversion
is included in the timed fit. This is a specific optimized control, not a
claim to have found an optimal all-Torch implementation.

All eighteen new pairs pass, with identical final RNG states. Maximum
returned-weight and best-ELBO differences remain 5.9e-16 and 3.6e-15.
Mean wall times over the three paired seeds are:

| Group / weight mode | NumPy | Shared-preparation Torch | NumPy speedup |
| --- | ---: | ---: | ---: |
| GMM / all components | 4.72 s | 5.02 s | 1.06x |
| GMM / posteriors only | 5.88 s | 7.02 s | 1.19x |
| GMM_noisy / all components | 4.34 s | 4.51 s | 1.04x |
| GMM_noisy / posteriors only | 4.30 s | 4.47 s | 1.04x |
| Ring / all components | 4.49 s | 5.72 s | 1.28x |
| Ring / posteriors only | 4.57 s | 5.71 s | 1.25x |

Aggregate wall time is 84.90 s for NumPy and 97.35 s for this Torch control,
a 1.15x speedup. Every observed pair favors NumPy, but the range is
1.005-1.685x and the median paired speedup is only 1.06x. Small differences
on the Gaussian mixtures should be treated as near parity on this machine;
Ring shows a larger advantage. All observations are retained, including
the unusually large paired ratio contributing to the GMM posterior-only
mean. That ratio also reflects a fast NumPy observation; no cause was
established. There are only three seeds per configuration, not enough to establish
small performance differences across machines or operating conditions.

Absolute NumPy time differs between the two campaigns (116.50 versus
84.90 s for the same cells), demonstrating timing variability in this laptop
study. Comparisons therefore use fresh pairs within each campaign; comparing
one campaign's NumPy times directly with the other's Torch times would be
misleading. No cause for this variation was established. The control's
combined imports took 2.83 s and its separate two-step-per-arm warmup 1.44 s.

## Evidence and limitations

The detailed measurements are retained in the
[primary comparison](../experiments/svbmc_numpy/comparison.json) and
[shared-preparation control](../experiments/svbmc_numpy/control.json).
The [execution plan](../plans/svbmc-numpy-prototype.md) records the bounded
scope and development checks. Both campaigns completed with zero validation
failures. Each measured script's archived bytes and recorded hash match its
live source after execution. Reproduction instructions and frozen sources
are in the [artifact directory](../experiments/svbmc_numpy/README.md).

These inputs cover real final-boost component counts and different posterior
transforms, but only three D=2 examples on one CPU. Matched stochastic
trajectories test implementation parity; they do not establish a general
posterior-quality improvement or convergence guarantee. This is not a GPU
study. Any later GPU comparison still needs float64 reliability checks,
synchronization and transfer/setup accounting; GPU gains cannot substitute
for CPU acceptance, and no 1.2x acceptance cutoff is assumed.

The earlier PyVBMC variational-step prototype already converted fixed GP
factors once per fit, cached posterior/bound templates, and used batched
GP integrals and chunked vectorized entropy. Its NumPy baseline was already
modernized. The specific S-VBMC preparation changes here concern a different
algorithm's sample/density construction. In the earlier study, a resident
K=50 CPU objective-plus-gradient check still took 11.95 ms in Torch versus
5.24 ms in NumPy, so repeated preparation/transfers were not the sole cause
of its slowdown. That was an eager feasibility prototype, not an exhaustive
Torch optimization or compilation study. See the
[Stage 4 evidence](../plans/stage4-torch-feasibility.md). These results do not
reverse the decision to retain NumPy/SciPy as PyVBMC's solver for 1.5.

The numerical NumPy module imports only NumPy and SciPy. Removing Torch from
an integrated S-VBMC would reduce its dependency requirements, but requires
an intentional API decision because upstream's direct numerical methods
return Torch tensors. Analytic derivatives and a maintained Adam update are
additional code responsibilities. The prototype demonstrates a possible
implementation, not a completed dependency or installation change.

The prototype canonicalizes SciPy draws to `(n_samples, D)`, addressing the
known D=1/single-draw shape issue locally. Exact public sample counts, RNG
ownership, canonical VP weights, posterior interfaces, import migration and
dependency-floor support remain integration work. PyVBMC's solver remains
NumPy/SciPy for 1.5. The user-facing skill and calibration follow-ups remain
recorded, and the final integrated 870-case benchmark has not been launched.
