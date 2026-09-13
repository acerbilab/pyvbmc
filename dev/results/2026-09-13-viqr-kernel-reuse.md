# VIQR prediction-kernel reuse: feasibility probe

Reusing prediction's training-to-candidate covariance matrix is promising,
with the clearest gain on the captured late noisy state. The prototype
preserves every checked output exactly on this machine. Retaining the
matrices across hyperparameter samples increases temporary memory.

## Method and coverage

The [probe](../scripts/probe_viqr_kernel_reuse.py) compares the guarded-sinh
source at `7361bb0` with isolated prototype subclasses of gpyreg's GP and
PyVBMC's VIQR. Checked source substitutions add an optional prediction
output containing the cross-kernel matrices, pass it through the acquisition
call, and replace VIQR's duplicated kernel calculation. Production source
and live classes are unchanged. The prototype covers the trained-GP,
separate-sample prediction path used by VIQR; it is not a complete public
gpyreg API implementation.

Two variants use either a transpose view of prediction's matrix or a
C-contiguous copy matching VIQR's previous layout. Every timed call includes
prediction, matrix retention, any copying, and the complete acquisition
wrapper. Matrix lists are local to the call and are not attached to the GP.

Inputs are the seven noisy checkpoints captured during the guarded-sinh
replay, plus its supplementary single-GP-sample Gaussian oracle state.
Each is evaluated with one candidate, a CMA-ES population of 6-10 candidates,
and 8192 candidates: 24 cases per variant. There are six early noisy states
at N=10 and one late noisy state at N=150; coverage of other late noisy
states remains missing. All captured factors use the Cholesky representation.
No targets are evaluated and no integration points are redrawn.

The final measurement sets BLAS thread limits before numerical imports and
asserts the runtime thread counts. Each variant is paired with the baseline
in alternating order over nine rounds, with 31 pairs per round for small
calls and one pair per round for sieve calls. Speedups are medians of paired
baseline/reuse time ratios. The
[raw artifact](../experiments/noisy-acquisition-efficiency/viqr_kernel_reuse_probe.json)
contains all repeats, source and capture hashes, library versions, thread
counts, numerical comparisons, and single-observation traced memory peaks.

## Results

Complete 8192-candidate call speedups:

| State | N / GP samples | Transpose view | Contiguous copy |
|---|---:|---:|---:|
| Logistic regression D5, noise 3, early | 10 / 8 | 1.037x | 1.036x |
| Multisensory D6, noise 1.3, early | 10 / 8 | 1.039x | 1.044x |
| Rosenbrock D2, noise 1, early | 10 / 8 | 1.008x | 1.018x |
| Rosenbrock D2, noise 3, early | 10 / 8 | 1.035x | 1.015x |
| Rosenbrock D2, noise 3, late | 150 / 6 | 1.205x | 1.060x |
| Student D8, noise 3, early | 10 / 8 | 1.034x | 1.026x |
| Timing D5, noise 2.2, early | 10 / 8 | 1.024x | 1.024x |
| Supplementary Gaussian D2 oracle | 65 / 1 | 1.139x | 1.078x |

On the seven noisy states, transpose-view singleton speedups are
1.069-1.083x and CMA-population speedups are 1.060-1.090x. The corresponding
supplementary oracle values are 1.030x and 1.033x. Early sieve gains are
small; nine pairs on one laptop do not establish a reliable gain for every
early state. These measurements do not estimate whole-run speedup.

All 24 cases reproduce the following exactly: prediction means and
variances with reuse disabled and enabled; transposed kernel values versus
VIQR's original calculation; and both variants' public acquisition outputs
versus the baseline. Consequently their selected candidate indices agree.
The baseline also reproduces every stored full-sieve output and three
repeats exactly. State and RNG digests are unchanged after the checks and
timings. This is evidence on the captured states and generating platform,
not a guarantee of bit identity on other BLAS implementations.

The late-state peak traced allocation rises from 64.44 MiB to 102.01 MiB
with views, or 111.38 MiB with copies. Retained kernels alone occupy
56.25 MiB there. Early-state views add about 4.38 MiB to the peak; the
single-sample oracle's view has the same measured peak as its baseline.
These are temporary traced allocations, not process RSS measurements.

## Recommended implementation direction

- Add a supported, opt-in gpyreg prediction output for cross-kernel matrices,
  with ordinary prediction calls retaining their existing return contract.
  Both repositories are maintained by the lab, so this interface can be
  implemented in gpyreg rather than duplicating its predictor in PyVBMC.
- Consume transpose views within the same VIQR call. The copying variant
  adds memory and substantially reduces the gain on the larger cases.
  Retain the existing criterion, samples, search, noise estimation and
  penalties. Preserve the existing custom-acquisition subclass interface.
- Make the temporary-memory policy explicit before production integration.
  Retaining every sample's matrix costs `8 * N * Nc * Ns` bytes. A generous
  bounded-memory fallback to ordinary evaluation is simpler than changing
  candidate batching; streaming one hyperparameter sample at a time is a
  possible alternative but requires a broader prediction interface.
- Validate both GP factor representations, all supported prediction return
  modes, custom kernels/subclasses, dtype, bounds, penalties and RNG
  preservation. Test gpyreg's unchanged default path as well as PyVBMC's
  new path; compare exact and tolerance oracles and run the normal replay
  gate before integration. No new golden campaign was launched for this
  feasibility probe.

Production API integration and its memory policy remain to be implemented.
The adaptive sieve, integration-budget and multistart L-BFGS-B experiments
remain independent work in the
[efficiency plan](../plans/noisy-acquisition-efficiency.md).
