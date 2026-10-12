# S-VBMC preparation and entropy speedups

The vectorized entropy preparation makes S-VBMC weight optimizations about
twice as fast on the three upstream ten-run stacks: 2.06x and 2.15x in
aggregate over two campaigns of 18 paired fits (2.0-2.4x on the two
Gaussian-mixture groups, 1.8-2.0x on Ring), with weights, ELBO, entropy,
evaluation counts and generator states identical to the previous
implementation in every pair. The regression references pass unchanged.
The [plan](../plans/svbmc-speedups.md) records the contract and decisions;
the evidence is in
[`experiments/svbmc_speedups/`](../experiments/svbmc_speedups/README.md).

## What changed

`SVBMC._stacked_entropy` estimates the entropy of the stacked mixture from
`n_samples` draws per component: it needs the log density of every
component at every draw, in the original space, an `(S, K_total)` matrix
with `S = K_total * n_samples`. The previous implementation built it one
component at a time: one inverse transform per component for its draws,
then, for every component, a forward transform of all `S` rows, a Jacobian
of all rows and `scipy.stats.norm.logpdf`, followed by a Python loop over
boolean masks to average the mixture log density per component in Torch.
The new `pyvbmc/svbmc/_entropy.py` does the shared work once per run: one
inverse transform for the draws of all the run's components, one forward
transform and one Jacobian of the `S` rows, and the diagonal-Gaussian log
densities of the run's components in one broadcast over a workspace
bounded at 32 MiB. The per-component means and variances are row
reductions of a `(K_total, n_samples)` view. The draws, one
`standard_normal((n_samples, D))` call per component from the object's
generator in run and component order, the row layout, the density formula
and the autograd path are unchanged. The method has no new state and its
public signatures are unchanged.

## Comparison design

`dev/scripts/svbmc_speedup_benchmark.py` keeps one worker process per
source tree, each importing its own `pyvbmc` with one BLAS thread and one
Torch thread and warmed by an unrecorded two-step fit before any cell is
timed; it alternates which tree runs first in every cell and never runs
the two at once. The before tree is a detached checkout of `dev-next` at
`83692ac`; the after tree is the `dev-svbmc-speedups` working tree with
the change (recorded as `83692ac` plus modifications). A cell constructs
`SVBMC(load_group(group, rng=0)[0], seed=seed)` and times the call
`optimize(n_samples=20, n_samples_final=100, lr=0.1, max_steps=500,
version=mode)`: every optimization step's fresh draws, the Adam updates
and the two final evaluations at 100 draws per component. Loading and
construction, which the change does not touch, are timed separately and
take 0.02-0.05 s. Cells: the three upstream groups (ten runs of K=50, D=2,
500 components), both weight modes, seeds 0-2, 18 pairs. The campaign was
run twice on 2026-09-13 on the Windows 11 laptop (Intel Core, family 6
model 170), Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1, Torch 2.7.0+cpu:
first from 16:28 to 16:34 with an earlier revision of the harness, then
from 16:51 to 16:59 with the final one, which adds provenance and
construction timing. Both processes ran at 98-99% of one CPU.

## Results

Mean wall time over the three seeds of one `optimize()` call in the final
campaign, with the first campaign's ratio of means for comparison:

| Group / weight mode | Before (s) | After (s) | Ratio of means | Paired speedups | First campaign |
| --- | ---: | ---: | ---: | --- | ---: |
| GMM / all components | 14.70 | 6.58 | 2.24x | 2.11x, 2.30x, 2.29x | 2.35x |
| GMM / posteriors only | 14.06 | 6.35 | 2.21x | 2.02x, 2.23x, 2.37x | 2.38x |
| GMM_noisy / all components | 16.51 | 8.32 | 1.98x | 1.87x, 1.87x, 2.26x | 2.22x |
| GMM_noisy / posteriors only | 17.21 | 8.11 | 2.12x | 2.34x, 1.99x, 2.11x | 2.20x |
| Ring / all components | 16.96 | 8.61 | 1.97x | 1.97x, 1.99x, 1.94x | 1.91x |
| Ring / posteriors only | 14.93 | 7.92 | 1.89x | 1.84x, 1.79x, 2.01x | 1.91x |

Aggregate wall time in the final campaign is 283.1 s before and 137.6 s
after, a 2.06x speedup; the paired speedups range from 1.79x to 2.37x with
median 2.01x. The first campaign gave 240.0 s before and 111.9 s after,
2.15x, paired speedups 1.89x to 2.44x with median 2.22x. The machine was
slower during the second campaign on both sides (its before times are
18% higher), which is the laptop timing variation the plan anticipated;
the paired design keeps the ratios comparable, and no cell was slower
after the change in either campaign. The fits take 13 to 25 entropy
evaluations, the same count in both trees of every pair. Per entropy
evaluation (counting the two larger final evaluations), the median fit
spends 0.85 s before and 0.40 s after in the final campaign, 0.75 s and
0.32 s in the first. The prototype control had suggested about 2.5x; the
Ring group, whose fits spend relatively more time in the Torch reduction,
pulls the aggregate below that.

## Numerical agreement

In all 18 pairs of both campaigns the returned weights, ELBO and entropy
are bit-identical (maximum absolute differences 0), the evaluation counts
agree and the SHA-256 digests of the generator state after the fit agree;
the two campaigns also agree with each other. The reported ELBO standard
deviation, whose entropy term is a row variance in Torch, differs in a
few pairs by at most 4e-16 relative, the reduction-order effect the plan
accepts. On this machine a one-off `np.array_equal` comparison at 3 draws
per component found the new density matrix bit-identical to a literal
per-component transcription of the previous code on every fixture group,
warped runs included; at 1 draw per component the warped runs differ by
up to 2.4e-13 relative, because the matrix product of the rotation in the
batched inverse transform rounds differently for a different number of
rows. The shipped equivalence tests (`pyvbmc/testing/svbmc/test_entropy.py`,
at 1 and 3 draws per component) therefore use `1e-12` rather than
exactness. The 18 reference cells of `test_svbmc_references.py` pass
unchanged at their `1e-8` gate; the references were generated under Torch
2.14.0 and this gate was run under Torch 2.7.0. The whole S-VBMC suite
takes 34 s for its 192 tests where the previous 173 took 67 s.

## Where the time goes now

These figures come from a short timing loop on the changed tree, not from
the benchmark: one thread, the 500-component GMM stack, medians of five
repetitions, with `cProfile` for the breakdown; they are indicative and
have no artifact. At 20 draws per component (a 10,000 by 500 matrix), one
entropy evaluation with gradient takes 0.238 s: 0.165 s in the preparation
and 0.073 s in the Torch `logsumexp`, reduction and backward pass. At the
final 100 draws per component (50,000 rows) the same split is 0.867 s and
0.362 s. Within the preparation, transforms and Jacobians take about 4 ms
in total; the rest is the elementwise arithmetic of the broadcast
densities (six passes over `rows x K_m x D` values) and the sum over `D`.
Writing the quadratic form as matrix products would remove most of that
but changes the rounding, so it is a separate decision, not part of this
change.

## Limitations

One laptop, one thread, three seeds per configuration, and only the
two-dimensional upstream corpus at 500 components; the supplementary
D=1-3 fixtures are covered by the equivalence tests but not timed. Matched
trajectories show implementation equivalence, not any change in posterior
quality. The measurement compares this change alone with its immediate
before-source; it says nothing about the original standalone package,
which the S-VBMC benchmark campaign in
[svbmc-integration.md](../plans/svbmc-integration.md#benchmark-campaign-required-for-15)
covers.
