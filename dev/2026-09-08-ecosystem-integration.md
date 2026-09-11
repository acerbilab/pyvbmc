# Integrating S-VBMC into PyVBMC

*Revised 9 September 2026; proposal for human review.*

Status (PI, 2026-09-11): approved and implemented the same day; the code
is `pyvbmc/svbmc/`. The open questions below were settled and are recorded,
with the execution record, in the
[integration plan](plans/svbmc-integration.md). Where the plan and this
proposal differ (independent sampling with a balanced option, a `seed`
argument replacing `testing`, the toy targets staying upstream, `ValueError`
for an unknown mode), the plan and the code are current.

The PI direction is to make the existing S-VBMC
implementation directly available from PyVBMC and, more generally, to avoid a
separate distribution for every closely related VBMC method. This note defines
a migration of the pinned S-VBMC 0.1.1 source (`13a78f6`) that preserves its
algorithm and existing user workflow.

The unchanged S-VBMC suite and checks of all thirty shipped PyVBMC posterior
files passed on Windows/Python 3.12 against the modernized PyVBMC code. The
[compatibility report](results/2026-09-08-svbmc-compatibility.md) records that
evidence and the known edge cases. The integration should preserve this tested
workflow while moving its ownership, documentation, and release process into
PyVBMC.

As a delivery principle, closely related methods should live in one
distribution with distinct classes, shared documentation, tests, and releases.
Method-specific dependencies can remain optional extras. Concrete methods can
establish common contracts as needed; no plugin registry or general algorithm
framework is required now.

## Proposed code and API boundary

Move the existing `SVBMC` class into `pyvbmc/svbmc/svbmc.py`, with a small
`pyvbmc/svbmc/__init__.py` as its canonical public import. Proposed usage
after integration:

```python
from pyvbmc.svbmc import SVBMC

stacked = SVBMC(vp_list)  # Existing fitted posteriors from the same problem.
stacked.optimize()
samples = stacked.sample(1000)
print(stacked.elbo["estimated"])
```

Also expose `SVBMC` lazily at the package root for discoverability, so
`from pyvbmc import SVBMC` loads the optional implementation only when it is
requested. Importing core PyVBMC must not import Torch. Requesting `SVBMC`
without the Torch extra should produce a concise installation error.

Keep the present class contract. Construction accepts finished
`VariationalPosterior` objects, filters unstable or excessively uncertain
runs, and extracts their component weights and stored expected-log-joint
terms. `optimize()` operates in place and returns `None`; it stores the final
NumPy weights in `stacked.w`, the entropy in `stacked.entropy`, and estimated
and debiased ELBO values in `stacked.elbo`. `sample(n)` returns an original-
space NumPy array, and `plot()` returns a Matplotlib figure. The direct
`stacked_ELBO()` and `maximize_ELBO()` methods retain their current Torch
tensor returns. Optimization retains the `all-weights`, `posterior-only`,
and `ns` modes, including their existing defaults and stopping policy.

This deliberately does not make `SVBMC.optimize()` imitate
`VBMC.optimize()`'s `(posterior, results)` return. It also does not require the
stacked object to implement the full `VariationalPosterior` surface (`pdf`,
`moments`, `mode`, exports, or `save`/`load`). Those methods do not exist in
S-VBMC 0.1.1 and are not needed to integrate the implemented algorithm. They
can be proposed later from demonstrated user needs.

The stacked posterior should remain a composite over its input posteriors.
Each VBMC run can have a different parameter transform, so its components do
not share one valid transformed coordinate system. The current implementation
handles that correctly by retaining the input objects and sampling each one
through its own transform. Flattening the components into one ordinary
`VariationalPosterior` would lose that information and would require a broad
posterior redesign without improving the existing S-VBMC workflow.

## Backend recommendation

S-VBMC holds component locations, scales and stored GP integrals fixed while
optimizing weights. Its sampling, transforms and component-density preparation
already use NumPy/SciPy; Torch evaluates the weight-dependent objective and
provides its gradients and Adam updates.

Retain the upstream Torch weight optimizer for the first integration, using
PyVBMC's existing `torch` extra (`torch>=2.7`) and lazy imports. This is the
smallest path from the validated source to a supported feature and preserves
the public tensor-returning numerical methods. Reviewer assent is required:
the earlier decision to retain NumPy/SciPy for PyVBMC's main solver did not
settle S-VBMC's backend.

The optimized NumPy prototype shows that a later backend change is feasible,
but does not compel it:

| Complete-fit comparison | NumPy result on the study CPU |
| --- | --- |
| Unchanged upstream Torch | 2.47x aggregate speedup |
| Torch with shared preparation and vectorized entropy reduction | 1.06x median paired; 1.15x aggregate |

Most of the first comparison's gain came from reusable sample and density
preparation that benefits either backend. The [full prototype results](results/2026-09-09-svbmc-numpy-prototype.md)
cover numerical parity, timing limits, and the additional maintenance implied
by an analytic gradient and a separate Adam implementation. Apply the shared
preparation and tested Torch reduction improvements in a separate performance
change after the source move; replacing Torch can remain a later API and
maintenance decision.

## Packaging and legacy imports

PyVBMC should own the implementation and must not depend on the standalone
`svbmc` distribution. It should not install a second top-level `svbmc`
package: two distributions owning the same import files would make upgrades
and uninstalls unreliable.

Instead, prepare a separate, temporary compatibility release of `svbmc` that
depends on `pyvbmc[torch]>=1.5`. It should forward both `svbmc.SVBMC` and the
documented `svbmc.svbmc.SVBMC` path to PyVBMC. The current package also
publicly exposes `svbmc.targets` (`GMM`, `Ring`) and `svbmc.utils`
(`overlay_corner_plot`, `find_init_bounds`), and its examples import them.
For a bounded transition, move those small modules to `pyvbmc.svbmc.targets`
and `pyvbmc.svbmc.utils`, keep them out of the PyVBMC root namespace, and
forward the old module paths from the compatibility release. This defines the
initial public support scope without preserving every incidental package
attribute. Preserve the explicitly exported `svbmc.__version__`, reporting
the compatibility distribution's own version; the integrated implementation
follows PyVBMC's release version.

Publish the forwarding release after the PyVBMC version it requires is
available. Coordinate both repositories' release notes and document the
migration and support window for existing users.

The thirty compatibility fixtures are input `VariationalPosterior` pickles.
They encode the existing
`pyvbmc.variational_posterior.variational_posterior.VariationalPosterior`
module path, which this migration does not change, and all loaded without
regeneration. They do not test pickled `SVBMC` instances. S-VBMC supplies no
`save`/`load` API or corpus of serialized stacked objects, so the integration
must not promise transparent migration of arbitrary pickled `SVBMC` objects.
The forwarding package may preserve imports needed by such objects, but that
behavior needs a dedicated fixture before it becomes a compatibility claim.

Both projects use BSD-3-Clause, with separate copyright notices. Retain the
S-VBMC 2025 copyright and license notice in the migrated source and source
distribution. Align the PyVBMC documentation, citation guidance, changelog,
and version metadata with the S-VBMC paper and original package rather than
silently absorbing its provenance.

## Behavioral fixes and policies

The source move does not depend on a new architecture, but reviewers should
choose which of these observable semantics to preserve for the first release:

- Canonicalize component draws to `(n_samples, D)` so D=1 and single-draw
  entropy calculations work. Preserve copied VP weights as `(1, K)`. These are
  narrow shape fixes; the D=1 failure was reproduced in the compatibility
  study and was already present upstream.
- Decide whether `sample(n)` must return exactly `n` rows. The current method
  rounds each posterior's allocation independently and can return a nearby
  count (65 for a request of 64 in the compatibility study). An exact
  allocation is preferable, but it changes existing behavior and needs a
  deterministic allocation rule.
- Document and decide RNG ownership. Entropy currently uses SciPy/NumPy's
  ambient state outside test mode. Sampling deep-copies input VPs, but PyVBMC
  deep copies intentionally share the VP generator, so sampling advances the
  input posteriors' streams. A local generator would improve isolation, but a
  new constructor `seed` argument is not required for migration and should not
  be added incidentally.
- Make the arithmetic policy explicit. Upstream mixes NumPy float64 with
  Torch's default float32 in some input and return paths, then widens final
  weights to NumPy float64. PyVBMC otherwise protects float64 numerics. Choose
  and test a consistent S-VBMC policy rather than describing the current cast
  sequence as guaranteed precision. Use explicit tensor dtypes if changing
  this policy, so S-VBMC does not change the application's global Torch default.

Input validation can be hardened alongside these fixes: reject an empty list,
inconsistent dimensions, malformed component shapes, and missing or nonfinite
`stable`, `elbo`, `I_sk`, or `J_sjk` statistics with useful messages. Users
must still attest that runs describe the same target and original parameter
space; object structure alone cannot establish that fact.

The standalone package declares Python >=3.11, NumPy >=2.3, SciPy >=1.16,
Matplotlib >=3.10, and older PyVBMC/gpyreg floors. PyVBMC currently supports
Python >=3.10, NumPy >=2.0, SciPy >=1.15, and Matplotlib >=3.9. Check the
migrated class and helper modules across these lower core floors and remove
declarations made redundant by integration. Do not raise PyVBMC's core floors
or claim the lower stack until that check passes.

## Review and validation scope

Reviewers need to decide: (1) whether to retain Torch for the initial merge;
(2) exact-count and RNG semantics; (3) the float64 policy; and (4) whether the
temporary forwarding release is maintained for one release or a stated time
window. The class location, preserved in-place workflow, composite posterior
representation, lazy optional dependency, forwarded helper modules, and
absence of a new general framework are proposed defaults.

Implementation validation can stay bounded: first check the moved code against
the pinned upstream outputs with matched random draws, independently of any
behavioral or performance changes. Port the upstream unit suite;
add real PyVBMC regressions for construction, all three optimization modes,
D=1, sampling count/RNG decisions, dtype, and heterogeneous transforms; load
and stack the thirty shipped input posteriors; and test imports both with and
without the Torch extra. Add focused tests for the compatibility package's
forwarded imports and version metadata, and
one deliberately serialized `SVBMC` fixture only if serialized-object support
is claimed. Update API documentation and one example, then coordinate the two
release notes and dependency metadata before any publication.
