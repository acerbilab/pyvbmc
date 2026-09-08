# Delivering algorithms within the PyVBMC ecosystem

PI direction (2026-09-08): make S-VBMC directly accessible from PyVBMC;
in principle it should not require a separate package. More generally,
avoid a separate distribution for every new algorithm. This note proposes
an approach for discussion; it does not authorize or claim an implementation.
The [initial S-VBMC compatibility check](2026-09-08-svbmc-compatibility.md)
provides the starting evidence.

PI follow-up: the direction sounds good; implementation is parked for
later. Preserve this proposal and the compatibility findings for resumption.

## Proposed delivery model

Maintain closely related VBMC methods in the PyVBMC repository and Python
distribution, with shared documentation, examples, CI and release version.
Give each substantial method its own internal module and public entry point.
Dependencies can remain optional without requiring separate algorithm
packages: an extra installs dependencies, while the code lives in PyVBMC.
For S-VBMC, use the existing Torch extra initially or an explicitly named
stacking extra if that makes installation clearer. Keep imports lazy until
the agreed PyTorch feasibility/port decision settles the core dependency.

Algorithms with distinct workflows should have distinct classes. For
example, S-VBMC consumes multiple finished runs whereas VBMC consumes a
target function. Keep those differences explicit rather than adding modes
to the main optimization loop. Small variations such as acquisitions can
continue using the existing extension points.

Use common input/output contracts where they improve composition: targets,
parameter bounds/transforms, explicit RNGs, posterior operations and result
diagnostics. Establish these from concrete methods, beginning with VBMC and
S-VBMC; a general plugin registry or framework is not needed at this stage.
Research additions can initially live under an experimental namespace with
documented stability, then move into the supported API after validation.

## S-VBMC as the first integration

A proposed user interface, following VBMC's return convention:

```python
from pyvbmc import SVBMC

stacker = SVBMC(posteriors, seed=123)
posterior, results = stacker.optimize()
samples, _ = posterior.sample(1000)
```

This is an API proposal, not currently executable PyVBMC syntax. Users
should pass posterior objects directly; saving and loading files remains
optional. A later convenience layer may orchestrate multiple VBMC runs,
but stacking existing runs should remain independently accessible.

The integration needs more than moving source files:

- Return a posterior with the familiar sampling, density, moments, plotting,
  serialization and optional export interface, and separate optimization
  diagnostics. Specify which methods are supported before shipping.
- Preserve each input posterior's transform. S-VBMC can combine runs with
  different transforms, so flattening everything into the current single-
  transformer `VariationalPosterior` is generally invalid. A stacked
  posterior can implement the common interface while retaining its inputs.
- Give stacking an explicit RNG contract, exact requested sample counts,
  float64 numerical behavior and tests with real PyVBMC objects. The current
  implementation rounds sample allocation independently and its sampling
  can advance input generators; those need intentional integrated semantics.
- Correct the confirmed D=1 component-draw shape defect, retain canonical
  VP weight shapes, and validate that inputs describe the same parameter
  space/problem. The current mock suite does not cover these contracts.
- Replace reliance on an undocumented collection of `vp.stats` keys with
  validated, documented stacking inputs. Keep uncertainty/filtering and
  debiased ELBO diagnostics visible. Validate algorithm behavior separately
  from API changes.
- Coordinate migration of the existing `svbmc` import and distribution,
  potentially through a compatibility shim, retaining attribution and
  existing saved-object support where applicable.

The old S-VBMC requirements are higher than core PyVBMC's Python/NumPy/SciPy
floors. Determine actual language/API needs before deciding whether optional
stacking can support core's floor; its current dependency declarations alone
are not a reason to raise the core requirements. Preserve the previously
agreed PyTorch feasibility criteria and make that decision separately.

## Next concrete design step

Agree the S-VBMC public API and posterior/result contracts, then prepare a
bounded integration plan covering source migration, real-object regression
tests, dependency/import behavior, documentation and legacy import support.
This becomes the worked example for future algorithm additions. The final
population validation still follows the remaining release implementation.
