# Delivering algorithms within the PyVBMC ecosystem

PI direction (2026-09-08): make S-VBMC directly accessible from PyVBMC;
in principle it should not require a separate package. More generally,
avoid a separate distribution for every new algorithm. This note proposes
an approach for discussion; it does not authorize or claim an implementation.
The [initial S-VBMC compatibility check](results/2026-09-08-svbmc-compatibility.md)
provides the starting evidence.

PI follow-up: the direction sounds good; implementation is parked for
later. Preserve this proposal and the compatibility findings for resumption.

## Proposed delivery model

Maintain closely related VBMC methods in the PyVBMC repository and Python
distribution, with shared documentation, examples, CI and release version.
Give each substantial method its own internal module and public entry point.
Dependencies can remain optional without requiring separate algorithm
packages: an extra installs dependencies, while the code lives in PyVBMC.
If S-VBMC retains its upstream Torch implementation, use the existing Torch
extra or an explicitly named stacking extra if that makes installation
clearer, with lazy imports. A NumPy implementation is now also a candidate
(see the 2026-09-09 follow-up below); a Torch dependency for stacking has not
been settled.

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

## NumPy alternative after the Stage 4 prototype (2026-09-09)

PI follow-up: the poor eager-Torch feasibility results suggest inspecting
S-VBMC and possibly porting it to NumPy. This records an investigation
direction, not a decision to port or resume the parked integration. The
[Stage 4 results](plans/stage4-torch-feasibility.md) concern PyVBMC's complete
variational step and do not establish S-VBMC's performance.

Static inspection of the pinned S-VBMC `13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01`
used in the compatibility check shows a substantially narrower backend
boundary (`src/svbmc/svbmc.py`):

- Component means/scales and their individual transforms remain fixed. The
  trainable variables are either all component-weight logits or one logit
  per input posterior. S-VBMC reuses stored GP integrals; it does not refit
  GPs or differentiate their integrals during stacking.
- Component sampling, original/transformed-coordinate mappings, Jacobian
  corrections and the component log-density matrix already use NumPy/SciPy
  (`stacked_entropy`, lines 178-227). Torch begins at matrix conversion
  (line 232), followed by weighted log-sum-exp, stratified entropy reduction,
  the linear expected log joint, softmax and Adam/backpropagation.
- A NumPy/SciPy implementation would need the gradient of this weight-only
  sampled objective and both optimizer modes. The entropy gradient must
  retain both the outer component-weight term and the mixture-density
  derivative for the actual samples. Replacing the latter by its population
  expectation would change the finite-sample estimator. Posterior-only
  gradients sum the corresponding component-logit gradients by input run.
- Removing Torch would reduce stacking's installation requirements even if
  runtime gains are modest. Static code inspection cannot establish the
  fraction of time spent in Torch: the existing NumPy transforms/density
  construction may dominate. Different input transforms must still be kept.

A useful next bounded comparison would first match objective values and
weight/logit gradients on identical stored draws and density matrices, then
compare complete stacking optimizations on the three shipped posterior
groups in both weight modes. Preserve the current Adam update, initialization,
rounded-loss stopping and best-iterate selection for that comparison;
PyVBMC's existing Adam helper has different policies and is not an automatic
replacement. Measure NumPy preparation, Torch conversion/reduction/backward,
and complete CPU wall separately. Keep ordinary fresh sampling per iteration:
caching one sample matrix for an entire fit changes the experiment.

The public `stacked_ELBO` and `maximize_ELBO` methods currently return Torch
tensors, and upstream tests assert that contract, although `optimize` stores
NumPy weights and Python scalars. Removing the dependency therefore needs an
intentional direct-method return-type/API decision as part of integration.
Existing tests check gradient existence and broad optimization outcomes,
not numerical gradient values or a matched optimizer trajectory.

Make float64 explicit in both comparison arms. Upstream S-VBMC has mixed
dtype paths and casts returned weights to Torch's default dtype before
widening to NumPy float64, so distinguish arithmetic/backend parity from an
intentional dtype-policy change. Include the established D=1 shape, exact
sample-count, RNG and canonical VP-weight follow-ups in integration design.
The already completed compatibility checks need not be repeated simply to
inspect this alternative. No S-VBMC source change, timing run, port or full
integration was performed for this static follow-up.
