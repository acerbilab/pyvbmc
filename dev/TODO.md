# Current pickup

Updated 2026-09-09. This file records only current actions and constraints;
completed work and evidence remain in the linked plans and reports.

## Deliver machine-local calibration

- Implementation commit `f2f99e3` is pushed on `dev-machine-calibration`.
- Feature CI run `34397131792` passed.
- Next, merge into `dev-next`, push, and delete the merged feature branch
  locally and remotely.
- Then run the full three-OS/three-Python matrix on `dev-next` and record the
  result. Another-machine calibration measurement is useful but not a blocker.
- Do not rerun completed local campaigns or validation unless a new failure or
  source change requires it.

Authoritative records:

- [Calibration plan and live checklist](plans/machine-local-calibration.md)
- [Calibration measurement report](results/2026-09-09-machine-local-calibration.md)
- [Calibration artifacts](experiments/machine_calibration/README.md)

## Work awaiting a separate decision

- S-VBMC integration is parked pending review by its main developer. Do not
  implement it before that review; use the
  [integration proposal](2026-09-08-ecosystem-integration.md).
- Optional runtime hints remain roadmap pickup 13. Preserve quiet mode, use no
  inference RNG, suppress repetition, and settle the open UX choices before
  implementation.
- The [user-facing agent skill proposal](2026-09-02-user-agent-skill.md) remains
  release work; keep it aligned with the settled 1.5 API and guidance.
- NumPy/SciPy is the settled PyVBMC 1.5 solver. Do not start a full Torch solver
  port; optional Torch and ArviZ exports remain.

## Release boundary and working rules

- Launch the final 870-case population and perform release publication only
  when separately authorized. Use the
  [current golden reference](golden/noisy_extension_20260907/README.md) and
  [latent-fix plan](plans/latent-bug-fixes.md) when that work starts.
- Run at most one heavy computation at a time, normally in the main thread.
- No old process, watcher, job, or agent session needs reattachment.
- Make planning, proposal, handoff, and status-only edits directly on
  `dev-next`; do not create branches for them. Remove completed feature
  branches after merging.
- The [modernization roadmap](plans/modernization-roadmap.md) is the complete
  tracker; the [1.5 overview](2026-09-06-pyvbmc-1.5-overview.md) gives release
  context. Do not copy completed handoffs back into this file.
