# Current pickup

Updated 2026-09-10. This file records only current actions and constraints;
completed work and evidence remain in the linked plans and reports.

Current work: commit and push `feat/runtime-tips`, require green branch CI,
then merge to `dev-next` and run the full matrix, as authorized on 2026-09-10.
Implementation, local verification and independent review are complete; the
[plan](plans/runtime-tips.md) tracks delivery. No benchmark jobs are in progress.

## Work awaiting a separate decision

- S-VBMC integration is parked pending review by its main developer. Do not
  implement it before that review; use the
  [integration proposal](2026-09-08-ecosystem-integration.md).
- The [user-facing agent skill proposal](2026-09-02-user-agent-skill.md) remains
  release work; keep it aligned with the settled 1.5 API and guidance.
- NumPy/SciPy is the settled PyVBMC 1.5 solver. Do not start a full Torch solver
  port; optional Torch and ArviZ exports remain.

## Release boundary and working rules

- After releasing 1.5, respond to issue #138 about explicit RNG control; see
  the [post-release follow-up](plans/modernization-roadmap.md#post-release-follow-up).
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
