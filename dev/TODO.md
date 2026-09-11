# Current pickup

Updated 2026-09-11. This file records only current actions and constraints;
completed work and evidence remain in the linked plans and reports.

The [overnight population benchmark](plans/final-population-benchmark.md)
completed all 273 cases without execution errors. The
[assessment](results/2026-09-11-overnight-population.md) finds 259 usable runs
versus 253 on matching reference seeds, with no rejections in either the
76-test KS screen or the 95-test paired analysis after Holm correction.
Noisy Rosenbrock posterior errors increased; noisy logistic lost two usable
runs. Decide whether to add seeds 15–29 for those two configurations
(30 runs, approximately 80 minutes) before reference promotion. Use the
campaign's frozen candidate for an extension of the same treatment; current
`dev-next` also includes the later noisy-acquisition changes. No further
sampling or reference promotion has been selected. No benchmark process
remains active.

## Work awaiting a separate decision

- Extend the core benchmark with at least one realistic noiseless target
  and 2–3 realistic noisy targets, drawing candidates from the lab-private
  [benchflow repository](https://github.com/acerbilab/benchflow). Choose the
  targets and which can move into public tests later. Also improve Slurm HPC
  support for benchmark runs. These are follow-up tasks before the final
  documentation review; see
  [benchmark coverage and HPC support](plans/modernization-roadmap.md#benchmark-coverage-and-hpc-support).
- S-VBMC integration is parked pending review by its main developer. Do not
  implement it before that review; use the
  [integration proposal](2026-09-08-ecosystem-integration.md).
- The [user-facing agent skill proposal](2026-09-02-user-agent-skill.md) remains
  release work; keep it aligned with the settled 1.5 API and guidance.
- NumPy/SciPy is the settled PyVBMC 1.5 solver. Do not start a full Torch solver
  port; optional Torch and ArviZ exports remain.
- The noisy-target acquisition search (a smaller sieve with a gated gradient
  refinement on a larger importance set) and the operation-level savings in
  the VIQR sieve call and the in-loop GP refits await a decision; see the
  [noisy-target acquisitions note](2026-09-08-noisy-acquisitions.md). The
  VIQR loss variants and the EIG acquisitions on that branch are options
  with no default changed; keeping any of them needs the hand-written API
  page AGENTS.md requires for a public class.

## Release boundary and working rules

- After all remaining 1.5 work is complete, but before release, review and
  update the user-facing documentation, especially the stale main
  [README.md](../README.md). Check the main docs and linked MATLAB VBMC wiki
  guidance; assess porting relevant wiki material into the PyVBMC repository
  with Python-specific adaptations. See the
  [pre-release documentation review](plans/modernization-roadmap.md#pre-release-documentation-review).
- After releasing 1.5, respond to issue #138 about explicit RNG control; see
  the [post-release follow-up](plans/modernization-roadmap.md#post-release-follow-up).
- Golden reference generation can involve many runs and substantial compute;
  checking a candidate against that reference can use fewer runs. Choose
  candidate coverage and seed counts for the question and available budget,
  and extend them when useful. A full 870-run candidate population is not
  required before reviewing the first stage. Release publication
  remains a separate action. Use the
  [current golden reference](golden/noisy_extension_20260907/README.md) and
  [latent-fix plan](plans/latent-bug-fixes.md) when that work starts.
- Run at most one heavy computation at a time, normally in the main thread.
- Preserve benchmark artifacts and frozen source checkouts under
  `dev/scripts/runs/population_overnight_20260910/`. These and the full golden
  traces in `dev/scripts/runs/golden/reference_870_20260907/` are gitignored
  and available on the original Windows benchmark machine only. A fresh
  clone has the tracked assessment JSON, comparison and report; to revalidate
  the raw artifacts, copy the campaign directory from that machine and follow
  the [population plan](plans/final-population-benchmark.md) for environment
  and source setup. Run `python dev/scripts/analyze_population_run.py` in
  that environment to reproduce the assessment. Frozen worktrees copied to
  another checkout need their Git links repaired or recreation at the recorded
  commits. No process or old agent session needs reattachment.
- Make planning, proposal, handoff, and status-only edits directly on
  `dev-next`; do not create branches for them. Remove completed feature
  branches after merging.
- The [modernization roadmap](plans/modernization-roadmap.md) is the complete
  tracker; the [1.5 overview](2026-09-06-pyvbmc-1.5-overview.md) gives release
  context. Do not copy completed handoffs back into this file.
