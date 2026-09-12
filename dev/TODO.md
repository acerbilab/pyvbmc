# Current pickup

Updated 2026-09-12. This file records only current actions and constraints;
completed work and evidence remain in the linked plans and reports.

The [overnight population benchmark](plans/final-population-benchmark.md)
(273 cases, seeds 0–14 of every configuration) and its
[follow-up extension](results/2026-09-11-noisy-follow-up-seeds-15-29.md)
(seeds 15–29 of noisy Rosenbrock SD3 and noisy logistic SD3, 30 cases)
completed without execution errors, giving a 303-run candidate population
of one frozen treatment. The pooled assessment finds 287 usable runs versus
273 on matching reference seeds, no rejections in the 76-test KS screen or
the 95-test paired family, and the seeds 0–14 shifts in the two noisy
configurations did not replicate on seeds 15–29: none of the eight
confirmatory tests rejects, and every Rosenbrock metric moved in the
candidate's favor. The PI accepts this evidence tentatively (2026-09-11):
the frozen treatment stands as the numerics of 1.5, and the confirmation
at equal sample sizes comes from the planned extension rather than from a
separate decision now. That extension, completing the remaining 567 seeds
of the same frozen treatment and then promoting the full 870-run candidate
population to the new golden reference in place of the current one, is
agreed in principle but not to be started without an explicit go-ahead;
its conditions and cost are in the plan's
[direction under consideration](plans/final-population-benchmark.md#direction-under-consideration--2026-09-11).
Order set by the PI (2026-09-11): the benchmark extension with realistic
noiseless and noisy targets first (done 2026-09-12, see below), then the
golden run on the current defaults; the noisy-acquisition options are taken
up only after both, so they can be analyzed on the richer benchmark. The frozen
checkout `68a43db` remains the generator as long as the defaults stay as
they are; if a noisy option later becomes a default, the noisy
configurations of the reference are regenerated from that code. No
benchmark process remains active.

S-VBMC is integrated as `pyvbmc.svbmc` (merged into `dev-next` at
`0b5af29` on 2026-09-11); the [integration plan](plans/svbmc-integration.md)
records the decisions, the worklog and the verification. Two items remain
from it, neither to be started now: the forwarding release of the
standalone `svbmc` package (depending on `pyvbmc[torch]>=1.5`, forwarding
the class and helpers, mapping `testing=True` to a seed with a deprecation
warning) once PyVBMC 1.5 is on PyPI, and the deferred preparation and
entropy speedups measured in the
[NumPy prototype](results/2026-09-09-svbmc-numpy-prototype.md), gated by the
regression references in `pyvbmc/testing/svbmc/`.
Separately, agreed 2026-09-12: the two-phase plan for
the stacked ELBO's optimism on noisy targets in the
[S-VBMC ELBO optimism note](2026-09-12-svbmc-elbo-optimism.md). All five
Phase 1 defaults are accepted; implementation and verification are tracked
in [svbmc-elbo-reporting.md](plans/svbmc-elbo-reporting.md). That phase
regenerates the same regression references, so it is ordered before or
together with the deferred speedups, never after them unnoticed.

The benchmark extension with real-data targets is complete (2026-09-12);
the decisions, definitions, method and evidence are in
[benchmark-realistic-targets.md](plans/benchmark-realistic-targets.md).
The targets are the Bayesian timing model and the multisensory
causal-inference model on two subjects from the lab-private
[benchflow repository](https://github.com/acerbilab/benchflow), on plain
`.npz` data with spline-trapezoidal priors and the paper's plausible boxes,
with importance-weighted ground truths committed under
`dev/scripts/data/truths/`. The reference campaign (four configurations,
seeds 0–29, 6 h 05 min on the benchmark machine) ran from a frozen checkout
at `fc50ee1` certified by the oracle `--check --exact` (11 of 11) and by
`golden_replay.py` reporting `identical` on the five default cases against
the `68a43db` traces, and was joined to the reference as
`reference_990_20260912` (`dev/scripts/reference_join.py`; record in
[realdata_extension_20260912](golden/realdata_extension_20260912/README.md)):
92-test even/odd null check with no flags, seed-0 replays of the four new
configurations identical. Findings to carry into the noisy-acquisition
work: the noisy runs stop on stability at about 205 to 215 evaluations
inside their budgets; noisy multisensory subject 1 has 5 of 30 usable runs
(median gsKL 2.4); every real-data posterior is under-dispersed, timing's
`w_s` and `sigma_p` by about 30 % over all 30 seeds. The frozen worktree
and campaign artifacts are under
`dev/scripts/runs/population_realdata_20260912/` (gitignored). No process
is running. Next in the PI's order: the golden run on the current defaults
(the 567-seed completion and promotion, not to be started without an
explicit go-ahead; see the population plan), then the noisy-acquisition
options. Goris stays deferred; which targets move into the public tests is
decided at the documentation review.

## Work awaiting a separate decision

- Look at golden run `logreg_D5` seed 5 of `reference_870_20260907`: a
  converged noiseless run that ends with an ELBO standard deviation of
  4.6 and an ELBO error of 2.7 while marked stable (noticed 2026-09-12
  while aggregating the sidecars' final ELBO standard deviations for the
  [S-VBMC ELBO optimism note](2026-09-12-svbmc-elbo-optimism.md)).
- Improve Slurm HPC support for benchmark runs, a follow-up task before
  the final documentation review; see
  [benchmark coverage and HPC support](plans/modernization-roadmap.md#benchmark-coverage-and-hpc-support).
- The [user-facing agent skill proposal](2026-09-02-user-agent-skill.md) remains
  release work; keep it aligned with the settled 1.5 API and guidance.
- NumPy/SciPy is the settled PyVBMC 1.5 solver. Do not start a full Torch solver
  port; optional Torch and ArviZ exports remain.
- The noisy-target acquisition search (a smaller sieve with a gated gradient
  refinement on a larger importance set) and the operation-level savings in
  the VIQR sieve call and the in-loop GP refits are deferred until the
  benchmark extension and the golden run are complete, so that they are
  analyzed on the extended noisy benchmark against the new reference; see
  the [noisy-target acquisitions note](2026-09-08-noisy-acquisitions.md).
  The VIQR loss variants and the EIG acquisitions on that branch are
  options with no default changed; keeping any of them needs the
  hand-written API page AGENTS.md requires for a public class.

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
  [current golden reference](golden/realdata_extension_20260912/README.md) and
  [latent-fix plan](plans/latent-bug-fixes.md) when that work starts.
- Run at most one heavy computation at a time, normally in the main thread.
- Preserve benchmark artifacts and frozen source checkouts under
  `dev/scripts/runs/population_overnight_20260910/`, the extension's
  artifacts under `dev/scripts/runs/population_extension_20260911/` and the
  real-data reference campaign under
  `dev/scripts/runs/population_realdata_20260912/`. These and the full
  golden traces in `dev/scripts/runs/golden/reference_990_20260912/` (and
  the previous `reference_870_20260907/`) are gitignored and available on
  the original Windows benchmark machine only. A fresh clone has the
  tracked assessment JSON, comparison, manifests and reports; to
  revalidate the raw artifacts, copy the campaign directories from that
  machine and follow the
  [population plan](plans/final-population-benchmark.md) for environment
  and source setup. Run `python dev/scripts/analyze_population_run.py` in
  that environment to reproduce the pooled assessment. Frozen worktrees
  copied to another checkout need their Git links repaired or recreation at
  the recorded commits. No process or old agent session needs reattachment.
- Make planning, proposal, handoff, and status-only edits directly on
  `dev-next`; do not create branches for them. Remove completed feature
  branches after merging.
- The [modernization roadmap](plans/modernization-roadmap.md) is the complete
  tracker; the [1.5 overview](2026-09-06-pyvbmc-1.5-overview.md) gives release
  context. Do not copy completed handoffs back into this file.
