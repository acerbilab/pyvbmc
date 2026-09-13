# Current pickup

Updated 2026-09-13. This file records only current actions and constraints;
completed work and evidence remain in the linked plans and reports.

The [population benchmark](plans/final-population-benchmark.md) is complete:
870 candidate cases, no execution failures, independently reviewed assessment
and targeted trajectory investigation. The PI accepted the evidence on
2026-09-13: no convincing evidence of meaningful degradation, with existing
occasional failures outside this campaign's correction scope. The promoted
[current reference](golden/promotion_20260913/README.md) is
`reference_990_20260913`: 870 candidate pairs plus 120 unchanged real-data
pairs. All 990 archives validate, the 92-test even/odd check has no flags,
and the five default replays match all non-timer NPZ arrays, semantic final
results and initial designs. The previous references remain preserved.

The noisy-acquisition options can be assessed against this extended
reference on PI instruction. The frozen treatment `68a43db` defines its
synthetic-target numerics; if a noisy option becomes a default, regenerate
the affected noisy configurations from that code.

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
The [S-VBMC ELBO reporting work](plans/svbmc-elbo-reporting.md) (Phase 1)
is complete and merged into `dev-next`; its regenerated regression
references are the gate for the deferred speedups. Phase 2, evaluating
stacked posteriors through other runs' GPs, remains a separate exploration
in the [ELBO optimism note](2026-09-12-svbmc-elbo-optimism.md), with
campaigns started on PI instruction.

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
`dev/scripts/runs/population_realdata_20260912/` (gitignored). That campaign
is complete. Its 120 pairs are retained in the promoted reference;
the noisy-acquisition options remain a separate task.
Goris stays deferred; which targets move into the public tests is
decided at the documentation review.

## Work awaiting a separate decision

- Assess a scoped PyMC integration for 1.5: structured ArviZ export, a
  target-function adapter and a worked example. The
  [proposal](2026-09-13-pymc-integration.md) records the comparison with
  PR #73, effort estimates and the proposed feasibility check.
- Improve Slurm HPC support for benchmark runs, a follow-up task before
  the final documentation review; see
  [benchmark coverage and HPC support](plans/modernization-roadmap.md#benchmark-coverage-and-hpc-support).
- NumPy/SciPy is the settled PyVBMC 1.5 solver. Do not start a full Torch solver
  port; optional Torch and ArviZ exports remain.
- The noisy-target acquisition search (a smaller sieve with a gated gradient
  refinement on a larger importance set) and the operation-level savings in
  the VIQR sieve call and the in-loop GP refits are ready for a separate
  assessment on the extended noisy benchmark against the promoted reference; see
  the [noisy-target acquisitions note](2026-09-08-noisy-acquisitions.md).
  The VIQR loss variants and the EIG acquisitions on that branch are
  options with no default changed; keeping any of them needs the
  hand-written API page AGENTS.md requires for a public class.

## Release boundary and working rules

- After all remaining 1.5 work is complete, but before release, review and
  update the user-facing documentation. A focused
  [README](../README.md), docs landing page, installation and quickstart
  pass was reviewed and approved by the PI on 2026-09-13.
  The [FAQ](../docsrc/source/faq.md) port and
  instructional link updates are complete; its full Sphinx build and
  inference-example checks remain to be run.
  Check the [agent skill](../skills/pyvbmc/SKILL.md) links against the
  release documentation. See the
  [pre-release documentation review](plans/modernization-roadmap.md#pre-release-documentation-review).
- After releasing 1.5, respond to issue #138 about explicit RNG control; see
  the [post-release follow-up](plans/modernization-roadmap.md#post-release-follow-up).
- Golden reference generation can involve many runs and substantial compute;
  checking a candidate against that reference can use fewer runs. Choose
  candidate coverage and seed counts for the question and available budget,
  and extend them when useful. A full 870-run candidate population is not
  required before reviewing the first stage. Release publication
  remains a separate action. Use the
  [current golden reference](golden/promotion_20260913/README.md) and
  [latent-fix plan](plans/latent-bug-fixes.md) when that work starts.
- Run at most one heavy computation at a time, normally in the main thread.
- Preserve benchmark artifacts and frozen source checkouts under
  `dev/scripts/runs/population_overnight_20260910/`, the extension's
  artifacts under `dev/scripts/runs/population_extension_20260911/`, the
  completion under `dev/scripts/runs/population_completion_20260912/`, and the
  real-data reference campaign under
  `dev/scripts/runs/population_realdata_20260912/`. These and the full
  golden traces in `dev/scripts/runs/golden/reference_990_20260913/` (and
  the previous `reference_990_20260912/` and `reference_870_20260907/`) are
  gitignored and available on the original Windows benchmark machine only.
  A fresh clone has the
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
