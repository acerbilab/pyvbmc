Stage 3 is integrated, verified, and ready for the reference extension.
Frozen checkout: reference/stage3-20260906 at 7314a6a (non-moving branch,
pushed). Only verification records changed after tested code 4bff1a5.
dev-next has subsequent documentation-only changes, including the human
release overview introduced in 2a9ab03; it is no longer at the frozen HEAD.
Read the current overview and this reminder before switching checkouts.

After explicit campaign authorization, start from the clean original worktree,
use `git switch --detach 7314a6a`, and keep that HEAD and the existing .venv
for both batches. Use .venv/Scripts/python.exe explicitly. After both finish,
return with `git switch dev-next` to record/publish results. Preserve the
generated sidecars and summary when returning; do not advance the frozen
reference branch or replace current release decisions with its older docs.

Passed: branch smoke 34043031387, full matrix 34043071150 (all nine jobs),
integrated smoke 34043979358; exact oracles 8/8; all five replays identical;
base-install suite 858 passed, 35 skipped, 2 xfailed. The feature-branch
suite with optional exports passed 957 tests. Full record:
dev/plans/stage3-pipeline-features.md.

Next: on explicit start authorization, run the two reference batches in
roadmap pickup 3f. One authorization may cover both as a sequential chain;
batch 2 starts only if batch 1 and its checks succeed. A ~20:00 start is
feasible: roughly 12-13 hours plus checks/overhead. Light browsing/docs are
fine; keep the laptop plugged in and awake, without competing heavy work.
Wall times are recorded but not gated; mixed-use timings are not a clean
speed benchmark. Nothing has been started or scheduled.

Batch 1 grows 14 original configurations from 20 to 50 seeds: 420 new runs,
about 7 hours. Batch 2 adds cigar_D8 and student_D8 at 50 seeds each, then
cigar_D15_exhaust at 10 seeds: 110 new runs, about 5 hours. Final population:
810 JSON/NPZ pairs, plus summary.md (811 published files). Check complete
pairs/counts, no error files, sidecar SHAs, and null comparisons before
publishing; do not mask compare failures. Pass shared --options JSON
{"vectorized_target": false} to every run invocation. Existing 280 records
retain metadata 18a236c, dirty false (numerical code bdaf322); new sidecars
record the frozen SHA. Exact commands: roadmap pickup 3f.

After both batches and their records, land the planned latent fixes,
then run final population validation and prepare dev-next -> main for 1.5.
Read-only bug investigation/planning can proceed beforehand. S-VBMC remains
planned alongside better connections to upcoming VBMC extensions (pickup 10).
Whether the PyTorch solver (Stage 4) joins 1.5 is TBD; it would not be a
JAX solver. Revisit core dependencies and the Python floor with that decision.

Local state at handoff (2026-09-06): no Python numerical jobs, scheduled
campaign, or session-bound watchers remain. No campaign launcher was created;
the next session must construct the sequential fail-fast command chain from
roadmap 3f when the start is authorized. There is no job ID to reattach to.

Artifacts deliberately kept out of Git:

- The 280 reference JSON/NPZ pairs are on this laptop under
  C:/Users/luigi/Documents/GitHub/pyvbmc/dev/scripts/runs/golden/item7_20260906/.
  A fresh clone must copy this directory from the laptop to use exact replay.
  The JSON sidecars are also tracked in dev/golden/baseline; the NPZ traces
  are not. Publishing traces as a 1.5 release asset remains future work.
- The original .venv is the base reference environment. Numerical versions
  and verification evidence are in plans/stage3-pipeline-features.md;
  recreate an environment from the documented setup and those versions if
  this venv is unavailable. Exact identity also requires the recorded platform.
- C:/Users/luigi/Documents/GitHub/pyvbmc-stage3 still exists at 7314a6a with
  its separate torch/JAX/ArviZ environment. It is optional for continuing;
  its runs directory is a junction to the original checkout's artifacts.
- Local verification logs are .venv/stage3-integrated-*.log in the original
  checkout and .venv/stage3-*.log in the Stage 3 worktree. Durable outcomes
  are in the plan; rerun the documented checks if detailed logs are needed
  elsewhere. No pending result exists only in a session process.

Read first: dev/2026-09-06-pyvbmc-1.5-overview.md, then this reminder;
dev/plans/modernization-roadmap.md pickups 3f, 5, 9, 10, 11;
dev/golden/README.md and dev/plans/stage3-pipeline-features.md. For bug work,
also read dev/2026-09-02-modernization-discussion.md section 9.
