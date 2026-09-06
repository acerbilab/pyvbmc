Stage 3 is integrated, verified, and ready for the reference extension.
Frozen checkout: reference/stage3-20260906 (non-moving branch).
Only verification records changed after tested integration code 4bff1a5.
Keep this frozen HEAD and the existing .venv for both reference batches.

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
planned (pickup 10); Stage 4 and the core Python-floor decision stay deferred.

Read first: dev/README.md; modernization-roadmap.md pickups 3f, 5, 9, 11;
dev/golden/README.md; stage3-pipeline-features.md; stage2-memory.md;
devlog section 9. Plans are under dev/plans/.
