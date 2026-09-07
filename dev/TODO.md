Reference extension complete and verified (2026-09-07, 08:55 UTC+03).
The checkout is back on dev-next, preserving its newer release decisions.
Frozen reference/stage3-20260906 remains at 7314a6a; it was used with the
original .venv for all 530 new runs, with explicit vectorized_target=False.

Result: 810 complete JSON/NPZ pairs (16 configurations at seeds 0-49,
cigar_D15_exhaust at seeds 0-9), no error files, original 280 pairs
hash-verified unchanged. Batch 1 added 420 runs and passed both null checks
(56 KS tests each). Batch 2 added 110 and passed the final even/odd null
check (68 tests). Fresh preflight and final replays: all five cases identical,
including initial designs. All 10 exhaust runs reached their intended budget;
all 800 other runs reported convergence. Total chain time including checks:
12 h 54 min, under light laptop use; timing is not a controlled speed test.

Durable records: dev/golden/extension_20260907/README.md, three statistical
comparison reports, preflight/final replay reports, validation.json and
sha256_manifest.json. Baseline: dev/golden/baseline (810 sidecars + summary).
Historical sidecars retain 18a236c, dirty=false (numerical code bdaf322).
New sidecars retain 7314a6a: 420 dirty=false; 110 dirty=true because batch-1
publication changed baseline documentation. Numerical source stayed frozen.

Next: prepare the latent-fix implementation plan required by roadmap pickup 9,
then carry out its approved fixes and replay each trajectory-moving change.
Complete S-VBMC integration and connections to upcoming extensions (pickup 10),
resolve the PyTorch feasibility decision below, and validate the final 1.5 code
as a population against this extended reference before dev-next -> main.
No latent fix or solver implementation was made during the reference work.

The first Stage 4 step is a PyTorch feasibility prototype tested on both CPU
and GPU in float64 against the modernized NumPy CPU baseline, including
transfer costs, followed by an explicit decision on the full port
(PI decision, 2026-09-06). Inclusion in 1.5 is preferred if feasible, subject
to numerical reliability, acceptable CPU performance, manageable installation
friction, and simpler future method development. Immediate speedup is not
required; roughly 3× runtime would be a reason to rethink the port or its
release scope, not an agreed numerical cutoff. The full port is not yet a
commitment. Settle the NumPy transition, core dependencies, and Python floor
in the design. Algorithmic extensions remain separate from the port. See
the roadmap Stage 4 entry and the 1.5 overview.

No campaign process, scheduler, or session-bound watcher remains. The launcher
finished both batches, published the local baseline, passed final replay,
returned to dev-next, and released its temporary keep-awake request.
Local operational state/logs: dev/scripts/runs/reference_extension_20260906/,
dev/scripts/runs/golden_{grow,extend}_20260906.log and
.venv/reference_extension_20260906.*.log. The permanent reports above contain
the outcomes; no new run or reattachment is needed.

The full NPZ traces remain gitignored on this laptop under
C:/Users/luigi/Documents/GitHub/pyvbmc/dev/scripts/runs/golden/item7_20260906/.
Copy that directory for exact replay elsewhere; publish the traces as a 1.5
release asset later. The original .venv is the reference environment:
Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1, gpyreg 1.1.0, cma 4.4.4.
The separate pyvbmc-stage3 worktree has its own optional-dependency environment;
its runs directory is a junction to these artifacts. Do not overwrite work in
that checkout. Stage 3 integration and CI evidence remain in
plans/stage3-pipeline-features.md.

Read first: dev/2026-09-06-pyvbmc-1.5-overview.md, this reminder,
dev/golden/extension_20260907/README.md, and roadmap pickups 3f, 9, 10, 11.
For bug work also read dev/2026-09-02-modernization-discussion.md section 9.
