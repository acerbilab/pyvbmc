Status: in progress.

Pick up PyVBMC at roadmap pickup point 3f in
dev/plans/modernization-roadmap.md. Stage 3 is implemented on
dev-next-stage3 at code 4ee612d with records/docs 285cd74. The branch is
pushed; smoke run 34043031387 and full-matrix run 34043071150 are green
(all nine matrix jobs). Stage 3 has not been
merged into dev-next, the integrated checks have not run, and there is no
freeze hash.

Next: after both Stage 3 CI runs pass, merge dev-next-stage3 into dev-next.
On that integrated checkout run the exact oracle check, the golden replay
(all five cases must be identical), and the full local suite. Record the
passing integrated commit and freeze that code for both reference-extension
nights. Every reference configuration must keep vectorized_target=False.
The existing sidecars and stored traces retain their recorded 18a236c (clean-tree)
provenance; its numerical code is that of bdaf322; every new sidecar records the actual frozen integrated code.

Night 1 grows the 14 original configurations to 50 seeds and takes about
7 h. Night 2 adds cigar_D8 and student_D8 at 50 seeds and
cigar_D15_exhaust at 10 seeds and takes about 5 h. Both must use the same
frozen code. They may run as one explicitly authorized chain, about 12–13 h
plus check and publication overhead; batch 2 starts only if batch 1 and all
of its checks succeed. Keep the laptop plugged in and awake, with no
competing heavy numerical job. Light browsing and documentation work are
fine, but mixed-use wall times are not a clean speed benchmark (`wall_s` is
not a result-gate metric). A proposed approximately 20:00 start is neither
scheduled nor authorized: launch only after the PI's explicit word. Use the
exact commands and checks in roadmap pickup 3f.

Only after both nights and their records are complete may latent bug work
start. This boundary applies to trajectory-neutral fixes, including the
float32/float16 widening cast, and trajectory-moving fixes. After those
fixes, run the final population night, then prepare the dev-next -> main PR;
everything ships in 1.5 (PI decisions 2026-09-06). Stage 4 remains deferred.
S-VBMC integration and compatibility remain planned, with their form still
to be decided (roadmap pickup 10).

Read first: dev/README.md; the roadmap's Stage 0 and Stage 2 bullets and
pickup points 5, 3e, 3f, 9 and 11; dev/golden/README.md;
dev/plans/stage2-memory.md §Summary and §Decisions; devlog
dev/2026-09-02-modernization-discussion.md §9, the 2026-09-06 entries.
