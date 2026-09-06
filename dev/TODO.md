Pick up PyVBMC at roadmap pickup point 3f in dev/plans/modernization-roadmap.md
(Stage 0 complete: the dtype canary is on dev-next, 282e0ed; nothing running),
or at pickup 11 for a Stage 3 session on the branch dev-next-stage3.
Read first: dev/README.md; the roadmap's Stage 0 and Stage 2 bullets and pickup
points 5, 3e, 3f; dev/golden/README.md; dev/plans/stage2-memory.md §Summary and
§Decisions; devlog dev/2026-09-02-modernization-discussion.md §9, the
2026-09-06 entries.

Next: night 1 of the golden-reference extension (grow the 14 original
configurations to 50 seeds; the exact chain, the pre-run replay check and the
expected outcome are in 3f (i)), started only when I say the laptop is free for
about 7 h. Then night 2 (3f (ii), about 5 h), then the records of 3f (iii),
then the merge of Stage 3 from branch dev-next-stage3 (replay identical, full
suite green) with the trajectory-neutral fixes of pickup 9 (the dtype canary's
widening cast for float32/float16 inputs is one of them), then the
trajectory-moving fixes of pickup 9, each replayed, then one population night
on the final code, then the PR dev-next -> main; everything ships in 1.5 (PI
decisions 2026-09-06). Nothing numerical may land on dev-next before the two
nights are done. The reference traces live only on this laptop under
dev/scripts/runs/golden/item7_20260906/.

In parallel, Stage 3 may start now on dev-next-stage3 (roadmap pickup 11: the
branch rules, a separate worktree, daytime only, and the decisions already
taken on the extras, the canary and the batched-target contract), plan first.
Also part of 1.5, form still to be decided: S-VBMC integration and
compatibility (roadmap pickup 10).
