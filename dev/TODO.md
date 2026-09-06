Pick up PyVBMC at roadmap pickup point 3f in dev/plans/modernization-roadmap.md
(Stage 0 complete: the dtype canary is on dev-next, 282e0ed; nothing running).
Read first: dev/README.md; the roadmap's Stage 0 and Stage 2 bullets and pickup
points 5, 3e, 3f; dev/golden/README.md; dev/plans/stage2-memory.md §Summary and
§Decisions; devlog dev/2026-09-02-modernization-discussion.md §9, the
2026-09-06 entries.

Next: night 1 of the golden-reference extension (grow the 14 original
configurations to 50 seeds; the exact chain, the pre-run replay check and the
expected outcome are in 3f (i)), started only when I say the laptop is free for
about 7 h. Then night 2 (3f (ii), about 5 h), then the records of 3f (iii),
then the latent bug fixes (roadmap pickup 9, plan first; the dtype canary's
widening cast for float32/float16 inputs is one of them; one more population
night at the end), then Stage 3 (plan first), both shipping in 1.5 with
everything else (PI decisions 2026-09-06), then the PR dev-next -> main.
Nothing numerical may land before the two nights are done. The reference traces
live only on this laptop under dev/scripts/runs/golden/item7_20260906/.
