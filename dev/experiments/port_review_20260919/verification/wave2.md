# Wave 2 verification: the findings that fire at the default options

The first group of wave-2 findings (slices P1a and P1b; reports under
`../reviews/`), verified on 2026-09-20 on `dev-port-review` at `51451dc`
against MATLAB VBMC at `396d649`. Three were verified by the orchestrator
with the scripts named below. Three were traced through both histories by
a read-only Opus agent, which also searched the code comments, the commit
and pull-request messages, the porting logs, the documentation, `dev/` and
the tests for a recorded reason (`wave2_warmup_history.md`); it found none
for any of them. The MATLAB lines involved all predate the Python lines,
so no row is a faithful port of an older MATLAB. The other wave-2 findings
are not in this file: they have been reported and not yet verified.

## Ledger

| id | reports | statement | verdict (class) | fires at defaults? | how verified | PI disposition |
|---|---|---|---|---|---|---|
| W2-1 | P1a comparison F1 | The "recent improvement" window of `_check_warmup_end_conditions` starts at `iteration - (T + 1)`, with `T = ceil(tol_stable_warmup / fun_evals_per_iter)`, where `private/vbmc_warmup.m:60-65` gives `iteration + 1 - T` in 0-based terms: five iterations at the defaults where MATLAB's window holds three, so the no-recent-improvement test is met less often and warm-up tends to end later | confirmed port discrepancy; never matched MATLAB (one iteration too long as written in `35be58b`, 2021-07-07; two since `9267816`, 2021-08-25, moved the loop to a 0-based counter, corrected the neighbouring floor and left this offset) | yes, from the fifth iteration of every run | both sources read; the window's index sets tabulated for both sides; history traced | fix |
| W2-2 | P1a internal F6, P1a comparison F2, P1b internal F8 | `_recompute_lcb_max` returns an empty array that nothing reads, and `_check_warmup_end_conditions` has no counterpart of MATLAB's preference for the recomputed vector (`private/vbmc_warmup.m:46-50`), while `recompute_lcb_max` defaults to `True` as `RecomputeLCBmax` does: both warm-up criteria rest on the per-iteration maxima, each from that iteration's GP, where MATLAB's rest on a cumulative maximum recomputed with the current GP (`private/recompute_lcbmax.m`) | confirmed port discrepancy; an unfinished placeholder (`pass` in the first commit, an empty array since `ec94eba`, 2021-06-08; the ToDo comment came with a pylint clean-up, `39b29e6`); the porting log and the counterpart map list the function as ported | yes, every run | both sources read; reads of `lcb_max_vec` searched; history traced | fix, the function and the consuming branch |
| W2-3 | P1a comparison F3 | `_setup_vbmc_after_warmup` sets `last_warmup` only, where `private/vbmc_warmup.m:97-102` also sets `LastWarping` and `LastSuccessfulWarping` to the current iteration. Both stay at minus infinity, so the first warp can come as soon as its other conditions hold, where MATLAB waits more than `WarpEveryIters` iterations after warm-up, and the stability termination is not held back for `TolStableIters/3` iterations after warm-up | confirmed port discrepancy; omitted in `35be58b`, eight months before warping existed on the Python side; the two readers came with `4aa4ca9` (2022-03-03) and `0a27ee3` (2023-02-15), neither of which touched the end of warm-up. The third MATLAB line, `LastNonlinearWarping`, is read nowhere in MATLAB | yes | both sources read; readers and writers of the two keys searched; history traced | fix |
| W2-4 | P1b internal F1 | `VBMC.__init__` builds the variational posterior from the untransformed `x0` and transforms `x0` afterwards, so the initial component means, a transformed-space quantity, hold original coordinates. `misc/setupvars_vbmc.m:66` transforms `x0` before `:84` sets `vp.mu`. With bounds 0 and 10, plausible bounds 4 and 6 and `x0 = 5`, the means map back to 9.94 | confirmed port discrepancy; in this order since the first commit of the constructor (`489598a`, 2021-06-03), against a MATLAB that already transformed first. The P1b comparison report lists the initial posterior as equivalent: it compared the tiling of the means, not their coordinates | yes, whenever `x0` and its transform differ; invisible when `x0` is the centre of a plausible box symmetric about it | `scripts/wave2_initial_vp_mu.py` (five cases); both sources read; `git log -S` | fix |
| W2-5 | P1a internal F4, P1a comparison F4, P1b comparison F9 | `_check_termination_conditions` holds termination while `iteration < min_iter`, with the 0-based index, where `private/vbmc_termination.m:98-99` compares the 1-based counter, and where the same function compares `iteration + 1` with `max_iter`: a run on which the minimum binds performs `min_iter + 1` iterations | confirmed port discrepancy; `test_vbmc_check_termination_conditions_prevent_early_termination` asserts the off-by-one | yes, when the minimum binds | `scripts/wave2_min_iter_guard.py` (iterations performed for six pairs of limits against `max(MinIter, MaxIter)`); both sources read | fix |
| W2-6 | P1a internal F8 | `warp_input` inverts the search bounds and the search cache with the transformer of the posterior it is handed, the best recorded one, while both live in the current inference space; a posterior recorded before an earlier warp carries another transformer. Handed such a posterior on a stored four-dimensional state, the second warp returns a search box 8.5 to 12.5 times wider per coordinate than the one the current transformer gives; handed the current posterior it returns that one | confirmed shared defect: `misc/warp_input_vbmc.m:8` and `:133` take the old transform from the posterior handed in, as PyVBMC does | latent: on 21 complete stored runs (42 warps, 13 of them after an earlier kept warp) the selection never returned a posterior from another space, under the rank criterion or under the look-back rule | `scripts/wave2_stale_transformer_mechanism.py`, `scripts/wave2_stale_transformer_frequency.py` | fix |

## Notes

- W2-1 to W2-5 move default trajectories when fixed: the end of warm-up
  (W2-1, W2-2), the first warp and the earliest stable termination
  (W2-3), the first variational optimization of a run whose start point
  changes under the transform (W2-4), and the length of a run on which
  `min_iter` binds (W2-5).
- W2-4 costs the first iteration one of its two slow optimizations: with
  two of them, `optimize_vp` starts the first from the candidate built on
  the incoming means, and the second from the best training points.
- The frequency script of W2-6 reads complete VBMC saves kept under the
  gitignored `dev/scripts/runs/` (the box-sampler runs and the S-VBMC
  pilot pool; `dev/scripts/runs/LOCAL.md` lists them). Of the 42 warps in
  those runs, 25 were undone by the undo check.
