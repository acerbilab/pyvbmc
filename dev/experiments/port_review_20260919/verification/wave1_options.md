# Wave 1 verification: the options surface and the best-posterior selection

A pass over the declared options that grew out of finding M F2 of
`wave1_M_P2.md` (six options that MATLAB VBMC deleted in 2021 survive as
`.ini` entries), made by the orchestrator and one Opus agent on 2026-09-19,
on `dev-port-review` at `0d24698` and MATLAB VBMC at `396d649`. Every
statement below was reproduced by a test that fails on the code before the
fix and passes after it; the commits named carry those tests.

Corrections by the orchestrator, from the independent check of the wave-1
pass (2026-09-22; `wave1.md`):

- **O-1.** The scan counted as read three options that no module read
  through the options at the time: `temperature` and `diagnostics`, which
  the wave-2 pass registered as inert, and `entropy_force_switch`, whose
  readers took a key of `optim_state` that nothing wrote until the wave-2
  pass read the option (`wave2.md`, rows W2-10 and W2-22).
  The declared options that nothing read were 28, not 25, and 25 after
  `bd47856`, not 22. `output_fcn` is not output warping's callback: MATLAB's
  `OutputFcn` is a generic output function, which MATLAB declares and never
  calls. "Whose consumer takes its value as an argument" fits
  `rank_criterion`, `best_safe_sd` and `best_frac_back`;
  `active_sample_fess_thresh` is hard-coded at its reader, and
  `search_cmaes_best` has no reader because cma returns its best-ever
  point.
- **O-2.** "`load` unaffected" no longer holds: since `9b5213d` `load` checks
  the option values, so a saved run that holds `noise_shaping=True` is
  refused.

## Ledger

| id | statement | verdict (class) | fires at defaults? | how verified | fix |
|---|---|---|---|---|---|
| O-1 | 25 options declared in the two `.ini` files are read by no module: the six of M F2, the knobs of unported features (acquisition-portfolio hedge, variational active sampling, variational sampler, the repository of previous posteriors, output warping's callback, per-stage overrides, noise shaping) and five whose consumer takes its value as an argument (`rank_criterion`, `best_safe_sd`, `best_frac_back`, `active_sample_fess_thresh`, `search_cmaes_best`) | confirmed (surface only) | no numerical effect; a user setting one gets nothing | a scan of the package for reads of every registered key, kept as a test that recomputes the set | `a63b17a`: `INERT_OPTIONS`, "(not used, ...)" markers, a warning at validation; `0d9a422`: a callable default is not compared; 22 names remain after `bd47856` made the three selection options live |
| O-2 | `noise_shaping=True` switches on an input-dependent GP noise function and disables the rank-one update without ever shaping the noise (`noiseshaping_vbmc.m` unported): a configuration of neither toolbox | confirmed port discrepancy (P2 F11) | no | reads of `noise_shaping` at `vbmc.py` and `active_sample.py` only; `noise_shaping_threshold`/`_factor` read nowhere | `bb6ab65`: `NotImplementedError` at construction, naming the missing MATLAB feature; `load` unaffected |
| O-3 | `determine_best_vp` takes `safe_sd`, `frac_back` and `rank_criterion_flag` as arguments, both call sites pass none, so `rank_criterion = True` (the declared default, MATLAB's too) never took effect | confirmed port discrepancy | yes, for a run ending without a stable iteration | MATLAB `vbmc.m` passes the three options to `best_vbmc` at both of its calls | `bd47856`: both call sites pass the options; three inert options become live |
| O-4 | the ranking branch of `determine_best_vp` had never executed: it indexed the iteration history's object-dtype arrays, which raise `IndexError` when used as a mask | confirmed Python-only defect | latent until O-3 | reproduced with a recorded history | `bd47856`: reads through `np.asarray` |
| O-5 | two counts in `determine_best_vp` use `max_idx` (the 0-based index of the last iteration) where `best_vbmc` uses the 1-based index as the number of iterations: the look-back window start and the rank penalty of a non-stable iteration | confirmed port discrepancy | yes, for a run ending without a stable iteration | `best_vbmc.m` read; the window start compared for 1 to 40 iterations and three fractions; a four-iteration ranking tie constructed | `4892fe3` |
| O-6 | `gp_int_mean_fun` is copied into `optim_state["int_mean_fun"]`, which nothing reads | confirmed (surface only) | no | grep | none; recorded by the sheet's `intmeanfun` entry |

## Notes

- `rank_criterion` on at the defaults changes which iteration's posterior a
  run returns when it ends without a stable iteration; a run whose last
  iteration is stable returns from it either way.
- The inert-option warning fires only for a value other than the declared
  default, so option dictionaries recorded by earlier runs pass through
  `VBMC(options=...)` silently.
