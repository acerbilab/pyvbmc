# Wave 6 fix pass: PyVBMC fix agent C

Raw report of fix agent C, wave 6 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-22. One Opus agent on its
own git worktree of PyVBMC, fast-forwarded to `b7622e51`, made the fixes of
the two PyVBMC-side rows of `../verification/wave6.md`, W6-23 and W6-37, one
commit per row as the PI ruled them; the orchestrator reviewed the diffs and
cherry-picked the commits onto `dev-port-review`, where they have other
hashes (`../verification/wave6.md`, "Fix commits"). The text below the rule
is the agent's final message, unedited.

---

# Wave 6 fix pass: report of PyVBMC fix agent C

## Setup

- `git merge --ff-only dev-port-review` → `Updating b2c16ebf..b7622e51` / `Fast-forward` (2682 files updated, no conflict).
- `git log -1 --format=%h` → `b7622e51`.
- `python -c "import os, sys; sys.path.insert(0, os.getcwd()); import pyvbmc; print(pyvbmc.__file__)"` →
  `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-abc7932b8fc368a51\pyvbmc\__init__.py`
  (under the worktree, as required).
- Interpreter `C:\Users\luigi\Documents\GitHub\pyvbmc\.venv\Scripts\python.exe`, every command with
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, one test file at a time from the worktree root.

## Commits

| hash | row | what it does | test file and name | failing assertion seen before the fix | test-file run after |
|---|---|---|---|---|---|
| `4d356e00` | W6-23 | Removes `hyp_dict["logp"]`: the `None` back-fill (`gaussian_process_train.py:70-71`), the `res["log_priors"]` assignment (`:178`) and its comment, and the `None` in the no-sampling branch (`:195`). Removes the two other readers in `pyvbmc/`: `"logp"` from the recorded outputs of `pyvbmc/testing/oracles/_gp_fit_history.py: fit_outputs`, and the `hyp_dict_logp` reference name from the three `gp_fit_history` fixture sidecars. Changes two shipped assertions (below). | `pyvbmc/testing/vbmc/test_gp_train_fit_inputs.py::test_the_fit_keeps_no_log_density_of_the_hyperparameter_samples` (new) and `::test_ending_warmup_clears_the_covariance_the_fit_reads` (tightened set) | new test: `assert "logp" not in hyp_dict` → `AssertionError: assert 'logp' not in {'hyp': array([[...]]), ...}`. Tightened set: `assert set(vbmc.hyp_dict) <= {"hyp", "warp", "full", "run_cov"}` → `AssertionError: ... Extra items in the left set: 'logp'` | `test_gp_train_fit_inputs.py` 14 passed; `pytest pyvbmc/testing/oracles` 143 passed, 15 skipped (the three platform-bound `gp_fit_history` replays among the passes); `test_gaussian_process_train.py` 12 passed; `test_gp_training_policy.py` 27 passed |
| `d98ff434` | W6-37 | `VBMC.load` refuses any name of `new_options` that is in the new module-level `_CONSTRUCTION_ONLY_OPTIONS` (`pyvbmc/vbmc/vbmc.py:61-110`), with a message naming it and saying to construct a new `VBMC` object. The `gp_mean_fun` value check and the `integer_vars` form check move out of `_init_optim_state` into `_validate_option_values` as `_validate_gp_mean_fun_option` / `_validate_integer_vars_option`, so both routes run them; the value checks run before the names are weighed. `load`'s `new_options` docstring and `Raises` block corrected, and the FAQ entry on continuing a run says the same. | `pyvbmc/testing/vbmc/test_vbmc_option_names.py::test_load_refuses_an_option_only_construction_reads` (11 parameters), `::test_load_takes_an_option_a_continued_run_reads`, `::test_load_refuses_a_gp_mean_fun_construction_refuses`, `::test_load_refuses_an_integer_vars_construction_refuses` | all thirteen: `Failed: DID NOT RAISE ValueError` at `with pytest.raises(ValueError)` around `VBMC.load(saved, new_options=...)` (13 failed, 38 passed) | `test_vbmc_option_names.py` 51 passed; `test_vbmc_init.py` 160 passed; `test_vbmc_save_and_load.py` 20 passed; `test_options.py` 76 passed; `test_vbmc_precomputed.py` 36 passed; `test_gp_records.py` 14 passed; `test_runtime_tips.py` + `test_gp_training_policy.py` 55 passed; `calibration/test_lifecycle.py` 14 passed; `test_acq_legacy_save.py` 2 passed; `test_vbmc_loop_state.py` 5 passed |

## The construction-only options, with the reader of each

Derived by a scripted search of `pyvbmc/` (excluding `pyvbmc/testing`) for every read of every option
declared in the two shipped `.ini` files, counting `options[...]`, `options.get(...)`, `"name" in options`
and the two `Options` methods that read one, and **not** counting a read of the `optim_state` entry of the
same name; then each candidate re-checked by grep and eyeballed. 19 names:

| option | reader |
|---|---|
| `uncertainty_handling` | `Options.update_defaults` (`options.py:386`) and `_init_optim_state` via `Options.uncertainty_handling_on` (`vbmc.py:1133`) |
| `specify_target_noise` | the same, plus `_init_optim_state` (`vbmc.py:1135`) and `_initialize_precomputed_evaluations` (`vbmc.py:806-815`, called from `__init__` only) |
| `gp_mean_fun` | `_init_optim_state` (`vbmc.py:1244`); the live reader is `optim_state["gp_mean_fun"]` |
| `gp_int_mean_fun` | `_init_optim_state` (`vbmc.py:1245`) |
| `f_vals` | `_init_optim_state` (`vbmc.py:966`) and `_initialize_precomputed_evaluations` (`vbmc.py:763`) |
| `tol_bound_x` | `_init_optim_state` (`vbmc.py:1017`) |
| `integer_vars` | `_init_optim_state` via `Options.integer_vars_mask` (`vbmc.py:996`) |
| `warmup` | `__init__` (`vbmc.py:523`) and `_init_optim_state` (`:1069-1070`) |
| `k_warmup` | `__init__` (`vbmc.py:516`) and `_init_optim_state` (`:1099`) |
| `entropy_switch` | `_init_optim_state` (`vbmc.py:1105`) |
| `det_entropy_min_d` | `_init_optim_state` (`vbmc.py:1108`) |
| `det_entropy_alpha` | `_init_optim_state` (`vbmc.py:1148`) |
| `tol_gp_var` | `_init_optim_state` (`vbmc.py:1112`); `tol_gp_var_mcmc` is a different option and is live |
| `proposal_fcn` | `_init_optim_state` (`vbmc.py:1080-1083`) |
| `fitness_shaping` | `_init_optim_state` (`vbmc.py:1207`) |
| `out_warp_thresh_base` | `_init_optim_state` (`vbmc.py:1209`) |
| `active_search_bound` | `_init_optim_state` (`vbmc.py:1163`, `:1168`); see the note below |
| `bounded_transform` | `__init__` (`vbmc.py:505`), the `ParameterTransformer` — **added by me**, not in the review's list |
| `cache_size` | `__init__` (`vbmc.py:554`), the `FunctionLogger` arrays — **added by me**, not in the review's list |

Every option the existing tests pass to `load` stays accepted: `max_fun_evals`, `max_iter`, `min_iter`,
`vectorized_target`, `show_tips`, `performance_calibration`, `search_optimizer`, and the values that
`test_new_options_are_checked_as_at_construction` expects to raise the construction message
(`gp_hyp_sampler`, `noise_shaping`, `acq_hedge`, `search_acq_fcn`, `search_cache_frac`, `cache_frac`).
Verified end to end on an instance saved after a 2-iteration run as well as on one saved without a run.

## Proposed changelog (CHANGELOG.md is the orchestrator's)

**Unreleased entry, for W6-37:**

> `VBMC.load(file, new_options=...)` refuses an option that PyVBMC reads only while it builds a `VBMC`
> object — `uncertainty_handling`, `specify_target_noise`, `gp_mean_fun`, `integer_vars`, `warmup`,
> `k_warmup`, `entropy_switch`, `active_search_bound` and their kin — with a message that names it and
> says to construct a new `VBMC` object: the saved run carries the state that was built from such an
> option, so a value given to `load` was stored and then ignored, leaving the options and the state in
> disagreement. The options a continued run reads, `max_fun_evals` and `max_iter` among them, are taken
> as before, and `gp_mean_fun` and `integer_vars` are checked for a value no run can use whichever way
> they are supplied.

**"Upgrading from 1.0.4" line, for W6-37:**

> `VBMC.load(file, new_options=...)` raises for an option that only construction reads
> (`uncertainty_handling`, `gp_mean_fun`, `integer_vars`, `warmup` and their kin; the message names
> them), where such a value used to be stored and ignored; construct a new `VBMC` object to run with
> one. A saved run whose stored `gp_mean_fun` or `integer_vars` holds a value PyVBMC refuses now raises
> on `load` as it does at construction.

**Optional, for W6-23** (the ledger asked for no changelog here, and I did not add one; the key was
reachable as `vbmc.hyp_dict["logp"]`, so it is arguably user-visible — the orchestrator's call):

> `vbmc.hyp_dict` no longer carries a `logp` entry. It held an array of zeros, the log prior densities
> the GP hyperparameter sampler reports rather than the log posterior MATLAB VBMC keeps there, and
> nothing read it.

## What I found on the way that the ledger does not have (mine)

1. **`hyp_dict["logp"]` was also an oracle reference.** `pyvbmc/testing/oracles/_gp_fit_history.py:
   fit_outputs` recorded the key among the outputs of a replayed fit, and the three `gp_fit_history`
   sidecars named `hyp_dict_logp` as a reference; `test_gp_fit_history_replay` compares the output key
   sets exactly (`_assert_outputs`: `set(actual) == set(expected)`). Leaving them alone would have broken
   the gate either way: with the write gone, `train_gp` passes the fixture's own pre-state `logp` through,
   and for `later_changing_ns` that is 70 entries against the reference's 65. I removed the three
   reference *names* from the `.json` sidecars (one line each, no reference value changed, the
   `rebaselined` records left as they stand) and left the now-dead arrays in the `.npz` files, which the
   capture script owns — I did not open it, per the brief's "open nothing else under `dev/`". If the
   orchestrator wants those three arrays dropped from the `.npz` files, that needs the capture tooling.
   All three replays pass, matching `gp_hyp`, `hyp_dict_full`, `hyp_dict_run_cov` and the predictions at
   rtol 1e-10.
2. **`noise_shaping` is not construction-only, contrary to the G1-X list.** `active_sample.py:828` reads
   the option in every iteration (it decides between the rank-one GP update and a full recompute), and
   `_validate_noise_shaping_option` already refuses every value but the default. Listing it would also
   have broken the shipped
   `test_new_options_are_checked_as_at_construction[{"noise_shaping": True}]`, which requires `load` to
   raise the *construction* message. Excluded, and the module comment says so.
3. **`active_search_bound` has a dead reader outside construction.** `active_sample.py:859-866` reads the
   option into `LB_searchmin` / `UB_searchmin`, which nothing uses (the source marks them "(unused)"). I
   list the option anyway, since the read has no effect, and the comment beside the list records it.
4. **Two options beyond the review's list:** `bounded_transform` and `cache_size`, each read once in
   `__init__` and nowhere else.
5. **Ordering.** The brief's two tests (`gp_mean_fun="nonsense"` raising the construction message, and
   `uncertainty_handling=True` raising the refusal) can both hold only if the value checks run *before*
   the construction-only names are weighed, since `gp_mean_fun` is itself construction-only. That is what
   I implemented: a value no run can use is reported as such, and a usable value in a construction-only
   option gets the interface refusal. A comment in `load` states the reason.
6. **`_validate_gp_mean_fun_option` reads `self.options.get("gp_mean_fun", "negquad")`**, following its
   neighbours (`show_tips`, `search_optimizer`, `gp_hyp_sampler`), so that a legacy file whose options
   lack the key still loads instead of being refused for a missing value.
7. **A consequence of moving the `integer_vars` form check:** `_validate_option_values` runs on load, so a
   saved run whose stored `integer_vars` has a form the current code refuses (an ambiguous `[1, 0, 1]`,
   say) now fails to load, and part (a) closes the `new_options={"integer_vars": ...}` escape hatch. Such
   a value has been refused at construction since before this wave — it is already in the
   "Upgrading from 1.0.4" list — so no file that today's PyVBMC could have written is affected; only a
   1.0.4 file could be. I judged that acceptable and proposed the second half of the upgrading line for
   it. Flagging it for the PI in case the ruling intended otherwise.
8. **FAQ.** `docsrc/source/faq.md`, "How do I save and continue a run?", said nothing about what
   `new_options` may change. I added one sentence there, since that is where a user meets the argument.
   No section title renamed, so `skills/pyvbmc/SKILL.md` is unaffected. `examples/` and the FAQ pass only
   `max_fun_evals` to `load`, so nothing else needed changing.

## Left undone, and why

- **CHANGELOG.md untouched** — the brief assigns it to the orchestrator; the sentences are proposed above.
- **The whole suite not run** — the brief restricts me to one file at a time and no uncapped `optimize()`.
  I ran every file that reads the changed code or loads a saved object, plus the full oracle directory.
- **`PYVBMC_ORACLES_ALL=1` not set** for `pyvbmc/testing/oracles` (the `gp_fit` and `active_sample_step`
  replays stay skipped, 15 skips) — to keep one heavy process off a shared machine. The three
  `gp_fit_history` captures are platform-matched on this machine and *did* run a full sampled `train_gp`,
  which is the decisive evidence for a change that removes only a dictionary write.
- **`dev/scripts/make_oracle_fixtures.py` not run or read** — the brief forbids opening anything else
  under `dev/`. Nothing in my change moves a numerical output, and the test-side oracle gate is green.

## Final state

```
$ git status --porcelain
(clean)

$ git log --oneline dev-port-review..HEAD
d98ff434 fix(vbmc): refuse a construction-only option in VBMC.load(new_options=)
4d356e00 fix(vbmc): drop the mislabelled log density of the GP hyperparameter samples
```

Nothing pushed. No `Claude-Session:` trailer; each commit carries a `Co-Authored-By:` line.
