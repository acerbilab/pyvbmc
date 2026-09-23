# Wave 7 fix round: PyVBMC fix agent A

Raw report of fix agent A, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent on a
worktree made by hand (`../pyvbmc-w7-A`, branch `w7-agent-A`, cut from
`dev-port-review-w7` at `ed9df90e`, whose package code is that of
`a65b96f4`), made the fixes of rows W7-1 and W7-2 of
`../verification/wave7.md` as the PI ruled them, and the test notes O3-8,
O3-9 and O3-10, one commit each. It ran only the test functions it added or
changed, by node id, while another session held the machine's heavy slot;
the test of W7-2 needs an `optimize()` run and was left for the gate. The
orchestrator reviewed the diffs and cherry-picked the commits onto
`dev-port-review-w7`, where they have other hashes
(`../verification/wave7.md`, "Fix commits"). The text below the rule is the
agent's final message, unedited.

---

# Wave 7 fix round: report of fix agent A

W7-1 and W7-2 are fixed, and test notes O3-8, O3-9 and O3-10 are done. There are four commits on `w7-agent-A`. Every test I could run passes. The W7-2 test needs an `optimize()` run, so I did not run it and it has not been seen to fail. The gate needs to run it (section 4).

## 1. Setup

- **Worktree:** `C:\Users\luigi\Documents\GitHub\pyvbmc-w7-A`, branch `w7-agent-A`.
- **Start:** `ed9df90e`. `git diff a65b96f4 HEAD -- pyvbmc` was empty at the start, so the package code was that of `a65b96f4`.
- **Import paths printed:**
  - `pyvbmc: C:\Users\luigi\Documents\GitHub\pyvbmc-w7-A\pyvbmc\__init__.py`
  - `gpyreg: C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\__init__.py`
- Everything ran with BLAS single-threaded and one pytest process at a time, by node id.
- **Scripts and logs**, in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\30d85518-7e4d-42ed-9a16-74a7fb47ace3\scratchpad\wave7\fix_A\`:
  - `w7_1_nothing_moves.py` / `.log`, with `whitening_a65b96f4.py` (the unfixed module)
  - `o3_8_mutation.py` / `.log`
  - `o3_9_mutation.py` / `.log`
  - `w7_1_zero_mean_capped_run.py` (not run)

## 2. Commits

| hash | row | what it does | test (node id) | seen to fail before | run after, result |
|---|---|---|---|---|---|
| `0d4316a7` | W7-1 (and O3-10) | `warp_gp_and_vp` gets a `ZeroMean` branch placed before the constant-mean branch. It warps the length scales as before, skips the mean block, and leaves the constant shift of the stored log joint to the refit that follows every warp. The docstring gains a `Raises` entry for the `ValueError` that remains for other mean functions. The constant and negative-quadratic code is untouched. The test helper `_bounded_warp_state` builds, without a run, a D=3 state with two bounded coordinates (probit) and one unbounded. | `pyvbmc/testing/whitening/test_rotoscaling.py::test_warp_gp_and_vp_warps_a_zero_mean_gp`; `::test_warp_gp_and_vp_moves_the_mean_with_the_stored_log_joint[const]` and `[negquad]` | The zero-mean test stopped before its assertions with `ValueError: Unsupported GP mean function for input warping.` (`whitening.py:472`). The const and negquad cases passed before the fix, as expected: they pin behaviour that was already right. | The 3 new cases, `test_warp_gp_and_vp` (MATLAB reference), `calibration/test_lifecycle.py::test_profile_survives_whitening_and_undo_snapshot` and `vbmc/test_vbmc_active_sample.py::test_a_cached_point_nothing_moved_keeps_its_value_after_a_warp`: all pass |
| `125cd1b7` | W7-2 | In `vbmc.py`, `self.optim_state["hyp_dict"] = self.hyp_dict` goes right after `warp_gp_and_vp`, with a comment. The undo path is unchanged, because it restores `optim_state_old` and active sampling links that one. The test shares the existing module fixture `warped_run` in `test_vbmc_warp_branch.py`, whose options now keep every warp (`warp_tol_improvement=-np.inf`, `warp_tol_sd_base=np.inf`); the module docstring says so. | `pyvbmc/testing/vbmc/test_vbmc_warp_branch.py::test_record_of_a_kept_warp_holds_the_warped_hyperparameters` | Not run (needs `optimize()`). Expected to fail at `assert np.array_equal(recorded["hyp"], vbmc.get_gp(iteration).get_hyperparameters(as_array=True))`, because before the fix the record holds the previous iteration's `hyp`. | Collection only (`--collect-only`: 4 tests collected) |
| `e50222be` | O3-8 | The probit and Student-t(4) loop of `test_inverse_type3_max_space` now uses `Y = 1e6` and asserts `X == np.nextafter(10, -np.inf)`. `1e6` is needed because Student-t(4) at `Y = 500` gives `10 - 9.6e-10`, which is not saturated; it saturates from about `3e4`. | `pyvbmc/testing/parameter_transformer/test_parameter_transformer.py::test_inverse_type3_max_space` | Test-only change. Mutation check (`o3_8_mutation.py`): with the upper clamp of `_from_unit_interval` removed, the new test fails and the `a65b96f4` version passes. | Passes, with `::test_inverse_type3_min_space` |
| `604a8695` | O3-9 | Both largeN tests now round-trip 1e5 distinct seeded rows, about 0.11 s each. Direct-inverse: `|x|` in [0.1, 9.9]. Inverse-direct: `|u|` in [0.05, 3]. Tolerances are unchanged; the comments say which regions are excluded and why. | `::test_transform_direct_inverse_largeN`, `::test_transform_inverse_direct_largeN` | Test-only change. Mutation check (`o3_9_mutation.py`): an `_inverse_probit` that gives every row the first row's value fails both new tests and passes both `a65b96f4` versions. | Both pass |

- **Re-run at HEAD:** all 7 changed or related test nodes pass together.
- **Readers of `optim_state["hyp_dict"]` (W7-2):**
  - In the package, only `active_sample` (`active_sample.py:799-800`, `:812`, `:918`) reads it. That is after its own relink at `vbmc.py:1578`, which sets the same object my relink sets.
  - `VBMC.load` (`vbmc.py:3259-3260`) reads it from the record, which is the intended change.
  - `train_gp` and `optimize_vp` receive `optim_state` but read no `hyp_dict` key.
  - `dev/scripts`: no reader.
  - `pyvbmc/testing/oracles/_oracles.py:563` and `:618` read it from stored fixtures. No fixture was captured at a kept-warp iteration: capture iterations 20 and 26 against last warps 15 and 14; the other fixtures never warped.

## 3. Evidence that nothing moved

- **W7-1, `w7_1_nothing_moves.py`:**
  - What it runs: the unfixed `warp_gp_and_vp` (from `git show a65b96f4:pyvbmc/whitening/whitening.py`) against the fixed one, on 400 random states per mean function.
  - What the states vary: D from 2 to 5, a mix of bounded and unbounded coordinates, the logit, probit and Student-t(4) transforms, 1 to 4 GP hyperparameter samples, K from 1 to 7, and temperature 1 or 2.
  - Largest absolute difference over the warped hyperparameters and the warped VP's `mu`, `sigma`, `lambd` and `w`: **0.0 for `const` and 0.0 for `negquad`**.
- **W7-2:** no script can show this without a run. Between the warp and the record nothing reads `optim_state["hyp_dict"]` (section 2), so only the recorded copy at a kept-warp iteration changes. The gate's seeded runs are the numerical check.

## 4. Tests written but not run

- **`test_record_of_a_kept_warp_holds_the_warped_hyperparameters`** needs the `warped_run` fixture (4-iteration seeded `optimize()`, D=2). It adds no new run.
  - The fixture change also alters the course of the three existing tests in that file after the warp, since the run now continues in the warped space. Their assertions should still hold, but the gate should run the whole file.
  - To see the new test fail, revert only the relink line in `vbmc.py` and keep the fixture change.
- **`w7_1_zero_mean_capped_run.py`** (scratch, optional gate check): a 4-iteration seeded run with `gp_mean_fun="zero"` and warp-forcing options.
  - It checks that the run finishes with a kept warp, and that the record of that iteration matches `get_gp(k)`.
  - At `a65b96f4` it should stop with the `ValueError`.
  - No test covers a zero-mean run past a warp end to end.

## 5. Proposed changelog text

Both defects are in `v1.0.4`, checked with `git show v1.0.4:...`:
- W7-1: v1.0.4 accepts `"zero"` and its `whitening.py:343` raises.
- W7-2: v1.0.4's `warp_input` deep-copies `optim_state` (`:116`) and its `load` restores `hyp_dict` from the record (`vbmc.py:2253-2254`).

Proposed Fixed entries:
- **W7-1:** "With `gp_mean_fun="zero"`, a run with two or more variables stopped with `ValueError` ("Unsupported GP mean function for input warping.") at its first input warp. The warp now re-expresses the length scales of a zero-mean GP as it does for the other mean functions, and the GP fit that follows the warp adapts the rest."
- **W7-2:** "`VBMC.load(file, iteration=k)` at an iteration that kept an input warp, or `VBMC.load(file)` of a run that stopped on one, restores that iteration's GP hyperparameters, from which a resumed run starts its GP fit. The record of such an iteration held the hyperparameters of the iteration before, in the coordinates before the warp. Files saved by earlier versions keep that record."

"Upgrading from 1.0.4": no new line. W7-1 lets a script that used to stop run on. For W7-2, the first point of the list ("Results differ from 1.0.4, also with a fixed seed") already covers resumed runs whose results change. The test notes need no entry.

## 6. What I noticed beyond my rows (not fixed)

- **The W7-2 fix repairs new records only.** A file saved before it still holds the stale `hyp_dict` at a kept-warp iteration, and `load(file, iteration=k)` still pairs it with the warped `get_gp(k)`. The ledger's other option, a repair inside `load`, would cover old files: for example, when the iteration's `logging_action` has `"rotoscale"` without `"undo rotoscale"`, take the hyperparameters from `get_gp(k)` and `gp_hyp_full[k]`. That is for the PI to decide; the changelog's last sentence depends on it.
- **The `warp_gp_and_vp` docstring is wrong about its return.** It documents `hyp_warped : dict` ("An updated copy of the dictionary..."), but the function returns an ndarray of shape `(Ns_gp, n_hyp)`.
- **The other saturation checks test nearness, not the clamp.** `test_inverse_type3_min_space` and the logit part of `max_space` use `np.allclose` (atol 1e-8), which accepts a missing clamp that returns the bound itself. For Student-t(4), `Y = -500` gives `-10 + 9.6e-10`, not the clamp.
- **Where round trips exceed the test tolerances on [-10, 10]** (measured, 1e5 random rows; consistent with W7-6 and W7-7, nothing to fix):
  - Student-t(4) direct-inverse (rtol 1e-11) fails for `|x| < 0.041`.
  - Inverse-direct (rtol 1e-12) fails for `|u| < 0.0142` with Student-t(4).
  - It also fails for `|u| > 4.30` with probit, because x near a bound resolves u only coarsely.
- **An exception elsewhere can still leave the instance half-warped.** The W7-1 fix removes the zero-mean exception, but `optimize` assigns the warped `optim_state` and function logger before `warp_gp_and_vp` runs. The only remaining refusal is for mean functions that construction no longer accepts.

## 7. Final state

`git status --porcelain`: empty.

`git log --oneline ed9df90e..HEAD`:
```
604a8695 test(parameter_transformer): round-trip many distinct rows in the largeN tests
e50222be test(parameter_transformer): test the upper saturation of the probit and Student's t inverses
125cd1b7 fix(vbmc): record the warped hyp_dict at an iteration that keeps a warp
0d4316a7 fix(whitening): warp a zero-mean GP instead of refusing it
```
