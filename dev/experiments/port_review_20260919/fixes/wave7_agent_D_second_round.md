# Wave 7 fix round: gpyreg fix agent D, second round

Raw report of fix agent D's second round, wave 7 of the port correctness
review (`dev/plans/port-correctness-review.md`), 2026-09-23. On the PI's
word of the same day, the same agent, resumed with its context on the same
worktree and branch (`../gpyreg-w7`, `w7-fixes`), fixed the two gpyreg
candidates it had found in its first round (`wave7_agent_D.md`, §6;
`../verification/wave7.md`, "Found during the fix round"): the space-filling
design of `fit` in a prior's upper tail, and an inverted smooth box accepted
by `set_priors`, one commit each with an entry in the release notes of
1.3.1, under the rules of its first round. The text below the rule is the
agent's final message, unedited.

---

# Wave 7 fix round: second report of gpyreg fix agent D

Both candidates are fixed on `w7-fixes`, one commit each, each with a release-notes entry in `1.3.1 (unreleased)`. On the 2304 GPs built the way PyVBMC builds them, the designs, fits and full sampler chains are unchanged to the last bit. Nothing was pushed, and the worktree is clean.

## 1. Setup
- **Worktree:** `C:\Users\luigi\Documents\GitHub\gpyreg-w7`, branch `w7-fixes`. This round started from `8be6ee4`, the head of the first round; the branch was cut from `0186d89`.
- **Imports:** `pytest` from the worktree root imports `…\gpyreg-w7\gpyreg\__init__.py`. Scripts used `PYTHONPATH="…/gpyreg-w7;…/pyvbmc-wave7"`, with pyvbmc from `…\pyvbmc-wave7\pyvbmc\__init__.py`. The unfixed comparison used `…\GitHub\gpyreg\gpyreg\__init__.py` (`0186d89`).
- **Checks:** BLAS single-threaded throughout. gpyreg's own hooks, run with `pre-commit run --files`, passed before each commit.

## 2. Commits

| hash | what it does | tests (node ids) | failure before the fix | run after |
|---|---|---|---|---|
| `0cf27e1` | **Space-filling design in the upper tail.**<br>• Where the lower bound is strictly above the prior's centre (`mu`, or `(a+b)/2` for a smooth box), `f_min_fill` maps its draws through `sf(lb) - (sf(lb) - sf(ub)) * S` and the inverse survival function. This is the same switch as the mass fix (W7-15), and it gives the same points in exact arithmetic.<br>• Every other design runs the old lines unchanged.<br>• Adds `smoothbox_isf` and `smoothbox_student_t_isf`. | • `gpyreg/testing/test_gaussian_process.py::test_space_filling_design_in_the_upper_tail`, 4 families. The upper-tail design lies inside the bounds and is the mirror image of the lower-tail design; the unscrambled Sobol draws of one coordinate are symmetric about 1/2.<br>• `::test_fit_with_samples_in_the_upper_tail_of_a_prior`: N(0,1) with bounds [9, 10], `n_samples=3`, against [-10, -9]. The design mirrors the lower-tail one and the samples stay inside the bounds.<br>• `test_smoothbox.py::test_isf`, `test_smoothbox_student_t.py::test_isf` | • Design test, all 4 families: `assert np.all((designs[0] >= bounds[0]) & …)` failed on designs of `[9.5, inf, inf, …]`.<br>• Fit test: `ValueError: The widths vector needs to be all positive real numbers`.<br>• The two `test_isf` tests import functions that are new in this commit. | 5 + 2 passed. The related `test_fit_does_not_write_into_the_space_filling_design`, `test_fit_with_targets_of_a_tiny_range` and `test_prior_mass_in_the_upper_tail` also pass (13 total). |
| `201b52b` | **Inverted smooth box.**<br>• `set_priors` refuses a smooth box of either family with `a > b`. The message names the hyperparameter and says "a lower end a of its box above its upper end b"; the general part of the message now also says "a <= b for a smooth box".<br>• `a == b` is meaningful and still accepted: a box with no plateau and a normalizer of 1 is the Gaussian or Student's t centred at `a`.<br>• `zero_width_box.py`: the CDF, survival function, percent point function and inverse survival function of a zero-width box equal scipy's norm/t exactly (difference 0.0).<br>• The docstring and comment say so. | • `::test_set_priors_refuses_an_inverted_smooth_box`: both families, plus one block coordinate beside a coordinate without a prior.<br>• `::test_smooth_box_of_zero_width`: the log prior equals `norm`/`t` `logpdf` to rtol 1e-12.<br>• `::test_get_priors_returns_what_set_priors_reads_back`: two zero-width cases added, one on a whole block and one on a single coordinate. | • `Failed: DID NOT RAISE ValueError` in all 3 refusal cases.<br>• The zero-width test and the round-trip test (12 cases) passed before and pass after. They record the `a == b` decision and the round trip. | 47 passed, together with the other `set_priors` and smooth-box tests |

## 3. Evidence that nothing moved
Script `pyvbmc_gp_outputs_designs.py` in `…\scratchpad\wave7\fix_D`. It uses the configurations of the first report: 2304 GPs from `_gp_hyp`, over D 1/2/3/5, levels 0/1/2, three mean functions, `noise_size` unset/0.2/5.0, the rectified-noise branch, 8 kinds of training set, and N 12/40. It additionally records:
- the design that each of 1008 seeded fits receives from `f_min_fill`, with its objective values;
- each fit's full slice-sampler chain;
- a 256-point `f_min_fill` design built from the priors, bounds and plausible bounds that `fit` passes to it.

Comparing `0186d89` with `201b52b` (`compare_designs_201b52b.log`):

| quantity | largest difference |
|---|---|
| designs (fit and 256-point) | 0.0 |
| design objective values | 0.0 |
| chains | 0.0 |
| fitted hyperparameters and objective | 0.0 |
| normalization constants, priors, bounds | 0.0 |
| log posterior and its gradient | 0.0 |

No configuration differs. Across these GPs no prior's lower bound lies above its centre: `(lb - mu)/sigma` ranges from -14.73 to 0.000.

On the new head, `f_min_fill_upper_tail.py` gives 64 finite design values out of 64 and samples of 9.05, 9.06 and 9.03. The lower-tail mirror is unchanged.

## 4. Release-notes lines (added to `1.3.1 (unreleased)`)
- "The space-filling design of `GP.fit` draws the starting values of a hyperparameter whose bounds both lie above the centre of its prior through the survival function of the prior and its inverse, where it drew them through the cumulative distribution function and the percent point function. With both bounds far in the upper tail, every point of that design lay at infinity, and a fit with hyperparameter samples raised `ValueError` because the widths of the slice sampler were NaN. Where the lower bound is not above the centre, the design is the same as before, to the last bit. `gpyreg.f_min_fill` has the inverse survival functions of the two smooth-box families, `smoothbox_isf` and `smoothbox_student_t_isf`."
- "`GP.set_priors` refuses, with a message that says what is wrong, a smooth box of either family whose lower end `a` is above its upper end `b`. 1.3.0 took such a box, whose normalization constant is then below one or negative, and gave a log prior that was wrong or NaN. A smooth box with `a == b`, which has no plateau and is the Gaussian or the Student's t centred at `a`, is taken as before."

## 5. Noticed
Nothing new beyond these two candidates.

## 6. Final state
`git status --porcelain`: empty.

`git log --oneline 0186d89..HEAD`:
```
201b52b fix: set_priors refuses an inverted smooth box
0cf27e1 fix: the space-filling design survives the upper tail of a prior
8be6ee4 test: seed test_gp_gradient_computations
dcaea44 test: the marginal likelihood just below the low-noise switch
14736b5 test: the likelihood gradient with a noise that varies by point
5499b59 fix: set_priors refuses a location that is not finite beside a finite sigma
5ae1c44 fix: the prior's mass inside the bounds survives the upper tail
e5b7238 fix: a hyperparameter with equal bounds gets no NaN in the prior's gradient
```
