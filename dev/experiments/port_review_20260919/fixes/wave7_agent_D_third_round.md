# Wave 7 fix round: gpyreg fix agent D, third round

Raw report of fix agent D's third round, wave 7 of the port correctness
review (`dev/plans/port-correctness-review.md`), 2026-09-23. On the PI's
rulings on the independent check of the pass (`../verification/wave7.md`,
"The independent check of the pass"), the same agent, resumed with its
context on the same worktree and branch (`../gpyreg-w7`, `w7-fixes`), added
a test that pins the design's unchanged path, gave the design of a
coordinate fixed by equal bounds that has a prior its value, and corrected
the release notes of 1.3.1 and two comments, one commit each. Its run of
gpyreg's whole suite finished before the orchestrator asked it to leave that
to the gates. The text below the rule is the agent's final message,
unedited.

---

# Wave 7 fix round: third report of gpyreg fix agent D

All three items are done on `w7-fixes`, one commit each, and the worktree is clean. Nothing was pushed.

The change-of-plan message arrived after gpyreg's whole suite had already finished. That run took 3 minutes, single process, BLAS single-threaded: **384 passed, 0 failed, 0 reruns**. Its log is `…\scratchpad\wave7\fix_D\suite_1f959de_1790158385.log`. After the message I ran nothing heavy.

## 1. Setup
- **Worktree:** `C:\Users\luigi\Documents\GitHub\gpyreg-w7`, branch `w7-fixes`. This round started from `201b52b`; the branch was cut from `0186d89`.
- **Imports:** `pytest` from the worktree root imports `…\gpyreg-w7\gpyreg\__init__.py`. Scripts used `PYTHONPATH="…/gpyreg-w7;…/pyvbmc-wave7"`.
- **Checks:** BLAS single-threaded throughout. gpyreg's own hooks, run with `pre-commit run --files`, passed before each commit.

## 2. Commits

| hash | item | what it does | tests (node ids) | before / after |
|---|---|---|---|---|
| `6ffff06` | 1 | Test only. Pins the design's unchanged path. | `gpyreg/testing/test_gaussian_process.py::test_space_filling_design_from_the_centre_down`: 4 families, lower bound exactly at the centre (`mu`, or `0.5 * (a + b)`) and 1.8 below it. The expected design is computed as the unchanged path computes it, `cdf(lb) + (cdf(ub) - cdf(lb)) * S` through the percent point function on the unscrambled Sobol draws; the test compares it with `f_min_fill`'s design using `np.array_equal`. | • 8 passed on the current code.<br>• In a scratch copy with the switch mutated to `>=`, the 4 at-centre cases fail (`assert np.array_equal(X[:, 0], expected)`). The mutant moves 102 to 199 of 257 design points per family.<br>• My first draft set the lower bound at 0.3, which is not the smooth-box centre in floating point (`0.5 * (-1.0 + 1.6)` is 0.30000000000000004), so the smooth-box cases missed the mutant. The committed test computes the centre. |
| `4eb8789` | 2 | Every prior branch of `f_min_fill` writes `LB` for a coordinate with `LB == UB`. Adds a release-notes entry. | `::test_space_filling_design_of_a_fixed_coordinate_with_a_prior`, 5 cases: Student's t and Gaussian noise priors of the form `_gp_hyp` writes, centred above and below the fixed value `log(sqrt(1e-5))`, and smooth boxes of both families. It checks that every design value of the fixed noise equals `LB` exactly and that every design objective value is finite. | • Before: 4 of 5 failed on `assert np.all(received["X0"][:, 2] == fixed)`. The design values were 0 to 1 ulp off, and 63 of 64 objective values were infinite. The smooth box around the value already passed, because its linear middle branch returns the value exactly.<br>• After: 5 passed, and 22 passed together with the other design and fixed-bound tests. |
| `1f959de` | 3 | Docs only: release notes and comments; no code changes. | — | — |

What `1f959de` changes:
- **Upgrading note of the location refusal:**
  - `None` replaces the prior of a hyperparameter none of whose coordinates has a prior.
  - One coordinate of a block without a prior takes NaN for both its location and its `sigma`, as the message says.
  - A smooth box with an infinite or NaN end has no replacement: the script writes the prior it means.
- **Inverted-box refusal:** it gets an **Upgrading** line.
- **Upper-tail design note:** it now says "every draw of that hyperparameter lay at infinity (the starting points the fit was given stayed finite)" and names `gpyreg.f_min_fill` beside `GP.fit`.
- **Switch comments** in `__recompute_normalization_constants` and `f_min_fill`: they no longer say the cumulative distribution function is close to one above the centre. They now say it is above one half there, where a double resolves it only to a fixed absolute step, and that far in the tail it rounds to one.

## 3. What item 2 changes on PyVBMC-built GPs
Script `pyvbmc_gp_outputs_designs.py`, the 2304 configurations of the earlier reports. I compared the working tree after the fix with `201b52b`, whose outputs are identical to `0186d89`. Files: `breakdown_changes.py`, `breakdown_fixedcoord.log`, `fixedcoord_effect.log`.

- **Which configurations change:** 144, exactly the predicted set. They are level 0 with `noise_size` 0.2 or 5.0, on the training sets whose target range is below `tol_gp_noise` (clustered, range 1e-3, range 1e-8). That is 3 × 2 × 3 means × 4 D × 2 N. In each of them the noise pair has collapsed and the noise prior is centred off the bound.
- **What changes in them:**
  - The 256-point design changes in all 144.
  - In the 72 that are fitted, the design changes in all 72. Before, 2232 of 2304 design objective values were infinite (31 of 32 per fit, all but the given starting point); after, none are. The fixed coordinate now sits exactly at its bound in all 72 designs, against none before.
  - The fitted hyperparameters and chains change in 58 of the 72.
  - The optimizer ends lower in 16, higher in 22, and equal within 1e-9 in 34. The optimizations now start from the ranked design; before, their starts were arbitrary.
- **The other 2160 configurations are bit-identical** in every recorded field: designs, design values, fits, chains, masses, bounds, log posterior and gradient.
  - This includes every level-1 and level-2 configuration and every configuration without `noise_size`. In those, the noise prior is centred on the fixed value, so the quantile function already returned it exactly.
  - It also includes the shared-coordinate sets. Their fixed length scale at (-inf, -inf) has a prior and was already drawn at -inf.

## 4. Release-notes lines
New entry, from `4eb8789`:

"The space-filling design of `GP.fit`, drawn by `gpyreg.f_min_fill`, gives a hyperparameter whose lower and upper bounds are equal their value at every point when the hyperparameter has a prior, as it did for one without. Mapped through the prior's quantile function, the value came back an ulp or two off, where the log prior is `-inf`, so that the objective was infinite at every point of the design but the starting points the fit was given, and the ranking of the design that picks the starts of the optimization was lost. A GP that PyVBMC builds meets this when the noise is held at its lower bound (targets of a range below about 3e-3) and the option `noise_size` moves the centre of the noise prior away from that bound."

The item-3 wording changes are listed under `1f959de` in section 2.

## 5. Noticed, not changed
- **The mass switch has the same gap for smooth boxes.** `test_prior_mass_from_the_centre_down` does catch a `>=` mutant of the prior-mass switch for the Gaussian and Student's t (I checked in a scratch copy). It has no smooth-box case, so a mutant of the smooth-box mass switch alone would go unseen.
- **A NaN design for fixed coordinates without a prior at (-inf, -inf)** (existing behaviour, not a regression). Such a coordinate, for example the negative-quadratic log scale on a zero-width input column, gets a NaN design from the plausible-box formula, because `PUB - PLB` is NaN. It happens on the shared-coordinate sets, where the fit already fails with the `KeyError` of W7-17.

## 6. Final state
`git status --porcelain`: empty.

`git log --oneline 0186d89..HEAD`:
```
1f959de docs: the 1.3.1 notes on the refusals and the design, and the switch comments
4eb8789 fix: the design gives a fixed coordinate with a prior its value
6ffff06 test: the design from the prior's centre down is the CDF path, bit for bit
201b52b fix: set_priors refuses an inverted smooth box
0cf27e1 fix: the space-filling design survives the upper tail of a prior
8be6ee4 test: seed test_gp_gradient_computations
dcaea44 test: the marginal likelihood just below the low-noise switch
14736b5 test: the likelihood gradient with a noise that varies by point
5499b59 fix: set_priors refuses a location that is not finite beside a finite sigma
5ae1c44 fix: the prior's mass inside the bounds survives the upper tail
e5b7238 fix: a hyperparameter with equal bounds gets no NaN in the prior's gradient
```
