# Wave 7 fix round: PyVBMC fix agent B

Raw report of fix agent B, wave 7 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-23. One Opus agent on a
worktree made by hand (`../pyvbmc-w7-B`, branch `w7-agent-B`, cut from
`dev-port-review-w7` at `ed9df90e`, whose package code is that of
`a65b96f4`), made the fixes of rows W7-3 and W7-4 of
`../verification/wave7.md` as the PI ruled them, and the test notes O2-4,
O2-6, O2-7 and O2-8, in five commits (O2-4 and O2-6 share one). It ran only the test functions it
added or changed, by node id, while another session held the machine's
heavy slot; it found that the fix of W7-3 moves the `vp_pdf` oracle on
three fixtures, at the rows whose density was below the smallest normal
number, which the ruling had not foreseen (`../verification/wave7.md`,
"Found during the fix round"). The orchestrator reviewed the diffs and
cherry-picked the commits onto `dev-port-review-w7`, where they have other
hashes. The text below the rule is the agent's final message, unedited.

---

# Wave 7 fix round: report of fix agent B

Both fixes (W7-3, W7-4) and the three test notes are committed on `w7-agent-B`. **One thing needs you before the gate:** the W7-3 fix changes the `vp_pdf` oracle on three fixtures, so those three oracle tests will fail until you re-baseline `vp_pdf` (section 3).

## 1. Setup
- Worktree `C:\Users\luigi\Documents\GitHub\pyvbmc-w7-B`, branch `w7-agent-B`, started from `ed9df90e`. `git diff a65b96f4 ed9df90e -- pyvbmc` is empty.
- Import paths, printed from the worktree:
  - `pyvbmc`: `C:\Users\luigi\Documents\GitHub\pyvbmc-w7-B\pyvbmc\__init__.py`
  - `gpyreg`: `C:\Users\luigi\Documents\GitHub\gpyreg\gpyreg\__init__.py`
  - numpy 2.5.2, scipy 1.18.1
- Every scratch script printed `pyvbmc.__file__` under the worktree (run with `PYTHONPATH` set), with BLAS single-threaded.
- The unfixed module is `git show a65b96f4:pyvbmc/variational_posterior/variational_posterior.py`, saved as `vp_unfixed.py` in the scratch dir. Its `_pdf` is called on the same instance as the fixed one.

## 2. Commits

**692fd72d — W7-3.** `fix(variational_posterior): keep the log density finite where the density underflows`
- **Change:** in `_pdf` with `log_flag`, a row whose linear sum is below `np.finfo(float).tiny` (and inside the bounds) gets:
  - its log density as a log-sum-exp of the component log densities, with the log weights taken in log space;
  - its gradient from the responsibilities.
  - Every other row, and the linear density and its gradient everywhere, run the old statements unchanged.
- The `log_flag` docstring now says this.
- **Changed existing test:** `test_proposal_pdf_gives_no_weight_where_the_proposal_is_zero` (`test_active_importance_sampling.py`) put its point at 1e5.
  - After the fix, the log density there is finite (-1.5e16), so the point's log weight is +1.5e16.
  - The point is now at 1e200, where the squared distance overflows and the log density is -inf. The docstring says why.
  - With the fix, the old version fails at `assert np.all(vp.pdf(far, ...) == -np.inf)`; the new version passes.
- **Tests and the assertions that failed before the fix:**
  - `test_variational_posterior.py::test_log_pdf_holds_where_the_density_underflows` failed at `assert np.all(np.isfinite(log_y)) and np.all(np.isfinite(dlog_y))`: `log_y` was -inf at 6 of 12 points.
  - `::test_log_pdf_in_the_original_space_holds_where_the_density_underflows` failed at `assert np.all(np.isfinite(log_p))`: -inf at 4 of 10 points.
  - `::test_mode_refines_on_a_narrow_posterior[False]` and `[True]` failed at `assert np.max(np.abs(mode - reference.x * scale)) < 1e-4 * scale`: 1.59e-5 against 3e-7, i.e. 5.3e-3 of the scale off.
  - After the fix: 3e-9 of the scale (`orig_flag=False`) and 1.6e-6 (`orig_flag=True`), with no `RuntimeWarning`.
- **Run after:** all 5 tests above pass.

**33b6b90b — W7-4.** `fix(variational_posterior): refuse points whose width is not the dimension`
- **Change:** `_pdf` raises `ValueError` when `x.shape[1] != self.D`. The message says each row is one point, and that for D = 1 the points form a column (`x.reshape(-1, 1)`).
- The `x` parameter and the Raises sections of the `pdf` and `log_pdf` docstrings are updated. The decorator is unchanged.
- **Tests and the assertions that failed before the fix:**
  - `::test_pdf_refuses_points_whose_width_is_not_the_dimension[df x orig_flag]` (6 cases):
    - `orig_flag=True`: an IndexError ("boolean index did not match").
    - df = inf, transformed space: regex mismatch on "cannot reshape array of size 1 into shape (1,1,3)".
    - df = ±3, transformed space: `Failed: DID NOT RAISE ValueError`. This is the silent case.
  - `::test_pdf_refuses_rows_whose_width_is_not_the_dimension`: regex mismatch.
- **Run after:** all 7 pass, and the W7-3 tests still pass.

**44655781 — test notes O2-4 and O2-6.** `test(entropy): check the Monte Carlo entropy's gradient exactly`
- **Change:** an independent implementation of the estimator at the call's draws, differentiated by complex step:
  - the location and scale parameters move only the samples;
  - the weights move the density and the average.
- New test `test_entmc_vbmc.py::test_entmc_vbmc_gradient_is_the_path_derivative`: 8 configurations, with and without the Jacobian (16 cases).
- The same check replaces the unasserted `check_grad` of `test_entmc_vbmc_nonoverlapping_mixture` (at the test's own Ns = 1e5, seed 42).
- **Fixed-draw checks at `:84` (single Gaussian) and `:150` (overlapping mixture):** kept as they are. They are asserted, they pass, and they check the estimator against the entropy's gradient within Monte Carlo error, which the exact check does not.
- **Before the fix:** the code is right, so these pass on unfixed code. Their sensitivity, from `o2_6_exact_check_margins.py`:
  - the worst gradient error is 1.6e-2 of the tolerance;
  - the gradient scaled by 1+1e-9 comes out at 99× the tolerance;
  - the fixed-draw derivative that the replaced `check_grad` compared with comes out at 7e8 to 1.3e9× the tolerance (up to 1.3% relative).
- **Run after:** 17 pass.

**89d3cf1e — test note O2-8.** `test(entropy): a budget case that splits a component's samples, and the flag subsets`
- **Change:**
  - Budget case `(8, 20, 738, _ALL_GRADS, (7777, 3000, 1000))`, appended last so existing node ids do not shift. Its canonical blocks split each component's samples into 409 and 329.
  - New test `test_entmc_vbmc_blocks_do_not_depend_on_the_other_flags`: all 16 flag subsets, with and without the Jacobian, on (4, 5, 60) and (8, 20, 738), checked bit for bit.
- **Tests:** `test_entmc_vbmc_is_independent_of_the_chunk_budget[8-20-738-grad_flags10-budgets10]` and the new test (4 cases).
- **Before the fix:** the code is right; they pass.
- **Run after:** 5 pass.

**f8e4d2d2 — test note O2-7.** `test(variational_posterior): finite differences of the density's gradient at K = 1, D = 1 over several rows`
- **Change:** new test `test_variational_posterior_grad_fd.py::test_pdf_grad_fd_one_component_several_rows[False/True]`.
  - It uses five rows at once and compares the diagonal of the Jacobian, with no Torch.
  - It covers the linear density and also the log density.
- **Before the fix:** the code is right; it passes. The finite-difference error is 0.009 (linear) and 0.076 (log) of the tolerance, and a 0.1% error on one row fails it.
- **Run after:** 2 pass.

At the final head, all 46 cases of the tests I added or changed passed in one pytest process.

## 3. Evidence that nothing moved (W7-3)

All scripts and logs are in `C:\Users\luigi\AppData\Local\Temp\claude\C--Users-luigi-Documents-GitHub-pyvbmc\30d85518-7e4d-42ed-9a16-74a7fb47ace3\scratchpad\wave7\fix_B\`. Each was rerun at the final head (`*_final.log`).

**`w7_3_bit_identity.py`**
- **Coverage:**
  - 259,200 rows over 30 random mixtures (D 1–10, K 1–50);
  - rays from the component centres out to 60 SDs, with extra points packed at 33–40 SDs;
  - both values of `orig_flag`: identity, and unbounded/logit/probit/student4 transforms;
  - linear and log, each with and without the gradient;
  - chunk budgets of 2^16, 7, 11 and 1000.
- **Rows normal at a65b96f4** (unfixed linear sum ≥ tiny): 217,829, of which 108,501 in the transformed space and 109,328 in the original space. Every output is bitwise equal: largest difference **0.0**.
- **Linear density and its gradient:** bitwise equal on all rows, **0.0**.
- **Student-t branches:** df 3, -3, 30 and 1e3, 864,000 row evaluations, **0.0**.
- **The 41,371 tail rows:** all finite. Against an independent log-sum-exp, the value agrees to 1.5e-16 relative and the gradient to 1.1e-13.

**`w7_3_oracles_fixed_vs_unfixed.py`** runs `vp_pdf` and every `acq_*` oracle on all 8 fixture states, once with the fixed `_pdf` and once with the a65b96f4 `_pdf` patched in.
- 39 of the 42 comparisons are bit-identical, including every acquisition oracle.
- `vp_pdf` moves on three fixtures, and only at the candidate rows whose density is below tiny:

| fixture | rows moved | `logpdf` entries | `dlogpdf` entries |
|---|---|---|---|
| `cigar_D4_boosted` | 6 | 6 | 24 |
| `cigar_D4_largeK` | 7 | 7 | 28 |
| `corr_D5_warped` | 76 | 75 | 380 |

- At those rows the old values were -inf (or the log of a subnormal) and NaN; the new ones are finite.
- The new `logpdf` there is at most -708.9, below log(tiny) = -708.4, so the floor in `acq_fcn_log` keeps `AcqFcnLog` bit-identical.

**`w7_3_oracle_states.py`**: the five fixtures with no tail rows equal their stored references bit for bit.

**Action needed:**
- Until you re-baseline, `test_oracle[cigar_D4_boosted-vp_pdf]`, `[cigar_D4_largeK-vp_pdf]` and `[corr_D5_warped-vp_pdf]` will fail on the inf/NaN pattern, and `--check --exact` will report `vp_pdf` moving. I was not allowed to run the generator.
- Proposed: `python dev/scripts/make_oracle_fixtures.py --rebaseline vp_pdf --reason "W7-3: the log density and its gradient by log-sum-exp where the mixture density is below the smallest normal double"`.
- The ledger's "proposed" text for W7-3 ("the `vp_pdf` oracle ... confirm[s] that nothing moves") does not hold for `vp_pdf`, and needs correcting.

**Student-t branches (finite df):**
- They also take the log of a linear sum, so they too give -inf once that sum underflows, but much farther out. The distance at which the density drops below tiny, in component scales:

| df | D = 2 | D = 10 | D = 20 |
|---|---|---|---|
| 3 | 4e61 | 5e23 | 3e13 |
| 10 | 1e26 | 5e15 | 4e10 |
| 30 | 2e10 | 2e8 | 6e6 |
| -3 (product form) | 5e38 | 2e8 | 4e4 |

- They get close to the Gaussian's ~37 only for large df (df = 1000: about 55) or for the product form with many coordinates (D = 20, df = -30: 71).
- They have no gradient (NotImplementedError), `mode` does not use them, and no package code evaluates the density with a finite df.
- Left as they are (`w7_3_student_t_tails.py`).

**Not run (by the brief):** the seeded gate runs. `active_importance_sampling.py:362` combines the value in a log-sum-exp with the box terms; it could change only if a box term were below about -670, or if a point failed its own box's membership test.

## 4. Tests written but not run
None needs `optimize()`. I did not run the other existing tests in the files I touched, as the brief requires. Those most exposed are the `test_mode_*` tests against MATLAB values: the objective changes only where it used to be -inf.

## 5. Proposed changelog text
Both defects are in `v1.0.4`: `v1.0.4:pyvbmc/variational_posterior/variational_posterior.py` has `dy = dy / y` and the log of the sum at `:531-537`, and the decorated `pdf` takes D from `x.shape` at `:426`.
- **W7-3 (Fixed):** "`vp.log_pdf` (and `vp.pdf` with `log_flag=True`) returned `-inf`, with a `NaN` gradient, far in the tails of the posterior, where the density is too small for a double (and lost precision just before), although both are finite; there they are now computed by log-sum-exp over the components. `vp.mode()` therefore no longer returns its starting point unrefined, with a NumPy `RuntimeWarning`, on a posterior whose standard deviation is small (below about 0.01) in the units of the parameters."
  - Upgrading line: "`vp.log_pdf` is finite in the far tails where it returned `-inf`, and `vp.mode()` of a narrow posterior returns the refined mode where it returned its starting point."
- **W7-4 (Fixed):** "`vp.pdf` and `vp.log_pdf` raise a `ValueError` when the rows of `x` do not have `D` coordinates. For a one-dimensional posterior, a flat array of several points was read as a single point and failed with an obscure error or, in the transformed space with a finite `df`, returned one meaningless number."
  - Upgrading line: "`vp.pdf` and `vp.log_pdf` refuse an `x` whose rows are not `D` wide; for a one-dimensional posterior, pass several points as a column (`x.reshape(-1, 1)`)."

## 6. Beyond my rows, not fixed
- **Interaction with W4-7 (`active_sample_proposal_pdf`).**
  - After W7-3, a point gets no weight only where the variational log density is itself -inf (overflowing or infinite distance), not merely where the density underflows.
  - So a box sample that fails its own box's strict `<` test and lies where the density underflows now gets a large finite log weight instead of -inf. Such a sample arises when `rect_delta` is 0 in a coordinate, when a draw lands exactly at `-rect_delta`, or through rounding at the edge.
  - Not reachable in practice: a zero `rect_delta` is the zero-width column of W7-17, which stops the GP fit first.
- **Overflow in `_pdf`.** Where `nf·w/σ^D` overflows (component scales around 1e-16 at D = 20), the linear sum is inf or NaN, and so is the log. The ruling covers only sums below tiny.
- **Docstring.** The Returns section of `log_pdf` calls its output "the probability density" rather than the log density.

## 7. Final state
`git status --porcelain`: empty.

`git log --oneline ed9df90e..HEAD`:
```
f8e4d2d2 test(variational_posterior): finite differences of the density's gradient at K = 1, D = 1 over several rows
89d3cf1e test(entropy): a budget case that splits a component's samples, and the flag subsets
44655781 test(entropy): check the Monte Carlo entropy's gradient exactly
33b6b90b fix(variational_posterior): refuse points whose width is not the dimension
692fd72d fix(variational_posterior): keep the log density finite where the density underflows
```
