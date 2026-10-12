# Wave 7 fix round: gpyreg fix agent D, fourth round

Raw report of fix agent D's fourth round, wave 7 of the port correctness
review (`dev/plans/port-correctness-review.md`), 2026-09-23. The same agent,
on the same worktree and branch (`../gpyreg-w7`, `w7-fixes`), added the
smooth-box families to the test that pins the unchanged path of the prior's
mass, a gap it had noted in its third round. The text below the rule is the
agent's final message, unedited.

---

# Wave 7 fix round: fourth report of gpyreg fix agent D

The smooth-box cases are in, in one commit, and a `>=` mutant of the smooth-box mass switch alone now fails the test.

**Commit:** `c81063c` "test: the prior's mass from the centre down, for the smooth boxes too" (test only).
- `gpyreg/testing/test_gaussian_process.py::test_prior_mass_from_the_centre_down` now covers four families: Gaussian, Student's t, smooth box, and smooth-box Student's t.
- Smooth-box parameters: `a = -1.0`, `b = 1.6`, `sigma = 1.2`, `df = 3`.
- The lower bound is placed exactly at the computed centre `0.5 * (a + b)` (0.30000000000000004), and 1.3 below it. The other bound pairs are (-5, -1) and (centre, inf).
- The expected mass is `smoothbox_cdf` or `smoothbox_student_t_cdf` at `ub` minus the same at `lb`, compared bit for bit.
- Run by node id on the committed code: 4 passed.

**Mutant, in a scratch copy** (`…\scratchpad\wave7\fix_D\mut`; imports confirmed from there):
- Only the smooth-box line of `__recompute_normalization_constants` was changed, `upper_half = lb > 0.5 * (a + b)` to `>=`.
- Result: 2 failed, 2 passed. The two smooth-box cases fail and the Gaussian and Student's t cases pass, since their switch is untouched.
  - `[smoothbox]`: `assert 0.48779744811547093 == (0.987797448115471 - 0.5)`
  - `[smoothbox_student_t]`: `assert 0.4612200056177151 == (0.961220005617715 - 0.5)`

**`git status --porcelain`** of `gpyreg-w7`: empty. Nothing was pushed.

**`git log --oneline 0186d89..HEAD`:**
```
c81063c test: the prior's mass from the centre down, for the smooth boxes too
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
