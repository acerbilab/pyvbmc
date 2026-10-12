# Wave 6, the fix round of the independent check: gpyreg fix agent D

Raw report of fix agent D, wave 6 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-22. One Opus agent on a
worktree of gpyreg made by hand (`../gpyreg-port-review-D`, branch
`port-review-wave6-D`, cut from the head of gpyreg's `port-review-wave6` at
`dc2a930`), made the fixes in gpyreg of the findings of the independent
check of the wave-6 pass that the PI ruled on (the check's raw reports are
on the orchestrator's machine, `dev/scripts/runs/LOCAL.md`, "Port
correctness review"; `../verification/wave6.md`, "The independent check of
the pass"), one commit per finding. The orchestrator reviewed the diffs
before the commits were taken onto `port-review-wave6`. The text below the
rule is the agent's final message, unedited.

---

# Fix agent D report

All items D1 to D5, the text items and the release notes are committed: 16 commits on `port-review-wave6-D` in `C:/Users/luigi/Documents/GitHub/gpyreg-port-review-D`, on top of `dc2a930`. Nothing is pushed, and the worktree is clean. No other checkout was modified: the main `gpyreg` checkout is clean and `../gpyreg-port-review` is still at `dc2a930`. In PyVBMC I only ran the one module. By the end, PyVBMC's working tree showed edits by someone else (`dev/TODO.md`, `matlab_side_defects.md`, two untracked `wave6_A1d_*.py` scripts); they are not mine.

## Commits

"Tests run" means run on the final head unless stated otherwise. All runs used single-threaded BLAS.

| hash | finding | what changed | test seen to fail, and how | tests run, result |
|---|---|---|---|---|
| c4a441e | D1 (R4-1, R2-2) | `set_priors` accepts a coordinate whose location and `sigma` are both NaN as having no prior. The location is `mu` for the Gaussian and Student's t families, and `[a, b]` for the two smooth-box families. It refuses a `sigma` that is NaN where the location is not NaN, infinite, zero or negative, with a message naming the hyperparameter and the problem. The docstring is updated. | New `test_set_priors_takes_a_coordinate_without_a_prior` (4 families) failed with the old ValueError. `..._refuses_a_nan_scale_beside_a_location` and the extended `..._refuses_a_scale_that_is_not_positive` failed on the message ("NaN" and "zero or negative" were missing). 11 failures before the fix. | PyVBMC `test_gp_train_fit_inputs.py`: `test_output_dependent_noise_is_bounded_by_the_training_values` failed against `dc2a930` (seen) and all 14 pass against the fix and the final head. |
| 97d2b1a | D2 (R1-1) | `__robust_cholesky(sigma, scale=None)`: the band is `10*n*eps*max(scale, max|D|)`; without a scale it is `max|D|` as before. `random_function` passes `max(diag(K_star))`. The `Raises` entry is added. | `test_random_function_on_a_dense_grid[grid1]` (R1's setup, 100-point grid in [-1, 1]) raised `LinAlgError` (-5.44e-15 against a band of 3.29e-18). The extended indefinite-matrix test failed on the missing `scale`. R1's probe `p2_w68.py` now draws in 108 of 108 cases. | `test_gaussian_process.py`: 105 passed. |
| 2e96c90 | D3 (R2-1) | `get_priors` reads a block's family from its coordinates that have a prior. It returns a NaN df as `("student_t", (mu, sigma, nan))` or `("smoothbox_student_t", (a, b, sigma, nan))`, and partial blocks with their NaN coordinates. The meaning of a NaN df is documented in `set_priors` and in fit's `df_base`. | New `test_get_priors_returns_what_set_priors_reads_back` (5 cases, before and after a fit): `get_priors` returned None in all 5. | same |
| ebc9dcd | D4 (R2-6), test-only | The fixture is D=3: `smoothbox` on the length-scale block, `smoothbox_student_t` on the mean's log scales, one box per coordinate (value above / inside / below). The Student's t moved to the noise. | Reverted `5a0ede8`'s log-prior hunks in the working tree only (its `get_priors` hunk no longer applies after D3). All 10 fixture tests then failed: `ValueError: operands could not be broadcast together with shapes (3,) (2,)`. With the old fixture and the same revert, all 10 passed. File restored. | same |
| 43bf232 | R1-5 | An unsupported `s2` type raises `TypeError` (as in 1.2.1); a wrong row count still raises `ValueError`. | `test_convert_shapes_input_checks` (string and list) failed with ValueError. | same |
| fa2c70c | R3-5 | `_target_spread` tests `height != 0`; a NaN range is returned as is. | New `test_targets_without_a_range_are_not_called_equal` (NaN, all +inf): the range came back as 1.0. | `test_bounds_info.py`: 15 passed. |
| ad685ef | R3-11 | The `thin` and `burn` checks test finiteness first. | `test_sample_sanity_checks` raised `OverflowError` on `thin=np.inf`. | `test_slice_sample.py`: 40 passed. |
| 2d5796d | R1-3, test-only | Inside the update, `predict` is patched on the instance to return the clamped value, computed as `update` computes `sn2_eff`. The docstring says the guard fires where rounding drives the latent variance to zero or below. | Not a fix. Probe over 5 seeds and 4 values of `log sf`: the unpatched update warned in 11 of 20 cases, the patched one in 20 of 20. | `test_gaussian_process.py`: 105 passed. |
| 4994bbe | R1-6, test-only | The test spies on `f_min_fill` and asserts `PLB <= PUB` and `LB <= PLB`, `PUB <= UB`. The vacuous line is gone. | n/a | same |
| a4f7969 | R1-2, R5-2, R5-14, docs | The comment in `fit` and the tiny-range test docstring give the true mechanism: the clips keep an ordered pair ordered; the noise's pair is inverted when std < 1e-3; it stays inverted unless the range is below 1e-6; gplite clips the same way (`gplite_train.m:157-158`). | n/a | same |
| 51e492f | R1-4, R2-4, R2-9, R2-11, R5-29 | `predict`'s pooled-variance text; the `Raises` sections of `update`, `get_recommended_bounds` and `fit`; the `nll` comment; the refusal message now says "are NaN (not set)"; the gamma ratio is about `sqrt(df / 2)`. | n/a | same |
| 7a48db8 | R2-5, docs | The comment in the W6-5 test says what a NaN df means outside `fit`. | n/a | same |
| c36bada | R3-1, docs | The `uuinv` caller's comment explains the weight `w` per coordinate. | n/a | — |
| f1415de | R3-6, R3-8, docs | `__effective_n` Notes: a constant trace gives rounding noise (finite, or NaN where the variance is exactly zero), and `sample` reports NaN. The BDA3/Stan sentence is dropped. | n/a | `test_slice_sample.py`: 40 passed. |
| 39ef329 | R3-9, docs | The isotropic kernels page names the three methods taken from `AbstractIsotropicKernel`. | n/a | — |
| d93f03c | release notes | The 1.3.0 section, detailed under Choices. RST checked with docutils: 26 points plus one nested list, no warnings. | n/a | — |

I also ran `test_gaussian_process_isotropic.py`, which I did not touch but which exercises `get_priors`, `set_priors` and `random_function`: 23 passed.

Formatting: isort is clean. black 26.5.1 flags the same hunks before and after in every touched file (pre-existing only). No added code line exceeds 79 characters; the one long line is the existing one-line paragraph of the isotropic .rst page. All 16 commits carry the Co-Authored-By line and none has a Claude-Session trailer. I amended only my own last commit, to fix a phrase in its message.

## Choices

- **D1:**
  - "Location NaN" means NaN, not "non-finite". A location of `inf` beside a NaN sigma is refused, and the message says "NaN sigma where its location is not NaN", which stays true in that case.
  - For the smooth-box families, a coordinate has no prior only when `a`, `b` and `sigma` are all NaN.
  - I did not refuse a NaN location beside a finite sigma; the item did not ask for it (see Found on the way).
- **D2:**
  - The band is `max(scale, max|D|)`, not `scale` alone. The eigensolver's own error scales with `max|D|`, which can exceed the diagonal of `K_star` by up to `n`. Measured cases have `max|D| < 1.44`, so the scale governs there.
  - The scale is the kernel diagonal alone. `random_function` adds the noise as separate normals after the factor, so the noise never enters `C`.
  - Guarded with `np.max(..., initial=0.0)` for zero test points.
- **D3:** I extended `get_priors` to blocks with a no-prior coordinate. D1 makes `set_priors` accept them, and PyVBMC's block (`df = [nan, 3]`) would otherwise round-trip to None. A mixed df such as `[0, 3]`, or `[0, nan]`, still returns None, as before.
- **D4:**
  - Moved both smooth boxes onto blocks.
  - Raised the fixture to D=3. At D=2 the reverted code only doubles a constant, which none of the fixture's self-comparison tests can see; with 3 coordinates it raises, so all 10 tests fail on the revert.
- **R1-6:** I kept the test and asserted on the recorded pair, rather than dropping the line.
- **R3-8:** I dropped the BDA3/Stan claim rather than restricting it, because I could not check Stan or BDA3 here.
- **Release notes:**
  - The opening paragraph is replaced by one sentence on what **Upgrading** marks, instead of regrouping.
  - The W6-1, isotropic, W6-24 and RationalQuadratic points now say which fits move. I also found that `fit` never reads a component's `x0`: its default start is the middle of the plausible box. So W6-1's kernel change reaches only callers of `get_bounds_info`, and for a gpyreg fit only the NegativeQuadratic bounds matter.
  - The Upgrading list goes slightly past R2-3's five items. It adds three refusals of the same kind that can stop a 1.2.1 script, each verified against the pin by a probe: quad with another mean function (1.2.1 treated it as a constant), an `s2` given as a row, and fractional `thin`/`burn`.
  - The inputs point also notes that the shape checks raise `ValueError` where they raised `AssertionError`.
  - An Upgrading line was added for a NaN df reading as Gaussian after a fit (1.2.1 kept `df_base`).
  - The notes probe (`scratchpad/doublecheck_w6/fix_D/probe_notes.py`), run on both the pin and the branch, confirmed every claim that compares to 1.2.1.
- **Citations:** I cited `gplite_noisefun.m:105-106` rather than the `103-104` given in R1 and the brief (see below).

## Found on the way

- **NaN location beside a finite sigma.** `set_priors` still accepts it and the log prior is NaN. Examples: `("gaussian", ([nan, 0.3], [1.0, 1.2]))`, or a smooth box with a NaN `a`. This is the mirror of D1 and is pre-existing (probe confirmed); `gaussian_process.py`, `set_priors` and `__prior_masks`.
- **Two citation slips.** `gplite_noisefun.m:103-104` (R1-2 and the brief) are the hard pair; the plausible pair is at `:105-106` at `396d649`. `gplite_pred.m:158-160` covers the pooled variances only partly; they are at `:158-161`.
- **Ledger row W6-15 and the plan.** They still carry agent B's note that BDA3 and Stan "differ only in taking the first pair unconditionally" (R3-8), which the Notes no longer claim.
- **Missing Raises entries in `fit`.** It does not list the `ValueError`s it passes through from `get_recommended_bounds` (an inverted pair) or from `update`.
- **`set_bounds` accepts an inverted pair.** It is refused only when `fit` or `get_recommended_bounds` processes it. This is consistent with the ruling, and the notes say "and so does fit".
- **`get_priors` edge case.** On hyper_priors written directly (a smooth-box coordinate with finite `a`, `b` and NaN `sigma`), it can return a block that `set_priors` refuses. Not reachable through `set_priors`.
- **Still open from the review:** R2-8 (`no_prior` set before the checks), R2-10 (dead `np.abs(sigma)`), R2-12, R2-13 and R1-7 (the duplicated `sl` fallback).

## Final state

```
$ git -C C:/Users/luigi/Documents/GitHub/gpyreg-port-review-D log --oneline dc2a930..HEAD
d93f03c docs: the 1.3.0 release notes, true point by point
39ef329 docs: what the isotropic kernels take from each parent
f1415de docs: the Notes of the effective sample size
c36bada docs: the weight of the plausible box in the space-filling design
7a48db8 docs: what a NaN df means in the test of the documented densities
51e492f docs: predict's pooled variance, the Raises sections and three comments of GP
a4f7969 docs: the mechanism behind an inverted plausible pair in fit
4994bbe test: the tiny-range fit test asserts on the plausible pair the design receives
2d5796d test: the low-noise duplicate update test fires the guard by construction
ad685ef fix: SliceSampler.sample refuses an infinite thin or burn
fa2c70c fix: targets with a NaN are not reported as all equal
43bf232 fix: an unsupported type of s2 raises TypeError
ebc9dcd test: the smooth boxes of the prior fixture are set over blocks
2e96c90 fix: get_priors returns a prior with NaN degrees of freedom
97d2b1a fix: random_function measures the rounding band against the prior variance
c4a441e fix: set_priors takes a coordinate of a block that has no prior

$ git status --short
(empty)
```

Probe scripts and run logs are in `C:/Users/luigi/AppData/Local/Temp/claude/C--Users-luigi-Documents-GitHub-pyvbmc/3d072054-989b-4149-86b7-253c1fc64af1/scratchpad/doublecheck_w6/fix_D/`: `probe_d2.py`, `probe_w69.py`, `probe_notes.py` and `run_*.log`.
