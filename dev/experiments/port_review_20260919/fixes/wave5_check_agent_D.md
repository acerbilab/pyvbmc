# Wave 5, fixes after the independent check: fix agent D

Raw report of fix agent D, wave 5 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-21. One Opus agent on its
own git worktree, cut at `b1bab4d`, made the fixes that the independent check
of the wave-5 pass asked for on the priors and the check of a prior against the hard bounds
(`../verification/wave5.md`, "The independent check of the pass"); the
orchestrator reviewed the diffs and cherry-picked the commits onto
`dev-port-review`, where they have other hashes. The text below the rule is
the agent's final message, unedited.

---

# Wave 5, fixes after the independent check: report of fix agent D

Worktree: `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-ad4a8ebc373a9f8f3`
Branch: `worktree-agent-ad4a8ebc373a9f8f3` (fast-forwarded from `b2c16ebf` to `b1bab4d2`; nothing pushed, no other branch touched)

## Setup outputs

- `git merge --ff-only dev-port-review` → fast-forward `b2c16ebf..b1bab4d2`; `git log -1 --format=%h` printed `b1bab4d2`.
- From the worktree root, `python -c "import os, sys; sys.path.insert(0, os.getcwd()); import pyvbmc; print(pyvbmc.__file__)"` printed
  `C:\Users\luigi\Documents\GitHub\pyvbmc\.claude\worktrees\agent-ad4a8ebc373a9f8f3\pyvbmc\__init__.py` (numpy 2.5.2, scipy 1.18.1, numpydoc 1.10.0).
- Every Python command ran with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. Three standalone probe scripts (scratchpad, `PYTHONPATH=<worktree>`) each printed the same `pyvbmc.__file__`: the bound-grid sweep, the support dtypes, and `numpydoc.validate` on the changed docstrings.
- `git status --porcelain` is empty at the end; HEAD is `5072f991`.

## `git log --oneline b1bab4d2..HEAD`

```
5072f991 docs(priors): correct the docstrings of the prior classes
264fb7ed fix(priors): give the shared finiteness check a private name
2d74c6d4 fix(priors): judge an argument of tile_inputs by the values it holds
7ee2030d fix(priors): take the value of a user marginal however it is written
d78573c1 fix(priors): read the support of a SciPy or Product prior from what it holds
ef5617cf fix(vbmc): give the prior-support check a slack for rounding
```

Diffstat over the range: 15 files, +483 / −47.

## Commits in order

| # | hash | item | what it does | files | tests added or changed (each seen to fail before) |
|---|---|---|---|---|---|
| 1 | `ef5617cf` | 1 | `_check_prior_covers_bounds` gives a coordinate whose two hard bounds are finite a slack of `1e-9 * (ub - lb)`; a coordinate with an infinite hard bound is compared exactly. The support is broadcast to the model's dimension before the message loop | `pyvbmc/vbmc/vbmc.py`, `pyvbmc/testing/vbmc/test_vbmc_init.py` | +`test_a_prior_built_from_the_hard_bounds_is_taken` (465 one-decimal pairs, step 1.3 over [−20, 20]), +`test_a_prior_built_from_random_hard_bounds_is_taken` (300 seeded pairs), +`test_a_vbmc_built_on_the_faq_prior_is_taken` (3 pairs through `VBMC`), +`test_a_support_short_of_a_hard_bound_is_refused` (both sides), +`test_an_infinite_hard_bound_against_a_finite_support_is_refused` (both sides), +`test_a_support_given_as_single_values_is_read_as_a_box`. Failing before: the three "taken" tests (67/465 grid pairs, 11/300 random pairs, 2 of the 3 `VBMC` cases) and the scalar-support one (`IndexError`). The two refusal tests guard the size of the slack and cannot fail on the stricter old code |
| 2 | `d78573c1` | 2 | `SciPy` and `Product` read the box of their support from the distribution, respectively the marginals, at every call; `a` and `b` are read-only properties, float64, shape `(D,)`, and are documented in the two class docstrings | `pyvbmc/priors/scipy.py`, `pyvbmc/priors/product.py`, `pyvbmc/testing/priors/test_scipy_prior.py` | +`test_a_stored_support_does_not_stand_for_the_distribution`, +`test_a_stored_support_does_not_refuse_the_hard_bounds`, +`test_a_prior_that_went_through_a_file_keeps_its_support` (pickle under `tmp_path`), +`test_the_support_is_float64` (3 distributions). Failing before: the first three and the `truncnorm` case of the dtype one |
| 3 | `7ee2030d` | 3 | the `UserFunction` branch of `Product._log_pdf` reads the return as an array, accepts one value, and raises a `ValueError` naming the marginal's position and the row otherwise; `UserFunction` documents the one value | `pyvbmc/priors/product.py`, `pyvbmc/priors/user_function.py`, `pyvbmc/testing/priors/test_product_prior.py` | +`test_a_user_function_marginal_may_return_its_value_as_an_array` (float, 0-d, `(1,)`, `(1, 1)`), +`test_a_user_function_marginal_that_returns_several_values_is_refused`. Failing before: the `(1,)` and `(1, 1)` cases and the refusal |
| 4 | `2d74c6d4` | 4 | an argument agrees with `size` when the two hold the same lengths once their axes of length one are dropped, whatever `squeeze` is | `pyvbmc/priors/tile_inputs.py`, `pyvbmc/testing/priors/test_tile_inputs.py` | +`test_tile_inputs_takes_a_shape_that_squeezes_to_size` (`(3,)`, `(1, 3)`, `(3, 1)`, `(1, 3, 1)` × both `squeeze`), **changed** `test_tile_inputs_shape_disagreeing_with_size` to run under both `squeeze` values. Failing before: the three non-flat shapes with `squeeze=False` |
| 5 | `264fb7ed` | 5 | `check_finite` → `_check_finite`, in `prior.py` and the four box modules that import it | `pyvbmc/priors/prior.py`, `uniform_box.py`, `trapezoidal.py`, `spline_trapezoidal.py`, `smooth_box.py`, `pyvbmc/testing/priors/test_priors.py` | +`test_the_finiteness_check_is_private_to_the_subpackage`. Failing before (with all five modules at the old revision): `ImportError`. No existing test named the function |
| 6 | `5072f991` | 6 | `Prior.pdf`'s `returns` header capitalized; `d` → `D` in `Prior.log_pdf` and `Prior._log_pdf`; `D` documented on `UniformBox` and `SmoothBox`; the shape refusal added to the four `Raises` blocks; the `tile_inputs` paragraph of commit 4 moved out of the parameter list, where numpydoc read it as nine parameters | `pyvbmc/priors/prior.py`, `uniform_box.py`, `smooth_box.py`, `trapezoidal.py`, `spline_trapezoidal.py`, `tile_inputs.py`, `pyvbmc/testing/priors/test_priors.py` | +`test_the_base_class_docstrings_use_the_numpydoc_section_headers`, +`test_the_base_class_docstrings_name_the_dimension_D`, +`test_a_box_constructor_documents_its_arguments_and_its_refusals` (4 classes: documented names equal the signature, and the `Raises` block mentions the shape the constructor is seen to refuse). All six fail before |

Every "fail before" was seen by copying the changed file(s) to the scratchpad, `git checkout --` on them, running the named tests, and copying back (never a bare `git stash`).

## Test runs

All from the worktree root, `-q -p no:cacheprovider`, with the three thread variables set.

- Final combined run of the ten modules of `pyvbmc/testing/priors/` and `pyvbmc/testing/vbmc/test_vbmc_init.py` at HEAD: **277 passed in 8.25 s** (`pyvbmc/testing/priors` alone: 117; `test_vbmc_init.py` alone: 160).
- Per item: after 1, `test_vbmc_init.py` 160 passed; after 2, priors 96 + init 160 = 256 passed; after 3, `test_product_prior.py` 12 passed; after 4, priors 110 passed; after 5, priors 111 passed; after 6, priors 117 passed.

No full suite, no oracle command, no golden replay, no `optimize()` call, no install.

## No number of a run moves at the default options

`prior.support()` has one caller in the package, `_check_prior_covers_bounds` (`vbmc.py:110`), which only accepts or raises; nothing else reads `prior.a` or `prior.b` outside `pyvbmc/priors/` (grep). Commit 3 writes `np.asarray(value).item()` where the old line assigned the same float. Commit 4 only widens the accepted set: every shape that was accepted has `shape == size`, hence equal squeezed shapes, and the final `reshape` is untouched. Commits 5 and 6 are a name and docstrings.

## Changelog sentences (for a user of release 1.0.4)

- **`ef5617cf`** — "A prior whose support is built from the hard bounds themselves, as `[uniform(loc=low, scale=high - low) for low, high in zip(LB, UB)]`, is accepted: the check that the hard bounds lie inside the support of `prior` allows a gap of a billionth of each coordinate's range, which covers the rounding of a support computed as `loc + scale`. A coordinate with an infinite hard bound is still compared exactly." Cannot stop a script; such a construction used to raise.
- **`d78573c1`** — "`SciPy` and `Product` report the support of the distribution, respectively the marginals, they carry, so a prior loaded from a file written by 1.0.4 gives the interval its distribution lives on instead of the one of the standardized distribution, and a run restored from such a file with `load(new_options=...)` is no longer refused. `prior.a` and `prior.b` are read-only float64 arrays." Can change what a script returns (`prior.a`, `prior.b`, `str(prior)` and `support()` of a shifted or scaled prior loaded from an older file) and can stop a script that assigns to `prior.a` or `prior.b`, so it belongs in the "Upgrading from 1.0.4" list.
- **`7ee2030d`** — "A marginal of a list-valued `prior=` given as a callable may return its log-density as a float or as an array of one element, so a function written `lambda x: scipy.stats.norm.logpdf(x)` works as a marginal as it does as a whole `log_prior=`; a return of several values raises a message naming the marginal and the point." Cannot stop a script; such a callable used to raise `ValueError: setting an array element with a sequence`.
- **`2d74c6d4`** — "`pyvbmc.priors.tile_inputs` takes a row, a column or a flat array of `D` values against `size=D` whatever `squeeze` is, where `tile_inputs(np.zeros((1, 3)), size=3)` raised; an array of another layout, such as `(2, 2)` against `size=4`, is still refused." Cannot stop a script.
- **`264fb7ed`** — "The helper that the box priors share to refuse an argument which is not finite is named `pyvbmc.priors.prior._check_finite`." It is neither exported from `pyvbmc.priors` nor on an API page, so it is outside the documented interface; an entry is worth having only if the section covers the private name, in which case `from pyvbmc.priors.prior import check_finite` stops working.
- **`5072f991`** — "The returns block of `Prior.pdf` appears on its documentation page, and the `D` argument and the shape refusal of the four box constructors are documented." Documentation only.

## What I stopped on

Nothing; all six items are done as ruled.

## What else I noticed

- **A record outside my area is now stale.** `dev/experiments/port_review_20260919/known_differences.md:2148` reads "Python: `pyvbmc/priors/prior.py: check_finite`"; the name is `_check_finite` after commit 5. (`fixes/wave5_agent_B.md:53` and `:116` also name it, but that file is the agent's own report of what it did then.)
- **A new pickle of a `SciPy` or `Product` prior no longer carries `a` and `b` in its dictionary**, since they are properties. A file written by this revision and read by 1.0.4 would find no `a`/`b`; files written by 1.0.4 and read here are what commit 2 fixes.
- **`tile_inputs` has no `Returns` section** (`numpydoc.validate` reports RT01), although it is exported and rendered on `docsrc/source/api/classes/priors.rst`. It is pre-existing and not in the ruling, so I left it.
- **`docsrc/source/api/classes/priors.rst:22`** says the hard bounds "must lie inside the support" (R5's finding 20). With the slack of commit 1, "inside" is looser still than that sentence suggests; the page is outside my area.
- **A 0-d array argument to a box constructor** is still refused with a message naming a shape the caller did not pass (`UniformBox(np.array(0.0), np.array(1.0), D=3)` → "found an input with shape (1,)"), because `np.isscalar` is False for a 0-d array and True for `np.float64` (R2's optional finding 11). Not in my items.
- **A support of the wrong length** (say two values for a model of three coordinates) now raises numpy's broadcast `ValueError` in `_check_prior_covers_bounds` rather than the check's own message. It is a `ValueError` either way, and the ruling asked only for the scalar case.
- **R2's findings 4, 5 (changelog wording) and 8 (the record that says the FAQ list "is taken")** fall outside my area; finding 8's statement is true after commit 1 for every pair I swept, but the record still needs the correction it asks for.
