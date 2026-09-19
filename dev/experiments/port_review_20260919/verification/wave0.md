# Wave 0 verification

Verification of the findings of the wave-0 reviewers (`../reviews/N1_internal.md`,
`N2_internal.md`, `N2_internal_rerun.md`, `N3_internal.md`) by the
orchestrator, 2026-09-19, on branch `dev-port-review` at `0e4e27a` (package
code identical to `f91fdf0`). Each finding was reproduced by running a small
script or by reading the cited lines; the scripts under `scripts/` are the
reviewers' own reproductions, read before reuse, plus one written here
(`n2s_f1_prior_draws.py`). Outputs are under `logs/`. All scripts insert the
repository root into `sys.path` or were run with `PYTHONPATH` set to it, so
they exercise this checkout.

Interpreters, from the repository root:

- core: `.venv/Scripts/python.exe`
- PyMC: `dev/scripts/runs/pymc_feasibility_20260914/venv/Scripts/python.exe`
  (PyMC 6.3.2, PyTensor 3.3.1, ArviZ 1.3.0)
- Torch: `../pyvbmc-stage3/.venv/Scripts/python.exe` (Torch 2.7 CPU, mpmath)

These slices have no MATLAB counterpart, so the classes are: **defect**
(the code contradicts its own contract), **as specified** (the code does
what its plan says; the question is whether the plan is what is wanted),
**documentation** (the text is wrong, the code is not), **cleanup**.

## S-VBMC (N1)

| Finding | Method | Outcome | Class |
| --- | --- | --- | --- |
| F1 runs with different original-space boxes give a silent NaN | `n1_f1_f2_f3_svbmc.py` part 1, Torch | `elbo nan entropy nan w [nan nan nan nan]`; `_validate_posteriors` compares no transformer fields | defect |
| F2 a second `optimize()` starts from the first solution | same script, part 2; `svbmc.py:605` reads `self.w`, `:747` overwrites it | per-run masses 0.5/0.5, then 0.116/0.884, then 0.066/0.934 (`posterior-only`); `all-weights` drifts likewise | defect |
| F3 the two shrinkage stages do not compose to the shrunk run level | same script, part 3 | corrected levels `[1.295, 20.981]` against shrunk targets `[1.132, 21.281]`; reproduces the module's value exactly | as specified: step 5 of `plans/svbmc-shrinkage-estimator.md` ("Add `E_shrunk_m - E_m` to every `within_m` component") and section 5 of the shrinkage note define exactly this |
| F4 `sample(balance_flag=True)` does not keep every component count within one draw | `n1_f4_balance.py`, Torch | worst deviation 1.8 at `n = 10`, 2.75 at `n = 37`, 0 at 100 and 1000; the within-run split delegates to `VariationalPosterior.sample`, which allocates its remainder by `rng.choice` | documentation |
| F5 `stacked_entropy` says `stacked_ELBO` subtracts the Jacobian corrections | reading `svbmc.py:428`, `:467`, `:552-558` | the subtraction happens once, in `__init__` (`I_corrected`) | documentation |

## PyMC adapter and exports (N2, static review and rerun)

| Finding | Method | Outcome | Class |
| --- | --- | --- | --- |
| rerun F1 support inferred from the transform kind | `n2r_f1_wald_support.py`, `n2r_f1_wald_abort.py`, PyMC | `pm.Wald(alpha=2)` reports support `(0, inf)`; `log_joint(log 1.9)` raises; a plausible box inside the claimed support is accepted and the run aborts with `Non-finite log density for model variables x=0.5895` | defect |
| rerun F2 only two of PyTensor's no-gradient signals are caught | `n2r_f2_rice_gradient.py`, PyMC; `_target.py:836`, `:859` | `pm.Rice` fails construction with `MethodNotDefined` (not a `NotImplementedError`) | defect |
| rerun F3 Torch density differs from `vp.pdf` near a hard bound | `n2r_f3_near_bound.py`, Torch, 60-digit reference | with bounds `(-2, 3)`, probit: NumPy `ParameterTransformer.__call__` error 1.6e-11 at a gap of 1e-6, 2.2e-5 at 1e-12, 3.0e-3 at 1e-14, 2.0e-2 within 2 ulp; the Torch inverse is exact to 1 ulp throughout | defect of the core transformer's precision near a nonzero bound; the Torch export is the accurate side |
| rerun F4 and static F2 one absorbed draw rejects the whole export batch (found by both reviewers) | `n2r_f4_onesided_absorb.py`, PyMC | coordinate -50 with `lower = 1.0` raises; 0, 50 and 500 map correctly; 800 overflows and raises | as specified (the adapter plan requires values strictly inside the support), with a reachable boundary case the plan did not consider |
| static F1 4000 prior draws on every default construction | `n2s_f1_prior_draws.py`, PyMC | one `_prior_draws` call of shape `(4000, D)` on all five accepted models, four of which need no curvature fallback; 0.08 to 0.11 s, 2.7 s of 3.7 s for the truncated-normal model | as specified: `plans/pymc-target-adapter.md` prescribes the location check for an automatically searched start and `check_location=False` for an explicit one |
| static F3 `UnsupportedModel` relabelled as a mode-search failure | reading `_target.py:608-612`, `_compat.py:16` | `UnsupportedModel` subclasses `ValueError` and the wrapper catches `ValueError` | defect (diagnostics only) |
| static F4 the no-gradient warning names the initial point when the user gave `start` | reading `_target.py:1187-1191`, `:619-621` | the message keys on the route, not on the start kind | defect (message only) |
| static F5 `unbounded` computed and never read | `grep` | one assignment at `_target.py:384`, no use | cleanup |

## Calibration (N3)

| Finding | Method | Outcome | Class |
| --- | --- | --- | --- |
| F1 non-default entropy budgets change the numbers | `n3_f1_entropy_budget.py`, core | the entropy gradient differs from the 2^14 reference by up to 6.7e-16 at every other budget; the value-only entropy differs by 8.9e-16 at 2^18 | as specified at kernel level: `plans/machine-local-calibration.md` sets `rtol = atol = 1e-12` for entropy and records a maximum difference of 7.11e-15, and notes that no optimizer trajectories were run. `golden_trace.py` and `svbmc_pool_run.py` pin `performance_calibration="off"`, so the references are unaffected |
| F2 `active_d4_k20` times 200 samples per component with gradients | reading `_campaign.py:204-211`, `variational_optimization.py:489-500`, `active_sample.py:786-791` | both call sites of `ns_ent_fine_active` pass `compute_grad=False` | defect (performance only) |
| F3 the summary reports a rejected candidate's speed-up | reading `_campaign.py:1293-1310` | `heldout_speedup` is the median of `group_speedups` whatever `pass` says | defect (report metadata) |
| F4 a `ValueError` from `make_record` or `write_record` discards a finished campaign | reading `_api.py:229-239` | `make_record` is outside any `try`; `write_record` is guarded for `OSError` only | defect, latent (no live trigger) |
| F5 the held-out gate needs all four rounds | reading `_campaign.py:883` | `ceil(0.80 * 4) = 4` | as specified or not: the plan does not say; conservative |
| F6 an incomplete campaign discards the groups that finished | reading `_campaign.py:1509-1532` | `settings = _defaults()` whenever the status is not complete; a test pins it | design choice |
| F7 the docstring promises output the code does not print | reading `__init__.py:14-16`, `_api.py:79-99` | neither the selected budgets nor a held-out number is printed | documentation |
| F8 `sieve_gradient` times a gradient the package takes one point at a time | reading `_campaign.py:171-178`; `grad_flag=True` occurs only in `get_mode` | confirmed | defect (performance only) |
| F9 the cache key omits the package version | reading `_cache.py:269-282` | schema, kernel and workload revisions plus the machine identity | design choice |
| F10 `_MAX_WORKLOAD_SLOWDOWN` and `_deduplicate_order` are unused | `grep` | one definition each, no use | cleanup |

All 24 findings are confirmed as facts about the code. None was a misreading.
