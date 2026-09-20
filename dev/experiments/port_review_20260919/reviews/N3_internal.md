# N3 — machine-local performance calibration, internal-correctness track

Raw reviewer report, wave 0 of the port correctness review
(`dev/plans/port-correctness-review.md`), 2026-09-19. One Opus agent,
fresh context, read-only on the repositories; its check scripts ran in the
session scratchpad and are not retained. The text below is the agent's
final message, unedited.

---

## 1. Coverage

Read completely:

- `pyvbmc/calibration/__init__.py`
- `pyvbmc/calibration/profile.py`
- `pyvbmc/calibration/_api.py`
- `pyvbmc/calibration/_cache.py`
- `pyvbmc/calibration/_campaign.py`
- `pyvbmc/entropy/entmc_vbmc.py` (consumer)
- `pyvbmc/_user_hints.py`, `pyvbmc/rng.py`

Read the relevant regions of (not under review):

- `pyvbmc/variational_posterior/variational_posterior.py` — `__init__`, `_set_calibration`, `_ensure_calibration_state`, `_resolve_calibration`, `_apply_calibration_load_override`, `pdf`/`_pdf`, `load`.
- `pyvbmc/vbmc/vbmc.py` — profile plumbing at lines 327–328, 414–416, 1181–1206, 2562–2571, 2881, 2916–2980, 3140–3142.
- `pyvbmc/vbmc/variational_optimization.py` — `_CANDIDATE_SHARED` (1034–1084), `_eval_vp` (480–500), `_sieve`/`optimize_vp` sample counts (208–260, 759–768), `_neg_elcbo` signature and entropy call (1087–1098, 1187–1196, 1258–1264).
- `pyvbmc/vbmc/active_sample.py` — option overrides at 276–282 and the fine-active ELBO call at 786–791.
- `pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini` — lines 43–63, 339.
- `pyvbmc/testing/oracles/_state.py::build_vp`, `.../oracles/_oracles.py` (how the oracles construct posteriors).
- `pyvbmc/testing/calibration/{test_profile,test_cache,test_cache_process,test_api,test_campaign,test_lifecycle}.py` — all test names, and the bodies of `test_campaign.py` and `test_lifecycle.py` in the regions cited below.

Skimmed: `pyvbmc/vbmc/_runtime_tips.py`, the acquisition functions' `vp.pdf` call sites.

Not reached: the PyMC/S-VBMC/Torch adapters (they do not read a profile); the MATLAB comparison fixtures; anything under `dev/` (excluded by the brief).

Small checks run (repository `.venv` interpreter; nothing written into the repository, no campaign run, no `optimize()` run, no cache file touched):

1. `check_layout.py` — enumerated `_layout_signature`/`_layout_aliases`/`_workspace_estimate` for every workload at every candidate budget, and compared each workload's `(D, K, count, effective_count)` against `_cache._WORKLOAD_SHAPES`. All ten shapes agree; the per-workload alias structure is used in F2/F8 below.
2. A snippet recomputing the package's per-component entropy sample counts from the `.ini` lambdas (`ns_ent`, `ns_ent_boost`, `ns_ent_fine`, `ns_ent_active`, `ns_ent_fine_active`) for `K ∈ {2, 5, 10, 20, 26, 50}`.
3. `check_exact.py` — `_entmc_vbmc` at all five candidate budgets on one fixed posterior (D=4, K=20, Ns=38 with gradients; and Ns=4096 value-only), identically seeded generators.
4. `check_pdf.py` — `vp._pdf` at three budgets on D=15, K=26, N=100000.
5. `python -c "import pyvbmc; ..."` — confirmed `pyvbmc.calibration._cache`, `pyvbmc.calibration._campaign`, `filelock` and `threadpoolctl` are absent from `sys.modules` after `import pyvbmc`.
6. `math.ceil(0.80 * r)` for `r ∈ {4, 5}`.

Confirmed working as documented (no finding): `import pyvbmc` neither runs the campaign nor reads the cache (check 5); the campaign never touches NumPy's global legacy state (`_make_problem` seeds dedicated `Generator`s, `make_vp` passes a `Generator` so `get_rng` returns it unchanged, and `run_campaign` snapshots and re-checks `np.random.get_state()` at `_campaign.py:1339` and `:1524`); the pdf budget is exactly result-neutral (check 4, matching the docstring at `variational_posterior.py:833–834`); a read-only cache directory degrades to an in-memory profile (`campaign_guard`'s `except OSError` at `_cache.py:1133`, then `source="memory"` at `_api.py:256` and the memory-wins branch at `_cache.py:964`); a corrupted, oversized, too-deep or wrong-fingerprint cache file is a plain cache miss (`read_record`, `_cache.py:931–951`); `write_record` writes through `mkstemp` in the destination directory plus `fsync` plus `os.replace`; and the campaign's own posteriors are built with `calibration="off"` (`_campaign.py:105`), so a campaign cannot recurse into the cache.

## 2. Findings

### F1. The selected entropy budgets change numerical results, not only speed
- Location: `pyvbmc/calibration/_campaign.py:608` (and the tolerance at `_campaign.py:416`); kernel at `pyvbmc/entropy/entmc_vbmc.py:120-156`
- Category: cross-module
- Proposed classification: possibly intentional
- Confidence: high
- The campaign validates a candidate budget against the default by `_output_comparison(..., exact=workload.group == "pdf" or budget == DEFAULT_BUDGET)`, i.e. bit-exactness is required for the pdf group and for the default budget only; every non-default *entropy* budget is accepted on `np.allclose(rtol=1e-12, atol=1e-12)`. The tolerance is deliberate, because the block structure in `_entmc_vbmc` changes the order of accumulations that run across blocks: `lambd_acc` (`entmc_vbmc.py:151`) and `w_acc` (`entmc_vbmc.py:156`) accumulate over component blocks, and `sum_logq` (`entmc_vbmc.py:140`) accumulates over sample blocks whenever `step < Ns`. Check 3 confirms this on a real posterior: with all gradients requested, `dH` differs between budgets by up to 6.7e-16 (`np.array_equal` False at every non-reference budget); value-only with Ns=4096, `H` differs between 2^14 and 2^18 (7.030979309895097 vs 7.0309793098950975). So a profile that selects a non-default `entropy_grad_chunk_elements` or `entropy_value_chunk_elements` makes the package compute different floating-point numbers from an uncalibrated install. The expectation stated in the review brief — and the module's own framing, "independent of user or solver random streams" (`_campaign.py:3-5`) — is that the settings are performance-only. `advanced_vbmc_options.ini:339` sets `performance_calibration = "cached"` by default, so this is the default path on any machine that has run `pyvbmc.calibrate()` once.
- Consequence if real: an ELBO/entropy gradient perturbed at the ulp level feeds Adam (`variational_optimization.py:1261`), the sieve comparison in `_eval_vp` (`variational_optimization.py:489`) and the active-phase accept/reject at `active_sample.py:789-793`. Over an `optimize()` run this amplifies chaotically: two machines differing only in their cached profile can return different component counts, different termination iterations and visibly different ELBOs. The `1e-12` acceptance tolerance is also roughly 3e4 ulps for `H ≈ 7` — four orders of magnitude looser than the differences actually observed — so in practice it does not bound how far a selected setting may move the result.
- Suggested reproduction: check 3 (`scratchpad/port_review/N3_internal/check_exact.py`) — build one posterior, call `_entmc_vbmc` with `budget=2**14` and `budget=2**16` from identically seeded generators, compare with `np.array_equal`.
- Test adequacy: no. `test_lifecycle.py::test_public_kernels_select_profile_budgets` only checks that the right budget reaches the kernel (it monkeypatches the kernel away). `test_campaign.py::test_small_real_helpers_preserve_numerics_rng_and_state` asserts `result["pass"]`, which for entropy is the same `allclose` test, not exactness. `test_vbmc_seed.py` pins `performance_calibration` to a fixed `TEST_CALIBRATION_PROFILE`, and `pyvbmc/testing/oracles/_state.py::build_vp` constructs posteriors with no `calibration=` argument (historical defaults), so neither the seed tests nor the oracles exercise a calibrated run.

### F2. `active_d4_k20` pairs a sample count the package only uses value-only with full gradient flags, and is scored in the `entropy_grad` group
- Location: `pyvbmc/calibration/_campaign.py:204-211`
- Category: indexing/shape
- Proposed classification: suspected defect
- Confidence: medium
- The workload is `_Workload("entropy_grad", "active_d4_k20", 4, 20, 200, (True, True, True, True))`: 200 samples per component with all four gradients, scored into `entropy_grad_chunk_elements`. The package's per-component entropy sample counts come from `advanced_vbmc_options.ini:43-63` divided by `K` (`variational_optimization.py:482` and `:759`). Check 2 gives, for `K=20`: `ns_ent/K = 37` (matches `adam_d4_k20`), `ns_ent_boost/K = 55` at `K=50` (matches the two `boost_*` workloads), `ns_ent_fine/K = 4096` (matches the two `fine_*` workloads), `ns_ent_active/K = 8`, and `ns_ent_fine_active/K = 200`. The only place 200 samples per component occurs is `ns_ent_fine_active`, and both of its call sites request no gradients: `_eval_vp` passes `compute_grad=False` positionally (`variational_optimization.py:489-500`, signature at `:1087-1098`) and `active_sample.py:789-791` does the same. `entmc_vbmc` (`entmc_vbmc.py:196-199`) therefore routes that call to `entropy_value_chunk_elements`. Conversely the active-phase *gradient* call uses `ns_ent_active/K` ∈ [6, 16], which this workload does not represent. So the shape is measured in the wrong group, with gradient flags that never co-occur with that sample count.
- Consequence if real: performance only. `active_d4_k20` is one of four workloads in the `entropy_grad` geometric mean (`_group_ratios`, `_campaign.py:861-866`) and one of four subject to the per-workload regression veto (`_gate_ratios`, `_campaign.py:887-889`), and check 1 shows it has five distinct layouts, i.e. full weight at every candidate. The `entropy_grad` selection is thus partly driven by a shape the package never executes, while the `entropy_value` group (only `fine_d4_k20` and `fine_d15_k26`, both Ns=4096) never measures the 200-sample active-phase shape it actually governs. The layout signature does not depend on the gradient flags, so the same `(D, K, count)` scored in `entropy_value` would remain equally discriminating.
- Suggested reproduction: check 2, plus reading the `compute_grad` argument at `variational_optimization.py:495` and `active_sample.py:790`.
- Test adequacy: no. `test_campaign.py` uses `_small_recipe()` and `_fake_kernels` for every selection test; the only test touching `campaign._workloads()` is `test_workspace_estimate_bounds_measured_incremental_peaks`, which checks memory bounds. Nothing asserts that a workload's `(group, count, grad_flags)` triple corresponds to a call the package makes.

### F3. A rejected candidate's speed-up is reported in the summary next to the default selection
- Location: `pyvbmc/calibration/_campaign.py:1295-1308`
- Category: formula/gradient
- Proposed classification: suspected defect
- Confidence: medium
- `_summary` computes `speedup = median(heldout["group_speedups"])` whenever that list is non-empty, independently of `heldout["pass"]`. When the held-out gate rejects the discovery candidate, `_validate_heldout` (`_campaign.py:1241-1248`) still returns the measured `group_speedups` for that candidate while setting `accepted_budget = DEFAULT_BUDGET`, and `run_campaign` then stores `settings[setting] = 2**16`. The resulting summary entry is `{"selected": 65536, "reason": "held-out gates failed", "heldout_speedup": <ratio of the rejected candidate>}`. The record validator accepts it (`_cache.py:492-499` requires only `None` or a positive finite number), so that is what lands in the persisted JSON. A reader of the summary would attribute the speed-up to the selected setting. The held-out gate can fail with a median well above 1 — for example when `winning_rounds < required_winning_rounds`, when one workload regressed (`_campaign.py:887-891`), or when either same-budget control band failed (`_campaign.py:1238-1240`) — so the reported number can contradict the selection.
- Consequence if real: report metadata only; no effect on the settings or on any numerical result. It misdescribes the outcome in the one field the schema reserves for the held-out comparison.
- Suggested reproduction: call `campaign._summary(report, settings, "complete")` with a `groups` entry whose `heldout_validation` has `pass=False`, `accepted_budget=2**16` and a non-empty `group_speedups`; inspect `summary[setting]["heldout_speedup"]`.
- Test adequacy: no. `test_campaign.py::test_heldout_rejects_unconfirmed_gain_and_bad_same_budget_control` asserts only `result["accepted_budget"] == campaign.DEFAULT_BUDGET`; the summary field is checked only in `test_fake_campaign_completes_after_thirty_second_estimate`, where every group retains the default and the field is `None`.

### F4. A successful campaign can be lost to an uncaught `ValueError` from record building or writing
- Location: `pyvbmc/calibration/_api.py:229-239`; raise sites at `pyvbmc/calibration/_cache.py:1017` and `pyvbmc/calibration/_cache.py:1037-1038`
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: medium
- After a `status == "complete"` campaign, `_api.calibrate` calls `make_record(...)` outside any `try`, then `write_record(record)` inside `try/except OSError`. Both can raise `ValueError`: `make_record` ends in `validate_record`, whose ~40 invariants include several tables duplicated from `_campaign` (`_cache._WORKLOAD_SHAPES`, `_GROUP_WORKLOADS`, the hard-coded round counts `5` and `4` at `_cache.py:618/621/633`, `CANDIDATE_BUDGETS`, `KERNEL_REVISION`, `WORKLOAD_REVISION`); and `write_record` raises `ValueError` if the serialized record exceeds `MAX_CACHE_BYTES`. Neither is caught, so instead of the documented behaviour ("If another calibration is active or the campaign is incomplete, this function returns a compatible prior profile or historical defaults with the outcome recorded in its `status` and `provenance`", `__init__.py:29-31`), `pyvbmc.calibrate()` raises and discards the finished measurements — `register_success` (`_api.py:262`) is never reached, so the result is not even retained in process memory.
- Consequence if real: a full campaign is thrown away and the caller sees a traceback rather than a fallback profile. I could not construct a live trigger: I verified by hand that the ten workload shapes agree with `_cache._WORKLOAD_SHAPES` (check 1), that `DISCOVERY_ROUNDS == 5` and `HELDOUT_ROUNDS == 4` match the literals in `_cache`, and that a complete report sits well under `MAX_REPORT_ITEMS` and `MAX_CACHE_BYTES` (estimated ~3200 items and a few hundred KB). The exposure is to future drift between the two modules' tables — exactly what the comment at `_cache.py:161-162` warns about.
- Suggested reproduction: monkeypatch `_cache.MAX_CACHE_BYTES` to a small value and call `_api.calibrate` with a stubbed `_run_campaign` returning a complete result; the call raises instead of returning a profile.
- Test adequacy: partly. `test_api.py::test_persistence_failure_keeps_success_in_memory` covers the `OSError` path only. No test drives a `ValueError` out of `make_record` or `write_record`.

### F5. The held-out win gate is effectively 100%, not the named 80%
- Location: `pyvbmc/calibration/_campaign.py:883` with `_MIN_WIN_FRACTION = 0.80` at `pyvbmc/calibration/_campaign.py:27` and `HELDOUT_ROUNDS = 4` at `:22`
- Category: formula/gradient
- Proposed classification: possibly intentional
- Confidence: high
- `needed_wins = math.ceil(_MIN_WIN_FRACTION * rounds)`. With `rounds = DISCOVERY_ROUNDS = 5` this is `ceil(4.0) = 4`, i.e. 80% as named. With `rounds = HELDOUT_ROUNDS = 4` it is `ceil(3.2) = 4`, i.e. all four rounds must show a ratio strictly greater than 1 (`_campaign.py:882`). The held-out phase therefore applies a stricter rule than the constant states. (Check 6 also shows the discovery case sits on a knife edge: `0.80 * 5` is exactly `4.0` in binary floating point only because the product rounds back down — `Fraction(0.8) * 5 != 4`.)
- Consequence if real: the gate is conservative, never permissive: a genuine but slightly noisy improvement is rejected and the historical default kept. Performance only.
- Suggested reproduction: `math.ceil(campaign._MIN_WIN_FRACTION * campaign.HELDOUT_ROUNDS)`.
- Test adequacy: no. `test_campaign.py::test_heldout_rejects_unconfirmed_gain_and_bad_same_budget_control` supplies timings that fail for other reasons; no test pins `required_winning_rounds`.

### F6. A watchdog stop or coverage failure discards every group that did complete
- Location: `pyvbmc/calibration/_campaign.py:1509-1532` (with `ESTIMATED_SECONDS = 30.0` at `:24` and `WATCHDOG_SECONDS = 300.0` at `:23`)
- Category: control flow
- Proposed classification: possibly intentional
- Confidence: high
- `run_campaign` catches `_WatchdogExpired`, `MemoryError` and `_IncompleteCoverage`, and in the `finally` block sets `settings = _defaults()` whenever `status != "complete"`. There is no partial outcome: if the `pdf` and `entropy_grad` groups finished and the watchdog fires during `entropy_value`, all three settings revert to `2**16` and nothing is persisted. `ESTIMATED_SECONDS` is a hard-coded constant that is exported in the report and required to be positive by `_cache.py:447-461`, but is never compared against the measured elapsed time and never displayed (see F7). The campaign performs roughly 67 timed calls per workload for a workload with five distinct layouts (5 first calls + 5 warmups + 1 control warmup + 5 rounds × 6 calls + 6 numerical-validation calls + 4 held-out warmups + 16 held-out calls), including the two heaviest shapes `fine_d15_k26` (K=26, Ns=4096, D=15) and `large_density` (N=100000, D=15, K=26). A machine several times slower than whatever produced the 30 s figure therefore gets nothing at all rather than the groups it did measure. (I did not time the real campaign — the brief forbids running it — so I make no claim about the actual duration.)
- Consequence if real: on a slow machine, `pyvbmc.calibrate()` spends up to five minutes and returns the historical defaults with `status="incomplete"`. Performance and user time only; no effect on results.
- Suggested reproduction: `campaign.run_campaign(deadline=..., _clock=..., _kernel_api=..., _recipe=...)` with a fake clock that crosses the deadline during the third group.
- Test adequacy: `test_campaign.py::test_watchdog_stops_only_after_call_and_returns_no_partial_winner` asserts exactly this behaviour, so it would not flag it (see §3).

### F7. `calibrate`'s docstring promises verbose output the code does not produce
- Location: `pyvbmc/calibration/__init__.py:14-16` and `:27-28`; `pyvbmc/calibration/_api.py:79-99` and `:140`, `:152-158`
- Category: defaults
- Proposed classification: suspected defect
- Confidence: high
- `pyvbmc.calibrate`'s docstring says `verbose` prints "the runtime estimate, campaign progress, selected settings, held-out results, and persistence status", and the Notes say "The displayed 30-second duration is an estimate rather than a deadline." `_api.calibrate` prints a fixed preamble ("Calibration usually takes tens of seconds.", `_api.py:158`) that never mentions 30 seconds, and `_print_report_summary` prints neither the three selected element budgets nor any held-out number — only "Faster settings were selected for this machine." or "The standard settings performed well; no reliable improvement was found." (`_api.py:79-88`). `_api.calibrate`'s own docstring ("Print progress and the resulting settings", `_api.py:140`) is overstated the same way.
- Consequence if real: documentation only. A user following the docstring to read off the selected settings or the measured held-out speed-up has to open the JSON report instead.
- Suggested reproduction: read the two `print` branches at `_api.py:79-99` against the docstring at `__init__.py:14-16`.
- Test adequacy: no. `test_api.py::test_verbose_feedback_reports_progress_values_and_location` and `::test_verbose_default_completion_uses_plain_summary` assert the strings the code actually prints.

### F8. `sieve_gradient` measures a pdf shape the package does not produce
- Location: `pyvbmc/calibration/_campaign.py:171-178`
- Category: indexing/shape
- Proposed classification: unsure
- Confidence: medium
- `sieve_gradient` is `(D=4, K=20, N=8192)` with `grad_flags` set, so `_invoke` calls `vp._pdf(..., grad_flag=True)` on 8192 rows. The only `grad_flag=True` pdf call in the package is inside `get_mode`'s objective (`variational_posterior.py:1241-1243`), which SciPy evaluates one point at a time; every other `vp.pdf` call site (`acquisition_functions/acq_fcn*.py:36/42`, `active_importance_sampling.py:378/488`, `kl_div`, `mtv`) is value-only. The other three pdf workloads do map onto real shapes: `sieve_value` (N=8192) is the acquisition sieve, `large_density` (N=100000, D=15, K=26) is `kl_div`/`mtv`/`get_mode`'s starting-point draw, and `small_value` (N=8) is the single-point acquisition evaluation. Check 1 additionally shows `small_value` has one layout at all five budgets, so it is filtered out of every ratio by the `affected` test at `_campaign.py:830-834` — correctly, since its execution is identical at every budget, but it means the pdf group's decision and its regression veto rest on `sieve_value`, `sieve_gradient` and `large_density`, two of which are the same `(D, K, N)`.
- Consequence if real: performance only. The `(4, 20, 8192)` shape carries two of the three effective votes in the pdf group's geometric mean, one of them in a gradient mode the package never runs at that size.
- Suggested reproduction: `grep -rn "grad_flag=True" pyvbmc --include=*.py` outside the tests and the calibration module returns only `variational_posterior.py:1242`.
- Test adequacy: no; same gap as F2.

### F9. The cached profile's compatibility key does not include the PyVBMC version
- Location: `pyvbmc/calibration/_cache.py:217-242` (`machine_identity`), `:269-282` (`fingerprint`), `:890-894` (version recorded in provenance), `:846-853` (`validate_record`)
- Category: state/caching
- Proposed classification: possibly intentional
- Confidence: high
- The fingerprint hashes `CACHE_SCHEMA_VERSION`, `KERNEL_REVISION`, `WORKLOAD_REVISION` and the machine identity (CPU model, architecture, host hash, OS system/release, Python major.minor, NumPy and SciPy versions, threadpool backend versions and thread counts, and eleven thread-related environment variables). `pyvbmc_version` is stored in the record's provenance but is neither hashed nor compared, and `validate_record` does not look at it. A cached profile therefore survives any PyVBMC upgrade whose `KERNEL_REVISION` and `WORKLOAD_REVISION` strings are unchanged. That is evidently the intended design — the two revision constants exist precisely to be bumped — but it makes correct reuse depend on a manual step rather than on the installed version, and the comment at `_cache.py:161-162` is the only place that says so.
- Consequence if real: performance only. A change to the chunking in `_pdf` or `_entmc_vbmc` that shifts which budget is fastest, shipped without bumping `KERNEL_REVISION`, leaves every previously calibrated machine on a stale setting.
- Suggested reproduction: build a record with `make_record`, change the reported package version, and observe that `validate_record` and `read_record` still accept it.
- Test adequacy: no. `test_cache.py::test_changed_identity_invalidates_record` perturbs the machine identity; nothing perturbs the package version.

### F10. Dead constant and dead helper in the campaign
- Location: `pyvbmc/calibration/_campaign.py:28` (`_MAX_WORKLOAD_SLOWDOWN`) and `pyvbmc/calibration/_campaign.py:353-361` (`_deduplicate_order`)
- Category: control flow
- Proposed classification: suspected defect
- Confidence: high
- `_MAX_WORKLOAD_SLOWDOWN = 1.05` is never referenced; the per-workload regression veto at `_campaign.py:887-889` compares against `_CONTROL_LOW = 1 / 1.05`, the constant named for the same-budget control band. The two are numerically the same threshold for a speed-up ratio, so the behaviour is right, but the named intent constant has no effect and the veto and the control band are now tied to one literal. `_deduplicate_order` is defined but never called — `_measure_discovery_group` iterates the orders from `_aligned_timing_orders` directly (`_campaign.py:1051-1066`), which already yields representatives.
- Consequence if real: none at runtime; it makes the two thresholds impossible to vary independently and leaves a reader unsure which rule is in force.
- Suggested reproduction: `grep -n "_MAX_WORKLOAD_SLOWDOWN\|_deduplicate_order" pyvbmc/calibration/_campaign.py` — one definition, no use, for each.
- Test adequacy: no test references either name.

## 3. Test adequacy notes

Tests under `pyvbmc/testing/calibration/` that in my judgement pin the implementation rather than a specification:

- `test_campaign.py` runs every scheduling and selection test through `_fake_kernels` and `_small_recipe()`. The real recipe (`campaign._workloads()`) is touched only by `test_workspace_estimate_bounds_measured_incremental_peaks`, and only for its memory bound, for 6 of the 50 `(workload, budget)` pairs. Nothing checks that a workload's group, sample count and gradient flags correspond to a call the package makes (F2, F8), nor that `_layout_signature` still matches the chunking in `_pdf` and `_entmc_vbmc`.
- `test_cache.py` builds its synthetic report from `_cache._GROUP_WORKLOADS`, `_cache._WORKLOAD_SHAPES` and `_cache.CANDIDATE_BUDGETS` (lines 63–179), i.e. from the same tables the validator reads. It therefore cannot detect divergence between those tables and `_campaign._workloads()` / `_campaign.CANDIDATE_BUDGETS` / `_campaign.DISCOVERY_ROUNDS` / `_campaign.HELDOUT_ROUNDS` — the drift that would turn every real campaign into the uncaught `ValueError` of F4. Likewise `profile.DEFAULT_CHUNK_ELEMENTS` and `_campaign.DEFAULT_BUDGET` are two independent literals with no test asserting they agree.
- `test_campaign.py::test_watchdog_stops_only_after_call_and_returns_no_partial_winner` asserts the all-or-nothing discard of F6 as the expected outcome, so the design choice is frozen by a test rather than stated anywhere a reader of the module would see it.
- `test_api.py::test_verbose_feedback_reports_progress_values_and_location` and `::test_verbose_default_completion_uses_plain_summary` assert the exact strings `_print_report_summary` emits, which is why the docstring drift in F7 is invisible to the suite.
- `test_lifecycle.py::test_public_kernels_select_profile_budgets` monkeypatches `_pdf` and `_entmc_vbmc` away and checks only which budget was forwarded. Combined with the oracles building posteriors at the historical defaults (`pyvbmc/testing/oracles/_state.py:302-308`) and `test_vbmc_seed.py` pinning a fixed profile, no test anywhere compares the numbers the package produces with a calibrated profile against the numbers it produces without one (F1).
