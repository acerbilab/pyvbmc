# Stage 2 items 6 and 7: memory (sieve candidates; the GPs retained in the iteration history)

Created: 2026-09-05 19:30. Status: **item 6 DONE 2026-09-05 (evening)**;
**item 7 DONE 2026-09-05 (night)**, decided with the PI the same evening
(§Decisions) and done in four steps (§Design, §Verification (item 7),
§Results).
Roadmap pickup point 3c (ii) (`plans/modernization-roadmap.md`); rationale
in `dev/2026-09-02-modernization-discussion.md` §4 (memory note), §9 ("GPs
retained for every iteration"), §10 (Stage 2 items 6 and 7). Method as in
items 1, 2, 5 and 8 (`plans/stage2-entmc.md`,
`plans/stage2-gpyreg-predict-and-sampler.md`): identity-preserving changes,
gated by the fixed-state oracles compared bit for bit with a dump of the
pre-change outputs, the golden replay reporting `identical`, and the full
suite. No profile campaign: neither item is a speed item (§Findings).

## Summary

**Item 6.** `_vb_init` (`variational_optimization.py`) built every sieve
candidate with `copy.deepcopy(vp)`, which copied the parameter transformer
and the variational parameters it then overwrote. It now builds a shell with
`_candidate_vp`: the generator and the transformer are shared with the base
posterior (as `VariationalPosterior.__deepcopy__` already shares the
generator), the attributes `_vb_init` assigns are left for it, everything
else (dimensions, the `optimize_*` flags, the cached mode) is deep-copied.
Candidates and the generator state are bit-identical to the deepcopy
version on 27,936 checked candidates; the replay is `identical` with finals
equal to item 5's in every metric. The whole `_vb_init` step goes from
31–43 to 12–20 µs per candidate, about 0.1 % of a run. Since `vbmc.py`
re-points the run's transformer at the winning posterior's after every
variational fit, the run now keeps one transformer object between warps
instead of replacing it with a copy every iteration.

**Item 7.** `iteration_history` deep-copies each iteration's GP with all
`Ns` Cholesky factors, Σ_i Ns_i N_i² doubles over a run: 323 MB on the 15-D
exhaust run, roughly 0.8 GB at D = 20 with the default budget. And
`IterationHistory._expand_array` grows a key's array through
`__setitem__`, which deep-copies the array *and every stored object in it*,
so each record re-copies the whole past: quadratic in the iteration count
(11.6 s of the exhaust run under cProfile) and a transient doubling of the
retained memory. Done, in this order: (0) the resume test compares the two
runs' ELBOs (it compared one with itself); (1) the arrays grow without the
re-copy; (2) each iteration's GP is recorded without its factors (training
data, hyperparameter samples and model) and the public
`VBMC.get_gp(iteration)` restores them, bit for bit, where a full GP is
needed (`final_boost`, `load`); (3) the recorded `optim_state` leaves out
the importance-sampling arrays, the largest key on noisy runs (56–70 % of
the retained bytes; a resumed run recomputes them before any read), unless
the new option `record_full_history_details` asks to keep them for
debugging. The rule behind (2) and (3), decided with the PI: what can be
rebuilt from the record is never stored, flag or no flag; what cannot be
rebuilt is dropped by default and kept under the flag. Each step is
identity-preserving (replay `identical`, full suite green after each); the
history of the noisy logreg run went from 117 to 4.6 MB (its RSS after the
run 332 → 163 MB, its peak 427 → 273 MB), `cigar_D4`'s from 9.4 to 1.9 MB
(§Results).

## Scope

- **In (item 6, done)**: `pyvbmc/vbmc/variational_optimization.py`
  (`_candidate_vp`, the one line of `_vb_init` that used it); a contract
  test in `pyvbmc/testing/vbmc/test_variational_optimization.py`; records.
- **In (item 7, planned)**: `pyvbmc/vbmc/iteration_history.py`
  (`_expand_array`), the recording seam in `vbmc.py` (`iteration_values`),
  the readers of stored GPs (`final_boost`'s call site, `load`, the
  `train_gp` warm start), tests for the history and for resume.
- **Out**: the two `vp0_fine` deep copies and the `vp_pruned` copies in
  `optimize_vp` (4 plus a few per call), `vp_old` per iteration, the
  `function_logger` copy per iteration (small), `GP.clean()` on live GPs;
  the latent `optimize_sigma=False` bug of `_vb_init` (unreachable, devlog
  §9); the 20-seed population (roadmap pickup 3c (i)).

## Findings the plan rests on

Measured 2026-09-05 on `2dcb51a` with gpyreg v1.1.0, one BLAS thread, from
item 5's profile campaign (`runs/profile_20260905_item5/*_cprof/profile.prof`,
read with `pstats`), the stored oracle snapshots, and scratch scripts
(`deepcopy_callers.py`, `bitcheck_vb_init.py`, `mem_history.py`; session
scratchpad, not committed; Follow-ups says how to rebuild them).

### Item 6: the per-candidate deep copy

- **How many candidates.** `_sieve` asks `_vb_init` for `init_N`
  candidates: `ceil(50 K)` at a full re-optimization (`recompute_var_post`
  or `always_refit_vp`; `elbo_starts = 2` slow starts, the candidates split
  over the three types), `ceil(0.1 · 50 K) = 5 K` on every other iteration
  (type 1 only); `final_boost` asks for 250 type-1 candidates at K = 50;
  the noisy path's per-sample update for `5 K` per sample. Item 5's
  cProfile counted 1055–2855 candidates per D = 4 run (26–42 `_vb_init`
  calls) and 21,660 on the exhaust run (177 calls); the devlog's "5K–50K
  candidate VPs in the sieve" (§2) is the per-call bound at full
  re-optimizations, not what a run does.
- **What the copy cost.** cProfile attributed 142–146 µs per candidate to
  `copy.deepcopy` inside `_vb_init` (0.15–0.41 s per D = 4 run, 3.17 s on
  the exhaust run), `_vb_init` in total 1.0–1.5 % of a profiled D = 4 run
  and 0.5 % of the exhaust run. Without the profiler the picture shrinks:
  on the eight snapshots (`5 K` type-1 candidates, median of 7 runs) the
  whole `_vb_init` step took 31–43 µs per candidate with the deepcopy and
  12–20 µs without it, so the copy was about 20–25 µs and cProfile
  overstated it about 6× (a deep copy is hundreds of tiny Python calls, the
  profiler's worst case). The in-situ saving is about 0.1 % of a run; the
  item is a tidiness and memory item, as the roadmap said.
- **What the copy copied.** The posterior's `__dict__`: `D`, `K`, `_rng`
  (shared by `__deepcopy__`), `w`, `eta`, `mu`, `sigma`, `lambd`
  (overwritten by `_vb_init` right after), the four `optimize_*` flags,
  `parameter_transformer` (arrays of size D: `lb_orig`, `ub_orig`, `type`,
  `mu`, `delta`; after a warp `R_mat (D, D)` and `scale`; the
  `_bounded_transforms` dict of closures), `bounds` (four arrays of size D;
  set to `None`), `stats` (a dict holding `I_sk (Ns, K)` and
  `J_sjk (Ns, K, K)`; set to `None`) and `_mode`.
- **The transformer is never mutated after construction**, so sharing it
  is safe: `ParameterTransformer.__init__` is the only writer of its
  attributes (`__call__`, `inverse`, `log_abs_det_jacobian` read);
  `whitening.warp_input` deep-copies it before changing `R_mat`, `scale`,
  `mu`, `delta` (`whitening.py:119, 166–171`) and installs fresh copies on
  the warped posterior and the logger (`:353`, `:204`); `vbmc.py:1068`
  assigns, `self.parameter_transformer = self.vp.parameter_transformer`,
  after every variational fit. With per-candidate copies that assignment
  replaced the run's transformer object every iteration by the winning
  candidate's copy; with sharing, the object returned by `optimize_vp` is
  the one it received, and `iteration_history["vp"]` copies (through
  `__deepcopy__`) still carry their own. `test_vbmc_seed.py::
  test_vp_deepcopy_shares_generator` asserts that `copy.deepcopy` of a
  posterior copies the transformer; that contract is untouched.
- **Nothing reads a candidate's `stats`, `bounds` or `_mode` before
  `_neg_elcbo` runs `set_parameters` on it** (which resets `_mode`); the
  shell keeps `_mode` as a deep copy anyway so the candidate's `__dict__`
  is the deepcopy's, key for key and in the same order.
- **A latent bug found by the bit-check**, not fixed (unreachable):
  `_vb_init(vb_type=3)` with `optimize_sigma=False` and `K_new > vp.K`
  keeps `sigma = np.zeros((1, K))` at the old `K` while `mu` grows to
  `K_new`, so the jitter `mu += sigma * lambd * randn(mu.shape)` raises
  `ValueError` (and would leave `sigma` at zero otherwise). `optimize_sigma`
  is always `True` in production (devlog §9). Both versions raise the same
  error on the 120 such calls of the check. Recorded in devlog §9.
- **No oracle reaches `_vb_init`** (`active_sample_step` runs only where the
  per-sample full update is off, and the `neg_elcbo` oracle starts from a
  stored `theta`), so the exact check against a dump is vacuous for this
  item and was run as the stage's standing gate only; the evidence is the
  bit-check of the candidates and the replay.

### Item 7: what the iteration history retains, and the re-copy

- **The recording seam** (`vbmc.py:1300–1322`, after the variational fit):
  `iter, vp, elbo, elbo_sd, var_ss, sKL, sKL_true, gp, gp_hyp_full, Ns_gp,
  pruned, timer, func_count, lcb_max, n_eff, function_logger`; at the end of
  the iteration (`:1521`) `optim_state` and `random_state`; `warmup` and
  `logging_action` separately. `IterationHistory.record` deep-copies the
  value.
- **`_expand_array` re-copies the whole past.** `record` at a new iteration
  calls `_expand_array`, which does `self[key] = np.append(self[key],
  [None])`; `__setitem__` deep-copies what it is given, and `ndarray.
  __deepcopy__` on an object array deep-copies every element, so at
  iteration `i` every one of the `i` stored GPs, posteriors, loggers and
  `optim_state`s of that key is copied again (and the old array garbage
  collected). Under cProfile: `iteration_history.__setitem__` 11.63 s of
  the 909 s exhaust run (1.3 %; `record` 12.48 s in total, 3450 calls),
  0.15–0.46 s per D = 4 run (0.5–0.8 %); `ndarray.__deepcopy__` 261k calls
  on the exhaust run. The `memcpy` of the factors is real time (cProfile
  inflates the Python-call part, as item 6 showed); the peak memory during
  the copy is twice the retained bytes of the largest key. The public
  `__setitem__` semantics are pinned by `test_iteration_history.py`
  (`test_iteration_history_set_item_deepcopy`, `..._record_iteration`);
  the fix is inside `_expand_array` (`dict.__setitem__` of an array that
  shares its element references), which keeps them.
- **Retained GP bytes, analytic** (from the campaign's per-iteration `N`
  and `Ns_gp`, `Σ_i max(Ns_i, 1) N_i²` doubles; `Ns_gp = 0` means one
  posterior): `cigar_D15_exhaust` 323 MB (150 iterations, N → 750, one
  posterior from N ≥ 350), `logreg_D5_noise3` 51 MB (52 iterations),
  `lumpy_D10` 23 MB, `cigar_D4` 10 MB, `rosenbrock_D2_noise1` 7 MB. At
  D = 20 with the default budget (1100 evaluations, sampling until
  N ≥ 400) the same sum is roughly 0.8 GB. Measured values for three
  configs are under "Measured" below.
- **Readers of the stored GPs.** (i) `train_gp`'s warm start
  (`gaussian_process_train.py:131–146`): `get_hyperparameters(as_array=
  True)` of the newer half of the stored GPs; `iteration_history[
  "gp_hyp_full"]` holds the same arrays, recorded from the same GP at the
  same time. (ii) `final_boost` (`vbmc.py:1535`): the best iteration's GP
  is handed to `optimize_vp`, whose `_gp_log_joint` reads the posteriors
  (`alpha`, `L`, `sW`, `L_chol`). (iii) `VBMC.load(iteration=)`
  (`vbmc.py:2296–2302`): `vbmc.gp = iteration_history["gp"][iteration]`,
  and `optimize()` then continues with active sampling on `self.gp` and
  the warp block copies it, so a full GP is needed there (`load` then
  truncates every history key through `__setitem__`, which deep-copies
  the entries, so the instance's GP is a copy of the record even today;
  the restore must therefore act on a copy and never refill the record,
  or the truncation would copy the factors back in). (iv) Plotting:
  `vbmc.py:1487` and `create_vbmc_animation.py:58–59` read `gp.X` and
  `previous_gp.X`; `vp.plot(gp=gp)` reads only `gp.X`
  (`variational_posterior.py:1284–1293`). (v) `active_sample.py:566`
  passes the history to `train_gp` in the noisy per-sample update, reader
  (i) again. Tests: `test_vbmc_finalboost.py` sets `iteration_history[
  "gp"] = np.arange(30)` and calls `final_boost(vp, dict())` with
  `optimize_vp` mocked, so a posterior restore must live at the call site
  in `optimize()`, not inside `final_boost`; `test_vbmc_save_and_load.py`
  loads the static pickle at `iteration=0` (full GPs in the file: a
  restore must be a no-op when the factors are present).
- **Where a recorded GP's posteriors come from.** `train_gp` builds a new
  `gpr.GP` every iteration and `fit` computes the posteriors at the sampled
  hyperparameters through gpyreg's core computation from `(X, y, s2, hyp)`;
  the rank-1 updates of active sampling happen on the previous GP, before
  `train_gp` replaces it, and warm-up trimming's `reupdate_gp` runs after
  the record on the live object. So a stored GP's factors should be exactly
  what `gp.clean()` followed by `gp.update(hyp=H, compute_posterior=True)`
  recomputes (the oracle fixtures rebuild GPs this way and reproduce every
  oracle output bit for bit, `plans/fixture-generator-and-oracles.md`
  §Findings). The measurement script checks this on every stored GP of its
  runs ("Measured").
- **`optim_state["active_importance_sampling"]`** on the noisy path holds
  `X (Na, D)`, `f_s2`, `ln_weights`, `K_Xa_X (Ns, Na, N)` and `C_tmp (Ns, N,
  Na)` with `Na` up to a few hundred; it is recomputed inside
  `active_sample` before the acquisition that needs it (`active_sample.py:
  330–333`), so the copies in the recorded `optim_state` are never read.
  Size measured below.
- **Everything else recorded is small**: `vp` (D K doubles plus `stats`),
  `function_logger` (the preallocated rows, `N_alloc (D + 4)` doubles),
  `optim_state` without the importance-sampling arrays (`vp_repo`,
  `hyp_dict`, scalars), `timer`. One latent exception, inert at the
  defaults: `optim_state["search_cache"]` is the whole `ns_search = 2^13`
  by `D` sieve when `search_cache_frac > 0` (`active_sample.py:343–345`),
  recorded every iteration and recomputed by a resumed run; the second
  candidate for `record_full_history_details` if that default changes.
- **Unpickling re-copies the history.** `IterationHistory.__setstate__`
  restores its entries through `update` → `__setitem__`, which
  deep-copies each one, so `dill.load` of a saved instance briefly holds
  two copies of the history (a second Σ Ns N² transient until the GP
  records are lean). Left as it is; a follow-up if it ever matters.
- **`optimize()` on a finished instance aliased the history.** The
  continuation (`vbmc.py:893–894`) set `self.vp` and `self.optim_state`
  to the last recorded entries themselves. Before step 1 the re-copy of
  `_expand_array` detached them at the next record, freezing the entries
  at a later, wrong state; after step 1 they kept following the live
  objects to the end of the run. Neither is the recorded iteration's
  state, and no computed output read those entries (the only reader of a
  past `optim_state` entry is commented out in `active_sample.py`), but
  `golden_trace.py` walks them and a later `save`/`load(iteration=)`
  would return them. Found by the review of this plan; fixed with step 2
  (the continuation works on deep copies) and pinned by the resume test,
  which asserts that every recorded `optim_state` carries its own
  iteration number.
- **The in-loop plot** (`vbmc.py:1511–1516`) calls `vp.plot` without
  `gp=`, so the `highlight_data` it computes from `previous_gp.X` a few
  lines above is never used. Not changed here (devlog §9).

### Measured (2026-09-05, 21:31–21:38; `mem_history.py`, seed 0, one BLAS thread, display off, code `2dcb51a` plus item 6)

Retained ndarray bytes in `iteration_history` at the end of the run, each
buffer counted once; RSS from `psutil`, the peak sampled every 0.2 s during
`optimize()`. `N` is the GP's training-set size at the end (live rows,
after warm-up trimming).

| config | iterations | N | history retained | `gp` (of which Cholesky factors) | `optim_state` (of which importance-sampling arrays) | `function_logger` | RSS before → after `optimize()` | peak RSS during |
|---|---|---|---|---|---|---|---|---|
| `cigar_D4` | 26 | 114 | 9.4 MB | 7.6 (7.3) MB | 0.2 (0) MB | 1.3 MB | 131 → 162 MB | 200 MB |
| `rosenbrock_D2_noise1` | 22 | 108 | 25.8 MB | 6.5 (6.2) MB | 18.3 (18.1) MB | 0.8 MB | 153 → 190 MB | 251 MB |
| `logreg_D5_noise3` | 52 | 247 | 117.3 MB | 47.7 (46.6) MB | 66.0 (65.5) MB | 3.2 MB | 153 → 332 MB | 427 MB |

Readings:

- **The factors are exactly Σ_i Ns_i N_i² doubles** (7.3, 6.2, 46.6 MB
  measured against the same sums computed from the stored GPs), so the
  analytic table above holds; its per-config numbers from the campaign
  used `func_count` for `N` and so count the rows warm-up trimming removed
  (10 MB estimated against 7.3 measured on `cigar_D4`). `vp`,
  `gp_hyp_full` and the scalar keys are negligible; `function_logger` is
  the preallocated rows, 0.8–3.2 MB.
- **On noisy runs the importance-sampling arrays in the recorded
  `optim_state` are the largest key**: 70 % of the retained bytes on
  `rosenbrock_D2_noise1` (18.1 of 25.8 MB) and 56 % on `logreg_D5_noise3`
  (65.5 MB, against 46.6 MB of factors). Step 3 is therefore part of the
  plan, not an option (Open question 3).
- **Peak RSS exceeds the post-run RSS by 38, 61 and 95 MB**, of the order
  of the retained history (9, 26, 117 MB) plus an iteration's working set
  (the sieve's candidates, the entropy blocks, the CMA-ES `predict`
  batches). The re-copy of `_expand_array` doubles the retained bytes
  while it runs; the same measurement after step 1 separates the two
  contributions.
- **Every stored GP rebuilds bit for bit**: 100 of 100 (26 + 22 + 52)
  through `deepcopy` → `clean()` → `update(hyp=H, compute_posterior=True)`,
  equal in `hyp`, `alpha`, `sW`, `L`, `sn2_mult` and `L_chol`, worst
  difference 0.0; the rebuild of a whole run's GPs took 0.02–0.17 s, so a
  restore at `final_boost` or in `load` costs milliseconds. This is the
  fact step 2 rests on.

## Design

### Item 6 (done)

```python
_CANDIDATE_ASSIGNED = ("w", "eta", "mu", "sigma", "lambd", "bounds", "stats")
_CANDIDATE_SHARED = ("_rng", "parameter_transformer")

def _candidate_vp(vp):
    new_vp = vp.__class__.__new__(vp.__class__)
    for key, value in vp.__dict__.items():
        if key in _CANDIDATE_SHARED:    setattr(new_vp, key, value)
        elif key in _CANDIDATE_ASSIGNED: setattr(new_vp, key, None)
        else:                            setattr(new_vp, key, copy.deepcopy(value))
    return new_vp
```

`_vb_init` then sets `K`, `w`, `mu`, `sigma`, `lambd`, `eta`, `bounds`,
`stats` exactly as before. Iterating `vp.__dict__` keeps the deepcopy's
attribute order and copies any attribute the class acquires later.

Gates run (the tracker has the times): the bit-check of old against new
`_vb_init` (the old module from `git show 2dcb51a:...`, states from the
eight oracle snapshots and a synthetic D = 15, K = 30 posterior, every
combination of the four `optimize_*` flags, the three candidate types,
`K_new ∈ {K, K + 1, K + 3}`, two seeds: 27,936 candidates equal attribute
by attribute with `np.array_equal`, the generator state equal after every
call, the base posterior untouched); `pytest pyvbmc/testing/oracles`;
`make_oracle_fixtures.py --check --exact --against` a dump of `2dcb51a`
(vacuous here, see Findings); the replay against the item 5 traces
(`--baseline runs/golden/replay_item5_step1`); the full suite. Test added:
`test_vb_init_candidates` (candidates share `rng` and the transformer, own
their arrays, have `bounds`/`stats` `None` and the requested `K`, the base
posterior is untouched, the first type-1 candidate at the same `K` carries
the base parameters verbatim, writing into a candidate leaves the base
alone).

### Item 7 (done)

Four steps, each with its own commit and gates, in this order.

0. **Make the existing resume test a real one.** `test_vbmc_resume_
   optimization` (`test_vbmc_optimize.py`) runs 8 iterations straight
   against 4 iterations, a `dill` round trip and 4 more, and asserts exact
   equality of `elbo_sd` and of the success flag; its ELBO assertion reads
   `elbo_1 == elbo_1`. Change it to `elbo_1 == elbo_2` and run it before
   any history change: if it passes, resume identity is guarded from then
   on at no cost; if it fails, that is a pre-existing non-identity to look
   at first, not something item 7 caused. The test resumes through
   `dill.load` and `optimize()` continuation, where the live GP is a
   pickled attribute, so it does not exercise the restore of step 2; the
   unit test there does.
1. **Grow the history without re-copying it.** `_expand_array` builds the
   longer object array and stores it with `dict.__setitem__` (no deep
   copy): the elements keep their identity, `record` still deep-copies the
   new value, and the public `__setitem__` keeps deep-copying what a caller
   assigns. Output-identical by construction. Test: after `record(key, v1,
   0)` and `record(key, v2, 1)`, `hist[key][0]` is the same object as
   before the second record. Removes the quadratic time and the transient
   peak.
2. **Lean GP records.** At the recording seam store `_lean_gp(self.gp)`: a
   shallow copy of the GP whose `posteriors` are `Posterior(hyp, None,
   None, None, None, None)` and whose `temporary_data` is `{}` (the
   kernel, mean, noise, priors, bounds, `X`, `y`, `s2` come along; `record`
   deep-copies the lean object, a few hundred KB at most). The restore is
   public: **`VBMC.get_gp(iteration)`** returns the GP of that iteration
   with its posteriors restored, as a copy, so the history stays lean
   (`gp.update(hyp=gp.get_hyperparameters(as_array=True),
   compute_posterior=True)` when `posteriors[0].alpha is None`, a no-op
   otherwise, so files saved with full GPs load unchanged); it is what a
   user or a debugger calls on a stored iteration, and what `optimize()`
   calls before `final_boost` and `load` calls for the GP it puts on the
   instance. The warm start of `train_gp` keeps reading the GP records: a
   lean GP answers `get_hyperparameters` unchanged, and reading
   `gp_hyp_full` instead would change the `gp_fit` oracle's inputs (its
   history stand-in has an empty `gp` array but `gp_hyp_full` filled with
   the current hyperparameters, because `_get_hyp_cov` indexes that key
   for every past iteration; the switch would add starting points to the
   fit and, once there are enough, consume a generator draw, moving the
   oracle for no numerical reason). Plotting and the animation read
   `gp.X`, which a lean GP has. `hyp_dict["hyp"]` and the `optim_state` copies are untouched.
   Memory: the GP key drops from Σ Ns N² to Σ N (D + 2) doubles plus the
   hyperparameters (323 MB → about 2 MB on the exhaust run). Cost: one
   restore of `Ns` Choleskys at the best iteration (milliseconds) and one
   per `load`. Identity: the restore recomputes exactly the factors the
   record dropped (Findings; verified on every stored GP of the
   measurement runs), so `final_boost` sees the same `alpha`, `L`, `sW`.
   Documentation: the method's docstring and a note in the `VBMC` API page
   that `iteration_history["gp"]` holds data and hyperparameters only.
3. **Drop what cannot be rebuilt, unless asked to keep it.** A new option
   in `advanced_vbmc_options.ini`, `record_full_history_details = False`
   ("Keep in the iteration history the intermediate arrays that a resumed
   run recomputes and that cannot be rebuilt from the record, for
   debugging: today the importance samples of the noisy acquisitions
   (`optim_state["active_importance_sampling"]`). What can be rebuilt from
   the record, such as the GP posteriors, is never kept."). By default the
   history's copy of `optim_state` has `active_importance_sampling = None`
   (the live `optim_state` keeps it until `active_sample` recomputes it);
   with the option on, the copy is recorded as it is today. The dict goes
   whole: the samples `X` are draws made mid-iteration, and the history
   holds the generator state only at iteration ends; `K_Xa_X` and `C_tmp`
   are deterministic in `X` and the GP that produced them, but that GP is
   the intermediate per-sample fit of the noisy path, whose hyperparameters
   are not recorded either. Set only for VIQR/IMIQR runs, so noiseless runs
   are untouched. It is the largest key of the history on noisy runs
   ("Measured": 18 MB of 26 on `rosenbrock_D2_noise1`, 66 MB of 117 on
   `logreg_D5_noise3`). The oracle fixtures already store `None` there and
   recompute the samples under their own seed.

Alternatives considered: `GP.clean()` on the recorded copy (same retained
memory as step 2 but still pays the deep copy of the factors on every
record, the transient step 1 removes); storing `(X, y, s2, hyp)` arrays
instead of a GP shell (readers would rebuild kernel, mean, noise and
priors: more code for the same memory); keeping only the best and last GPs
(changes `load(iteration=)` semantics, not needed once the records are
lean).

Gates for each step: `pytest pyvbmc/testing/oracles` and `--check --exact
--against` a fresh dump (vacuous for the history, run as the standing
gate); the replay `identical` on all five configs (step 2 changes what
`final_boost` receives, which the replay's finals exercise); the full
suite, with the resume test of step 0 now comparing the ELBO; a unit test
for step 2 (a GP from an oracle snapshot, made lean and restored through
the same helper: `hyp`, `alpha`, `sW`, `L`, `sn2_mult`, `L_chol` equal bit
for bit; `test_vbmc_load_static` at `iteration=0` asserting the loaded
`vbmc.gp` has its factors); a test for step 3 (the history's `optim_state`
copy has `active_importance_sampling` `None` by default and the recorded
dict with the option on, on the noisy oracle snapshot's state or a short
noisy run already in the suite); `mem_history.py` before and after on
`cigar_D4`, `rosenbrock_D2_noise1` and `logreg_D5_noise3` (retained MB per
key and peak RSS).

## Steps

- [x] Item 6: readings (`_sieve`, `_vb_init`, `__deepcopy__`, the
      transformer's writers, whitening, the recording seam)
- [x] Item 6: cProfile attribution of `copy.deepcopy` to its callers
- [x] Item 6: pre-change dump; `_candidate_vp`; the contract test
- [x] Item 6: bit-check old vs new; oracles; exact check; replay; full suite
- [x] Item 7: findings (readers, seam, re-copy, analytic memory), design
      options, gates; the measurement script
- [x] Item 7: measurement runs (`cigar_D4`, `rosenbrock_D2_noise1`,
      `logreg_D5_noise3`) recorded under "Measured"
- [x] Item 7: PI's decisions on the open questions (§Decisions)
- [x] Item 7 step 0: `elbo_1 == elbo_2` in the resume test; passes
      (`564f53a`)
- [x] Item 7 step 1: `_expand_array` without the re-copy; test (`e357216`)
- [x] Item 7 step 2: lean GP records, `VBMC.get_gp`, tests, docs (the warm
      start stays on the GP records, see the tracker); `1aa1933`
- [x] Item 7 step 3: `record_full_history_details`, the `.ini` entry,
      test; the in-loop plot fix; committed with it
- [x] Item 7: `mem_history.py` after each step (§Results); records

## Verification (item 6)

- [x] Bit-check: 27,936 candidates equal attribute by attribute, generator
      state equal after every call, base posterior untouched; 120 calls
      raise identically in both versions (the latent bug)
- [x] `pytest pyvbmc/testing/vbmc/test_variational_optimization.py`: 13
      passed (the new test included)
- [x] `pytest pyvbmc/testing/oracles`: 116 passed, 15 skipped
- [x] `--check --exact --against runs/oracle_outputs_prechange_2dcb51a`:
      8 of 8 fixtures bit-identical (before and after the `gp_nlZ` switch
      below)
- [x] Replay against the item 5 traces (`runs/golden/replay_item6_step1`,
      2.7 min): `identical` on all five configs, initial design identical,
      and the finals (`elbo_err`, `gskl`, `mmtv`, evaluations, iterations,
      final K) equal to item 5's replay row for row
- [x] Full suite `pytest --reruns=5 -x` after both changes (item 6 and
      the `gp_nlZ` switch): **542 passed, 15 skipped, 0 reruns, 5:02**
      (`runs/pytest_full_item6_step1_1788632726.log`)

## Decisions

- **Share the transformer, do not copy it once per `_vb_init` call.** A
  per-call copy would have kept today's object churn (a new transformer
  object per iteration through `vbmc.py:1068`) for no benefit; the
  transformer is immutable after construction and every code path that
  needs a different one makes its own copy.
- **A shell built from `__dict__`, not a hand-written constructor call.**
  Attribute order and set stay those of `copy.deepcopy`, and an attribute
  added to `VariationalPosterior` later is copied rather than silently
  dropped.
- **`_mode` is copied, not reset**, although `set_parameters` resets it
  before any candidate is used: the shell's job is to equal the deep copy.
- **The `vp0_fine` and `vp_pruned` copies stay** as `copy.deepcopy`: a
  handful per `optimize_vp`, and they must own their transformer as any
  posterior returned to the caller does today.
- **Item 7 gets a plan and measurements before code** (PI, pickup 3c):
  it changes what a resumed run and `final_boost` receive, and three
  readers plus the save format depend on the record.
- **The retention rule** (PI, 2026-09-05, late evening): the history is
  kept for resume and for debugging, so nothing is dropped for being
  unread. What can be rebuilt from the record (the GP posteriors) is never
  stored and is rebuilt on demand, whether or not the option is set; what
  cannot be rebuilt (the importance samples, drawn mid-iteration) is
  dropped by default and kept under `record_full_history_details`.
- **The restore is public** (PI): `VBMC.get_gp(iteration)`, returning a
  restored copy, rather than a private helper, since it is also what one
  calls when inspecting a stored iteration.
- **The option is named for what it adds**, `record_full_history_details`
  (PI): the history is always recorded; the option keeps the details a
  resumed run would otherwise recompute.

## Open questions

All four closed with the PI on 2026-09-05 (late evening); the answers are
in §Design and §Decisions.

1. ~~Lean GP records restored on demand, `GP.clean()` on the recorded deep
   copy, or best and last GPs only?~~ **Lean records restored on demand**
   (`GP.clean()` on a copy keeps the per-record copy of the factors;
   best/last changes what `load(iteration=)` can do).
2. ~~Switch the `train_gp` warm start to `gp_hyp_full`?~~ **No.** Agreed
   as "yes" at first (identical arrays, one reader fewer); reversed while
   implementing step 2, when the `gp_fit` oracle's history stand-in turned
   out to make the switch move that oracle for no numerical reason
   (§Design, step 2). The reader works on lean records unchanged, so the
   gain would have been cosmetic.
3. ~~Drop `active_importance_sampling` from the recorded `optim_state`?~~
   **Yes by default, kept under `record_full_history_details`** (the
   retention rule above); the measurement made it the largest key on noisy
   runs.
4. ~~A resume-identity test in the suite?~~ **Use the one that exists**:
   `test_vbmc_resume_optimization`, with its `elbo_1 == elbo_1` fixed to
   compare the two runs (step 0), plus the unit-level restore test of
   step 2.

## Risks

- Item 7 step 2: a stored GP whose factors were *not* a fresh core
  computation (a code path not found in Findings) would make the restore
  differ at rounding level, and `final_boost` with it; the per-GP rebuild
  check in `mem_history.py` and the replay's finals are the detectors.
- Item 7 step 2: `load()` of a file written by the new code on an old
  PyVBMC would set `vbmc.gp` to a record without factors and fail at the
  first acquisition of a continued run (`gp.predict` with `alpha =
  None`); the save format changes in that direction only (old files load
  on new code). Release note for the eventual PR:
  `iteration_history["gp"][i]` holds data and hyperparameters, and
  `vbmc.get_gp(i)` is the usable GP of iteration `i`.
- Item 7 step 1 on its own changed what a continued run leaves in the
  history (Findings, "`optimize()` on a finished instance aliased the
  history"); step 2's continuation copies close that. The history's
  public API is unchanged.

## Verification (item 7)

Each step: `pytest pyvbmc/testing/oracles` (116 passed, 15 skipped every
time; no oracle reaches the history, the standing gate), the replay
against the item 6 traces (`identical` on all five configs, initial design
identical, every time: `runs/golden/replay_item7_step{1,2,3}`), the full
suite (`pytest --reruns=5 -x`: 543, 556 and 560 passed with 0 reruns as
the tests were added; 15 skipped), `mem_history.py` on the three configs
(§Results), pre-commit clean. Step 0's resume test passes at every step
with the ELBO compared; after step 2 it also asserts the lean records,
`get_gp`, a `VBMC.load(iteration=2)` round trip on a file written by the
new code and that every recorded `optim_state` carries its own iteration
number. Step 2's unit tests (`test_gp_records.py`, 13) check the lean copy
and the bit-exact restore on three oracle snapshots and `get_gp` on the
static pickle; step 3's (`test_optim_state_record.py`, 4) the recorded
`optim_state` with and without the option. A read-only Opus review of
step 1 and the step 2/3 design ran before the gates (tracker).

## Results

Item 6 has no speed result worth a table: `_vb_init` is about 0.1 % of a
run after the change (Findings). Memory per candidate: the transformer copy
(a few KB) and the copied `stats` (`Ns K²` doubles at large K: 20 KB at
K = 50, Ns = 1) are gone; candidates hold only their own parameters.

Item 7, `mem_history.py` after each step (seed 0, one BLAS thread; the
same runs as "Measured" in §Findings, identical trajectories throughout):

| config | step | history retained | `gp` | `optim_state` | RSS after `optimize()` | peak RSS | wall |
|---|---|---|---|---|---|---|---|
| `cigar_D4` | before | 9.4 MB | 7.6 MB | 0.2 MB | 162 MB | 200 MB | 46 s |
| | 1 | 9.4 MB | 7.6 MB | 0.2 MB | 163 MB | 200 MB | 44 s |
| | 2 | 1.9 MB | 0.1 MB | 0.2 MB | 153 MB | 191 MB | 43 s |
| | 3 | 1.9 MB | 0.1 MB | 0.2 MB | 153 MB | 191 MB | 42 s |
| `rosenbrock_D2_noise1` | before | 25.8 MB | 6.5 MB | 18.3 MB | 190 MB | 251 MB | 74 s |
| | 1 | 25.8 MB | 6.5 MB | 18.3 MB | 188 MB | 244 MB | 72 s |
| | 2 | 19.4 MB | 0.1 MB | 18.3 MB | 174 MB | 244 MB | 70 s |
| | 3 | 1.3 MB | 0.1 MB | 0.1 MB | 158 MB | 225 MB | 71 s |
| `logreg_D5_noise3` | before | 117.3 MB | 47.7 MB | 66.0 MB | 332 MB | 427 MB | 254 s |
| | 1 | 117.3 MB | 47.7 MB | 66.0 MB | 279 MB | 383 MB | 208 s |
| | 2 | 70.1 MB | 0.5 MB | 66.0 MB | 229 MB | 338 MB | 197 s |
| | 3 | 4.6 MB | 0.5 MB | 0.5 MB | 163 MB | 273 MB | 198 s |

Reading: step 1 changes nothing retained and removes the quadratic
re-copy, visible only where the history is large (`logreg_D5_noise3`: RSS
after the run 332 → 279 MB, peak 427 → 383 MB, wall 254 → 208 s; within
noise on the two small configs). Step 2 takes the GP key from Σ Ns N²
doubles to the data and hyperparameters (47.7 → 0.5 MB on logreg, 7.6 →
0.1 MB on cigar). Step 3 removes what remained on the noisy runs, the
importance-sampling arrays: the history of `logreg_D5_noise3` ends at
4.6 MB (3.2 of them the logger's preallocated rows), its RSS after the run
at 163 MB against 332 before the item and its peak at 273 MB against 427;
`rosenbrock_D2_noise1` 25.8 → 1.3 MB. The peak RSS still exceeds the
post-run RSS by 40–110 MB: the working set of an iteration (the sieve, the
entropy blocks, the CMA-ES `predict` batches), not the history. The
exhaust run (323 MB of factors by the analytic sum) was not
re-measured; the overnight population run records a `peakRSS` per run,
so its report against the item 1 population shows the in-situ effect on
280 runs. `mem_history.py`'s rebuild check compares each stored GP with
its own rebuild and so reports "differ" on lean records (no factors to
compare); the bit-exactness of the restore is pinned by
`test_gp_records.py` instead.

## Follow-ups

- **Reproducing the scratch checks.** `deepcopy_callers.py` (pstats over
  the `*_cprof/profile.prof` files of a campaign: callers of
  `copy.deepcopy` outside `copy.py`, cumulative time and calls, plus the
  `_vb_init` / `record` / `_expand_array` / `__setitem__` rows),
  `bitcheck_vb_init.py` (the old module from `git show 2dcb51a:pyvbmc/vbmc/
  variational_optimization.py` with its relative imports made absolute;
  states from `pyvbmc.testing.oracles._state.build_state(load_snapshot(
  FIXTURES / name))`, `X_star, y_star` from `get_hpd(gp.X, gp.y,
  options["hpd_frac"])`; both versions run on deep copies of the same base
  posterior with `default_rng(seed)`; candidates compared attribute by
  attribute with `np.array_equal`, dicts and lists recursively; timing
  `perf_counter` medians of 7 at `5 K` type-1 candidates), `mem_history.py`
  (a benchmark config through `benchmark_targets.find_config(label).make(
  seed=0)`, `psutil` RSS sampled every 0.2 s in a thread during
  `optimize()`, then a walk of every object in `iteration_history` summing
  ndarray bytes with each buffer counted once, the GP key split into
  Cholesky factors, other posterior arrays and data, and for every stored
  GP a `deepcopy` → `clean()` → `update(hyp=..., compute_posterior=True)`
  compared to the original with `np.array_equal`).
- The `optimize_sigma=False` path of `_vb_init` (devlog §9) if that flag is
  ever exposed.

## Execution tracker

Legend: `[ ]` not started, `[~]` in progress, `[x]` done, `[!]` needs
attention. Times are wall clock on 2026-09-05.

- [x] Reading list of pickup point 3c; the transformer's writers, the
  recording seam, the readers of stored GPs, `IterationHistory` — until
  about 21:00
- [x] `copy.deepcopy` attributed to its callers in item 5's five cProfile
  listings (`deepcopy_callers.py`): `_vb_init` 142–146 µs per candidate
  under the profiler, `iteration_history.__setitem__` 11.6 s on the
  exhaust run — 21:05
- [x] Pre-change oracle dump `runs/oracle_outputs_prechange_2dcb51a/`
  (53 arrays per snapshot) — 21:12
- [x] `_candidate_vp` in `variational_optimization.py`; `test_vb_init_
  candidates`; the module's 13 tests pass — 21:14
- [x] Bit-check old vs new (`bitcheck_vb_init.py`): 27,936 candidates
  equal, generator states equal, 120 calls raising identically (the
  latent bug); timing 31–43 → 12–20 µs per candidate — 21:18
- [x] `pytest pyvbmc/testing/oracles` 116 passed, 15 skipped; `--check
  --exact --against` the dump 8 of 8 (`runs/oracle_exact_item6_step1.log`);
  pre-commit clean — 21:20–21:21
- [x] Replay against the item 5 traces (`runs/golden/replay_item6_step1`,
  2.7 min): `identical` on all five, finals equal to item 5's — 21:23
- [x] `gp_nlZ` oracle through `gp.log_likelihood` / `gp.log_posterior`
  (`_lz_and_grad`); `pytest -k gp_nlZ` 8 passed; `--check --exact
  --against` the dump 8 of 8 (`runs/oracle_exact_item6_step2.log`) — 21:25
- [x] Full suite 542 passed, 15 skipped, 0 reruns, 5:02 — 21:31
- [x] Plan written (this file), records (roadmap, `dev/README.md`, devlog
  §9 and §10, the oracle table, the item 8 plan's follow-up, `AGENTS.md`)
  — 21:30
- [x] Memory measurement runs (`mem_history.py` on `cigar_D4`,
  `rosenbrock_D2_noise1`, `logreg_D5_noise3`; one process,
  `runs/mem_history_item7_1788632726.log`) — 21:31–21:38: 100 of 100
  stored GPs rebuilt bit-identically; the importance-sampling arrays the
  largest key on the noisy runs; "Measured" filled in, Open question 3
  answered — 21:42
- [x] Item 6 and the `gp_nlZ` switch committed (`d144a83`, `affca4d`),
  the records (`138215f`) — 21:46
- [x] The four open questions discussed with the PI and closed: lean GP
  records, the warm start through `gp_hyp_full`, the retention rule with
  `record_full_history_details`, the existing resume test fixed and used;
  `VBMC.get_gp` public — 22:00–22:15; §Design, §Decisions, §Steps updated;
  committed `181e489`
- [x] **Step 0** — 22:20: `elbo_1 == elbo_2` in
  `test_vbmc_resume_optimization`; the test passes (a run resumed through a
  `dill` round trip reproduces the straight run's ELBO and ELBO sd
  exactly); `564f53a`
- [x] **Step 1** — 22:25–22:50: `_expand_array` stores the grown array
  with `dict.__setitem__`; `test_iteration_history_record_keeps_earlier_
  entries`; history tests 13 passed; oracles 116 passed, 15 skipped;
  replay against the item 6 traces (`runs/golden/replay_item7_step1`):
  `identical` on all five, initial design identical; full suite **543
  passed, 15 skipped, 0 reruns, 4:42**
  (`runs/pytest_full_item7_step1_1788638671.log`); `mem_history.py`
  after (`runs/mem_history_item7_step1_1788638671.log`): retained bytes
  unchanged (9.4 / 25.8 / 117.3 MB, as expected: the fix removes a
  transient, not what is retained); on `logreg_D5_noise3` the RSS after
  `optimize()` 332 → 279 MB, the peak 427 → 383 MB and the wall 254 →
  208 s; on the two small configs (history 9–26 MB) peak and RSS within
  noise (200 → 200, 251 → 244 MB), so their peak excess over the final
  RSS (38–56 MB) is the iteration's working set, not the re-copy;
  100 of 100 stored GPs again rebuilt bit-identically; committed
  `e357216`
- [~] **Step 2** — implemented by an Opus agent in an isolated worktree
  from a written spec (helpers `_lean_gp` / `_restore_gp_posteriors` in
  `gaussian_process_train.py`, the seam, `VBMC.get_gp`, the `final_boost`
  call site, `load`, `test_gp_records.py` with 13 tests on the oracle
  snapshots and the static pickle, `test_vbmc_load_static` assertions,
  `AGENTS.md`); the agent verified against gpyreg's `update` source that
  the shallow copy cannot mutate the original and ran only the new
  module and `-k load_static` (13 + 1 passed); patch applied to the main
  tree 23:00 with two wording fixes. The warm start of `train_gp` stays on
  the (lean) GP records rather than `gp_hyp_full`: the `gp_fit` oracle's
  history stand-in provides an empty `gp` array but `gp_hyp_full` filled
  with the current hyperparameters, so the switch would feed the warm
  start extra starting points and draw from the generator
  (`rng.choice`), moving the oracle for no numerical reason; identical
  arrays either way.
- [x] **Read-only Opus review of step 1 and the step 2/3 design** (ran in
  parallel with the implementation, static analysis only) — 22:35–22:50:
  no blocker; every reader in the plan's list confirmed and the rebuild's
  exactness traced through gpyreg (`fit` ends in `update(hyp=...)` with no
  new data on every exit, so no recorded GP carries rank-1 factors).
  Should-fixes, applied: the continuation of `optimize()` aliased the
  live VP and `optim_state` to the history's last entries, which step 1
  stopped detaching → deep copies, and the resume test asserts every
  recorded `optim_state` carries its own iteration number; `load` does
  not alias (its truncation deep-copies every entry), so the plan's
  aliasing claim was wrong → corrected, with the rule that the restore
  acts on a copy; the warm start must stay on the GP records (the
  `gp_hyp_full` switch would move the `gp_fit` oracle) → plan and
  records corrected, Open question 2 reversed; step 3 must read the
  option with `.get` and leave the key set of a noiseless `optim_state`
  alone → the draft already did; the resume test gains a
  `VBMC.load(iteration=2)` round trip on a file written by the new code
  (the static pickle only exercises the no-op branch) → added. Notes,
  applied: the history test covers a growth by several slots and the
  object dtype; the `clean()`-on-a-shallow-copy trap is in `_lean_gp`'s
  comment; `search_cache`, the unpickle re-copy and the in-loop plot's
  unused `highlight_data` are in §Findings. Light tests after the edits:
  32 passed (the resume test included); step 2 gate chain restarted on
  the final code — 23:05
- [x] **Step 2 gates** — 23:05–23:30: oracles 116 passed, 15 skipped;
  replay against the item 6 traces (`runs/golden/replay_item7_step2`)
  `identical` on all five, initial design identical, 0 flagged; full suite
  **556 passed, 15 skipped, 0 reruns, 4:43**
  (`runs/pytest_full_item7_step2_1788639831.log`); `mem_history.py`
  (`runs/mem_history_item7_step2_1788639831.log`): §Results, the GP key
  47.7 → 0.5 MB on logreg; committed `1aa1933` — 23:31
- [x] **Step 3** — 23:35–23:51: `_optim_state_record` and the seam in
  `vbmc.py`, `record_full_history_details = False` in
  `advanced_vbmc_options.ini` (the options page includes the file, no
  other docs change), `test_optim_state_record.py` (4 tests),
  `AGENTS.md`; at the PI's request the in-loop plot passes `gp=self.gp`
  (it drew no training points; smoke-tested with `plot=True` under the
  Agg backend, 3 iterations); light tests 37 passed; oracles 116 passed;
  replay `identical` on all five (`runs/golden/replay_item7_step3`); full
  suite **560 passed, 15 skipped, 0 reruns, 4:36**
  (`runs/pytest_full_item7_step3_*.log`); `mem_history.py`: §Results, the
  logreg history 70.1 → 4.6 MB; committed `ffbd4b2` — 23:52
- [x] Records (this file, roadmap, devlog §9/§10, `dev/README.md`) —
  23:55
- [ ] `/doublecheck` of the item 7 commits and records (read-only
  reviewers); then the 20-seed population overnight on the PI's word
