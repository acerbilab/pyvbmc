# Wave 2: history of three warm-up discrepancies

Raw report of a read-only history search, wave 2 of the port correctness
review (`dev/plans/port-correctness-review.md`), 2026-09-20. One Opus
agent, asked by the PI whether three warm-up discrepancies of
`reviews/P1a_comparison.md` (F1, F2, F3) were ported that way from the
start or changed later, and whether a rationale is written down anywhere.
The orchestrator spot-checked the dating of `35be58b`, `9267816`,
`4aa4ca9`, `0a27ee3` and `39b29e6` against the history and found it as
stated. The text below is the agent's final message, unedited.

---

# History and rationale search — three warm-up discrepancies in PyVBMC

Scope: read-only inspection of `pyvbmc` (branch `dev-port-review` @ `51451dc`) and the sibling MATLAB checkout `../vbmc` (`master` @ `396d649`). No repository state was changed. One arithmetic script was run (window sizes, item 1). `gh` **is** available and authenticated (account `lacerbi`); I searched issues and pull requests read-only.

A note that applies to all three items: `private/vbmc_warmup.m` has not been touched since `d2989b1` (2021-01-12), and `private/recompute_lcbmax.m` has not been touched since `97ce2ec` (2020-04-11). Both therefore looked exactly as they look today on **every** date on which the Python lines were written (2021-01-19, 2021-06-08, 2021-07-07, 2021-08-25, 2022-03-03, 2023-02-15). There is no "older MATLAB" defence available for any of the three items.

Second general note: the initial Python commit `d974e20` (2021-01-19, Marlon Tobaben, "feat: initial commit of first version of code port") was a signature skeleton. `initialCodePort/vbmc.py:83-101` declares `__4vbmc_warmup` and `__4recompute_lcbmax` with bodies of `pass`; `initialCodePort/loop.py:250-257` transliterates the caller faithfully, including `if options.RecomputeLCBmax: optimState.lcbmax_vec = recompute_lcbmax(gp, optimState, stats, options)`. So the bodies at issue were all first written in mid-2021, not in the January transliteration.

---

## Item 1 — the warm-up "recent improvement" window

### 1. Is the discrepancy real as described? Yes, exactly as described.

Current Python, `pyvbmc/vbmc/vbmc.py:1975-1993`:

```python
lcb_max_vec = self.iteration_history.get("lcb_max")[: iteration + 1]
...
if self.options.get("warmup_check_max"):
    idx_last = np.full(lcb_max_vec.shape, False)
    recent_past = iteration - int(
        math.ceil(
            self.options.get("tol_stable_warmup")
            / self.options.get("fun_evals_per_iter")
        )
        + 1
    )
    idx_last[max(1, recent_past) :] = True
    impro_fcn = max(
        0,
        np.amax(lcb_max_vec[idx_last]) - np.amax(lcb_max_vec[~idx_last]),
    )
```

MATLAB, `private/vbmc_warmup.m:60-64`:

```matlab
idx_last = false(size(lcbmax_vec));
RecentPast = iter-ceil(options.TolStableWarmup/options.FunEvalsPerIter)+1;
idx_last(max(2,RecentPast):end) = true;
improFcn = max(0,max(lcbmax_vec(idx_last)) - max(lcbmax_vec(~idx_last)));
```

With `T = ceil(TolStableWarmup/FunEvalsPerIter)`, MATLAB's 1-based start is `iter - T + 1`, whose 0-based equivalent is `iter - T = iteration + 1 - T` (PyVBMC's `iteration` is 0-based; `vbmc.py:1960` and its comment "MATLAB has +1 on the right side due to different indexing" confirm the convention). The Python computes `iteration - T - 1`, two entries earlier. The `+ 1` of MATLAB's expression was moved inside the parentheses, flipping its sign, and the 1-based→0-based shift was never applied.

Independent arithmetic check (defaults `tol_stable_warmup = 15`, `advanced_vbmc_options.ini:141`; `fun_evals_per_iter = 5`, `basic_vbmc_options.ini:13`; so `T = 3`), window sizes as a function of 0-based `iteration`:

```
iter_py  len  py_start  py_n  ml_start  ml_n
      1    2         1     1         1     1
      2    3         1     2         1     2
      3    4         1     3         1     3
      4    5         1     4         2     3
      5    6         1     5         3     3
      6    7         2     5         4     3
      9   10         5     5         7     3
```

The two agree through `iteration = 3` and part from `iteration = 4`; from `iteration = 6` on the Python "recent" set holds 5 entries against MATLAB's 3. Since the Python "recent" set is a superset of MATLAB's and its "before" set a subset, `impro_fcn` is always ≥ MATLAB's, so `no_recent_improvement_flag` is satisfied strictly less often and warm-up tends to run longer.

### 2. Timeline of the Python lines

Every commit that touched `vbmc.py` in the `recent_past` / `idx_last` block (`git log -L 1974,1996`, `git log -G'idx_last'`, `git log -S'recent_past'`):

| commit | date | author | subject |
|---|---|---|---|
| `ec94eba` | 2021-06-08 | Marlon (Tobaben) | feat: add skeleton of vbmc.optimize() — method stub only |
| **`35be58b`** | **2021-07-07** | **Marlon** | **feat: handle end of vbmc warmup (#15)** — writes `recent_past = iteration - int(ceil(...) + 1)` and `idx_last[max(2, recent_past):]` |
| `4b94e1e` | 2021-07-17 | Marlon | feat: load and validate basic and advanced vbmc options (#18) — option plumbing |
| **`9267816`** | **2021-08-25** | **Mikko Aarnos** | **feat: added VP optimization (#26)** — `max(2, ·)` → `max(1, ·)`, `[:iteration]` → `[:iteration+1]`, `np.ceil` → `math.ceil`; `recent_past` untouched |
| `abf5c3e` | 2021-10-31 | Marlon | feat: enhance variational_optimization code… (#29) — black reflow only |
| `d9bfe16`, `3f9c93a` | 2022-08-28/30 | Bobby-Huggins | Snake_case (#91), Snake case (#92) — identifier renames |
| `343b9c7`, `85ba77b`, `10d94b6` | 2022-10-04 … 2024-06-28 | Bobby-Huggins, pipme | signature/history renames, NumPy 2.0 — no change to the expression |

The reviewer's dating is correct. The expression `iteration - (T + 1)` has stood unchanged since `35be58b`; the code acquired its **present** meaning at `9267816` (2021-08-25), when the iteration counter became 0-based and the floor was adjusted but the offset was not.

**Did the Python ever match MATLAB? No.** At `35be58b` the counter was still 1-based (`if iteration > tol_stable_warmup_iters + 1` and `lcb_max_vec = history[:iteration]` are the MATLAB expressions verbatim), so the correct 0-based start was `iteration - T`, and the code used `iteration - T - 1` — one entry too many (4 instead of 3 at the defaults). `9267816` rebased everything around it to 0-based, which turned the one-entry error into a two-entry error (5 instead of 3). The commit message of `9267816` explicitly claims this class of fix — "fix: fixed some off by one errors caused by a previous change of starting iterations from 0 instead of from 1" — and the floor `max(2,·)→max(1,·)` in the very same hunk is one of them; `recent_past` was simply missed.

### 3. The MATLAB lines on the day of the port (2021-07-07)

`git log --before=2021-07-07 -1 -- private/vbmc_warmup.m` → `d2989b1` (2021-01-12). Its lines 61-64 are byte-identical to the block quoted in point 1. `git blame` dates `RecentPast = iter-ceil(...)+1;` and `idx_last(max(2,RecentPast):end) = true;` to `d6f1188` (2019-09-20, lacerbi, "changed interface and tweaked warmup") — nineteen months before the port and unchanged since.

### 4. Written rationale

**None found.** Searched:
- Code comments around the lines, now and in `35be58b` and `9267816`: the only comments are the MATLAB-derived "Second requirement, also no substantial improvement of max fcn value in recent iters (unless already performing BO-like warmup)" and, two blocks up, "NB: Take care with MATLAB 'end' indexing and off-by-one errors" (added with the ELCBO window, which *is* translated correctly).
- Full commit messages (`git log --format=fuller`, `--grep` on `warmup|warm-up|lcb|warp`): `35be58b`'s message is a list of sub-commit subjects ("feat: check conditions that end vbmc warmup", "refactor: rename booleans for readability", …) with no discussion of indexing.
- PR #15 (`gh pr view 15 --repo acerbilab/pyvbmc`). The author's own discussion points were: (1) introducing named booleans instead of MATLAB's two-line `stopWarmup` expression, (2) "We could split the method into two… (This means I would split at line 91 of MATLAB.)". The two follow-up comments are about the boolean names and "coverage is at 100%". The PI's review is an empty-bodied APPROVED. No inline review comments exist on the PR (`gh api repos/acerbilab/pyvbmc/pulls/15/comments` is empty). Nothing about the window.
- `pyvbmc/vbmc/README.md:60` lists `_check_warmup_end_conditions(): vbmc_warmup.m` as a plain reference, with no caveat; the "Porting status" section says nothing about warm-up.
- `docsrc/source/`: no mention.
- `dev/` (all of it, including `dev/2026-09-02-modernization-discussion.md`, `dev/plans/`, `dev/results/`, `dev/TODO.md`): grep for `recent_past`, `lcb_max`, `warmup` returns only generic prose about warm-up plus the port-review documents of 2026-09-19. `dev/experiments/port_review_20260919/known_differences.md` (the pre-existing differences sheet) has **no** entry for this.
- Tests: `pyvbmc/testing/vbmc/test_vbmc_loop_termination.py:396-468` — all four `test_check_warmup_end_conditions_*` set `iteration_history["lcb_max"] = np.ones(101)`, a constant vector, so `impro_fcn` is 0 under any window and the index arithmetic is unobservable. No test comment speaks to intended behavior. No skipped or xfail test touches it.

### 5. Verdict

**Accidental — mistranslation of a 1-based index expression, high confidence.** The evidence that decides it: the same hunk that fixed the neighbouring floor for 0-based indexing (`9267816`) left this offset alone, under a commit message that names off-by-one fixes as its purpose; the ELCBO window five lines above it *is* translated correctly with an explicit warning comment about MATLAB `end` indexing; and the two windows disagree already at `35be58b` under the then-current 1-based convention, i.e. the expression was never right under either convention. No deliberate-difference record exists anywhere.

---

## Item 2 — `_recompute_lcb_max` is an unimplemented stub

### 1. Is the discrepancy real as described? Yes.

`pyvbmc/vbmc/vbmc.py:2403-2408`:

```python
    def _recompute_lcb_max(self):
        """
        RECOMPUTE_LCB_MAX Recompute moving LCB maximum based on current GP.
        """
        # ToDo: Recompute_lcb_max needs to be implemented.
        return np.array([])
```

Called at `vbmc.py:1641-1645` (`if self.options.get("recompute_lcb_max"): self.optim_state["lcb_max_vec"] = self._recompute_lcb_max().T`). `grep -rn 'lcb_max_vec' pyvbmc/ --include=*.py` gives exactly one write (`vbmc.py:1644`) and no read: the `lcb_max_vec` occurrences at `:1975-2001` are a **local variable of the same name**, assigned from `self.iteration_history.get("lcb_max")` at `:1975`. The option defaults to `True` (`pyvbmc/vbmc/option_configs/advanced_vbmc_options.ini:308-309`, "Recompute LCB max for each iteration based True current GP estimate" — the mangled description is an artifact of the automated `yes`→`True` conversion of the MATLAB defaults).

MATLAB: `private/recompute_lcbmax.m` is a real 23-line implementation (predict with the current GP on all active training inputs, `lcb = fmu - options.ELCBOImproWeight*sqrt(fs2)`, `movmax(lcb,[numel(lcb),0])`, sampled at `stats.N`); `vbmc.m:815-817` stores it in `optimState.lcbmax_vec`; `private/vbmc_warmup.m:46-50` prefers it to `stats.lcbmax`:

```matlab
if isfield(optimState,'lcbmax_vec') && ~isempty(optimState.lcbmax_vec)
    lcbmax_vec = optimState.lcbmax_vec(1:iter);
else
    lcbmax_vec = stats.lcbmax(1:iter);
end
```

`defopts.RecomputeLCBmax = 'yes'` at `vbmc.m:343`. So PyVBMC permanently takes MATLAB's `'no'` branch, while MATLAB's default is `'yes'`. A second, subtler half of the same omission: PyVBMC's `_check_warmup_end_conditions` has no equivalent of the `isfield/isempty` preference at all — `vbmc.py:1975` reads `iteration_history["lcb_max"]` unconditionally, so even a correctly filled `optim_state["lcb_max_vec"]` would be ignored. The recomputed vector feeds **both** warm-up criteria (`impro_fcn` and the long-term `max_thresh`/`idx_1st`/`pos` test at `vbmc.py:2000-2007`), and MATLAB's version is a cumulative maximum (non-decreasing) while the recorded per-iteration `lcb_max` values are not.

### 2. Timeline of the Python lines

| commit | date | author | what it did |
|---|---|---|---|
| `d974e20` | 2021-01-19 | Marlon Tobaben | `initialCodePort/vbmc.py:95` declares `__4recompute_lcbmax(self, gp, optimState, stats, options)` with body `pass`; `loop.py:250-257` transliterates the MATLAB call site faithfully |
| `c272fa7` | 2021-01-24 | Marlon Tobaben | package split, stub carried over |
| **`ec94eba`** | **2021-06-08** | **Marlon** | **feat: add skeleton of vbmc.optimize()** — renames it `_recompute_lcbmax`, changes `pass` to `return np.array([])`, and writes the live call site `if self.options.get("recomputelcbmax"): self.optim_state["lcbmax_vec"] = self._recompute_lcbmax().T` |
| `35be58b` | 2021-07-07 | Marlon | writes `_check_warmup_end_conditions` reading `iteration_history["lcbmax"]` unconditionally — the `isfield/isempty` preference is dropped here |
| **`39b29e6`** | **2021-12-24** | **Marlon** | **fix: solve several warnings raised by pylint (#65)** — adds the line `# ToDo: Recompute_lcbmax needs to be implemented.` above the `return` |
| `d9bfe16` / `3f9c93a` | 2022-08-28 / 2022-08-30 | Bobby-Huggins | Snake_case (#91) renames the option to `recompute_lcb_max`; Snake case (#92) renames the method to `_recompute_lcb_max` and the state key to `lcb_max_vec`, and rewords the comment to "Recompute_lcb_max needs to be implemented." |

**The Python never matched MATLAB.** The body has returned an empty array since `ec94eba` (2021-06-08) and was `pass` before that. It has never contained a GP prediction. Nothing has ever read its output.

### 3. The MATLAB lines on the day of the port

On 2021-06-08, `git log --before=2021-06-08 -1 -- vbmc.m` → `0917171`, which contains at line 343 `defopts.RecomputeLCBmax = 'yes ...'` and at lines 818-821:

```matlab
    if optimState.Warmup && iter > 1
        if options.RecomputeLCBmax
        	optimState.lcbmax_vec = recompute_lcbmax(gp,optimState,stats,options)';
        end
```

`private/recompute_lcbmax.m` existed in full (added `97ce2ec`, 2020-04-11, "added recomputation of lcbmax"; never changed since), and `private/vbmc_warmup.m:46-50` already preferred it (`git blame`: all five lines `97ce2ec`, 2020-04-11). The same is true on 2021-07-07. Nothing here can be read as a faithful port of an older MATLAB.

### 4. Written rationale — searched hard, and there is none

- **In-code**: the only marker is `vbmc.py:2407`, `# ToDo: Recompute_lcb_max needs to be implemented.` It records the intent to come back, not a reason to stay away. Note how it entered: in a *pylint clean-up* (`39b29e6`), not in a design discussion.
- **Porting log**: `pyvbmc/vbmc/README.md:62` lists `_recompute_lcb_max(): recompute_lcbmax.m` among the "Matlab references", in the same format as fully ported functions, with no caveat. The file's own "Porting status" section (lines 21-28) lists what is unported — the experimental features, `acqhedge_vbmc.m`, `vp_repo`, the compute-vargrad branches — and does **not** mention `recompute_lcbmax`. That is an omission in the log, not a rationale.
- **`dev/`**: no hit for `recompute_lcb`, `lcbmax` or `lcb_max` outside the 2026-09-19 port-review documents. `dev/2026-09-02-modernization-discussion.md` §9 (known latent bugs) does not list it. `dev/TODO.md` and `dev/plans/latent-bug-fixes.md` do not mention it. The pre-existing differences sheet `dev/experiments/port_review_20260919/known_differences.md` has no entry; `dev/experiments/port_review_20260919/counterpart_map.md:91` actively mis-states it: `| private/recompute_lcbmax.m | pyvbmc/vbmc/vbmc.py:2403: _recompute_lcb_max | P1a | ported | |`.
- **Issues / PRs** (`gh`, authenticated): the repository has 13 issues in total. `gh pr list --search recompute` and `--search lcb` both return empty; `--search warmup` returns only #15 and #31. No issue or PR text mentions the function. The nearest miss is worth recording: JOSS reviewer issue **#139** ("Some options appear to not have any effect", 2023-05-18) listed `retry_max_fun_evals`, `optim_toolbox`, `repeated_acq_discount`; the answer was "these options are from the MATLAB implementation but are unused in this implementation, at least for the time being. To avoid confusion we removed them just now in pull #140." `recompute_lcb_max` was not among them and survived that sweep — plausibly because the option *is* read at `vbmc.py:1642`, so a scan for unread options does not find it.
- The same blind spot recurs in the recent audit: `a63b17a` (2026-09-19, "docs(options): mark the declared options that nothing reads") registered 25 inert options in `INERT_OPTIONS` (`pyvbmc/vbmc/options.py:21-46`); `recompute_lcb_max` is **not** in that set, and the `.ini` line for it was left untouched by that commit, because its test "recomputes the set by scanning the package for reads" and this option is read.
- **Tests**: no test under `pyvbmc/testing/` names `_recompute_lcb_max` or reads `optim_state["lcb_max_vec"]` (the only matches are oracle fixture snapshots of `optim_state`, which carry the empty array). No skipped or xfail test. The warm-up tests supply `iteration_history["lcb_max"]` directly.
- **Docs**: `docsrc/source/` says nothing; `docsrc/source/faq.md` documents *other* unported features (`RetryMaxFunEvals`, `results["overhead"]`) but not this one.

### 5. Verdict

**Accidental — a placeholder that was never filled in; no rationale exists anywhere.** Confidence high. The decisive evidence: the call site was transliterated faithfully on day one and kept live behind a default-`True` option; the body has been an empty-array stub since the skeleton commit; the only written trace is a ToDo added as a lint clean-up; the porting log lists the function as if ported; and three separate later sweeps that would plausibly have caught it (the JOSS dead-option review of 2023, the 2026-09-19 inert-option audit, the counterpart map) each passed it by. The additional half of the omission — `_check_warmup_end_conditions` having no branch on `lcb_max_vec` at all — means that implementing the function alone would not restore MATLAB's behavior; `vbmc.py:1975` would also need MATLAB's preference. There is no evidence anyone decided this; there is evidence (the ToDo) that someone intended the opposite.

---

## Item 3 — the warping clocks at the end of warm-up

### 1. Is the discrepancy real as described? Yes, with two refinements.

MATLAB, `private/vbmc_warmup.m:91-102` (the branch taken when warm-up really ends):

```matlab
        optimState.Warmup = false;
        if isempty(action); action = 'end warm-up'; else; action = [action ', end warm-up']; end
        threshold = options.WarmupKeepThreshold * (numel(optimState.DataTrimList)+1);
        optimState.LastWarmup = optimState.iter;

        % Start warping
        optimState.LastWarping = optimState.iter;
        optimState.LastSuccessfulWarping = optimState.iter;
        optimState.LastNonlinearWarping = optimState.iter;
```

Python, `vbmc.py:2032-2042`, ends at `self.optim_state["last_warmup"] = iteration`. Both sides initialize the clocks to `-Inf` (`misc/setupvars_vbmc.m:156`, `:159`; `vbmc.py:933`, `:937`) and both set them inside the warp itself (`misc/warp_input_vbmc.m:161-162`; `pyvbmc/whitening/whitening.py:240-241`), so this is the only missing update. The two live Python readers see the stale `-inf`:

- `vbmc.py:1248-1250`, `self.iteration - self.optim_state["last_warping"] > WarpDelay`, is `inf > WarpDelay` → always true. MATLAB requires `iter - LastWarping > WarpEveryIters*max(1,WarpingCount) = 5` after warm-up ends (`vbmc.m:533-541`), so it cannot warp for five more iterations. Both sides default `warp_rotoscaling = True` (`advanced_vbmc_options.ini:321`; `vbmc.m:348`), so this reader is live at the defaults.
- `vbmc.py:2176-2180`, `iteration - self.optim_state.get("last_successful_warping", 0) >= tol_stable_iters / 3`, is `inf >= 4` → always true. MATLAB (`private/vbmc_termination.m:85`) blocks stability-based termination for `TolStableIters/3 = ceil(60/5)/3 = 4` iterations after warm-up ends. This reader is live at the defaults regardless of whether warping is enabled, because MATLAB sets the clock unconditionally.

Refinement (a): `LastNonlinearWarping` is dead in MATLAB too — `grep -rn 'LastNonlinearWarping' ../vbmc/` returns exactly one line, `private/vbmc_warmup.m:102`; it is never initialized and never read. Only two of the three omitted lines have any effect.

Refinement (b): the reviewer's framing "both sides update them inside the warp itself" is right, but PyVBMC's warp-undo path at `vbmc.py:1376-1380` also sets `last_warping` (mirroring `vbmc.m:619`) while deliberately leaving `last_successful_warping`, which matches MATLAB.

### 2. Timeline of the Python lines

| commit | date | author | what it did |
|---|---|---|---|
| `489598a` | 2021-06-03 | Marlon | feat: add init to VBMC — ports `setupvars_vbmc.m`: `optim_state["last_warping"] = -np.inf`, `optim_state["last_successful_warping"] = -np.inf`, `optim_state["warping_count"] = 0`, with MATLAB's comments. **The fields existed a month before the warm-up port.** |
| **`35be58b`** | **2021-07-07** | **Marlon** | **feat: handle end of vbmc warmup (#15)** — writes `_setup_vbmc_after_warmup`; the end-warm-up branch stops at `self.optim_state["lastwarmup"] = iteration`. The three MATLAB lines and the `% Start warping` comment are absent, with no note. |
| `f1e6a0b` | 2021-06-30 | Marlon | feat: add loop termination methods (#10) — ports `vbmc_termination.m` **without** the `LastSuccessfulWarping` guard (`is_finished_flag = True` unconditionally) |
| **`4aa4ca9`** | **2022-03-03** | **Bobby-Huggins** | **Whitening (#69)** — adds `pyvbmc/whitening/whitening.py` (first appearance), the `doWarping` branch in `optimize` with `iteration - self.optim_state["last_warping"] > WarpDelay`, the warp-undo write, and `optim_state["last_successful_warping"] = optim_state["iter"]` inside the warp. **It does not touch `_setup_vbmc_after_warmup`** (the string does not occur in its diff, and the function's own line history does not include this commit). |
| **`0a27ee3`** | **2023-02-15** | **Bobby-Huggins** | **refactor: clarify `VBMC._check_termination_conditions()`. (#130)** — adds the reader `iteration - last_successful_warping >= tol_stable_iters / 3` |
| `dd0e17e`, `ad96d50`, `c3d823a`, `8f906d8` | 2022-11-23 … 2026-09-08 | various | packaging, pre-release, oracle fixtures, GP fix — no change to the end-warm-up branch |

Line history of the function body itself (`git log -L 2025,2090:pyvbmc/vbmc/vbmc.py`) — `ec94eba`, `35be58b`, `357ce06`, `aa45473`, `9267816`, `abf5c3e`, `c477a7b`, `4949c82`, `b8f8802`, `d9bfe16`, `3f9c93a`, `343b9c7`, `85ba77b`, `10d94b6` — contains neither `4aa4ca9` nor `0a27ee3`. **The Python never set the clocks at warm-up end, in any revision.**

**On the PI's hypothesis**: it is half right and instructive. Input warping did indeed arrive eight months after the warm-up port (`whitening.py` first appears in `4aa4ca9`, 2022-03-03), so at the time `_setup_vbmc_after_warmup` was written there was no warping code for the clocks to govern — the natural explanation for dropping three lines that looked inert. But the `optim_state` fields themselves already existed (`489598a`, 2021-06-03), so the lines were not untranslatable, merely pointless-looking. And the hypothesis's second half is confirmed: neither commit that later brought a *reader* went back to add the writer. `0a27ee3`'s PR body makes the pattern explicit — "Adds a heuristic to avoid exiting after a recent warping to `VBMC._check_termination_conditions()` (same check as exists in MATLAB version, **but it wasn't present in Python**)" — i.e. the author was porting a missing MATLAB check, from `vbmc_termination.m`, and did not check `vbmc_warmup.m` for the matching assignment.

### 3. The MATLAB lines on the day of the port (2021-07-07)

The current revision was `d2989b1` (2021-01-12); the block quoted in point 1 is exactly what `git show d2989b1:private/vbmc_warmup.m` contains. `git blame` of those lines: `optimState.LastWarmup` from `aa7e5d5` (2020-04-07), the `% Start warping` comment from `c56d034` (2019-12-04), `LastWarping` and `LastNonlinearWarping` from `8b266a3` (2020-02-07), `LastSuccessfulWarping` from `d2989b1` (2021-01-12). All four predate 2021-07-07, and the reviewer's dating (`8b266a3`, `d2989b1`) is correct.

### 4. Written rationale

**None found.**
- No comment in `_setup_vbmc_after_warmup` mentions warping, in the current source or in `35be58b`. The MATLAB `% Start warping` comment was not carried over even as a "Missing port" marker — and the codebase does use such markers freely elsewhere (23 of them in `pyvbmc/vbmc/`, e.g. `active_sample.py:343` "Missing port: noise_shaping", `gaussian_process_train.py:334` "Missing port: output warping hyperparameters not implemented").
- `35be58b`'s message and PR #15's discussion (see item 1) are about naming and splitting the function; the split point chosen — "line 91 of MATLAB" — is exactly the line above the block in question, and no one remarked on its contents.
- PR #69 (Whitening) body: "I still have one additional test that I am writing… I am not aware of any bugs or otherwise unusual behavior besides the 'invalid value in sqrt'…". No mention of warm-up. Approved with an empty review body; no inline comments.
- PR #130 body: quoted above; it notes the *reader* was missing in Python but says nothing about the warm-up-end writer.
- `pyvbmc/whitening/README.md` records one porting limitation only: "Rotating and rescaling has been implemented, but nonlinear warping has not." That covers `LastNonlinearWarping` incidentally, but not the two live clocks.
- `dev/`: no hit for `last_warping` or `last_successful_warping` outside the 2026-09-19 port-review documents; the differences sheet has no entry.
- Tests: `pyvbmc/testing/vbmc/test_vbmc_loop_termination.py:470-575` exercises `_setup_vbmc_after_warmup` and asserts `vbmc.optim_state.get("last_warmup") == 100` (`:499`) but never the warping clocks; `pyvbmc/testing/vbmc/test_vbmc_init.py:515-516` asserts only that both start at `-inf`. `test_vbmc_check_termination_conditions_stability` leaves `last_successful_warping` at `-inf`, so the guard is vacuously true there.

### 5. Verdict

**Accidental — an omission at port time (2021-07-07) that was never repaired when the readers arrived (2022-03-03, 2023-02-15).** Confidence high for "no recorded rationale" and high for "accidental". The mitigating context is real and worth stating to the PI: when the lines were dropped, PyVBMC had no warping code at all, so two of the three assignments were genuinely inert on that day and the third (`LastNonlinearWarping`) is inert even in MATLAB; what makes it a defect is that neither `4aa4ca9` nor `0a27ee3` revisited `vbmc_warmup.m` when it stopped being inert. This is the one of the three items where "a faithful port of a then-incomplete Python" is a partial explanation — but it is an explanation reconstructed from dates, not one anybody wrote down.

---

## Summary

1. **Recent-improvement window**: ported wrong from the start (`35be58b`, 2021-07-07) and made worse, not better, by the 0-based rebasing of `9267816` (2021-08-25) which fixed the neighbouring floor and missed this offset; **no rationale anywhere**; verdict **accidental mistranslation**, high confidence — the Python has never matched MATLAB under either indexing convention.
2. **`_recompute_lcb_max`**: a stub since the first skeleton (`ec94eba`, 2021-06-08, body `return np.array([])`; `pass` before that), with the call site and the default-`True` option transliterated faithfully around it and the consuming branch (`vbmc_warmup.m:46-50`) never ported either; **no rationale anywhere** — only the in-code ToDo added during a pylint clean-up (`39b29e6`), while `pyvbmc/vbmc/README.md:62` lists the function as ported and three later dead-code sweeps passed it by; verdict **accidental, an unfinished placeholder**, high confidence.
3. **Warping clocks at warm-up end**: never set in any Python revision — dropped when `_setup_vbmc_after_warmup` was written (`35be58b`, 2021-07-07), eight months before warping existed on the Python side (`4aa4ca9`, 2022-03-03) and twenty months before the second reader was added (`0a27ee3`, 2023-02-15), neither of which went back to `vbmc_warmup.m`; **no rationale anywhere**, not even a "Missing port" marker; verdict **accidental omission**, high confidence, with the date sequence as a partial explanation and the note that one of the three MATLAB lines (`LastNonlinearWarping`) is dead in MATLAB as well.
