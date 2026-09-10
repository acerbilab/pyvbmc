# Acquisition search on the noisy path: sieve, CMA-ES, gradient refinement

Detailed report behind the [summary note](../2026-09-08-noisy-acquisitions.md);
the experiments on the acquisitions themselves are in
[2026-09-08-noisy-acquisition-experiments.md](2026-09-08-noisy-acquisition-experiments.md).
Saved mid-run states (seed 0, budget-limited, no final boost) are the test
bed; every strategy is scored on the same VIQR surface, later on an
independent importance set. Same container and branch as the companion
report.

## The sieve and CMA-ES on saved states

The PI recalled that CMA-ES did poorly on the acquisition and that a larger
sieve kept helping, which is why `ns_search` is 8192. Measured on saved
mid-run states (seed 0, budget-limited, no final boost) at D = 2 (N = 86),
5 (N = 139), 8 (N = 200) and 10 (N = 246), one fixed VIQR surface per state
(same GP, VP and importance set), `scratch: acq_opt_bench.py`. The score is
the integrated-IQR *reduction* at the returned point as a fraction of the
best any method found (a 65536-point sieve, then L-BFGS-B from eight
mutually distant tops and from every strategy's result); the cost is
acquisition evaluations (points) and batched calls.

| strategy | D = 2 | D = 5 | D = 8 | points D = 2 / 5 / 8 |
|---|---|---|---|---|
| sieve of 8192, no refinement (production without CMA-ES) | 100 % | 77 % | 60 % | 8192 |
| CMA-ES as in production (log post-IQR, `tolfun` 1e-2, noise handler, sigma0 = VP sd) | 100 % | 77 % | 60 % | 10 / 12 / 15 |
| CMA-ES on the log-reduction, `tolfun` 1e-3, no noise handler, sigma0 = VP sd | 100 % | 100 % | 100 % | 158 / 442 / 1012 |
| same, sigma0 = 0.3 length scales | 100 % | 77 % | 73 % | 236 / 626 / 1172 |
| L-BFGS-B from the sieve best, batched finite-difference gradient | 100 % | 100 % | 100 % | 18 / 72 / 279 |
| L-BFGS-B from 8 mutually distant sieve tops | 100 % | 100 % | 100 % | 270 / 786 / 2142 |
| coarse-to-fine resampling around the top 32, 4 rounds of 1024 | 100 % | 99 % | 97 % | 4096 |

The sieve alone, median best reduction over five draws as a fraction of
the reference:

| `ns_search` | D = 2 | D = 5 | D = 8 |
|---|---|---|---|
| 512 | 99.9 % | 60 % | 59 % |
| 2048 | 100 % | 61 % | 60 % |
| 8192 | 100 % | 72 % | 63 % |
| 32768 | 100 % | 76 % | 68 % |

Reading:

- **At D = 2 the sieve is the optimizer and it is enough**: 512 points
  already sit at the optimum, so nothing downstream can matter, which is
  why the CMA-ES diagnostics on the D = 2 snapshot showed no gain.
- **At D = 5 and 8 the sieve leaves 25–40 % of the achievable reduction
  on the table and grows only logarithmically with its size** (D = 8:
  59 % at 512, 68 % at 32768). That is the "bigger sieve keeps helping"
  memory: the sieve has been doing the optimizer's job.
- **Production CMA-ES never improves on the sieve point**: it stops on
  `tolfun` after two generations (10–15 evaluations) on every state. Two
  compounding causes. The objective is the log of the *residual*
  interquantile range, which varies by 5e-3 across the whole candidate set
  (the reduction is 1e-3 of the range) against an absolute `tolfun` of
  1e-2; and `sigma0` is the VP's largest standard deviation, so the first
  generations sample the whole posterior width around the sieve point and
  find nothing better. (The noise handler is pointless on a deterministic
  objective but was not isolated as a cause.)
- **A local gradient step from the sieve point recovers the whole gap at
  1–3 % of a sieve's cost**: L-BFGS-B with a forward-difference gradient
  from one batched call of D + 1 rows reaches the reference in 72 (D = 5)
  and 279 (D = 8) evaluations, 12 and 31 calls, 0.02 and 0.07 s. Eight
  restarts found nothing better on these states. CMA-ES also gets there
  once it optimizes the log-reduction without the noise handler, at 3–4×
  the evaluations, and only with the wide `sigma0` (the length scales here
  are far larger than the posterior width, so 0.3 ℓ is not a local step).
- **`lumpy_D10_noise3` at N = 246 is a different failure and is excluded**:
  the GP's length scales collapse in two dimensions (per-sample minima of
  1e-4 and 0 against a posterior width of order 1), the surface is flat to
  8e-13 across the sieve and the best reduction is 1.5e-13 of the range,
  ten orders of magnitude below D = 5 and 8. No optimizer can act on that;
  the GP hyperparameter fit under heavy noise is the problem there.

What to do, in order:

1. **Optimize the log-reduction, not the log residual.** Same minimizer,
   O(1) dynamic range, so any tolerance means what it says. Available as
   `AcqFcnVIQR(loss="iqr_reduction")`; the search stage should use it as
   the optimizer's objective while the sieve can keep either.
2. **Replace CMA-ES by L-BFGS-B from the sieve's best point (optionally a
   few mutually distant tops), with the gradient from one batched
   acquisition call of D + 1 rows.** The batched objective already exists
   for CMA-ES. This is the standard recipe of modern BO libraries (raw
   samples, top-k restarts, L-BFGS-B). Analytic gradients (every operation
   in the VIQR core is closed form) or autodiff in the torch port would
   halve the cost and remove the finite-difference step.
3. **Then shrink the sieve.** With refinement in place, 2048 candidates
   plus L-BFGS-B should match 8192 plus nothing on these states; the
   remaining role of the sieve is coverage of separate basins, which eight
   restarts did not find here. To be confirmed end to end, since the
   `ns_search = 2048` arm lost accuracy at σ = 3 without refinement.
4. If CMA-ES stays: log-reduction objective, `tolfun` 1e-3 or relative, no
   noise handler, `sigma0` from the VP width, a minimum number of
   generations before `tolfun` can fire.
5. The coarse-to-fine batched resampling (97–99 %) is the gradient-free
   fallback where the surface is not smooth (integer variables, the
   piecewise-constant nearest-neighbour noise estimate).

### Sieve size before L-BFGS-B

Same states and scoring (`scratch: sieve_size_lbfgs.py`): a sieve of
`n` candidates, then L-BFGS-B from its best point or from 2, 4 or 8
mutually distant tops; five draws each; median [min] of the reduction as
a fraction of the reference, and the median total evaluations (sieve plus
refinement).

| state | sieve n | sieve alone | 1 start, evals | 4 starts, evals | 8 starts, evals |
|---|---|---|---|---|---|
| D = 2 | 128 | 99.6 % [99.0] | 100 % [100], 156 | 100 % [100], 258 | 100 % [100], 447 |
| D = 2 | 1024 | 100 % [99.8] | 100 % [100], 1043 | 100 % [100], 1154 | 100 % [100], 1307 |
| D = 5 | 128 | 59 % [58] | 74.5 % [70], 231 | 100 % [74.5], 681 | 100 % [100], 1209 |
| D = 5 | 256 | 60 % [59] | 74.5 % [60], 383 | 100 % [100], 701 | 100 % [100], 1295 |
| D = 5 | 512 | 60 % [59] | 74.5 % [60], 597 | 100 % [100], 1005 | 100 % [100], 1551 |
| D = 5 | 1024 | 61 % [60] | 70 % [70], 1115 | 100 % [100], 1487 | 100 % [100], 2009 |
| D = 5 | 8192 | 72 % [66] | 100 % [70], 8277 | 100 % [100], 8595 | 100 % [100], 9009 |
| D = 8 | 128 | 52 % [51] | 100 % [100], 363 | 100 % [100], 1056 | 100 % [100], 2073 |
| D = 8 | 256 | 54 % [52] | 100 % [100], 500 | 100 % [100], 1148 | 100 % [100], 2183 |
| D = 8 | 1024 | 57 % [53] | 100 % [100], 1241 | 100 % [100], 1934 | 100 % [100], 2933 |
| D = 8 | 8192 | 63 % [61] | 100 % [100], 8409 | 100 % [100], 9084 | 100 % [100], 10002 |

- The sieve size barely matters once a local optimizer follows: at D = 8
  a 128-point sieve and one start reach the optimum every time (363
  evaluations against 8192 for a sieve that reaches 63 %).
- The number of restarts matters where the surface has several basins:
  the D = 5 state has a second basin at 74.5 % that catches a single start
  in most draws whatever the sieve size (even the 8192 sieve's own best
  point sits in it in some draws); four mutually distant starts reach the
  optimum from every sieve of 256 points or more.
- A defensible default is therefore a sieve of 512–1024 candidates and
  four restarts from mutually distant tops, about 1000–2000 evaluations at
  D = 5–8, a fifth of the present sieve, reaching the optimum where the
  present search stops at 60–75 %. The distance criterion for "mutually
  distant" should be scaled by the VP width rather than the GP length
  scales, which can be far larger than the posterior.
- Caveat: one state per dimension, seed 0; the end-to-end arms decide.

### Bumpiness and Monte Carlo overfitting of the refinement

The PI's caution about generalizing from smooth synthetic states: checked
on the three states above plus three built to be harder (`scratch:
landscape.py`, `refine_na.py`): a noisy multimodal `lumpy_D4` (σ = 1), a
correlated `banana_D6` (σ = 2), and the Student target late in its run
(N = 325, one hyperparameter sample, K = 29).

**Basins**, from L-BFGS-B started at 32 VP draws on one importance set:

| state | GP length scale / VP width, min and median | distinct optima | starts within 5 % of the best |
|---|---|---|---|
| rosenbrock_D2_noise3 | 2.1, 8.3 | 2 | 59 % |
| logreg_D5_noise3 | 5.0, 6.6 | 3 | 94 % |
| student_D8_noise3, N = 200 | 4.1, 7.4 | 2 | 91 % |
| student_D8_noise3, N = 325 | 3.6, 4.0 | 5 | 3 % |
| banana_D6_noise2 | 1.8, 274 | 7 | 9 % |
| lumpy_D4_noise1 | 0.05, 0.95 | 31 | 6 % |

Once the length scales approach the posterior width, or N grows, the
surface fragments and random starts land in poor basins. The earlier
D = 5 and 8 states were the smooth end.

**Monte Carlo overfitting.** Refine on the 100-sample importance set,
judge the point on an independent 4000-sample set (median of five draws;
the refined point's reduction over the sieve point's):

| state | gain on its own 100-sample set | gain under 4000 samples | refined point worse than the sieve's |
|---|---|---|---|
| logreg_D5_noise3 | ×1.18 | ×1.06 | 2 / 5 |
| student_D8_noise3, N = 200 | ×1.36 | ×0.96 | 3 / 5 |
| student_D8_noise3, N = 325 | ×1.40 | ×0.82 | 3 / 5 |
| banana_D6_noise2 | ×1.18 | ×1.16 | 1 / 5 |
| lumpy_D4_noise1 | ×1.00 | ×1.00 | 0 / 5 |

Per draw the judged gain ranges from ×0.57 to ×1.28 on the Student state.
The "100 % of the reference" numbers of the two subsections above were
measured on the surface the optimizer climbed, so they measured how well
L-BFGS-B fits the sampling noise of 100 importance points, not how much
better the chosen point is. The coarse-to-fine resampling overfits the
same way; the sieve is robust because it does not resolve the fine
structure.

**Refining on a larger set fixes it.** Sieve and one-start L-BFGS-B on a
set of `Na` samples, judged on an independent 6400-sample set (median
[min, max] over five draws):

| state | Na = 100 | Na = 400 | Na = 1600 |
|---|---|---|---|
| logreg_D5_noise3 | ×1.02 [0.96, 1.26], worse 2/5 | ×1.22 [1.00, 1.35], worse 1/5 | ×1.21 [0.99, 1.24], worse 1/5 |
| student_D8_noise3, N = 200 | ×0.90 [0.54, 1.27], worse 3/5 | ×1.22 [1.17, 1.38], worse 0/5 | ×1.27 [1.18, 1.31], worse 0/5 |
| banana_D6_noise2 | ×1.21 [0.96, 1.25], worse 1/5 | ×1.20 [0.97, 1.25], worse 2/5 | ×1.24 [1.05, 1.34], worse 0/5 |
| student_D8_noise3, N = 325 | ×0.87 [0.67, 1.16], worse 3/5 | ×1.07 [1.02, 1.11], worse 0/5 | ×1.08 [1.03, 1.09], worse 0/5 |

Revised recommendation (superseding the search items in the companion report's conclusions):

1. The sieve stays the coarse selector, on the 100-sample set as now (its
   choice was the best of the compared points under the judge in most
   draws); its size can come down once a refinement follows.
2. Refinement only on an importance set of at least 400, preferably 1600
   samples (or a deterministic quadrature of that accuracy), which costs
   little because the refinement makes tens of batched calls of D + 1
   rows: re-evaluate the sieve's top few on the large set, then L-BFGS-B
   on the log-reduction from the best of them, and keep the sieve point
   if the large-set value does not improve. Judged independently this
   gains ×1.07–1.27 over the sieve point on four states with no draw
   worse at 1600 samples.
3. The gain is modest and state-dependent; whether it is worth anything
   end to end is still the open question, and the synthetic targets
   remain the caveat.

### Measured cost of the search pipelines

Per new point, on the four states, one BLAS thread with four processes
sharing four cores (`scratch: pipeline_cost.py`); the chosen point judged
on an independent 6400-sample set, gain relative to today's pipeline A,
median [min, max] over five draws. Set-up is `active_importance_sampling`
(one GP prediction on the Na points and two triangular solves with Na
right-hand sides per hyperparameter sample, so it scales as Ns · N² · Na):
Na = 100: 5–12 ms, Na = 400: 18–43 ms, Na = 1600: 83–267 ms on these
states (N = 139–325, Ns = 1–7).

| pipeline | logreg D = 5 | Student D = 8, N = 200 | banana D = 6 | Student D = 8, N = 325 |
|---|---|---|---|---|
| A: today, sieve 8192 on Na = 100 | ×1.00, 601 ms | ×1.00, 834 ms | ×1.00, 766 ms | ×1.00, 234 ms |
| B: sieve 1024 on Na = 100; top 8 re-scored and L-BFGS-B on Na = 1600 | ×1.42 [1.31, 2.41], 365 ms | ×1.84 [1.17, 2.57], 535 ms | ×1.26 [1.05, 2.79], 571 ms | ×1.95 [1.31, 2.34], 164 ms |
| C: one set Na = 400, sieve 1024, L-BFGS-B | ×1.39 [1.00, 2.40], 349 ms | ×1.53 [1.17, 2.49], 321 ms | ×1.22 [0.99, 5.23], 435 ms | ×1.76 [1.23, 2.20], 91 ms |
| D: one set Na = 1600, sieve 1024, L-BFGS-B | ×1.41 [1.30, 1.88], 930 ms | ×1.85 [1.21, 2.36], 925 ms | ×1.28 [1.06, 5.67], 1024 ms | ×1.87 [1.25, 2.33], 263 ms |
| E: sieve 8192 on Na = 400, no refinement | ×1.31 [1.00, 1.95], 1822 ms | ×1.24 [0.99, 2.15], 1822 ms | ×1.00 [1.00, 5.40], 1617 ms | ×1.72 [1.11, 2.09], 1061 ms |
| F: one set Na = 400, sieve 2048, L-BFGS-B from top 4 | ×1.39, 694 ms | ×1.53, 856 ms | ×1.22, 1159 ms | ×1.76, 220 ms |

Reading: today's pipeline loses a third to a half of the achievable
reduction to the Monte Carlo noise of its 100-sample set (its sieve pick
overfits that set as much as a refinement would) and to the missing
refinement. The 1600-sample set-up is a real cost, about a third of
today's sieve call, but pipeline B still comes in at 0.6–0.7× today's
time because the sieve shrinks eightfold, and its judged gain is
×1.26–1.95 with no draw below ×1.05. Pipeline C (a single 400-sample set)
is cheaper still, 0.4–0.6×, with slightly smaller gains and one draw at
×0.99. Sieving on a 1600-sample set (D) or a larger sieve on a better set
without refinement (E) cost more than today for no more gain. Four
restarts (F) added nothing over one on these states.

Revised recommendation: pipeline B, or C where the set-up matters
(large N with several hyperparameter samples); the set-up cost scales as
Ns · N², so B's grows toward the end of a long run while Ns shrinks.

### Sizing the refinement set as a function of N

Two measurements set the rule (`scratch: pipeline_cost.py`, the CV probe
and the candidate-side timing in the session log).

**How many samples the refinement needs.** The per-sample interquantile
reduction at the sieve's best point, on a 1600-sample set, first
hyperparameter sample:

| state | N | Ns | CV of the per-sample reduction | Na for a 5 % standard error | share of the reduction in the top 10 % of samples |
|---|---|---|---|---|---|
| rosenbrock_D2_noise3 | 86 | 8 | 1.26 | 638 | 36 % |
| logreg_D5_noise3 | 139 | 7 | 0.87 | 302 | 29 % |
| student_D8_noise3 | 200 | 6 | 1.34 | 720 | 42 % |
| student_D8_noise3, N = 325 | 325 | 1 | 1.84 | 1358 | 55 % |
| banana_D6_noise2 | 158 | 6 | 8.31 | 27624 | 98 % |
| lumpy_D4_noise1 | 110 | 8 | 4.20 | 7061 | 98 % |

On the smooth states a 5 % standard error needs 300–1400 samples, the
range that worked; on the bumpy ones the reduction is carried by a few
importance points and no affordable Na fixes that by brute force. The CV
is free to compute from the sieve's own 100-sample evaluation (the
spread of the per-sample terms at its best candidate), so it can gate
the refinement: `Na_refine = clip((CV / 0.05)^2, 400, Na_cap)`, and skip
the refinement when `(CV / 0.05)^2` exceeds the cap, since a gradient
step on an estimate that noisy climbs the sample (the overfitting above).

**What the set costs, and a formulation that removes the N² term.**
Today's set-up (`active_importance_sampling`) precomputes
`(K + Σ)^-1 k(X, Xa)`: two triangular solves with Na right-hand sides
per hyperparameter sample, about 7e-7 ms × Ns · N² · Na on this machine
(267–281 ms at N = 200, Ns = 6, Na = 1600). The refinement evaluates D + 1
points per call, so it can instead solve candidate-side, `(K + Σ)^-1
k(X, x)` for the D + 1 points (what `predict` does anyway) and contract
with `k(X, Xa)` formed once (Ns · N · Na · D):

| state | Na | today: set-up + 60 calls | candidate-side: `k(X, Xa)` + 60 calls |
|---|---|---|---|
| student_D8_noise3, N = 200, Ns = 6 | 1600 | 281 + 60 × 8.9 = 815 ms | 29 + 60 × 4.4 = 291 ms |
| student_D8_noise3, N = 200, Ns = 6 | 6400 | 1140 + 60 × 23 = 2526 ms | 58 + 60 × 12 = 775 ms |
| student_D8_noise3, N = 325, Ns = 1 | 1600 | 83 + 60 × 2.5 = 232 ms | 3 + 60 × 1.3 = 78 ms |
| logreg_D5_noise3, N = 139, Ns = 7 | 1600 | 91 + 60 × 8.0 = 570 ms | 9 + 60 × 3.0 = 190 ms |

With the candidate-side evaluation the refinement's cost is
`Ns · (D + 1) · (N² + N · Na)` per call and linear in N in the Na term, so
the cap is on the per-call budget rather than on a set-up:
`Na_cap = clip(B / (max(Ns, 1) · N), 400, 6400)` with B = 1.9e6 giving 1600
at N = 200, Ns = 6 (1370 at N = 350, Ns = 4; 1900 at N = 500, Ns = 2). With
today's route the same cap reads `B' / (max(Ns, 1) · N²)`, B' = 3.8e8.
Either way the sieve stays on the 100-sample set and at 1024 candidates.

