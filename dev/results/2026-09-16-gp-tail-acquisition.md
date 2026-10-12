# Tail acquisition and GP refitting on noisy Rosenbrock

The first extreme tail acquisitions in all three corrected box-sampler
runs are explained by VIQR's covariance objective. Uniform-box candidates
win the sieve, their preference over central candidates survives a
50-digit recomputation, and the following GP fits substantially increase
signal variance, length scales and matrix conditioning. This examination
finds no further implementation bug in those decisions.

The [box-sampler report](2026-09-16-gp-box-sampler.md) establishes the
porting fix and the three paired runs. This investigation uses the
corrected runs only: Rosenbrock D2, known observation SD 3, seeds
1000–1002. An extreme evaluation is one whose observed log density is
more than 1000 below the largest observed value in that run, as in the
box-study diagnostics. It addresses the onset of extreme tail exploration. It does
not certify the numerical accuracy of later decisions once condition
numbers reach 1e13–1e14.

## Exact reconstruction

`dev/scripts/gp_tail_probe.py` resumes the saved iteration immediately
before each first extreme evaluation: iterations 14, 24 and 17 for
seeds 1000, 1001 and 1002. A Python trace callback captures the sieve
before selection and the acquisition state immediately before each
target evaluation. The target callback requires exact agreement with
the original evaluation coordinates and supplies the saved noisy value
and SD. There are no fresh noisy observations.

All five calls of each batch reproduce exactly: calls 76–80, 131–135
and 96–100. At the following recorded iteration, GP inputs, targets and
hyperparameter samples, VP mixture parameters and RNG state are also
exact. The acquisition batches hold GP hyperparameters fixed and update
the posterior factors between observations. Their importance points are
drawn afresh from the VP; the replay reconstructs these otherwise omitted
history details.

## Why the tail points win

The comparison uses the selected point, the best sieve point, and the
best-scoring central sieve point. Here “central” means that the analytic
noiseless log density lies within 20 of the largest value among that
sieve's candidates. This density-based definition is a diagnostic using
benchmark truth, not information available to VBMC.

| Seed / first tail call | Selected original coordinates | Noiseless log density | GP predicted mean | VIQR IQR reduction: selected / central |
|---|---|---:|---:|---:|
| 1000 / 80 | (6.3474, 7.1079) | −1110.35 | −882.99 | 2.013% / 1.560% |
| 1001 / 134 | (−7.6609, 9.6556) | −2417.60 | −2276.19 | 1.043% / 0.709% |
| 1002 / 98 | (−7.3650, 0.2986) | −2917.76 | −2654.45 | 1.417% / 1.136% |

GP means are averaged over hyperparameter samples and expressed in
original coordinates by subtracting the transform's log Jacobian.

Each sieve winner comes from the uniform box. In seed 1000, CMA-ES
improves its score from 4.843207263 to 4.843184471 and moves it from
(6.4162, 7.3906) to the selected point above. In seeds 1001 and 1002,
CMA-ES returns a worse score and the sieve point is retained. Thus an
optimizer excursion beyond the proposal box does not explain these
first tail evaluations.

VIQR evaluates the remaining uncertainty at the VP integration points
after a hypothetical observation. For each GP hyperparameter sample,
the variance reduction is

\[
\tau^2(x_a;x_*) =
\frac{\operatorname{Cov}[f(x_a),f(x_*)\mid\mathcal D]^2}
{\operatorname{Var}[f(x_*)\mid\mathcal D]+\sigma^2(x_*)}.
\]

The candidate's predicted mean is absent from this expression. A point
with tiny predicted posterior density can therefore be useful for
reducing covariance uncertainty in the central region. In these states,
the GP already has large signal variance and long correlation lengths;
estimated observation variance remains approximately 9 at both tail and
central candidates. All 100 integration points are central: their true
log densities range approximately from −10 to −4. The preference is
not explained by integration points landing in the extreme tails.

The table reports reduction of the IQR sum, averaged over the GP
hyperparameter samples, relative to the pre-observation sum. These are
predictions under the fitted GP with fixed hyperparameters and fixed
integration points. They are not measured improvements in inference.

## Numerical check

The independent calculation rebuilds the squared-exponential kernel
from the exact stored float64 coordinates and hyperparameters, forms a
fresh Cholesky factor in 50-digit arithmetic, and recomputes candidate
and integration-point posterior variances, cross-covariances and VIQR.
Training and candidate observation-noise values are held at their
float64 values. Every stored GP hyperparameter sample and the same 100
integration points contribute. No stored inverse or Cholesky factor is
used in this calculation.

| Seed | Central minus selected log-VIQR | Largest float64 versus 50-digit score difference |
|---|---:|---:|
| 1000 | 0.00461265 | 1.84e−10 |
| 1001 | 0.00336954 | 2.63e−9 |
| 1002 | 0.00284127 | 7.39e−10 |

Lower VIQR is better. The selected point beats the central comparator
at both precisions in every case. The seed-1000 CMA-ES improvement also
survives. None of the checked look-ahead variances is clipped to zero
in either calculation. These checks establish the ranking among the
three examined candidates, not the ranking of the entire sieve or the
global acquisition optimum. Fresh integration samples were not tested.

## What changes at the GP refit

Holding the preceding iteration's hyperparameters fixed while adding
the new batch's training data isolates the covariance change caused by
the additional inputs. Refitting on those same data produces the large
conditioning increase:

| Seed | Previous GP condition | New data, old hyperparameters | New data, refitted hyperparameters | Largest signal variance before / after |
|---|---:|---:|---:|---:|
| 1000 | 4.03e7 | 4.23e7 | 8.93e8 | 6.10e6 / 1.18e8 |
| 1001 | 3.53e8 | 3.63e8 | 2.62e9 | 2.81e7 / 2.02e8 |
| 1002 | 2.25e8 | 2.35e8 | 7.10e9 | 2.45e7 / 7.15e8 |

Condition numbers are the maximum over hyperparameter samples, computed
as the squared condition number of the Cholesky factor. Fitted length
scales increase as well. For seed 1000, their ranges in transformed
coordinates move from about 5–7 to 8–11. Observation variances remain
approximately 9, and no Cholesky retry noise multiplier is activated.
The refit therefore increases the signal-to-noise ratio and correlations
while retaining the extreme observations in the training set.

This supports a feedback mechanism: a globally correlated GP values tail
observations for central uncertainty reduction; fitting increasingly
large log-density ranges then produces still larger scales and worse
conditioning. The fixed-hyperparameter comparison identifies the refit
as the immediate source of the condition-number jump. It does not
isolate the effect of the single tail observation from the rest of the
batch, hyperparameter optimization and sampling.

## Disposition and reproduction

The bounded investigation is complete. The evidence supports treating
the remaining robustness issue as an algorithmic follow-up under the
PI's existing decision to defer such changes beyond 1.5. A follow-up
could evaluate MATLAB's optional noise shaping, which PyVBMC has not
ported, or other controls on how very low-density observations influence
the GP. This study has not compared remedies or established their effect
on inference accuracy. The box-sampler bug fix remains independently
justified by the source comparison.

Tracked evidence is in `dev/experiments/gp_tail_20260916/`: the full
15-call diagnostic summary, per-seed replay and precision checks, GP
conditioning histories and a manifest with input and script hashes.
The raw captures and dependency installation are listed in
`dev/scripts/runs/LOCAL.md`. The numerical source is `e1ed79f` (the
box fix is `40a6f18`); gpyreg is 1.2.1 at `9e70e6b`.

For each seed, run the probe's `replay` mode with `--seed`, `--runs`
pointing to the corrected box-study output, `--out` to a fresh capture
directory and `--gpyreg-source` to the pinned checkout. Run `analyze`
and `refit` on that output directory, then `precision --seed` for each
seed. All modes take the three path arguments. `precision` additionally
accepts `--mpmath-source` for an isolated mpmath installation. The final
probe was exercised in all four modes; its replay checks passed for
all three seeds. Package numerical code and oracle references are
unchanged by this investigation.
