# PyVBMC 1.5 overnight population assessment

## Findings

The first stage shows no broad deterioration in the integrated solver. All
273 runs completed, the existing population comparison flags no configuration,
and overall accuracy-based usability is similar to the full golden reference.
On the same seed subset, usability improves and optimizer time decreases.
The difficult noisy configurations have mixed outcomes: Student D8 improves,
while Rosenbrock SD3 has larger median posterior errors and logistic SD3 loses
two usable runs. These are the useful targets for any additional sampling.

The final-boost guard follows its specified rule in every case. It rejects
five candidates, retaining valid pre-boost posteriors. The boost improves
posterior metrics in most runs, with some tradeoffs; no accepted boost turns
a usable pre-boost posterior into an unusable returned posterior in this
population. The earlier Normal D5 accuracy-fence exceedance remains small in
the context of the full reference distribution.

There is no evidence here that warrants changing the production algorithm or
running the entire 870-case candidate allocation. A focused extension of
noisy Rosenbrock SD3 and logistic SD3, seeds 15–29, would provide a more useful
next check: 30 additional runs, approximately 80 minutes including overhead.
Keep the current treatment fixed and report seeds 15–29 separately as a
follow-up sample, alongside cumulative summaries of seeds 0–29. Reference
promotion can be considered after this targeted question is resolved or the
current evidence is accepted as sufficient.

## Design and artifact validation

The candidate consists of 15 seeds for each of 18 ordinary configurations and
3 deliberate D15 exhaust cases. It is compared with all 870 golden reference
runs; same-seed summaries are reported separately. The full-reference and
same-seed comparisons answer different descriptive questions and must not be
interchanged when interpreting an apparent gain.

Candidate source: `68a43db`, with numerical code identical to `de686f7`.
Gpyreg: `a2f8ddc`, version 1.1.0. Environment: Python 3.12.6, NumPy 2.5.2,
SciPy 1.18.1 and cma 4.4.4. The campaign used one worker, one BLAS thread,
scalar targets, historical chunk settings (`performance_calibration="off"`),
and the joint 0.1 final-boost guard with boost-only weight shrinkage disabled.

All 273 completion records, artifact hashes, NPZ arrays, posterior captures,
transformers, scores and options were revalidated. All 870 tracked reference
sidecars match the published reference hashes. The 76-test comparison was
recomputed and matches the overnight report exactly. Returned metrics in the
boost reports exactly match the main sidecars. All 273 boost decisions were
independently recomputed from the stored production scores; there are no
discrepancies or diagnostic metric errors.

No inference, additional target evaluations or diagnostic rescoring was
performed for this assessment. The retained boost metrics were computed by
the overnight worker using the existing benchmark's diagnostic generators.

## Overall outcomes

"Usable" preserves the existing benchmark criterion: absolute log-evidence
error < 1, Gaussianized symmetric KL (gsKL) < 1, and mean marginal total
variation (MMTV) < 0.2. It is separate from reported convergence.

| Outcome | Full reference | Reference, matching seeds | Candidate |
| --- | ---: | ---: | ---: |
| Runs | 870 | 273 | 273 |
| Converged | 853 | 266 | 266 |
| Usable | 825 (94.8%) | 253 (92.7%) | 259 (94.9%) |
| Target evaluations | 131,375 | 42,985 | 42,040 |
| Summed optimizer time | 21.638 h | 7.139 h | 6.555 h |

There are 12 same-seed transitions into usability and six out of it, for a
net gain of six. This is a comparison of the integrated algorithms on selected
seeds, not an isolated estimate of a particular fix's effect.

All seven candidate budget terminations produced complete valid outputs:
the three D15 cases deliberately use 750 evaluations, and noisy Rosenbrock
SD3 seeds 3, 5, 8 and 13 reached their ordinary evaluation limit. There are
no execution failures. The matching reference also has 266 converged runs.

The overnight supervisor took 6 h 49 min 50 s, finishing at 05:48:58 UTC+03
on September 11. Its elapsed time excludes the first Gaussian case's earlier
21.7-second worker check, whose result was reused. Summed optimizer time
includes all 273 cases and is **8.2% lower** than the matching reference;
evaluation count is **2.2% lower**. These observations span separate laptop
sessions and are not a controlled speed measurement.

Peak process memory is recorded but should not be treated as a paired memory
benchmark: the candidate uses fresh processes per case, while historical
workers were reused and their process-lifetime peak can include earlier cases.

## Statistical screening and difficult configurations

The existing two-sample KS comparison covers evidence error, gsKL, MMTV and
evaluation count for all 19 configurations, with a single Holm correction at
alpha 0.05. None of the 76 tests is rejected. The smallest unadjusted p-value
is 0.028 for an improvement in the three-case exhaust cohort's evidence error.
The screening does not establish equivalence. Effect sizes, tail cases and
practical accuracy inform the staged assessment alongside these tests.

### Paired analysis

Within each configuration, candidate and reference runs are paired by seed.
Evidence error, gsKL, MMTV and evaluation count use two-sided Wilcoxon
signed-rank tests, with exhaustive sign permutations of the nonzero
differences so that tied ranks are handled exactly. Usability uses the exact
McNemar test on gains and losses. One Holm correction covers these 95 tests
(19 configurations times five outcomes), separately from the historical KS
screen. This paired analysis is exploratory, added during the results review.

The signed-rank null assumes independent seed differences symmetric about
zero; identical seeds do not preserve alignment of random draws after the
algorithms diverge. Exact enumeration removes approximation error in the
p-value, but does not remove those assumptions. See the
[SciPy signed-rank documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
Seed pairs are analyzed within configurations, without pooling them as
independent replicates across configurations.

**None of the 95 paired tests rejects after correction.** The following
selected effect sizes give the median of individual candidate-minus-reference
differences, which differs from subtracting the two cohort medians. Positive
changes mean worse accuracy. All adjusted p-values in this family equal 1.

| Configuration and metric | Median paired change | Improved / worsened | Raw paired p-value |
| --- | ---: | ---: | ---: |
| Student D4, MMTV | +0.00796 | 3 / 12 | 0.0256 |
| Rosenbrock SD3, MMTV | +0.05198 | 5 / 10 | 0.0554 |
| Rosenbrock SD3, gsKL | +0.30710 | 5 / 10 | 0.1070 |
| Logistic SD3, MMTV | +0.02657 | 7 / 8 | 0.3591 |
| Student D8 SD3, evidence error | -0.24182 | 9 / 6 | 0.0833 |

Student D4 MMTV has the smallest raw paired p-value; all 15 runs remain
usable. Exact McNemar raw p-values for noisy logistic, Rosenbrock and Student
D8 are 0.625, 1.0 and 0.125 respectively. The paired results leave the noisy
Rosenbrock posterior shift as a useful follow-up question, without establishing
a population regression or improvement in any configuration.

### Difficult configurations

| Configuration | Usable, full reference | Usable, matching reference | Usable, candidate |
| --- | ---: | ---: | ---: |
| Logistic D5, noise SD3 | 39/50 | 13/15 | 11/15 |
| Rosenbrock D2, noise SD3 | 23/30 | 13/15 | 13/15 |
| Student D8, noise SD3 | 14/30 | 3/15 | 8/15 |

For **noisy Rosenbrock**, median gsKL is 0.109 in the full reference, 0.076
on matching reference seeds, and 0.321 in the candidate. Median MMTV is
0.076, 0.061 and 0.112 respectively. Ten of the 15 matching seeds worsen
on each posterior metric. Median evidence error changes less, from 0.226 to
0.247 on matching seeds; median evaluations decrease from 190 to 180.
Usability and convergence counts stay unchanged, but the thresholded count
hides a shift in typical posterior accuracy. The candidate's pre-boost median
gsKL is already 0.321 and its MMTV is 0.114. This identifies the main-loop
solution as a place to investigate; it does not identify which numerical
change caused the difference.

For **noisy logistic regression**, median matching-seed evidence error changes
from 0.167 to 0.218, gsKL from 0.427 to 0.466, and MMTV from 0.156 to 0.160.
Median evaluations decrease from 240 to 220. Three previously usable seeds
(1, 4 and 14) become unusable, while seed 3 becomes usable. Seeds 1 and 4
miss the MMTV threshold narrowly (0.211 and 0.205); seed 14 has evidence error
1.282 and MMTV 0.262. All 15 converge. The mixed shifts merit tracking without
attributing them solely to fewer evaluations.

For **noisy Student D8**, median matching-seed evidence error improves from
1.462 to 0.923 and MMTV from 0.157 to 0.142; median gsKL improves from 0.654
to 0.595. The full reference is less difficult than its selected first 15
seeds: median evidence error is 1.082 and 14/30 are usable. The candidate's
8/15 usable outcomes therefore represent a more modest improvement against
the whole reference than the 3/15-to-8/15 matched comparison suggests.

The two largest same-seed gsKL increases are noisy Rosenbrock seed 11
(0.027 to 1.622) and Banana D10 seed 5 (0.160 to 1.193). Their pre-boost
values are already 1.840 and 1.614; boosting improves each. Banana D10 still
has 14/15 usable outcomes on both sides and its median posterior errors
improve against the full reference. Preserve these cases for diagnosis rather
than interpreting a seed-level change as a population regression by itself.

## Final-boost behavior

All 273 runs attempted a boost: **268 accepted, five rejected**. The guard
compares the worst stored score change across the ELBO and ELBO minus five
SDs with the strict threshold of -0.1. Every recorded decision agrees with
that rule, including the validity checks.

Across all cases, boosting improves returned evidence error in 187, gsKL in
215 and MMTV in 217. It worsens those metrics in 81, 53 and 51 cases,
respectively; the five rejected boosts leave each metric unchanged. These
counts compare each run's stored pre-boost and returned posterior. They are
descriptive and do not imply that the surrogate guard guarantees true
posterior improvement.

Usability increases from **257 pre-boost to 259 returned**, with two gains
(noisy logistic seed 8 and noisy Rosenbrock SD3 seed 4) and no losses. All
five rejected candidates and their fallbacks are usable, so their rejection
does not change that binary count. Their tradeoffs are:

| Rejected candidate | Effect of keeping the pre-boost posterior |
| --- | --- |
| Logistic D5 seed 7 | Better evidence error and gsKL; slightly worse MMTV. |
| Logistic D5 seed 11 | Better gsKL; worse evidence error and MMTV. The candidate's ELBO SD rises from 0.022 to 0.128. |
| Rosenbrock SD3 seed 5 | Better evidence error; worse gsKL and MMTV. |
| Student D8 seed 5 | Better evidence error, gsKL and MMTV. |
| Student D8 seed 13 | Better evidence error, gsKL and MMTV. |

This supports retaining the selected guard for the next assessment. The
stored raw candidates allow later policy comparisons without regenerating
boosts; there is no need to change the threshold based on these five cases.

## Normal D5 follow-up

Seed 0 reproduces the retained gsKL value **0.000490743**, above the historical
fixed fence of **0.000429099**. It is the only candidate Normal seed above
that fence. All 15 Normal runs converge and remain usable; median gsKL is
0.0000825 versus 0.0000869 in the full reference. The candidate maximum is
0.000491 versus the full reference maximum of 0.001581. Thus the original
seed-level flag persists, but this population does not show a broad Normal
posterior regression. Preserve the flag and its original threshold.

## Reproduction and artifacts

Run `.venv/Scripts/python.exe dev/scripts/analyze_population_run.py` from the
repository root. It reads the saved artifacts, verifies them, reconstructs the
comparison and writes
[assessment.json](../experiments/population_assessment_20260911/assessment.json)
and [comparison.md](../experiments/population_assessment_20260911/comparison.md).
The JSON includes full-reference and matched summaries for every configuration,
all 273 paired changes, all boost decisions/metrics, the 76 KS statistics and
the 95 paired tests with raw and Holm-adjusted p-values.

Raw traces, captures and completion records remain under
`dev/scripts/runs/population_overnight_20260910/results/`. Source provenance,
allocation and execution are recorded in the
[population plan](../plans/final-population-benchmark.md).
