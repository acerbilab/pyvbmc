# Golden reference promotion, 2026-09-13

`reference_990_20260913` contains 990 JSON/NPZ pairs across 23 configurations.
It replaces the 870 historical synthetic-target pairs with the completed
candidate population and retains the 120 real-data pairs byte for byte.
The previous `reference_990_20260912` remains intact on the generating
machine, with its [manifest](../realdata_extension_20260912/sha256_manifest.json)
and [execution record](../realdata_extension_20260912/README.md).

The PI accepted the completed assessment and authorized promotion on
2026-09-13. The acceptance criterion is preservation of algorithmic behavior
without evidence of meaningful degradation. Correcting existing occasional
inference failures is outside this campaign's scope. The population evidence
and targeted investigation provide no convincing evidence of degradation
or a newly introduced implementation defect. Finite seed counts and the
absence of equivalence margins limit stronger claims.

The 870 candidates comprise 273 first-stage cases, 30 prespecified noisy
follow-up cases, and 567 completion cases. All use frozen PyVBMC harness
`68a43db18f69cf1a99aa44779503c8a1b861891b`, numerical base
`de686f712c6822968f0b98779469ec9e6152d0f4`, and gpyreg
`a2f8ddce867f502e29717959cf0ff3529f598618`. Execution used one worker,
single-threaded BLAS, `vectorized_target=False`, calibration off and
`tol_elcbo_boost=0.1`. Python is 3.12.6, NumPy 2.5.2 and SciPy 1.18.1.
The real-data pairs retain `fc50ee1`, whose default-path numerical agreement
with this treatment was certified in the
[real-data record](../realdata_extension_20260912/README.md).

| Full matched assessment | Historical | Candidate |
|---|---:|---:|
| Complete cases | 870 | 870 |
| Converged | 853 | 855 |
| Usable | 825 | 837 |
| Target evaluations | 131,375 | 131,075 |
| Summed optimizer hours | 21.638 | 21.860 |

Usability requires evidence error < 1, gsKL < 1 and MMTV < 0.2. There are
33 paired gains and 21 losses. None of the 76 KS tests rejects after Holm
correction. Four of 95 paired tests reject, all favoring the candidate:
the three accuracy metrics on Banana D2 and MMTV on Rosenbrock D2.
None of the 19 usability tests rejects. The pooled families are descriptive
full-allocation evidence following staged inspection; only the earlier
eight-test noisy follow-up was prespecified as confirmatory, and none of
those tests rejects. The completion cohort alone has 550/567 usable fits
versus 552/567 historical fits. Interactive laptop use makes wall time an
uncontrolled observation rather than a speed comparison.

Final boost accepts 858 of 870 proposals and rejects 12. Usability rises
from 829 before boost to 837 returned, with eight gains and no losses.
Independent review reproduced all 95 paired probabilities and Holm
decisions, all 76 KS statistics and probabilities, aggregate counts and
every boost decision. The original [assessment JSON](../../experiments/population_completion_20260912/assessment.json)
is preserved without edits; SHA-256 is
`dd4b97cbe86fdd6bb5350ca8005d92237fdf451cad498e448073e3b82a9799fe`.
The [comparison table](../../experiments/population_completion_20260912/comparison.md)
and [verification](../../experiments/population_completion_20260912/assessment_verification.json)
provide the supporting results.

Targeted investigation covered the noiseless Student D8 and Banana D10
usability flips. Student remains 48/50 usable: historical failures at seeds
12 and 15 become usable, while seeds 18 and 44 become unusable. Banana
changes from 46/50 to 43/50, with four gains and seven losses. These seed
exchanges are compatible with occasional stochastic failures; the observed
counts do not establish a higher underlying failure probability. Candidate
Student seed 18 is more severe than the investigated historical failures
and remains part of the reference rather than being excluded.

Saved-posterior evaluation attributes Student's reported ELBO optimism
mostly to the GP expected-log-joint estimate. Returned optimism is
6.299 ± 0.021 nats for seed 18 and 1.141 ± 0.008 for seed 44, using
100,000 independent draws per state (± denotes Monte Carlo standard error).
Historical selected pre-boost failures also show optimism, 1.575 and
0.814 nats. Seed 18 remains internally stable at a poor broad fit; seed 44
deteriorates late and terminates within the existing stability rule's
exception allowance. Both run longer than their historical usable matches.
Banana failures concentrate around the lower bend and underrepresent the
curved tails. Retained rotoscaling is associated with these fits, but the
trajectories diverge and narrow before transformation. The trace's `warped`
flag includes undone attempts; actual transformer changes were checked.
Failures precede final boost. No individual numerical change was shown
to cause them, and no failure-mode correction is required for promotion.

The Student reconstruction and interpretation received an independent
review. Banana's 100 returned means and covariances reproduce stored
moments within 1e-12; independently recomputed gsKL differs by at most
3.2e-15. Durable diagnostic evidence is in
[true-target estimates](../../experiments/population_completion_20260912/student_true_elbo.json),
[first-divergence checks](../../experiments/population_completion_20260912/student_first_divergence.json)
and [endpoint verification](../../experiments/population_completion_20260912/investigation_verification.json).
Detailed trajectories and diagnostic scripts remain under the campaign's
gitignored `investigation_20260913/` directory.

Promotion revalidates all 870 candidate completion records and their four
artifact hashes, checks the previous 990 pairs against their manifest,
copies the approved pairs without regeneration, and tests all 990 NPZ
archives for ZIP integrity. The new [manifest](sha256_manifest.json) hashes
NPZ bytes directly and normalizes JSON/Markdown line endings to LF.
The [even/odd check](even_vs_odd.md) has zero flags across 92 KS tests.
The [five-case replay](final_replay.md) checks current code against the
prepared reference: every non-timer NPZ loop/final array, semantic final-result
field and initial design must match exactly. Timing, memory and provenance
metadata are excluded. All five cases passed with zero flags at current
code `27f6564` in 5.5 minutes. The returned posterior's transformer is absent
from these traces and remains explicitly uncertifiable.

The reproducible assembly and publication procedure is [promote.py](promote.py):

```console
python dev/golden/promotion_20260913/promote.py prepare
python dev/scripts/golden_replay.py --baseline dev/scripts/runs/golden/reference_990_20260913 --sidecars dev/scripts/runs/golden/reference_990_20260913 --out dev/scripts/runs/population_completion_20260912/promotion_replay
```

Use the pinned gpyreg checkout on the generating machine for the replay.
Copy its `replay.json` and `replay.md` to this record as `final_replay.json`
and `final_replay.md`, then run `promote.py publish`. Preparation can resume
only when existing copied pairs match their sources. Publication rejects
optimized Python (`-O`), checks the exact replay results, revalidates the
prepared and historical references, and verifies that the active sidecars
still match the historical manifest before replacing them. The
[validation record](validation.json) contains provenance and gate results.

Active sidecars and summary are tracked in `dev/golden/baseline/`; full
traces remain local under `dev/scripts/runs/golden/reference_990_20260913/`.
The preceding references and campaign artifacts are retained. This
promotion publishes no release asset and changes no solver numerics.

Independent promotion review checked allocation, preservation, hashes,
publication guards, replay provenance, current reference pointers and the
scientific interpretation. The guards reject optimized Python, active
sidecar drift and a replay with mismatched source or reference provenance.
