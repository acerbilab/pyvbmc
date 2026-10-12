# S-VBMC speed on the release benchmark

Checked 2026-10-07 from the recorded October 2 release-gate timings.
Across eight benchmark conditions, the integrated S-VBMC optimization had
a **1.9- to 5.2-fold speedup** over standalone `svbmc` 0.1.1, using the
reciprocal of each condition's median paired runtime ratio. This replaces
the provisional 1.9-to-4 figure from the September pools in `CHANGELOG.md`.

## Measurement and provenance

The [campaign record](../experiments/release_gate_20261002/README.md)
identifies PyVBMC `ff3ed01463b3c2a031dd2225c9f487dc369d2397`, gpyreg
1.4.0 and standalone `svbmc` 0.1.1 at
`13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01`. It ran on the Turso cluster's
AMD EPYC 7452 nodes with one physical core per task, one BLAS/OpenMP
thread and CPU Torch 2.14.0. Each pair fitted the same subset of posteriors,
with the same retained runs and component counts. The arms ran one at a
time within a task, alternating which ran first across repetitions, after
a discarded warm-up fit of each implementation.

The measurement is `optimize_seconds`, timed around `SVBMC.optimize()`
in each arm. It includes all work inside that call, including final ELBO
evaluation and the integrated arm's shrinkage computation. It
excludes importing, loading and constructing the stack, drawing posterior
samples, and the harness's reference and accuracy scoring. Both arms use
`n_samples=20`, `lr=0.1`, `max_steps=500` and `version="all-weights"`.
The integrated arm uses `n_samples_final=100` for its fresh final estimate.
The [harness](../scripts/svbmc_pool_stack.py), `fit_integrated`,
`fit_original` and `runtime_ratios`, defines the timing boundary and ratio.

There are **560 paired fits**: 70 per condition, comprising 20 subsets
each at `M = 2, 4, 8` and 10 at `M = 16`. The other 400 of the campaign's
960 cells, at `M = 3, 5, 32`, ran only the integrated arm and contribute
nothing to the speed comparison. Subsets are disjoint within each `M`;
they can overlap across `M`.

## Results

For a paired cell, the ratio is integrated `optimize_seconds` divided by
original `optimize_seconds`. The table aggregates all 70 pairs of a
condition, retaining their prescribed weighting across `M`. Speedup is
the reciprocal of that median, not the ratio of two independently
aggregated times. The 95% intervals invert the recorded bootstrap
intervals for the median ratio; they are conditional on these pools and
timing observations.

| Condition | Median integrated / original | Speedup | 95% interval |
|---|---:|---:|---:|
| Multisensory D6, noise 3 | 0.193200 | 5.18 | 4.76–5.61 |
| Multisensory D6, noise 1.3 | 0.205316 | 4.87 | 4.37–5.86 |
| Rosenbrock D2, noise 3 | 0.362509 | 2.76 | 2.56–3.25 |
| Gaussian mixture D2, noise 3 | 0.434463 | 2.30 | 1.99–2.65 |
| Ring D2, noise 3 | 0.460294 | 2.17 | 1.99–2.38 |
| Student D8, noise 3 | 0.433387 | 2.31 | 2.03–2.80 |
| Gaussian mixture D2, noiseless | 0.525672 | 1.90 | 1.78–2.23 |
| Multisensory D6, noiseless | 0.234534 | 4.26 | 3.77–4.74 |

The medians recomputed from
[`stacking/results.json`](../experiments/release_gate_20261002/stacking/results.json)
equal all eight medians in
[`stacking/summary.json`](../experiments/release_gate_20261002/stacking/summary.json)
exactly. The paired cells have matching retained-run and component counts.
The campaign's verification record reports all 200 tasks verified with no
failed, missing or partial task. The numeric records' hashes match
[`redaction.json`](../experiments/release_gate_20261002/stacking/redaction.json).

The calculation can be repeated from the repository root without running
inference:

```python
import json
from pathlib import Path
from statistics import median

path = Path("dev/experiments/release_gate_20261002/stacking")
cells = json.loads((path / "results.json").read_text())["cells"]
for condition in sorted({cell["condition"] for cell in cells}):
    ratios = [
        cell["arms"]["integrated"]["optimize_seconds"]
        / cell["arms"]["original"]["optimize_seconds"]
        for cell in cells
        if cell["condition"] == condition
        and "original" in cell["arms"]
        and "integrated" in cell["arms"]
    ]
    print(condition, len(ratios), 1 / median(ratios))
```

## Applicability to the release

Between the measured commit and `276ebca2`, the S-VBMC changes select the
two-level estimate as the noisy headline and update its reporting,
docstrings and startup tip. The weight optimizer, entropy implementation
and shrinkage calculation are unchanged. The benchmark already computed
`shrunk_two_level` inside the timed call; selecting it as the headline
adds no new estimator computation. These recorded timings support the
release figure without another timing campaign.

The [October 5/6 repeat on Arm 3's pools](../experiments/release_gate_20261005/README.md#stacking-comparison-on-arm-3s-pools-stacking_arm3)
provides a check on another set of inputs: its 560 pairs give condition
speedups of 2.1 to 5.3 with identical S-VBMC code. Arm 3's warm-up variant
was not selected for the release, so the changelog uses the October 2
release-code pools.

The range describes this benchmark on one CPU family, using warm calls
and the settings above. It is neither a promise for every stack nor a
measurement of an entire VBMC-plus-stacking workflow. It does not cover
GPU execution, first-call overhead or larger stacks. The arms use
different random-number streams, so their fitted weights need not be
identical; paired subsets control the input workload.
