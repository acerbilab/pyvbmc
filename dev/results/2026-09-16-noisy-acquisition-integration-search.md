# Noisy-acquisition integration and search experiment

**Status:** paused after development-panel selections following a usage-limit
interruption. No treatment or production-default recommendation has been
established.

This experiment evaluates the cost and selection quality of positive-weight
integration rules and smaller candidate searches for standard VIQR. The
[experimental plan](../plans/noisy-acquisition-efficiency.md) specifies the
allocation, independent judging, timing protocol and promotion criteria.
GP fitting and Bayesian quadrature are outside its scope.

## Sources and environment

The numerical baseline is PyVBMC commit
`9cc6882768ff682ae892a6b453c42e0f2d03d5fa`. The capture harness is committed
as `3fe616c477130625bb01d30d29785773fe2e0e66`. Production `pyvbmc/` source
is unchanged by these experiments.

The frozen gpyreg source is commit
`9e70e6ba53f7607d05c2d9cc2fa9f41cd12b8f3b`, imported from the archived
`gpyreg_1.2.1` checkout. Its installed distribution metadata reports 1.2.0;
the source hashes and resolved import path identify the actual dependency.
The environment uses Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1 and CMA 4.4.4
on Windows. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS`
are all fixed at one. Numerical processes run sequentially.

The [environment preflight](../experiments/noisy-acquisition-efficiency/integration-search/environment_preflight.json)
records the dependency paths, build information and five data/truth archive
hashes. Raw states and per-cell evidence are stored under the local artifact
directory `dev/scripts/runs/noisy_acq_efficiency_20260916/`, indexed in
`dev/scripts/runs/LOCAL.md`.

## State coverage and verification

All seven historical VIQR captures reproduced their stored full-sieve
scores exactly, with no change to their state or RNG. The
[historical-state preflight](../experiments/noisy-acquisition-efficiency/integration-search/historical_state_preflight.json)
records these checks. Its single-call timings are feasibility measurements,
not performance evidence.

The [capture manifest](../experiments/noisy-acquisition-efficiency/integration-search/capture_manifest.json)
allocates six target/noise configurations, development seed 0 and holdout
seed 1, with early and late checkpoints for each trajectory. A trajectory
that terminates before the late threshold contributes its last distinct
eligible state. Captures retain live GP factors and the complete acquisition
setup; exact public-score replay is required before a state is accepted.
All twelve allocated capture runs completed without execution errors, and
all twenty-four states passed exact replay. The [capture inventory](../experiments/noisy-acquisition-efficiency/integration-search/capture_inventory.json)
records coverage and each checkpoint's charged evaluation count.

The expanded developer suite passed 133 checks covering capture identity,
snapshot reconstruction, weighted VIQR parity, numerical stress cases,
judge uncertainty, split locks, grouped denominators, target-free search,
exact CMA-ES baseline agreement, search/crosscheck provenance, failure
denominators, judging escalation and timing boundaries. Two warnings came
from deliberately invalid inputs in the failure-reporting test.
The existing acquisition/oracle suite passed 234 checks, with 15 skips and
three existing VP-density gradient warnings. The initial numerical tools
are committed as `8a13215`; the additional campaign, crosscheck and evidence
tools accompany this report's recovery record.

Independent static reviews covered the numerical protocol and tools before
the interruption, then the fixture repairs, saved-result provenance and
restart instructions during recovery. No review findings remain open.
The documented next-stage pilot has been statically checked but not run.

## Experimental results

Development-panel selections are complete. Full independent judging,
paired timing, full-sieve comparisons and search allocations have not run.
Holdout treatment choices must be frozen before their evaluation. Failed,
missing and unresolved comparisons remain in the allocated denominators.

The [development panel manifest](../experiments/noisy-acquisition-efficiency/integration-search/panel_manifest.json)
allocates 864 treatment selections and 96 paired fresh production MC100
baselines. All 960 cells succeeded, with no failed, missing or partial
records. Morning recovery validated every terminal record and its bound
payload hashes against the frozen manifest and source identity; the
[completion record](../experiments/noisy-acquisition-efficiency/integration-search/panel_completion.json)
records those counts and hashes. The first paired cell passed the smoke, including
eight independent judge replicates on its 34-coordinate union. This smoke
establishes pipeline operation and provides no method-selection conclusion.

The authorized first execution window was 2026-09-16 18:34:41 UTC through
2026-09-17 05:34:41 UTC. A usage limit interrupted active work after screening.
The 04:46 UTC recovery check found no running Python experiment process.
Recovery completed verification and preserved the restart point; the user
needs the laptop approximately 45 minutes after that check. E2-E4 remain
incomplete. E5 requires authorization for a separate execution window.

## Resuming the saved experiment

Run from the existing `pyvbmc-stage3` checkout on
`dev-noisy-acquisition-efficiency`, reusing its `.venv`.
No background job or session needs reattachment.
The ignored `dev/scripts/runs` junction points to the shared local artifact
directory in the sibling checkout, `../pyvbmc/dev/scripts/runs`.
The `noisy_acq_efficiency_20260916/` results, frozen dependency at
`svbmc_pool_20260913/gpyreg_1.2.1/`, and `.venv` are local by design and
are not included in the Git push. They remain available on this laptop;
moving machines requires transferring these artifacts and matching the
recorded environment. The capture and panel runs do not need to be repeated.
Published JSON records redact user-directory paths. Execute with the
original manifests under the local artifact directory, whose hashes remain
recorded in the published summaries. Historical local checkpoint IDs are
retained on the local-only `local/noisy-acquisition-pre-redaction-20260917`
branch; the public feature branch consolidates the same experiment sources.
Use:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYVBMC_GPYREG_SOURCE = (Resolve-Path 'dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1').Path
$env:PYTHONPATH = "$PWD;$env:PYVBMC_GPYREG_SOURCE;$PWD\dev\scripts"
$runRoot = 'dev/scripts/runs/noisy_acq_efficiency_20260916'
```

First measure one development state at the initial 4096-node judge budget.
The following pilot records elapsed time excluding Python imports, uses the
full frozen candidate union for that state, and writes into the controller's
expected layout. A completed pilot is not rerun for timing:

```powershell
@'
import time
from pathlib import Path
import noisy_acq_integration as experiment
from noisy_acq_campaign import write_immutable

root = Path("dev/scripts/runs/noisy_acq_efficiency_20260916")
out = root / "panel_judging"
manifest = experiment.prepare_judge_manifest(
    root / "panel_manifest.json", root / "panel", 4096, None
)
manifest["launch_ready"] = True
write_immutable(out / "b4096_manifest.json", manifest)
state_id = "student_D8_noise3_seed0_late"
union = next(item for item in manifest["unions"] if item["state_id"] == state_id)
cell_out = out / "b4096"
if (cell_out / "records" / f"{state_id}.complete.json").exists():
    raise RuntimeError("Pilot already completed; consult its saved cost record.")
started = time.perf_counter()
record = experiment.run_judge_cell(manifest, cell_out, union)
cost = {"state_id": state_id, "budget": 4096,
        "seconds_excluding_imports": time.perf_counter() - started,
        "status": record["status"]}
experiment.write_json(out / "pilot_cost.json", cost)
print(cost)
if record["status"] != "succeeded":
    raise RuntimeError("Inspect the failed pilot before allocating further work.")
'@ | .venv/Scripts/python.exe -
```

Use `pilot_cost.json` to estimate the larger node budgets and remaining
states before proceeding. The following command launches the **full judging
allocation**, potentially through 65536 nodes for unresolved comparisons.
It resumes the matching pilot under `panel_judging/b4096/`:

```powershell
.venv/Scripts/python.exe dev/scripts/noisy_acq_campaign.py integration --manifest "$runRoot/panel_manifest.json" --results "$runRoot/panel" --out "$runRoot/panel_judging"
```

Its `completed_ladder.json` identifies the final judge summary. Use that
summary for the diagnostic MC crosscheck; measure warmed panel timings
sequentially with the same manifest:

```powershell
.venv/Scripts/python.exe dev/scripts/noisy_acq_campaign.py mc --manifest "$runRoot/panel_manifest.json" --judge-summary '<final-judge-summary>' --out "$runRoot/panel_mc"
.venv/Scripts/python.exe dev/scripts/noisy_acq_timing.py run --manifest "$runRoot/panel_manifest.json" --out "$runRoot/panel_timing"
```

Reduce these outputs with `noisy_acq_evidence.py`, then apply the plan's
development selection rules before preparing a full-sieve manifest.
Later E2/E3 allocations depend on those results and remain unfrozen.
Do not use single-shot screening durations as paired timing evidence or
the one-cell judge smoke as the panel's quality assessment.
