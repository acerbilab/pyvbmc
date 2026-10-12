"""Derive the report tables from the two balanced measurement artifacts."""

import hashlib
import json
import math
from pathlib import Path
from statistics import median


def summarize(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["status"] == "complete", path
    assert raw["campaign_numpy_global_rng_unchanged"]
    rows = []
    for kernel, cases in raw["cases"].items():
        for case, heldout, trace in zip(
            cases,
            raw["heldout"][kernel]["cases"],
            raw["traced_allocations"][kernel],
            strict=True,
        ):
            assert case["name"] == heldout["name"] == trace["name"]
            assert case["complete"] and case["numerical_validation"]["pass"]
            times = heldout["round_seconds"]
            selection = raw["selection"][kernel]["by_regime"][case["name"]]
            tentative = raw["tentative_selection"][kernel][case["name"]]
            default = median(times["default"])
            selected = median(times["selected"])
            saving = default - selected
            # Same-setting fluctuations never count as a tuning gain.
            repay = (
                math.ceil(case["measurement_seconds"] / saving)
                if selection["budget"] != 65536 and saving > 0
                else None
            )
            controls = {}
            for arm in ("default", "selected"):
                ratios = [
                    a / b
                    for a, b in zip(
                        times[arm],
                        times[arm + "_same_budget_control"],
                        strict=True,
                    )
                ]
                controls[arm] = {
                    "median_ratio": median(ratios),
                    "range": [min(ratios), max(ratios)],
                }
            aliases = case["budget_aliases"]
            rows.append(
                {
                    "kernel": kernel,
                    "name": case["name"],
                    "shape_D_K_count": [
                        case["D"],
                        case["K"],
                        case["effective_count"],
                    ],
                    "requested_count": case["requested_count"],
                    "count_unit": (
                        "rows" if kernel == "pdf" else "per_component"
                    ),
                    "tentative_budget": tentative["budget"],
                    "final_budget": selection["budget"],
                    "candidate_median_ms": {
                        budget: 1000 * median(case["round_seconds"][str(rep)])
                        for budget, rep in aliases.items()
                    },
                    "heldout_default_ms": 1000 * default,
                    "heldout_selected_ms": 1000 * selected,
                    "heldout_paired_median_speedup": median(
                        a / b
                        for a, b in zip(
                            times["default"], times["selected"], strict=True
                        )
                    ),
                    "same_budget_controls": controls,
                    "discovery_wall_seconds_excluding_diagnostics": case[
                        "measurement_seconds"
                    ],
                    "selection_warmups_and_rounds_seconds": case[
                        "selection_wall_seconds"
                    ],
                    "first_default_call_ms": 1000
                    * case["first_use_seconds"]["65536"],
                    "synthetic_setup_seconds": case["synthetic_setup_seconds"],
                    "diagnostic_seconds": case["diagnostic_seconds"],
                    "discovery_only_amortization_calls": repay,
                    "maximum_absolute_error": max(
                        check["max_abs_difference"]
                        for check in case["numerical_validation"]["checks"]
                    ),
                    "traced_peak_mib": trace["peak_bytes"] / 2**20,
                }
            )
    return {
        "artifact": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "script_sha256": raw["provenance"]["script_sha256"],
        "timings_seconds": raw["timings_seconds"],
        "rows": rows,
    }


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    output = [
        summarize(directory / name) for name in ("primary.json", "repeat.json")
    ]
    (directory / "summary.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
