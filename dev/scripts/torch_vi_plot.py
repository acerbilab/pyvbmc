"""Plot the completed Stage 4 control artifacts without numerical reruns."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = (
    ("warmup", "Warmup"),
    ("deterministic", "Deterministic K=1"),
    ("bounded", "Bounded"),
    ("warped", "Warped"),
    ("noisy", "Noisy fixed GP"),
    ("boost_sampled", "Boost K=50 (sampled GP)"),
    ("medium_synthetic", "Synthetic D=10, K=25"),
    ("boost_single", "Synthetic D=15, K=50 boost"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    values = {}
    for path in (args.controls / "runs").glob("*/result.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "ok" and data["seed"] == 1701
        assert data["complete_run"]["trace_enabled"] is False
        key = (data["case"], data["backend"], data["device"])
        if key in values:
            raise ValueError(f"duplicate control: {key}")
        values[key] = data["complete_run"]["timing"]["total"]
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for offset, backend, device, color, label in (
        (-0.23, "numpy", "cpu", "#334155", "NumPy CPU"),
        (0, "torch", "cpu", "#d97706", "Torch CPU"),
        (0.23, "torch", "cuda", "#2563eb", "Torch CUDA"),
    ):
        times = [values[(case, backend, device)] for case, _ in CASES]
        ax.barh(
            [i + offset for i in range(len(CASES))],
            times,
            height=0.22,
            color=color,
            label=label,
        )
    ax.set_yticks(range(len(CASES)), [label for _, label in CASES])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Complete variational fit (seconds, log scale)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_title(
        "Eager Torch feasibility: complete-fit controls", fontweight="bold"
    )
    fig.suptitle(
        "One warmed observation per workload / arm | float64 | one CPU thread"
        " | synchronized CUDA\nWindows 11 | Core Ultra 7 155H | RTX 4060 Laptop"
        " | fixed GP factors",
        fontsize=10,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            args.out / f"complete-fit-timings.{extension}",
            dpi=170,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
