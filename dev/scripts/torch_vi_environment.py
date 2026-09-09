"""Record the isolated Stage 4 environment and a tiny float64 device smoke.

Run with the reference venv and an explicit --overlay; no package installation
or global configuration changes are made. This is developer tooling only.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(args.overlay.resolve()))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "device": args.device,
        "overlay": str(args.overlay.resolve()),
        "modules": {},
        "free_disk_bytes": shutil.disk_usage(root).free,
    }
    start = time.perf_counter()
    for name in ("numpy", "scipy", "gpyreg", "pyvbmc", "torch"):
        tick = time.perf_counter()
        module = importlib.import_module(name)
        result["modules"][name] = {
            "version": importlib.metadata.version(name),
            "file": module.__file__,
            "import_seconds": time.perf_counter() - tick,
        }
    result["imports_seconds"] = time.perf_counter() - start
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    result["torch_configuration"] = torch.__config__.show()
    result["cuda_runtime"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    result["torch_threads"] = [
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
    ]
    result["default_dtype"] = str(torch.get_default_dtype())
    result["driver"] = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if args.device == "cuda" and not result["cuda_available"]:
        raise RuntimeError("Requested CUDA device is unavailable")
    tick = time.perf_counter()
    values = np.arange(1.0, 10.0).reshape(3, 3) / 10
    x = torch.tensor(
        values, dtype=torch.float64, device=args.device, requires_grad=True
    )
    loss = (x.sin() * x).sum()
    (grad,) = torch.autograd.grad(loss, x)
    matrix = torch.triu(x.detach()) + torch.eye(
        3, dtype=torch.float64, device=args.device
    )
    rhs = torch.ones((3, 2), dtype=torch.float64, device=args.device)
    sol = torch.linalg.solve_triangular(matrix, rhs, upper=True)
    if args.device == "cuda":
        torch.cuda.synchronize()
        result["gpu_name"] = torch.cuda.get_device_name()
        result["gpu_capability"] = list(torch.cuda.get_device_capability())
        result["gpu_total_memory"] = torch.cuda.get_device_properties(
            0
        ).total_memory
    result["first_device_smoke_seconds"] = time.perf_counter() - tick
    error = np.max(
        np.abs(grad.cpu().numpy() - (np.sin(values) + values * np.cos(values)))
    )
    residual = (matrix @ sol - rhs).abs().max().item()
    assert x.dtype == grad.dtype == sol.dtype == torch.float64
    assert error < 1e-14 and residual < 1e-14
    result["float64_smoke"] = {
        "gradient_max_abs_error": float(error),
        "triangular_solve_max_abs_residual": residual,
        "passed": True,
    }
    result["overlay_installed_bytes"] = sum(
        p.stat().st_size for p in args.overlay.rglob("*") if p.is_file()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
