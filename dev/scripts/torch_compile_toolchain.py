"""Serial, bounded float64 Inductor toolchain smoke; not a VBMC benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


def msvc_environment():
    """Activate an existing MSVC installation for this process only."""
    from setuptools._distutils._msvccompiler import _get_vc_env

    env = _get_vc_env("x64")
    os.environ.update({key.upper(): value for key, value in env.items()})
    return {
        key: os.environ.get(key)
        for key in ("VCToolsVersion", "WindowsSDKVersion")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--compiler-overlay")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if not args.child:
        out.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        with (out / "smoke.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, *sys.argv[1:], "--child"],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                status = {"exit_code": process.wait(timeout=600)}
            except subprocess.TimeoutExpired:
                status = {"timeout_seconds": 600}
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                else:
                    process.kill()
                process.wait(timeout=10)
        status["process_wall_seconds"] = time.perf_counter() - started
        (out / "launcher.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        print(json.dumps(status), flush=True)
        return status.get("exit_code", 1)
    out.mkdir(parents=True, exist_ok=True)
    if any(
        (out / name).exists()
        for name in ("inductor_cache", "triton_cache", "smoke.json")
    ):
        raise FileExistsError("Smoke worker requires fresh compiler caches.")
    report = {"device": args.device, "python": sys.version}
    start = time.perf_counter()
    try:
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "TORCHINDUCTOR_COMPILE_THREADS",
        ):
            os.environ[key] = "1"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(out / "inductor_cache")
        os.environ["TRITON_CACHE_DIR"] = str(out / "triton_cache")
        sys.path.insert(0, str(Path(args.overlay).resolve()))
        if args.compiler_overlay:
            sys.path.insert(0, str(Path(args.compiler_overlay).resolve()))
            # Triton discovers bundled CUDA relative to platlib; a --target
            # overlay needs this explicit process-local location instead.
            os.environ["CUDA_PATH"] = str(
                Path(args.compiler_overlay).resolve()
                / "triton/backends/nvidia"
            )
            report["cuda_path"] = os.environ["CUDA_PATH"]
        report["msvc"] = msvc_environment()
        report["cl"] = shutil.which("cl")
        imported = time.perf_counter()
        import torch

        report["torch_import_seconds"] = time.perf_counter() - imported
        report["torch"] = torch.__version__
        report["torch_path"] = torch.__file__
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float64)
        x = torch.linspace(-2, 2, 128, device=args.device, requires_grad=True)

        def objective(value):
            return torch.logsumexp(value.sin() + value.square(), dim=0)

        eager = objective(x)
        eager_grad = torch.autograd.grad(eager, x)[0]
        if args.device == "cuda":
            torch.cuda.synchronize()
        compile_start = time.perf_counter()
        compiled = torch.compile(objective, backend="inductor", fullgraph=True)
        actual = compiled(x)
        actual_grad = torch.autograd.grad(actual, x)[0]
        if args.device == "cuda":
            torch.cuda.synchronize()
        report["cold_forward_backward_seconds"] = (
            time.perf_counter() - compile_start
        )
        report["backward_type"] = type(actual.grad_fn).__name__
        report["value_error"] = abs((actual - eager).item())
        report["gradient_max_error"] = (
            (actual_grad - eager_grad).abs().max().item()
        )
        torch.testing.assert_close(actual, eager, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            actual_grad, eager_grad, rtol=1e-12, atol=1e-12
        )
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
    report["child_wall_seconds"] = time.perf_counter() - start
    (out / "smoke.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
