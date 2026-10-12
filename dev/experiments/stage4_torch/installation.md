# Prototype environment and installation evidence

Measured on 2026-09-08. This is an experiment environment, not a proposed
change to PyVBMC's dependencies or Python support policy.

## Tested configuration

Windows 11 build 26200, CPython 3.12.6, Intel Core Ultra 7 155H (16 physical
cores, 22 logical processors), NVIDIA RTX 4060 Laptop GPU (8 GiB), driver
576.80. NumPy 2.5.2 and SciPy 1.18.1 come from the existing editable development
environment. BLAS and Torch use one CPU thread. The gpyreg checkout is 1.1.0
at `a2f8ddce867f502e29717959cf0ff3529f598618`.

The PyVBMC checkout starts at
`bf43c20327f5684e6a22e801f4e6f3964a23cd98` on `dev-next`; its installed
distribution version metadata lags the checkout. File hashes and Git
revisions, rather than that stale distribution label, identify the source.
Full preflight results are in [environment.json](environment.json).

## Isolated Torch installations

The experiment uses two separate import overlays. It does not reinstall
NumPy, SciPy, gpyreg or PyVBMC, and does not change package requirements.

| Build | Role | Downloaded Torch wheel | Installed overlay including dependencies |
| --- | --- | ---: | ---: |
| Torch 2.14.0+cpu | Primary Torch CPU arm | About 124 MB (existing install log) | 634,416,193 bytes |
| Torch 2.14.0+cu126 | CUDA arm | About 2,602.8 MB | 4,350,399,261 bytes |

The CUDA installation took about 5 minutes 20 seconds, including a
3 minute 22 second wheel download. This is one network/machine observation,
not a general installation-time estimate. Its wheel bundles CUDA 12.6;
the existing NVIDIA driver was used. The CPU-only wheel needs no CUDA
runtime. The CUDA wheel also passed a CPU smoke, but its CPU timings are
not pooled with the CPU-only build's primary measurements.

Commands used or reproduced by the overlays:

```powershell
python -m pip install --target CPU_OVERLAY 'torch==2.14.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install --target CUDA_OVERLAY 'torch==2.14.0+cu126' --index-url https://download.pytorch.org/whl/cu126
```

Set `PYTHONPATH` to the chosen overlay followed by `dev/scripts` and the
repository root when invoking an experiment script. Use the ordinary
development environment without either overlay for the NumPy arm. The
machine-local overlays are under the ignored `dev/scripts/runs/` directory.
Raw installation reports/logs and environment manifests are retained there.

The tested CUDA wheel's metadata declares Python >=3.10. Declared
dependencies are filelock, typing-extensions >=4.10.0, setuptools >=77.0.3,
sympy >=1.13.3, networkx >=2.5.1, jinja2 and fsspec >=0.8.5. Only Windows and
Python 3.12 were exercised in this prototype; this is not a cross-platform
installation certification or a decision on PyVBMC's Python floor.

The reviewed [CPU wheel index](https://download.pytorch.org/whl/cpu/torch/)
lists 2.14.0 wheels for Python 3.10, 3.11 and 3.12 on Windows x86-64,
Linux x86-64/AArch64 (manylinux 2.28), and Apple silicon (macOS 14+).
It does not list a 2.14.0 Intel Mac wheel. These are availability checks,
not successful installation tests. The CUDA 12.6 index likewise supplies
the Windows x86-64 wheels used here; no Mac GPU feasibility was tested.

An explicit CPU wheel index is an installation instruction that a dependent
library cannot encode in its ordinary dependency metadata. A future hard
Torch dependency therefore needs an intentional pip/conda packaging policy.
The official [previous-version installation instructions](https://docs.pytorch.org/get-started/previous-versions/)
document the separate CPU and CUDA wheel indexes. PyTorch stopped publishing
its own conda packages starting with 2.6; conda-forge maintains its own
[CPU/GPU feedstock](https://github.com/conda-forge/pytorch-cpu-feedstock).
Those conda variants were not installed or timed here. See the
[PyTorch 2.6 announcement](https://pytorch.org/blog/pytorch2-6/) for the
official-channel change.

An untested conda CPU route is:

```console
conda create -n pyvbmc-torch -c conda-forge --strict-channel-priority python=3.12 pytorch-cpu
```

The [feedstock recipe](https://github.com/conda-forge/pytorch-cpu-feedstock/blob/main/recipe/meta.yaml)
selects a CPU-constrained PyTorch build through `pytorch-cpu`. Its reviewed
recipe was at 2.13.0, so this route is not a reproduction of the measured
2.14.0 pip environment. Resolve and record actual packages before using a
conda environment for future comparisons.

## First-use measurements

The first import immediately after installing the CUDA overlay took
10.736 seconds; a later fresh-process import of that build took 1.582
seconds. The CPU-only build imported in 1.446 seconds. These include
machine-specific DLL loading/cache effects. The first tiny CUDA allocation,
float64 gradient and triangular-solve smoke took 0.158 seconds; the CPU-only
smoke took 0.00188 seconds. The smoke gradient error was zero and the maximum
triangular-solve residual was 1.11e-16 on each device.

All numerical tensors are explicitly float64. Torch's global default dtype
was left at float32, providing an additional check that the prototype does
not depend on changing process-wide defaults. Complete-fit measurements
record imports, state reconstruction, CUDA context creation, GP extraction
and transfers separately from the variational fit itself.
