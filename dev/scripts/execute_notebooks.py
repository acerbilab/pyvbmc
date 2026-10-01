"""Execute the example notebooks and store their outputs.

The docs build renders the outputs stored in ``examples/*.ipynb`` without
executing the notebooks (``nb_execution_mode = "off"`` in
``docsrc/source/conf.py``), so the stored outputs come from this script. It
runs the notebooks one at a time, in order, in one scratch directory, where
Example 6 finds the posterior that Example 4 saves, through a kernel of the
Python environment given by ``--python``::

    python dev/scripts/execute_notebooks.py --python PATH/TO/python.exe
    python dev/scripts/execute_notebooks.py --python ... --only 4 6 --no-write

The script needs ``nbclient`` and ``nbformat``; the kernel's environment
needs ``ipykernel`` and every notebook's dependencies (Examples 7 to 9 need
the ``torch``, ``arviz`` and ``pymc`` extras and JAX). The kernel runs with
BLAS single-threaded, with this checkout first on ``PYTHONPATH``, with its
own IPython profile directory, with ``PLOTLY_RENDERER`` set so that
Example 2's figure is stored as HTML, which the docs render, beside its
Plotly JSON, which they skip, with ``PYVBMC_NO_UPDATE_REMINDER`` set so that
no old-release reminder enters the stored outputs, and, on a machine without
``g++``, with ``PYTENSOR_FLAGS`` stating that PyTensor has no C compiler.

After a notebook's cells, and after the cells that ``CHECKS`` names, the
script runs check cells that assert what the notebook's text says about its
results, and a cell that prints the versions the notebook ran with. It
removes them before storing the notebook, renumbers the executions, merges
adjacent pieces of one output stream, removes from the HTML of a Plotly
figure the MathJax it loads (which breaks the math of the docs page that
shows it), and writes the standard kernel metadata. A notebook is written back to ``examples/`` only when its cells
run without error and its checks pass; ``--no-write`` leaves every executed
notebook in the scratch directory only. Each run writes a record (commit,
environment, durations, versions, check results) to ``--record-dir``.
"""

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
KERNEL_NAME = "pyvbmc-notebooks"
KERNEL_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PLOTLY_RENDERER": "plotly_mimetype+notebook_connected",
    # The stored outputs ship with the release: no old-release reminder.
    "PYVBMC_NO_UPDATE_REMINDER": "1",
}
STANDARD_KERNELSPEC = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
CHECK_MARK = "__check__"
# The MathJax that Plotly's notebook HTML loads (it always asks for it from
# the CDN) breaks the math of a docs page that loads its own MathJax.
PLOTLY_MATHJAX = re.compile(
    r'<script[^>]*src="[^"]*mathjax[^"]*"[^>]*></script>\s*'
)
PROVENANCE_MARK = "__provenance__"

# What each notebook's text states about its results, as (anchor, code):
# the check runs right after the one cell whose source contains the anchor,
# or after the last cell when the anchor is None.
CHECKS = {
    1: [
        (
            None,
            """
            assert results["success_flag"]
            assert abs(results["elbo"] - lml_true) < 0.1, results["elbo"]
            """,
        ),
    ],
    2: [
        (
            None,
            """
            assert results["success_flag"]
            assert abs(results["elbo"] - lml_true) < 0.1, results["elbo"]
            """,
        ),
    ],
    3: [
        (
            'print(results["success_flag"])',
            """
            assert results["success_flag"] is False
            assert results["convergence_status"] == "no"
            assert results["r_index"] > 1, results["r_index"]
            """,
        ),
        (
            None,
            """
            assert results["success_flag"]
            assert results["convergence_status"] == "probable"
            assert results["r_index"] < 1, results["r_index"]
            assert results["func_count"] < 50 * (D + 2), results["func_count"]
            """,
        ),
    ],
    4: [
        (
            None,
            """
            assert all(success_flags), success_flags
            assert max(elbos) - min(elbos) < 0.1, elbos
            assert np.max(kl_matrix) < 0.25, kl_matrix
            """,
        ),
    ],
    5: [
        (
            "ub = 10 * np.ones((1, D))",
            """
            assert results["success_flag"]
            """,
        ),
        (
            "scs.multivariate_normal(mean=np.zeros(D)",
            """
            assert results["success_flag"]
            """,
        ),
    ],
    6: [
        (
            None,
            """
            assert results["success_flag"]
            kl = vbmc.vp.kl_div(vp2=noise_free_vp)
            assert np.max(kl) < 0.2, kl
            assert abs(results["elbo"] - noise_free_vp.stats["elbo"]) < 0.5
            """,
        ),
    ],
    7: [
        (
            None,
            """
            assert all(vp.stats["stable"] for vp in vps)
            run_elbos = [vp.stats["elbo"] for vp in vps]
            assert any(abs(e - np.log(0.5)) < 0.1 for e in run_elbos), run_elbos
            assert stacked.elbo_details["headline_method"] == "raw"
            assert abs(stacked.elbo) < 0.1, stacked.elbo
            share = np.mean(stacked.sample(20000)[:, 0] > 0)
            assert abs(share - 0.5) < 0.1, share
            """,
        ),
    ],
    8: [
        (
            None,
            """
            assert results["convergence_status"] == "probable"
            assert budget["used"] <= budget["limit"], budget
            nuts_summary = az.summary(
                nuts_data, var_names=["beta", "sigma"], kind="stats"
            )
            for name in vbmc_summary.index:
                gap = abs(
                    vbmc_summary.loc[name, "mean"]
                    - nuts_summary.loc[name, "mean"]
                )
                assert gap < 0.5 * nuts_summary.loc[name, "sd"], name
            """,
        ),
    ],
    9: [
        (
            None,
            """
            from scipy.special import logsumexp

            assert results["success_flag"]
            size = 400
            axes = [
                lo + (np.arange(size) + 0.5) * (hi - lo) / size
                for lo, hi in zip(lower_bounds, upper_bounds)
            ]
            grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
            log_lik = torch_log_likelihood(grid)
            weights = np.exp(log_lik - logsumexp(log_lik))
            mean = weights @ grid
            sd = np.sqrt(weights @ (grid - mean) ** 2)
            log_evidence = logsumexp(log_lik) - np.log(size**2)
            assert abs(results["elbo"] - log_evidence) < 0.1, results["elbo"]
            draws = parameter_draws_t.numpy()
            assert np.max(np.abs(draws.mean(0) - mean) / sd) < 0.2
            assert np.max(np.abs(draws.std(0) / sd - 1)) < 0.2
            """,
        ),
    ],
}

PROVENANCE = f"""
import json as _json
import sys as _sys
from importlib.metadata import version as _version

import gpyreg as _gpyreg
import pyvbmc as _pyvbmc

_provenance = {{
    "python": _sys.version.split()[0],
    "pyvbmc": _version("pyvbmc"),
    "pyvbmc_file": _pyvbmc.__file__,
    "gpyreg": _version("gpyreg"),
    "gpyreg_file": _gpyreg.__file__,
}}
for _name in (
    "numpy", "scipy", "matplotlib", "plotly", "torch", "jax", "pymc",
    "pytensor", "arviz",
):
    if _name in _sys.modules:
        _provenance[_name] = _version(_name)
print({PROVENANCE_MARK!r}, _json.dumps(_provenance))
"""


def _check_cell(code):
    """A cell that runs ``code`` and prints whether its assertions held."""
    body = textwrap.indent(textwrap.dedent(code).strip(), "    ")
    source = (
        "try:\n"
        f"{body}\n"
        f"    print({CHECK_MARK!r}, 'ok')\n"
        "except AssertionError as _error:\n"
        f"    print({CHECK_MARK!r}, 'FAILED', repr(_error))\n"
    )
    cell = nbformat.v4.new_code_cell(source)
    cell.metadata["inserted_by_execute_notebooks"] = True
    return cell


def _insert_checks(notebook, number):
    """Insert the check cells and the provenance cell; return their count."""
    inserted = 0
    for anchor, code in reversed(CHECKS.get(number, [])):
        if anchor is None:
            position = len(notebook.cells)
        else:
            matches = [
                i
                for i, cell in enumerate(notebook.cells)
                if cell.cell_type == "code"
                and anchor in cell.source
                and not cell.metadata.get("inserted_by_execute_notebooks")
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Example {number}: the check anchor {anchor!r} matches "
                    f"{len(matches)} code cells, not one."
                )
            position = matches[0] + 1
        notebook.cells.insert(position, _check_cell(code))
        inserted += 1
    provenance = nbformat.v4.new_code_cell(PROVENANCE)
    provenance.metadata["inserted_by_execute_notebooks"] = True
    notebook.cells.append(provenance)
    return inserted + 1


def _collect_and_remove_inserted(notebook):
    """Read the check results and versions, then remove the inserted cells."""
    checks, provenance = [], None
    kept = []
    for cell in notebook.cells:
        if not cell.metadata.get("inserted_by_execute_notebooks"):
            kept.append(cell)
            continue
        text = "".join(
            output.get("text", "")
            for output in cell.get("outputs", [])
            if output.output_type == "stream"
        )
        for line in text.splitlines():
            if line.startswith(CHECK_MARK):
                checks.append(line[len(CHECK_MARK) :].strip())
            elif line.startswith(PROVENANCE_MARK):
                provenance = json.loads(line[len(PROVENANCE_MARK) :])
    notebook.cells = kept
    return checks, provenance


def _tidy(notebook):
    """Renumber executions, merge adjacent stream pieces, drop timings."""
    count = 0
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
        if cell.cell_type != "code":
            continue
        count += 1
        cell.execution_count = count
        merged = []
        for output in cell.outputs:
            if (
                output.output_type == "stream"
                and merged
                and merged[-1].output_type == "stream"
                and merged[-1].name == output.name
            ):
                merged[-1].text += output.text
            else:
                if output.output_type == "execute_result":
                    output.execution_count = count
                merged.append(output)
        cell.outputs = merged
    notebook.metadata.pop("interpreter", None)
    notebook.metadata["kernelspec"] = dict(STANDARD_KERNELSPEC)


def _strip_plotly_mathjax(notebook):
    """Remove the MathJax script from the HTML of Plotly figures."""
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if (
                "application/vnd.plotly.v1+json" in data
                and "text/html" in data
            ):
                html = data["text/html"]
                if not isinstance(html, str):
                    html = "".join(html)
                data["text/html"] = PLOTLY_MATHJAX.sub("", html)


def _output_problems(number, notebook):
    """What the stored outputs must not show, or must show."""
    problems = []
    texts = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                texts.append(output.text)
            elif "data" in output:
                texts.append(str(output.data.get("text/plain", "")))
    if any("np.float64(" in text for text in texts):
        problems.append("an output shows a NumPy scalar as np.float64(...)")
    if number == 2:
        has_html_figure = any(
            "plotly" in str(output.get("data", {}).get("text/html", ""))
            and "application/vnd.plotly.v1+json" in output.get("data", {})
            for cell in notebook.cells
            for output in cell.get("outputs", [])
        )
        if not has_html_figure:
            problems.append("the Plotly figure has no HTML output")
    if any(
        PLOTLY_MATHJAX.search(str(output.get("data", {}).get("text/html", "")))
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if "application/vnd.plotly.v1+json" in output.get("data", {})
    ):
        problems.append("a Plotly figure's HTML still loads MathJax")
    return problems


def _kernel_spec(data_dir, python):
    kernel_dir = data_dir / "kernels" / KERNEL_NAME
    kernel_dir.mkdir(parents=True, exist_ok=True)
    env = dict(KERNEL_ENV, PYTHONPATH=str(ROOT))
    if shutil.which("g++") is None:
        # Without a C compiler PyTensor (Example 8) warns about it on import,
        # and the warning would be stored; stating that there is none keeps
        # the same behaviour without the warning.
        env["PYTENSOR_FLAGS"] = "cxx="
    spec = {
        "argv": [
            str(python),
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ],
        "display_name": "PyVBMC notebooks",
        "language": "python",
        "env": env,
    }
    (kernel_dir / "kernel.json").write_text(
        json.dumps(spec, indent=1), encoding="utf-8"
    )
    return env


def _git(*args):
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _notebook_path(number):
    matches = sorted(EXAMPLES.glob(f"pyvbmc_example_{number}_*.ipynb"))
    if len(matches) != 1:
        raise RuntimeError(f"Example {number}: found {len(matches)} files.")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="the interpreter the kernel runs (default: this one)",
    )
    parser.add_argument(
        "--only",
        type=int,
        nargs="+",
        default=sorted(CHECKS),
        help="the example numbers to run (default: all)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="leave the executed notebooks in the scratch directory only",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=ROOT
        / "dev/scripts/runs"
        / f"notebooks_{datetime.date.today():%Y%m%d}",
        help="where the record and the scratch directory go",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="the limit on one cell's execution, in seconds",
    )
    args = parser.parse_args()
    numbers = sorted(set(args.only))
    if 6 in numbers and 4 not in numbers:
        parser.error("Example 6 loads the posterior that Example 4 saves.")

    stamp = f"{time.time():.0f}"
    args.record_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(
        tempfile.mkdtemp(prefix=f"work_{stamp}_", dir=args.record_dir)
    )
    data_dir = workdir / "_jupyter"
    kernel_env = _kernel_spec(data_dir, args.python.resolve())
    os.environ["JUPYTER_PATH"] = str(data_dir)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(data_dir / "runtime")
    os.environ["IPYTHONDIR"] = str(data_dir / "ipython")

    record = {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "tracked_changes": _git("status", "--porcelain", "--untracked=no"),
        "host": platform.node(),
        "kernel_python": str(args.python),
        "kernel_env": kernel_env,
        "workdir": str(workdir),
        "written": not args.no_write,
        "notebooks": {},
    }
    record_path = args.record_dir / f"record_{stamp}.json"
    print(f"Record: {record_path}", flush=True)
    print(f"Scratch directory: {workdir}", flush=True)

    all_passed = True
    for number in numbers:
        source_path = _notebook_path(number)
        notebook = nbformat.read(source_path, as_version=4)
        _insert_checks(notebook, number)
        entry = {"file": source_path.name}
        start = time.monotonic()
        print(f"Example {number}: {source_path.name} ...", flush=True)
        try:
            NotebookClient(
                notebook,
                timeout=args.timeout,
                kernel_name=KERNEL_NAME,
                record_timing=False,
                resources={"metadata": {"path": str(workdir)}},
            ).execute()
            entry["error"] = None
        except CellExecutionError as error:
            entry["error"] = str(error).splitlines()[-1] if str(error) else ""
        entry["seconds"] = round(time.monotonic() - start, 1)
        checks, provenance = _collect_and_remove_inserted(notebook)
        entry["checks"] = checks
        entry["provenance"] = provenance
        _tidy(notebook)
        _strip_plotly_mathjax(notebook)
        entry["output_problems"] = _output_problems(number, notebook)
        executed_path = workdir / source_path.name
        nbformat.write(notebook, executed_path)
        entry["bytes"] = executed_path.stat().st_size
        passed = (
            entry["error"] is None
            and bool(checks)
            and all(check == "ok" for check in checks)
            and not entry["output_problems"]
        )
        entry["passed"] = passed
        all_passed &= passed
        if passed and not args.no_write:
            shutil.copyfile(executed_path, source_path)
            entry["written_to"] = str(source_path.relative_to(ROOT))
        record["notebooks"][number] = entry
        record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        verdict = "passed" if passed else "FAILED"
        print(
            f"Example {number}: {verdict} in {entry['seconds']} s, "
            f"{entry['bytes'] / 1e6:.2f} MB; checks {checks}; "
            f"error {entry['error']}; problems {entry['output_problems']}",
            flush=True,
        )
    shutil.rmtree(data_dir, ignore_errors=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
