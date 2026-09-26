"""A minimal harness that meets the campaign contract, for the driver's tests.

``test_campaign_driver.py`` runs the scripts of ``dev/scripts/hpc/`` on it,
in a scratch git repository with stub Slurm commands
(``campaign_slurm_stubs.py``). It is also the shortest example of a harness
built on ``campaign_contract.py``. Its cases take no time: each writes one
text file, ``<tag>.out``, which its tracked copies hold with each case's
completion record and the summary.

Subcommands::

    python campaign_stub_harness.py prepare --out DIR [--cases N] \\
        [--fail I,J]
    python campaign_stub_harness.py cases --out DIR [--subset NAME]
    python campaign_stub_harness.py worker --out DIR --case LINE
    python campaign_stub_harness.py verify --out DIR
    python campaign_stub_harness.py summarize --out DIR

Case ``i`` is the line ``g<k>/c<iii> <i>``, five cases per group ``k``.
``--fail`` names the case indices whose worker raises after writing its
file. The subsets are ``odd``, ``first3`` and ``g0``, ``g1``, ... (one per
group). ``summarize`` is the finishing step. With ``STUB_FAKE_AFFINITY``
set, the worker records its CPU affinity as one physical core, which the
host a test runs on need not have. Two variables stop a case in the middle
of its run, after it has written its file: with ``STUB_HOLD`` set it waits
there (up to a minute) for a test to signal or kill it, and with
``STUB_SIGNAL_SELF`` set it sends itself SIGTERM, which is how a test
delivers the signal on Windows, where one process cannot send another
SIGTERM.
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import campaign_contract as contract  # noqa: E402

FILES = (
    "dev/scripts/campaign_stub_harness.py",
    "dev/scripts/campaign_contract.py",
)


def case_lines(count):
    return [f"g{(i - 1) // 5}/c{i:03d} {i}" for i in range(1, count + 1)]


def subset_indices(name, count):
    if name == "odd":
        return list(range(1, count + 1, 2))
    if name == "first3":
        return list(range(1, min(count, 3) + 1))
    if name.startswith("g") and name[1:].isdigit():
        group = int(name[1:])
        return [i for i in range(1, count + 1) if (i - 1) // 5 == group]
    raise SystemExit(f"no subset {name!r}")


def this_identity():
    record = contract.identity({"harness": ROOT}, FILES)
    if os.environ.get("STUB_FAKE_AFFINITY"):
        # campaign_slurm_stubs.ONE_CORE_AFFINITY, which this harness's
        # scratch repositories do not hold.
        record["host"]["cpu_affinity"] = {
            "cpus": [3],
            "physical_cores": ["0:3"],
            "core_threads": {"0:3": [3]},
            "job_cpuset": None,
        }
    return record


def read_manifest(out):
    return contract.read_json(Path(out) / "manifest.json")


def cmd_prepare(args):
    out = args.out.resolve()
    fail = [int(i) for i in args.fail.split(",") if i] if args.fail else []
    manifest = {
        "campaign": "stub",
        "cases": args.cases,
        "fail": fail,
        "identity": this_identity(),
        "site": contract.site_block(),
        "pip_freeze": contract.pip_freeze(),
        "finishing_steps": [["summarize"]],
        "tracked_copies": {
            "files": ["summary.json"],
            "cases": {"record": True, "artifacts": ["*.out"]},
        },
        "created": contract.now(),
    }
    path = out / "manifest.json"
    if path.exists():
        previous = contract.read_json(path)
        differing = contract.source_differences(
            manifest["identity"], previous["identity"]
        )
        if differing:
            raise SystemExit(
                f"{path} is prepared with another identity: {differing}"
            )
        return 0
    contract.write_json(path, manifest)
    print(f"{path}: {args.cases} cases", flush=True)
    return 0


def cmd_cases(args):
    manifest = read_manifest(args.out)
    lines = case_lines(manifest["cases"])
    if args.subset:
        for index in subset_indices(args.subset, len(lines)):
            print(f"{index} {lines[index - 1]}")
    else:
        for line in lines:
            print(line)
    return 0


def cmd_worker(args):
    out = args.out.resolve()
    manifest = read_manifest(out)
    tag = contract.case_tag(args.case)
    index = int(args.case.split()[1])

    def run(identity):
        path = out / f"{tag}.out"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"case {index}\n", encoding="utf-8")
        if os.environ.get("STUB_SIGNAL_SELF"):
            signal.raise_signal(signal.SIGTERM)
        if os.environ.get("STUB_HOLD"):
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                time.sleep(0.05)
            raise RuntimeError("the stub was held for a minute")
        if index in manifest["fail"]:
            raise RuntimeError(f"the stub fails case {index} on purpose")
        return [path], {"index": index}

    return contract.run_worker(
        out,
        args.case,
        manifest["identity"],
        this_identity,
        run,
        lambda: [out / f"{tag}.out"],
    )


def cmd_verify(args):
    out = args.out.resolve()
    manifest = read_manifest(out)
    lines = case_lines(manifest["cases"])
    tags = [contract.case_tag(line) for line in lines]

    def check(tag):
        contract.check_completion(
            out,
            tag,
            manifest["identity"],
            required=[f"{tag}.out"],
            node_feature=manifest["site"]["NODE_FEATURE"],
        )

    def partial(tag):
        return [f"{tag}.out"] if (out / f"{tag}.out").exists() else []

    groups = sorted({tag.split("/")[0] for tag in tags})
    stray = [
        f"{group}/{path.name}"
        for group in groups
        for path in sorted((out / group).glob("*.out"))
        if f"{group}/{path.stem}" not in tags
    ]
    report = contract.reconcile(out, lines, check, partial, stray=stray)
    contract.write_json(
        out / "verification.json", {"campaign": "stub", **report}
    )
    for case in report["cases"]:
        if case["status"] != "verified":
            print(json.dumps(case), flush=True)
    print(json.dumps(report["counts"]), flush=True)
    return report["exit_code"]


def cmd_summarize(args):
    out = args.out.resolve()
    report = contract.read_json(out / "verification.json")
    contract.write_json(out / "summary.json", {"counts": report["counts"]})
    print(f"summary of {out}: {report['counts']}", flush=True)
    return 0


def main(argv=None):
    sys.stdout.reconfigure(newline="\n")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--cases", type=int, default=12)
    prepare.add_argument("--fail")
    cases = sub.add_parser("cases")
    cases.add_argument("--out", type=Path, required=True)
    cases.add_argument("--subset")
    worker = sub.add_parser("worker")
    worker.add_argument("--out", type=Path, required=True)
    worker.add_argument("--case", required=True)
    for name in ("verify", "summarize"):
        sub.add_parser(name).add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    commands = {
        "prepare": cmd_prepare,
        "cases": cmd_cases,
        "worker": cmd_worker,
        "verify": cmd_verify,
        "summarize": cmd_summarize,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
