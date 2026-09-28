"""Promote the reference population of the release gate to the golden reference.

The reference that replaces ``reference_990_20260913`` is the after arm of
the release gate's population campaigns (``dev/plans/slurm-benchmark-
support.md``): every configuration of the ``production`` suite at seeds
0-99, run on a Slurm cluster by ``population_run.py``'s array mode with the
release code. Exact replay depends on the machine and its BLAS, so the
traces ``golden_replay.py`` compares with are the reference's replay
fingerprints instead: one run at seed 0 of each configuration made with the
same code on the developer's machine, and the six seeded gate runs
(``seeded_gate_runs.py``) recorded twice there. ``reference_join.py``
cannot make the promotion, since it extends a reference and refuses any
overlap with it. The procedure, from the repository root, on the machine
of the fingerprints, as one process at a time::

    python -u dev/scripts/seeded_gate_runs.py run --out GATES
    python -u dev/scripts/reference_promote.py fingerprints \\
        --after AFTER --out FINGERPRINTS
    python dev/scripts/reference_promote.py prepare --after AFTER \\
        --assessment ASSESSMENT --accepted-assessment SHA256 \\
        --fingerprints FINGERPRINTS --gate-runs GATES --record RECORD \\
        [--name NAME] [--rulings FILE]
    python -u dev/scripts/reference_promote.py replay --record RECORD \\
        --out REPLAY
    python dev/scripts/reference_promote.py publish --record RECORD \\
        --final-replay REPLAY [--public-traces TEXT]

``AFTER`` is the after arm's redacted tracked copies,
``dev/experiments/release_gate_<date>/population_after/``
(``campaign_contract.redact``), tracked by git: a campaign directory is
refused, since ``publish`` copies its sidecars into the repository.
``ASSESSMENT`` is the output of ``analyze_population_run.py --arms BEFORE
AFTER``, whose SHA-256 with LF line endings the PI accepted
(``--accepted-assessment``; a refusal prints it), and ``RECORD`` the
promotion's record, ``dev/golden/promotion_<date>/``. These three lie in
the checkout. The reference is named ``reference_<runs>_<date>``, the date
of the first ``prepare``, unless ``--name`` names it.

Every step that runs code first checks that this process runs the after
arm's code (:func:`check_code`): ``HEAD`` equals the after arm's commit in
:data:`NUMERIC_PATHS` (the package, but for its tests and S-VBMC, which no
VBMC run imports, and the harness files that build a run), the working tree
equals ``HEAD`` there, and the imported gpyreg is the after arm's commit,
clean.

``fingerprints`` runs ``golden_replay.py`` on every configuration of the
after arm at seed 0, with the after arm as ``--sidecars`` and no baseline
traces, so that each run is judged against the population's accuracy
envelope alone; its ``--out`` holds the fingerprints and the report.

``prepare`` changes nothing that is tracked but the record. It checks:

- that ``dev/golden/baseline/`` holds the previous reference, as its
  promotion's manifest says, and, where this machine holds them, the
  previous reference's traces;
- the after arm: its redaction record, its allocation (the whole
  ``production`` suite at seeds 0-99), its options
  (``population_run.DEFAULT_OPTIONS``), that it ran its harness checkout's
  own package from clean trees, its verification report, which must still
  describe it, the record and rescored metrics of every verified case
  (``analyze_population_run.load_array_campaign``), and that every case it
  does not place as verified has a ruling in ``--rulings`` (a JSON object
  of case tags and rulings), which leaves that case out of the reference;
- that the assessment is the accepted file and compares this after arm;
- the fingerprints: every configuration at seed 0, each a pair with no
  error file and an intact archive, run with one BLAS thread, historical
  calibration budgets and the options of the after arm's runs of it, from
  clean checkouts of the after arm's code and gpyreg, completed with finite
  metrics and consistent files, and judged by the replay against the after
  arm's envelopes (:func:`check_fingerprints`): some may lie outside their
  envelopes, as some of the population's own runs lie outside theirs, and
  more of them than that rate makes plausible refuses (a binomial tail
  below :data:`FINGERPRINT_ALPHA`);
- the gate runs (``seeded_gate_runs.check_record``): the gate script,
  identical recordings from clean checkouts of the after arm's code and
  gpyreg, on this host.

It then copies the fingerprints and the gate runs' directory into
``dev/scripts/runs/golden/<name>_fingerprints/`` (gitignored; a copy that
exists must equal its source) with their summary, and writes the record:
``previous_reference_README.md`` (``dev/golden/README.md`` as it stands),
the fingerprints' replay report, the gate runs' record, the even/odd null
check of the population, ``sha256_manifest.json`` (the sidecars to
publish, the fingerprints and the gate runs' files; JSON and Markdown
hashed with LF line endings) and ``validation.json``. It exits nonzero if
the null check flags a configuration, having written everything.

``replay`` runs ``golden_replay.py`` on the defaults of the promotion
(:data:`DEFAULT_CONFIGS`, seed 0) against the prepared fingerprints and the
after arm's envelopes. Write the record's ``README.md`` from the record
before ``publish``: the generated documents link to it.

``publish`` requires the prepared record, its README naming the reference,
the README of the release gate's records beside ``AFTER``, which names the
campaign's archive, a replay of the defaults that is identical in every
case from ``HEAD``, and the previous reference still in
``dev/golden/baseline/``, and computes everything it writes before it
writes anything. It then replaces the sidecars and the summary in
``dev/golden/baseline/`` with the reference's; rewrites, together,
``golden_replay.py``'s ``DEFAULT_BASELINE`` and ``DEFAULT_CONFIGS``, the
``Trajectories`` entry of ``AGENTS.md``, the current reference's section
and the ``golden_replay.py`` entry of ``dev/README.md``, and
``dev/golden/README.md``; names the reference where ``dev/README.md`` and
``analyze_population_run.py`` say what ``dev/golden/baseline/`` holds, and
drops the previous population's examples from ``golden_replay.py``; copies
this script into the record as ``promote.py``; and removes its entry from
``dev/README.md``. ``--public-traces`` names where the population's traces
are published, for ``dev/golden/README.md``. Each rewritten passage must be
the text this script was written against (:data:`PASSAGES`, compared by
SHA-256 with LF line endings; ``test_reference_promote.py`` checks the
checkout's files), so that an edit made to one since is carried into its
template rather than overwritten. The commit is the operator's: it removes
this script and ``test_reference_promote.py``, whose promotion is done, and
carries by hand into the documents of :data:`HAND_EDITS` and the machine's
``dev/scripts/runs/LOCAL.md`` what they say of the reference.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: The checkout whose history the commit checks read.
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import population_run as runner  # isort: skip  (single-thread env first)

# isort: split
import analyze_population_run as analysis
import golden_replay
import numpy as np
import reference_join as join
import seeded_gate_runs as gates

contract = runner.contract
golden_trace = runner.golden_trace

#: The allocation of the reference: every configuration of this suite, at
#: these seeds.
SUITE = "production"
LABELS = None
SEEDS = "0-99"
#: The configurations ``golden_replay.py`` replays by default after the
#: promotion, at seed 0: those of before, with the noisy Rosenbrock at the
#: production budget.
DEFAULT_CONFIGS = (
    "normal_D5",
    "banana_D2",
    "halfnormal_D2",
    "cigar_D4",
    "rosenbrock_D2_noise1_production",
)
PREVIOUS = "reference_990_20260913"
PREVIOUS_RECORD = "dev/golden/promotion_20260913"
BASELINE = "dev/golden/baseline"
GOLDEN_RUNS = "dev/scripts/runs/golden"
#: The code whose equality between two commits makes their runs the same
#: runs, as git pathspecs: the package, but for its tests and S-VBMC, which
#: no VBMC run imports, and the harness files that build a run.
NUMERIC_PATHS = (
    "pyvbmc",
    ":(exclude)pyvbmc/testing",
    ":(exclude)pyvbmc/svbmc",
    "dev/scripts/golden_trace.py",
    "dev/scripts/benchmark_targets.py",
    "dev/scripts/profile_run.py",
    "dev/scripts/data",
)
#: The ``--baseline`` of the fingerprints' replay, a directory that does not
#: exist, so that no trace is compared and every run meets the envelope.
NO_TRACES = "no-traces"
#: The probability under which the count of fingerprints outside their
#: envelopes refuses the promotion: the binomial tail of that count, at the
#: rate at which the population's own runs lie outside theirs.
FINGERPRINT_ALPHA = 0.01
#: What the assessment's before arm is, by its package commit.
BEFORE_CODE = {
    "f91fdf0": "the code at the start of the port correctness review"
}
NORMALIZATION = join.NORMALIZATION
USABLE = join.USABLE
#: The passages ``publish`` rewrites: file, the start of their first line
#: and of the line after them (None: the whole file), and the SHA-256 of
#: the text this script was written against, with LF line endings.
PASSAGES = {
    "agents": (
        "AGENTS.md",
        "- **Trajectories.**",
        "- **S-VBMC.**",
        "c36b469fea993ffeeec8fcf733fa40550fcb6b405e9d28d848051237d926c57d",
    ),
    "readme_reference": (
        "dev/README.md",
        "The golden suite's noisy configurations pin",
        "[`golden/README.md`](golden/README.md) is the human-facing",
        "8def777ce5c2f4db89b0bf656a2c6f95fe4eabc11d28a9b31772e930f182730a",
    ),
    "readme_replay": (
        "dev/README.md",
        "- `scripts/golden_replay.py` —",
        "- `scripts/regenerate_baseline.sh` —",
        "ebcfd9c777bfd65371ff9a78760192f29fe4bf04340ac665fb76df7a2dd118ef",
    ),
    "golden_readme": (
        "dev/golden/README.md",
        None,
        None,
        "1cfc0d84325c8a51591498cbbf6469680d5c1a835dee00ac9798711688ea2446",
    ),
    "replay_configs": (
        "dev/scripts/golden_replay.py",
        "# Cheap configurations covering",
        "ACCURACY = (",
        "2f443d0b88df41fa4b4b5f0c14cb63e074149415ca51a98f5b23be1b8aae95e0",
    ),
}
#: This script's own entry in ``dev/README.md``, which ``publish`` removes.
OWN_ENTRY = (
    "dev/README.md",
    "- `scripts/reference_promote.py` —",
    "- `scripts/seeded_gate_runs.py` —",
)
#: The lines ``publish`` replaces where they name the previous reference or
#: its population (file, lines, their replacement, a template).
REPLACEMENTS = (
    (
        "dev/scripts/golden_replay.py",
        f'DEFAULT_BASELINE = DEFAULT_BASELINE / "{PREVIOUS}"\n',
        'DEFAULT_BASELINE = DEFAULT_BASELINE / "{traces}"\n',
    ),
    (
        "dev/scripts/golden_replay.py",
        "  known failure, e.g. ``student_D4`` seed 19).\n",
        "  known failure).\n",
    ),
    (
        "dev/scripts/golden_replay.py",
        "        # (the reference seed may itself be the population's far "
        "outlier,\n"
        "        # e.g. student_D4 seed 19), so the envelope applies only to "
        "runs\n",
        "        # (the reference seed may itself be the population's far "
        "outlier),\n"
        "        # so the envelope applies only to runs\n",
    ),
    (
        "dev/README.md",
        f"  holds `{PREVIOUS}`, so the default command stops at that\n",
        "  holds `{name}`, so the default command stops at that\n",
    ),
    (
        "dev/README.md",
        "seed allocation differs from the extended reference and it masks "
        "comparison\n"
        "failures, so it must not be used to republish this reference. "
        "Follow the\n"
        "extension record's commands and fail-fast checks instead. The "
        "benchmark follows\n",
        "seed allocation differs from the reference's and it masks "
        "comparison\n"
        "failures, so it must not be used to republish this reference, which "
        "the\n"
        "release gate's campaigns made. The benchmark follows\n",
    ),
    (
        "dev/scripts/analyze_population_run.py",
        f"  ``{PREVIOUS}``, so the default command line stops at that\n",
        "  ``{name}``, so the default command line stops at that\n",
    ),
    (
        "dev/scripts/analyze_population_run.py",
        f"#: ``{PREVIOUS}`` now, which :func:`verify_reference` refuses.\n",
        "#: ``{name}`` now, which :func:`verify_reference` refuses.\n",
    ),
)
#: What else names the current reference and is carried into the new one
#: by hand, since each is a record or a plan whose other text the promotion
#: does not know; and the record-only scripts that read or hash
#: ``golden_replay.py``'s defaults, which name the previous reference's
#: directories from then on.
HAND_EDITS = (
    "dev/TODO.md",
    "dev/plans/modernization-roadmap.md",
    "dev/plans/final-population-benchmark.md",
    "dev/2026-09-06-pyvbmc-1.5-overview.md",
    "dev/scripts/validate_viqr_sinh.py (record-only; reads the defaults)",
    "dev/scripts/noisy_acq_inference.py (record-only; hashes golden_replay.py)",
)
VALIDATION = "validation.json"
MANIFEST = "sha256_manifest.json"
NUMBER_WORDS = (
    "no",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)


class PromotionError(RuntimeError):
    """A check of the promotion failed; nothing after it runs."""


def check(condition, message):
    if not condition:
        raise PromotionError(message)


# --------------------------------------------------------------------------
# Git, and the checks that tests stand in for
# --------------------------------------------------------------------------


def git(*args):
    return contract.git(REPO, *args)


def head_commit():
    return git("rev-parse", "HEAD")


def full_commit(commit):
    """The full commit a short one names, in this checkout's history."""
    return git("rev-parse", "--verify", f"{commit}^{{commit}}")


def numerics_differ(a, b):
    """The paths of :data:`NUMERIC_PATHS` in which commits ``a`` and ``b``
    differ."""
    return git(
        "diff",
        "--name-only",
        full_commit(a),
        full_commit(b),
        "--",
        *NUMERIC_PATHS,
    ).splitlines()


def working_changes():
    """What of :data:`NUMERIC_PATHS` differs from ``HEAD`` in the working
    tree, untracked files included."""
    return git("status", "--porcelain", "--", *NUMERIC_PATHS).splitlines()


def is_tracked(path):
    """Whether git tracks the file ``path`` in this checkout."""
    try:
        git("ls-files", "--error-unmatch", str(Path(path).resolve()))
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def clean_required():
    """Whether the checkouts of the runs must be clean (tests say not)."""
    return True


def same_commit(a, b):
    """Whether two commits, either one abbreviated, are the same."""
    a, b = str(a or ""), str(b or "")
    return len(min(a, b, key=len)) >= 7 and (
        a.startswith(b) or b.startswith(a)
    )


def same_float(a, b):
    """Equal floats, NaN equal to NaN."""
    a, b = float(a), float(b)
    return a == b or (a != a and b != b)


def check_code(source, what):
    """This process runs the after arm's code (module docstring).

    ``source`` is the after arm's source identity; returns ``HEAD``.
    """
    from profile_run import module_source

    head = head_commit()
    differing = numerics_differ(head, source["trees"]["pyvbmc"]["commit"])
    check(
        not differing,
        f"{what}: HEAD's code differs from the after arm's in {differing}",
    )
    gpyreg = (module_source("gpyreg") or {}).get("git") or {}
    check(
        same_commit(gpyreg.get("sha"), source["trees"]["gpyreg"]["commit"]),
        f"{what}: the imported gpyreg ({gpyreg.get('sha')}) is not the after "
        f"arm's ({source['trees']['gpyreg']['commit']})",
    )
    if clean_required():
        changes = working_changes()
        check(
            not changes,
            f"{what}: the working tree differs from HEAD: {changes}",
        )
        check(not gpyreg.get("dirty"), f"{what}: the gpyreg checkout is dirty")
    return head


# --------------------------------------------------------------------------
# Files and text
# --------------------------------------------------------------------------


def sha256(path, normalize_text=False):
    return join.sha256(path, normalize_text)


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bytes_sha256(data):
    """The SHA-256 of text bytes with their line endings read as LF."""
    data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_text(path):
    """``(text with LF line endings, the file's line ending)``."""
    data = Path(path).read_bytes().decode("utf-8")
    return data.replace("\r\n", "\n"), ("\r\n" if "\r\n" in data else "\n")


def write_text(path, text, eol="\n"):
    Path(path).write_bytes(text.replace("\n", eol).encode("utf-8"))


def read_json(path):
    return contract.read_json(path)


def write_json(path, value):
    join.write_json(path, value)


def inside(root, path, what):
    """``path`` relative to ``root`` as POSIX; it must lie in the checkout."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        raise PromotionError(
            f"{what} {path} lies outside the checkout {root}, where the "
            "records that link to it are"
        ) from None


def span(text, start, end):
    """Where the passage from the line beginning with ``start`` up to the
    line beginning with ``end`` lies in ``text``; the whole text for a
    ``start`` of None."""
    if start is None:
        return 0, len(text)
    found = [m.start() for m in re.finditer(rf"(?m)^{re.escape(start)}", text)]
    check(len(found) == 1, f"{len(found)} lines begin with {start!r}")
    following = re.compile(rf"(?m)^{re.escape(end)}").search(
        text, found[0] + 1
    )
    check(
        following is not None, f"no line begins with {end!r} after {start!r}"
    )
    return found[0], following.start()


def passage(root, key):
    """The passage :data:`PASSAGES` names, as the checkout holds it."""
    file, start, end, _ = PASSAGES[key]
    text, _ = read_text(Path(root) / file)
    i, j = span(text, start, end)
    return text[i:j]


def passage_problems(root):
    """The passages that are not the text this script was written against."""
    problems = []
    for key, (file, start, _, expected) in PASSAGES.items():
        found = text_sha256(passage(root, key))
        if found != expected:
            problems.append(
                f"{file}: the passage {start or '(the file)'!r} has changed "
                "since this script was written; carry the change into its "
                f"template and its SHA-256 into PASSAGES ({found})"
            )
    return problems


#: What opens a Markdown block at the start of a line, which a wrapped line
#: must not begin with.
BLOCK_MARKER = re.compile(r"(\d+[.)]|[-+*>]|#+)(\s|$)")


def reflow(text, width=79):
    """Wrap the paragraphs and bullets of Markdown ``text`` at ``width``.

    Headings, tables and fenced code are kept as they are. `` + `` does not
    break (``75 (D + 2)``), and no wrapped line begins with what would open
    a Markdown block (a list marker, a heading, a quote).
    """
    out, block, fenced = [], [], False

    def flush():
        if not block:
            return
        bullet = re.match(r"(\s*)- ", block[0])
        lead = bullet.group(1) if bullet else ""
        indent = lead + "  " if bullet else lead
        words = " ".join(line.strip() for line in block)
        lines = textwrap.wrap(
            words.replace(" + ", "\0+\0"),
            width=width,
            initial_indent=lead,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        i = 1
        while i < len(lines):
            body = lines[i][len(indent) :]
            if BLOCK_MARKER.match(body):
                first, _, rest = body.partition(" ")
                lines[i - 1] += " " + first
                if rest:
                    lines[i] = indent + rest
                else:
                    del lines[i]
                continue
            i += 1
        out.extend(line.replace("\0", " ") for line in lines)
        block.clear()

    for line in text.split("\n"):
        if line.startswith("```"):
            flush()
            fenced = not fenced
            out.append(line)
        elif fenced or line.startswith(("#", "|")) or not line.strip():
            flush()
            out.append(line)
        elif re.match(r"\s*- ", line):
            flush()
            block.append(line)
        else:
            block.append(line)
    flush()
    return "\n".join(out)


# --------------------------------------------------------------------------
# The previous reference
# --------------------------------------------------------------------------


def previous_manifest(root):
    return Path(root) / PREVIOUS_RECORD / MANIFEST


def check_active_baseline(root):
    """``dev/golden/baseline/`` holds the previous reference exactly."""
    manifest = read_json(previous_manifest(root))
    baseline = Path(root) / BASELINE
    check(
        join.tags_of(baseline, ".json") == set(manifest["files"]),
        f"{BASELINE} does not hold the sidecars of {PREVIOUS}",
    )
    for tag, hashes in manifest["files"].items():
        check(
            sha256(baseline / f"{tag}.json", True) == hashes["json_sha256"],
            f"{BASELINE}/{tag}.json is not {PREVIOUS}'s",
        )
    check(
        sha256(baseline / "summary.md", True) == manifest["summary_sha256"],
        f"{BASELINE}/summary.md is not {PREVIOUS}'s",
    )
    return manifest


def check_previous_traces(root):
    """The previous reference's traces, where this machine holds them."""
    traces = Path(root) / GOLDEN_RUNS / PREVIOUS
    if not traces.is_dir():
        return "absent from this machine"
    join.verify_previous(traces, previous_manifest(root))
    return "verified against its manifest"


# --------------------------------------------------------------------------
# The after arm
# --------------------------------------------------------------------------


def expected_allocation():
    labels = None if LABELS is None else list(LABELS)
    return runner.allocation(SUITE, labels, SEEDS)


def read_manifest(after):
    """The after arm's manifest, once it is known to be the redacted
    tracked copies of a campaign of the expected allocation."""
    after = Path(after)
    check(
        contract.read_redaction(after) is not None,
        f"{after} holds no {contract.REDACTION}: AFTER is the after arm's "
        "redacted tracked copies (campaign_contract.redact), not its "
        "campaign directory, whose sidecars hold the operator's paths",
    )
    check(
        is_tracked(after / "manifest.json"),
        f"{after} is not tracked by git: the documents that publish writes "
        "link to it",
    )
    manifest = read_json(after / "manifest.json")
    check(
        manifest.get("harness") == "population_run",
        f"{after} is not a campaign of population_run.py",
    )
    check(
        manifest["allocation"] == expected_allocation(),
        f"{after} does not allocate the {SUITE} suite at seeds {SEEDS}",
    )
    return manifest


def read_after(after, rulings):
    """The after arm, checked (module docstring); the verified cases' rows,
    sidecars and the SHA-256 of their files."""
    manifest = read_manifest(after)
    check(
        manifest["options"] == runner.DEFAULT_OPTIONS,
        f"{after} ran with other options than population_run's defaults",
    )
    trees = manifest["identity"]["source"]["trees"]
    check(
        trees["pyvbmc"] == trees["harness"],
        f"{after} did not run its harness checkout's own package",
    )
    if clean_required():
        dirty = sorted(
            name for name, tree in trees.items() if not tree["clean"]
        )
        check(not dirty, f"{after} ran from the dirty trees {dirty}")
    campaign = analysis.load_array_campaign(
        after, after, analysis.read_rescoring(after)
    )
    unequal = sorted(
        stem
        for stem, row in campaign["rows"].items()
        if not all(row["equal_to_in_run"].values())
    )
    check(not unequal, f"rescored metrics differ from the runs' in {unequal}")
    unverified = {
        f"{label}/{stem}"
        for stem, (label, _, status) in campaign["cases"].items()
        if status != "verified"
    }
    check(
        set(rulings) == unverified,
        f"cases not verified without a ruling: "
        f"{sorted(unverified - set(rulings))}; rulings of cases that are "
        f"verified or unknown: {sorted(set(rulings) - unverified)}",
    )
    sidecars, files = {}, {}
    for stem, row in sorted(campaign["rows"].items()):
        rel = runner.case_files(row["label"], row["seed"])["sidecar"]
        path = Path(after) / rel
        sidecars[stem] = json.loads(path.read_text(encoding="utf-8"))
        files[stem] = {
            "file": rel,
            "sha256": sha256(path, True),
            "source_sha256": contract.source_sha256(after, rel),
        }
    return campaign, sidecars, files


def population_of(sidecars):
    """The reference population as ``golden_trace.load_population`` reads
    it from a directory of these sidecars, in the order of their files."""
    entries = {}
    for stem in sorted(sidecars, key=lambda stem: f"{stem}.json"):
        side = sidecars[stem]
        entry = entries.setdefault(
            side["label"], {"seeds": [], "rows": [], "fails": 0}
        )
        entry["seeds"].append(side["seed"])
        entry["rows"].append(side["final"])
    return golden_trace.merge_populations([entries])


def outcomes(sidecars, labels):
    """Per configuration: runs, converged, reached budget, usable."""
    table = {}
    for label in labels:
        sides = [s for s in sidecars.values() if s["label"] == label]
        budget = (
            sides[0]["effective_options"]["max_fun_evals"] if sides else None
        )
        finals = [s["final"] for s in sides]
        table[label] = {
            "runs": len(sides),
            "noisy": bool(sides and sides[0].get("noise_sd")),
            "max_fun_evals": budget,
            "converged": sum(bool(f["success_flag"]) for f in finals),
            "reached_budget": sum(
                budget is not None and f["func_count"] >= budget
                for f in finals
            ),
            "usable": sum(
                all(f[k] < v for k, v in USABLE.items()) for f in finals
            ),
        }
    return table


# --------------------------------------------------------------------------
# The assessment, the fingerprints and the gate runs
# --------------------------------------------------------------------------


def check_assessment(directory, accepted, campaign):
    path = Path(directory) / "assessment.json"
    digest = sha256(path, True)
    check(
        digest == accepted,
        f"{path} is not the accepted assessment (its SHA-256 with LF line "
        f"endings is {digest})",
    )
    assessment = read_json(path)
    candidate = assessment["arms"]["candidate"]
    check(
        candidate["name"] == campaign["name"]
        and candidate["source"] == campaign["identity"]["source"],
        f"{path} does not assess {campaign['name']} as its candidate",
    )
    check(
        candidate["verified"] == len(campaign["rows"]),
        f"{path} assesses another count of verified cases",
    )
    reference = assessment["arms"]["reference"]
    tests = assessment["confirmatory_tests"]
    return {
        "sha256": digest,
        "comparison_sha256": sha256(Path(directory) / "comparison.md", True),
        "before": {
            "name": reference["name"],
            "arm": reference["arm"],
            "pyvbmc": reference["source"]["trees"]["pyvbmc"]["commit"],
            "gpyreg": reference["source"]["trees"]["gpyreg"]["commit"],
            "verified": reference["verified"],
        },
        "paired_cases": assessment["paired_cases"],
        "flagged_configurations": assessment["flagged_configurations"],
        "confirmatory_tests": len(tests),
        "confirmatory_rejected": sum(
            bool(t.get("holm_rejected")) for t in tests
        ),
        "usability_gains": len(assessment["usability_gains"]),
        "usability_losses": len(assessment["usability_losses"]),
    }


def option_differences(fingerprint, cluster):
    """The options in which a fingerprint's run and the after arm's runs of
    its configuration differ.

    Each sidecar holds the effective value of the options it requested
    beside those of ``profile_run.EFFECTIVE_OPTION_KEYS``, and the after
    arm's runs request more (``population_run.DEFAULT_OPTIONS``). An option
    only they hold is compared with its value in a VBMC built as the
    fingerprint's run built one, from its configuration and its requested
    options, by the package of this process, which :func:`check_code` binds
    to the after arm's code.
    """
    ours = fingerprint["effective_options"]
    theirs = cluster["effective_options"]
    differing = sorted(
        k for k in set(ours) & set(theirs) if ours[k] != theirs[k]
    )
    only = sorted(set(theirs) - set(ours))
    if only:
        from benchmark_targets import find_config
        from profile_run import jsonable

        from pyvbmc import VBMC

        problem = find_config(fingerprint["label"]).make(
            seed=fingerprint["seed"]
        )
        args, _ = problem.vbmc_args()
        vbmc = VBMC(
            *args,
            options=dict(fingerprint["requested_options"]),
            seed=fingerprint["seed"],
        )
        differing += [
            k for k in only if jsonable(vbmc.options.get(k)) != theirs[k]
        ]
    return differing + sorted(set(ours) - set(theirs))


def outlier_rate(envelopes):
    """The fraction of a population's runs that lie outside their own
    configuration's envelope in some accuracy metric."""
    outside = total = 0
    for entry in envelopes.values():
        fences = {
            m: golden_replay.envelope(entry[m]) for m in golden_replay.ACCURACY
        }
        for i in range(len(entry["seeds"])):
            total += 1
            outside += any(
                entry[m][i] > fences[m] for m in golden_replay.ACCURACY
            )
    return outside / total if total else 0.0


def outside_probability(outside, n, rate):
    """The probability that ``outside`` or more of ``n`` runs lie outside
    their envelopes, each with probability ``rate``."""
    from scipy.stats import binom

    return float(binom.sf(outside - 1, n, rate)) if outside else 1.0


def check_fingerprints(directory, after, campaign, sidecars):
    """The fingerprints (module docstring); their summary for the record."""
    directory = Path(directory)
    labels = campaign["manifest"]["allocation"]["labels"]
    source = campaign["identity"]["source"]["trees"]
    report = read_json(directory / "replay.json")
    check(report["threads"] == 1, "the fingerprints ran with other threads")
    check(
        report["calibration_budget"] is None,
        "the fingerprints ran with pinned calibration budgets",
    )
    rows = {(row["label"], row["seed"]): row for row in report["rows"]}
    check(
        set(rows) == {(label, 0) for label in labels},
        "the fingerprints are not every configuration at seed 0",
    )
    envelopes = golden_replay.load_envelopes(after, labels)
    commits = set()
    seed0_identical = 0
    outside = {}
    for label in labels:
        row = rows[(label, 0)]
        tag = golden_trace._tag(label, 0)
        final = row.get("final_new") or {}
        check(
            row["ok"]
            and not row.get("consistency_issues")
            and not row.get("final_validity_issues")
            and all(
                np.isfinite(float(final.get(m) or np.nan))
                for m in golden_replay.ACCURACY
            ),
            f"{tag}: the fingerprint failed: {row.get('verdict')}",
        )
        check(
            "elbo_exact_iter" not in row,
            f"{tag}: the fingerprint's replay compared a trace",
        )
        check(
            all(
                same_float(
                    row["pop_fence"][m],
                    golden_replay.envelope(envelopes[label][m]),
                )
                for m in golden_replay.ACCURACY
            ),
            f"{tag}: the fingerprint was judged by another population",
        )
        if row["outside"]:
            outside[tag] = row["outside"]
        check(
            not (directory / f"{tag}.error.txt").exists(),
            f"{tag}: the fingerprint has an error file",
        )
        side = read_json(directory / f"{tag}.json")
        check(
            {k: side["final"].get(k) for k in golden_replay.FINAL_KEYS}
            == row["final_new"],
            f"{tag}: the sidecar is not the run the replay judged",
        )
        with zipfile.ZipFile(directory / f"{tag}.npz") as archive:
            check(archive.testzip() is None, f"{tag}: the trace is damaged")
        meta = side["meta"]
        check(
            meta["threads"] == {k: "1" for k in runner.THREAD_KEYS},
            f"{tag}: the fingerprint ran with other threads",
        )
        gpyreg = (meta.get("gpyreg_source") or {}).get("git") or {}
        package = (meta.get("pyvbmc_source") or {}).get("git") or {}
        check(
            same_commit(gpyreg.get("sha"), source["gpyreg"]["commit"]),
            f"{tag}: the fingerprint's gpyreg is not the after arm's",
        )
        check(
            same_commit(package.get("sha"), meta["git"]["sha"]),
            f"{tag}: the fingerprint's package is not its checkout's",
        )
        if clean_required():
            check(
                not meta["git"]["dirty"] and not gpyreg.get("dirty"),
                f"{tag}: the fingerprint ran from a dirty checkout",
            )
        cluster = [s for s in sidecars.values() if s["label"] == label]
        differing = option_differences(side, cluster[0]) if cluster else None
        check(
            differing == [],
            f"{tag}: the fingerprint ran with other options than the after "
            f"arm's runs: {differing}",
        )
        commits.add(meta["git"]["sha"])
        seed0_identical += bool(row.get("semantic_final_identical"))
    rate = outlier_rate(envelopes)
    probability = outside_probability(len(outside), len(labels), rate)
    check(
        probability >= FINGERPRINT_ALPHA,
        f"{len(outside)} of {len(labels)} fingerprints lie outside their "
        f"envelopes ({outside}), which the population's own rate of runs "
        f"outside theirs, {rate:.3f}, makes improbable (p = "
        f"{probability:.2g} < {FINGERPRINT_ALPHA})",
    )
    check(len(commits) == 1, f"the fingerprints ran at commits {commits}")
    commit = commits.pop()
    differing = numerics_differ(commit, source["pyvbmc"]["commit"])
    check(
        not differing,
        f"the fingerprints' code differs from the after arm's in {differing}",
    )
    first = read_json(directory / f"{golden_trace._tag(labels[0], 0)}.json")
    return {
        "cases": len(labels),
        "commit": full_commit(commit),
        "gpyreg": first["meta"]["gpyreg_source"]["git"]["sha"],
        "versions": {
            k: first["meta"][k] for k in ("python", "numpy", "scipy", "cma")
        },
        "outside_envelope": outside,
        "population_outlier_rate": rate,
        "outside_probability": probability,
        "semantic_finals_identical_to_cluster_seed0": seed0_identical,
        "minutes": report["minutes"],
    }


def check_gate_runs(directory, campaign):
    record, problems = gates.check_record(directory)
    check(not problems, f"the gate runs fail their check: {problems}")
    check(
        record["script"]["path"] == gates.GATE_SCRIPT,
        f"the gate runs ran {record['script']['path']}",
    )
    source = campaign["identity"]["source"]["trees"]
    first = record["recordings"][gates.RECORDINGS[0]]["record"]
    trees = first["identity"]["source"]["trees"]
    if clean_required():
        check(
            not record["allow_dirty"]
            and all(tree["clean"] for tree in trees.values()),
            "the gate runs ran from a dirty checkout",
        )
    check(
        trees["gpyreg"]["commit"] == source["gpyreg"]["commit"],
        "the gate runs' gpyreg is not the after arm's",
    )
    differing = numerics_differ(
        trees["harness"]["commit"], source["pyvbmc"]["commit"]
    )
    check(
        not differing,
        f"the gate runs' code differs from the after arm's in {differing}",
    )
    host = first["identity"]["host"]["hostname"]
    check(
        host == socket.gethostname(),
        f"the gate runs ran on {host}, not on this machine",
    )
    return record, {
        "runs": len(first["runs"]),
        "identical": record["identical"],
        "commit": trees["harness"]["commit"],
        "gpyreg": trees["gpyreg"]["commit"],
        "host": host,
        "script": record["script"],
    }


# --------------------------------------------------------------------------
# Copies
# --------------------------------------------------------------------------


def copy_checked(source, target):
    """Copy a file, or check that an existing copy equals it."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        check(
            sha256(target) == sha256(source),
            f"{target} exists and is not a copy of {source}",
        )
        return
    shutil.copyfile(source, target)
    check(sha256(target) == sha256(source), f"{target} differs from {source}")


def gate_files(record):
    return sorted(
        {
            entry["file"]
            for recording in record["recordings"].values()
            for entry in recording["files"].values()
        }
        | {gates.RECORD}
    )


def ks_tests(report):
    """The KS tests an even/odd comparison's report holds."""
    return sum(
        1
        for line in report.splitlines()
        if re.match(r"^\| \S+ \| (elbo_err|gskl|mmtv|func_count) \|", line)
    )


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def cmd_fingerprints(args, root):
    after = Path(args.after).resolve()
    out = Path(args.out).resolve()
    check(
        not out.exists() or not any(out.iterdir()),
        f"{out} is not empty; the fingerprints are made once",
    )
    manifest = read_manifest(after)
    check_code(manifest["identity"]["source"], "fingerprints")
    return golden_replay.main(
        [
            "--configs",
            ",".join(manifest["allocation"]["labels"]),
            "--seeds",
            "0",
            "--baseline",
            str(out / NO_TRACES),
            "--sidecars",
            str(after),
            "--out",
            str(out),
        ]
    )


def reference_name(runs, record):
    """The reference's name: that of an earlier ``prepare`` of the record,
    or ``reference_<runs>_<today>``."""
    path = Path(record) / VALIDATION
    if path.is_file():
        return read_json(path)["reference"]
    return f"reference_{runs}_{time.strftime('%Y%m%d')}"


def cmd_prepare(args, root):
    root = Path(root)
    after = Path(args.after).resolve()
    record = Path(args.record).resolve()
    after_rel = inside(root, after, "--after")
    record_rel = inside(root, record, "--record")
    assessment_rel = inside(root, args.assessment, "--assessment")
    if (record / VALIDATION).exists():
        status = read_json(record / VALIDATION)["status"]
        check(status != "promoted", f"{record} is of a promotion made")
    rulings = read_json(args.rulings) if args.rulings else {}

    previous = check_active_baseline(root)
    previous_traces = check_previous_traces(root)
    campaign, sidecars, files = read_after(after, rulings)
    head = check_code(campaign["identity"]["source"], "prepare")
    assessment = check_assessment(
        args.assessment, args.accepted_assessment, campaign
    )
    fingerprints = check_fingerprints(
        args.fingerprints, after, campaign, sidecars
    )
    gate_record, gate_summary = check_gate_runs(args.gate_runs, campaign)
    labels = campaign["manifest"]["allocation"]["labels"]
    name = args.name or reference_name(len(sidecars), record)
    traces = root / GOLDEN_RUNS / f"{name}_fingerprints"
    print(
        f"Checked {len(sidecars)} cases, the fingerprints and the gate "
        f"runs; the reference is {name}",
        flush=True,
    )

    fingerprint_dir = Path(args.fingerprints)
    traces.mkdir(parents=True, exist_ok=True)
    for label in labels:
        tag = golden_trace._tag(label, 0)
        for suffix in (".json", ".npz"):
            copy_checked(
                fingerprint_dir / f"{tag}{suffix}", traces / f"{tag}{suffix}"
            )
    for file in gate_files(gate_record):
        copy_checked(Path(args.gate_runs) / file, traces / "gate_runs" / file)
    golden_trace.cmd_summary(argparse.Namespace(dir=str(traces)))

    record.mkdir(parents=True, exist_ok=True)
    readme = record / "previous_reference_README.md"
    if readme.exists():
        check(
            sha256(readme, True)
            == sha256(root / "dev/golden/README.md", True),
            f"{readme} is not dev/golden/README.md as it stands",
        )
    else:
        shutil.copyfile(root / "dev/golden/README.md", readme)
    for source, target in (
        ("replay.json", "fingerprint_replay.json"),
        ("replay.md", "fingerprint_replay.md"),
    ):
        shutil.copyfile(fingerprint_dir / source, record / target)
    shutil.copyfile(Path(args.gate_runs) / gates.RECORD, record / gates.RECORD)

    population = population_of(sidecars)
    even, odd = join.split_population(population)
    text, flagged = golden_trace.compare_populations(even, odd)
    (record / "even_vs_odd.md").write_text(
        f"# Even/odd check: {name}\n\n{text}\n", encoding="utf-8"
    )
    summary = golden_trace.summary_text(population, name) + "\n"
    manifest = {
        "population": name,
        "sidecars": files,
        "summary_sha256": text_sha256(summary),
        "fingerprints": {
            golden_trace._tag(label, 0): {
                "json_sha256": sha256(
                    traces / f"{golden_trace._tag(label, 0)}.json", True
                ),
                "npz_sha256": sha256(
                    traces / f"{golden_trace._tag(label, 0)}.npz"
                ),
            }
            for label in labels
        },
        "fingerprints_summary_sha256": sha256(traces / "summary.md", True),
        "gate_runs": {
            file: sha256(traces / "gate_runs" / file)
            for file in gate_files(gate_record)
        },
        "text_hash_normalization": NORMALIZATION,
    }
    write_json(record / MANIFEST, manifest)
    table = outcomes(sidecars, labels)
    validation = {
        "reference": name,
        "status": "prepared",
        "prepared": contract.now(),
        "after": {
            "path": after_rel,
            "name": campaign["name"],
            "arm": campaign["manifest"].get("arm"),
            "manifest_sha256": contract.source_sha256(after, "manifest.json"),
            "verification_sha256": contract.source_sha256(
                after, "verification.json"
            ),
            "source": campaign["identity"]["source"],
            "allocation": campaign["manifest"]["allocation"],
            "options": campaign["manifest"]["options"],
            "counts": campaign["counts"],
            "rulings": rulings,
        },
        "assessment": {"path": assessment_rel, **assessment},
        "population": {
            "runs": len(sidecars),
            "configs": {label: table[label]["runs"] for label in labels},
            "outcomes": table,
            "even_odd_tests": ks_tests(text),
            "even_odd_flagged": sorted(flagged),
        },
        "fingerprints": {
            "source": str(Path(args.fingerprints).resolve()),
            "traces": inside(root, traces, "the fingerprints' traces"),
            **fingerprints,
            "report_sha256": sha256(record / "fingerprint_replay.json", True),
        },
        "gate_runs": {
            "source": str(Path(args.gate_runs).resolve()),
            **gate_summary,
            "record_sha256": sha256(record / gates.RECORD, True),
        },
        "previous": {
            "name": PREVIOUS,
            "manifest": inside(root, previous_manifest(root), "the manifest"),
            "manifest_sha256": sha256(previous_manifest(root), True),
            "pairs": len(previous["files"]),
            "baseline": f"{BASELINE} holds it",
            "traces": previous_traces,
        },
        "record": record_rel,
        "host": contract.host_part(strict=False),
        "promotion_script": {
            "sha256": sha256(Path(__file__), True),
            "head": head,
        },
    }
    write_json(record / VALIDATION, validation)
    print(
        f"Prepared {name}: {len(sidecars)} sidecars to publish, "
        f"{len(labels)} fingerprints in {traces}, "
        f"{len(fingerprints['outside_envelope'])} outside their envelopes; "
        f"even/odd flags: {sorted(flagged)}",
        flush=True,
    )
    if flagged:
        raise PromotionError("the even/odd check flags configurations")
    return 0


def prepared(root, record):
    record = Path(record).resolve()
    validation = read_json(record / VALIDATION)
    check(
        validation["status"] == "prepared",
        f"{record} is {validation['status']}, not prepared",
    )
    check(
        not validation["population"]["even_odd_flagged"],
        "the even/odd check flagged configurations",
    )
    return record, validation


def cmd_replay(args, root):
    record, validation = prepared(root, args.record)
    check_code(validation["after"]["source"], "replay")
    return golden_replay.main(
        [
            "--configs",
            ",".join(DEFAULT_CONFIGS),
            "--seeds",
            "0",
            "--baseline",
            str(Path(root) / validation["fingerprints"]["traces"]),
            "--sidecars",
            str(Path(root) / validation["after"]["path"]),
            "--out",
            str(Path(args.out).resolve()),
        ]
    )


def check_final_replay(root, directory, validation):
    directory = Path(directory)
    report = read_json(directory / "replay.json")
    traces = Path(validation["fingerprints"]["traces"]).name
    check(report["calibration_budget"] is None, "the replay pinned budgets")
    check(report["threads"] == 1, "the replay ran with other threads")
    head = check_code(validation["after"]["source"], "publish")
    check(
        same_commit(report["git"]["sha"], head),
        f"the replay ran at {report['git']['sha']}, not at HEAD {head}",
    )
    if clean_required():
        check(
            not report["git"]["dirty"], "the replay ran from a dirty checkout"
        )
    header = (directory / "replay.md").read_text(encoding="utf-8")
    check(
        f"baseline `{traces}`" in header,
        f"the replay's baseline is not {traces}",
    )
    rows = report["rows"]
    check(
        sorted((r["label"], r["seed"]) for r in rows)
        == sorted((label, 0) for label in DEFAULT_CONFIGS),
        "the replay is not of the default configurations at seed 0",
    )
    for row in rows:
        check(
            row["ok"]
            and row["identical"]
            and row["semantic_final_identical"] is True
            and row["initial_design_ok"] is True
            and row["design"] == "identical (X_init in both traces)"
            and not row["flagged"],
            f"{row['label']}: the replay is not identical: {row['verdict']}",
        )
        meta = read_json(
            directory / f"{golden_trace._tag(row['label'], 0)}.json"
        )["meta"]
        check(
            same_commit(
                meta["gpyreg_source"]["git"]["sha"],
                validation["after"]["source"]["trees"]["gpyreg"]["commit"],
            ),
            f"{row['label']}: the replay's gpyreg is not the after arm's",
        )
    return report, head


def check_prepared_files(root, validation, manifest):
    """The prepared fingerprints and gate runs are as the record hashes
    them; returns the new sidecars' bytes, checked, by stem."""
    traces = Path(root) / validation["fingerprints"]["traces"]
    for tag, hashes in manifest["fingerprints"].items():
        check(
            sha256(traces / f"{tag}.json", True) == hashes["json_sha256"]
            and sha256(traces / f"{tag}.npz") == hashes["npz_sha256"],
            f"{traces}: {tag} is not the prepared fingerprint",
        )
    for name, digest in manifest["gate_runs"].items():
        check(
            sha256(traces / "gate_runs" / name) == digest,
            f"{traces}: gate_runs/{name} is not the prepared file",
        )
    after = Path(root) / validation["after"]["path"]
    new = {}
    for stem, entry in manifest["sidecars"].items():
        data = (after / entry["file"]).read_bytes()
        check(
            bytes_sha256(data) == entry["sha256"],
            f"{after / entry['file']} is not the prepared sidecar",
        )
        new[stem] = data
    return new


def number(n):
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)


def facts(root, validation, report, head, public_traces):
    """What the rewritten documents say, from the record."""
    after = validation["after"]
    trees = after["source"]["trees"]
    versions = after["source"]["versions"]
    population = validation["population"]
    fingerprints = validation["fingerprints"]
    table = population["outcomes"]
    labels = list(table)
    noisy = [label for label in labels if table[label]["noisy"]]
    budget = {
        label: row["reached_budget"]
        for label, row in table.items()
        if row["reached_budget"]
    }
    exhaust = budget.pop("cigar_D15_exhaust", 0)
    parts = []
    if exhaust:
        parts.append(f"the {exhaust} runs of `cigar_D15_exhaust`, by design")
    if budget:
        parts.append(
            f"{sum(budget.values())} others: "
            + ", ".join(f"{n} of `{label}`" for label, n in budget.items())
        )
    reached = sum(row["reached_budget"] for row in table.values())
    excluded = len(after["rulings"])
    outside = len(fingerprints["outside_envelope"])
    before = validation["assessment"]["before"]["pyvbmc"]
    root = Path(root)

    def link(target, start):
        return Path(os.path.relpath(root / target, root / start)).as_posix()

    return {
        "name": validation["reference"],
        "traces": Path(fingerprints["traces"]).name,
        "runs": population["runs"],
        "configs": len(labels),
        "seeds": SEEDS.replace("-", "–"),
        "noisy_runs": sum(table[label]["runs"] for label in noisy),
        "noisy_configs": len(noisy),
        "tests": population["even_odd_tests"],
        "converged": sum(row["converged"] for row in table.values()),
        "budgets": f"{reached} reached their budgets"
        + (f" ({', and '.join(parts)})" if parts else ""),
        "usable": sum(row["usable"] for row in table.values()),
        "excluded_sentence": (
            f"{excluded} case{'s' if excluded > 1 else ''} of the allocation "
            f"{'are' if excluded > 1 else 'is'} not in the reference; the "
            "promotion record gives the ruling on each. "
            if excluded
            else ""
        ),
        "commit": trees["pyvbmc"]["commit"][:8],
        "gpyreg": trees["gpyreg"]["commit"][:8],
        "python": versions.get("python"),
        "numpy": versions.get("numpy"),
        "scipy": versions.get("scipy"),
        "cma": versions.get("cma"),
        "before": next(
            (
                f"{text} (`{before[:8]}`)"
                for prefix, text in BEFORE_CODE.items()
                if before.startswith(prefix)
            ),
            f"the code of `{before[:8]}`",
        ),
        "after_rel": after["path"],
        "after_dev": link(after["path"], "dev"),
        "after_link": link(after["path"], "dev/golden"),
        "gate_readme_link": link(
            f"{Path(after['path']).parent.as_posix()}/README.md", "dev/golden"
        ),
        "public_traces": (
            f" They are published as {public_traces}." if public_traces else ""
        ),
        "record_link": link(validation["record"], "dev/golden"),
        "readme_record_link": link(validation["record"], "dev"),
        "envelope_sentence": (
            "none lies outside its configuration's accuracy envelope"
            if not outside
            else f"{outside} of them lie outside their configuration's "
            "accuracy envelope, which the population's own rate of runs "
            f"outside theirs ({fingerprints['population_outlier_rate']:.3f}) "
            "makes plausible"
        ),
        "seed0_identical": fingerprints[
            "semantic_finals_identical_to_cluster_seed0"
        ],
        "duration": (
            "about a minute"
            if report["minutes"] < 1.5
            else f"about {round(report['minutes'])} minutes"
        ),
        "date": time.strftime("%Y-%m-%d"),
        "previous_commit": head[:8],
        "defaults": number(len(DEFAULT_CONFIGS)),
        "default_configs": ", ".join(f"`{c}`" for c in DEFAULT_CONFIGS),
    }


def cmd_publish(args, root):
    root = Path(root)
    record, validation = prepared(root, args.record)
    manifest = read_json(record / MANIFEST)
    name = validation["reference"]
    check(
        (record / "README.md").is_file()
        and name in (record / "README.md").read_text(encoding="utf-8"),
        f"{record}/README.md, which the documents link to, does not name "
        f"{name}",
    )
    gate_readme = Path(validation["after"]["path"]).parent / "README.md"
    check(
        (root / gate_readme).is_file(),
        f"{gate_readme}, which names the campaign's archive, does not exist",
    )
    report, head = check_final_replay(root, args.final_replay, validation)
    new = check_prepared_files(root, validation, manifest)
    check_active_baseline(root)
    check(
        sha256(record / "previous_reference_README.md", True)
        == sha256(root / "dev/golden/README.md", True),
        "dev/golden/README.md has changed since prepare copied it",
    )
    problems = passage_problems(root)
    check(not problems, "\n".join(problems))

    # Everything written is computed and checked first.
    sidecars = {stem: json.loads(data) for stem, data in new.items()}
    summary = golden_trace.summary_text(population_of(sidecars), name) + "\n"
    check(
        text_sha256(summary) == manifest["summary_sha256"],
        "the summary of the sidecars is not the prepared one",
    )
    values = facts(root, validation, report, head, args.public_traces)
    rewrites = {}

    def rewritten(file):
        if file not in rewrites:
            rewrites[file] = list(read_text(root / file))
        return rewrites[file]

    for key, template in (
        ("agents", AGENTS_TEMPLATE),
        ("readme_reference", README_REFERENCE_TEMPLATE),
        ("readme_replay", README_REPLAY_TEMPLATE),
        ("golden_readme", GOLDEN_README_TEMPLATE),
    ):
        file, start, end, _ = PASSAGES[key]
        entry = rewritten(file)
        i, j = span(entry[0], start, end)
        text = reflow(template.format(**values)).rstrip("\n") + "\n"
        if end is not None and entry[0][j - 2 : j] == "\n\n":
            text += "\n"
        entry[0] = entry[0][:i] + text + entry[0][j:]
    file, start, end, _ = PASSAGES["replay_configs"]
    entry = rewritten(file)
    i, j = span(entry[0], start, end)
    entry[0] = entry[0][:i] + replay_configs(values) + entry[0][j:]
    for file, old, new_text in REPLACEMENTS:
        entry = rewritten(file)
        count = entry[0].count(old)
        check(count == 1, f"{file} holds {old!r} {count} times")
        entry[0] = entry[0].replace(old, new_text.format(**values))
    file, start, end = OWN_ENTRY
    entry = rewritten(file)
    i, j = span(entry[0], start, end)
    entry[0] = entry[0][:i] + entry[0][j:]
    compile(
        rewrites["dev/scripts/golden_replay.py"][0], "golden_replay.py", "exec"
    )

    # The writes.
    baseline = root / BASELINE
    for path in baseline.glob("*_seed*.json"):
        path.unlink()
    for stem, data in new.items():
        (baseline / f"{stem}.json").write_bytes(data)
    write_text(baseline / "summary.md", summary)
    for file, (text, eol) in rewrites.items():
        write_text(root / file, text, eol)
    shutil.copyfile(Path(__file__), record / "promote.py")
    previous_traces = root / GOLDEN_RUNS / PREVIOUS
    if (
        previous_traces.is_dir()
        and not (previous_traces / "README.md").exists()
    ):
        shutil.copyfile(
            record / "previous_reference_README.md",
            previous_traces / "README.md",
        )
    for source, target in (
        ("replay.json", "final_replay.json"),
        ("replay.md", "final_replay.md"),
    ):
        shutil.copyfile(Path(args.final_replay) / source, record / target)
    validation["status"] = "promoted"
    validation["publication"] = {
        "published": contract.now(),
        "head": head,
        "previous_sidecars_commit": head,
        "script_sha256": sha256(Path(__file__), True),
        "public_traces": args.public_traces,
        "final_replay": {
            "cases": len(report["rows"]),
            "identical": sum(bool(r["identical"]) for r in report["rows"]),
            "minutes": report["minutes"],
            "report_sha256": sha256(record / "final_replay.json", True),
        },
        "rewritten": sorted(rewrites),
    }
    write_json(record / VALIDATION, validation)
    print(
        f"Published {name}: {len(new)} sidecars in {BASELINE}; rewrote "
        f"{', '.join(sorted(rewrites))}; copied this script to "
        f"{record / 'promote.py'}. In the promotion commit, remove "
        "dev/scripts/reference_promote.py and "
        "dev/scripts/test_reference_promote.py, carry the reference into "
        f"{', '.join(HAND_EDITS)}, and list the fingerprints in this "
        "machine's dev/scripts/runs/LOCAL.md.",
        flush=True,
    )
    return 0


# --------------------------------------------------------------------------
# The rewritten passages
# --------------------------------------------------------------------------


def replay_configs(values):
    comment = textwrap.wrap(
        "Cheap configurations covering the regimes the Stage 2 items touch: a "
        "Gaussian at D = 5, the D = 2 banana, the bounded (probit) "
        "half-normal, the warped large-K cigar and the noisy VIQR path at the "
        f"production budget. {values['duration'].capitalize()} at seed 0 on "
        "the machine of the reference's replay fingerprints.",
        width=77,
    )
    lines = [
        *[f"# {line}" for line in comment],
        "DEFAULT_CONFIGS = (",
        *[f'    "{label}",' for label in DEFAULT_CONFIGS],
        ")",
    ]
    return "\n".join(lines) + "\n"


AGENTS_TEMPLATE = """\
- **Trajectories.** `python dev/scripts/golden_replay.py` replays golden
configurations and compares each run step by step with its stored trace; a
run that parts from its trace passes when its finals stay inside the
population's envelope. The traces it compares with by default, the replay
fingerprints of the reference `{name}`, were made on the machine that
`dev/scripts/runs/LOCAL.md` lists and exist only there. There, a change
that must move nothing leaves every default case identical; elsewhere,
replay at the parent commit first and pass that replay's `--out` directory
as `--baseline`. A change that moves the default trajectories is
assessed on the benchmark suite before it is accepted; the golden
references are then updated and the old ones preserved (`dev/README.md`).
"""

README_REFERENCE_TEMPLATE = """\
The current golden reference is `{name}`: **{runs} runs of the {configs}
configurations of the `production` suite at seeds {seeds}, {noisy_runs} of
them noisy, with {tests} population KS tests**. Release checks and new
experiments use the `production` suite: the golden suite with its noisy
entries freed of the 2020 paper's budget of 50 (D + 2) evaluations and
given `production`-tagged labels, so that they run at the package's
defaults for a specified-noise target (75 (D + 2) evaluations); its
noiseless entries are the golden ones. The runs were made on a Slurm
cluster by `population_run.py`'s array mode from PyVBMC `{commit}` with
gpyreg `{gpyreg}`
([plans/slurm-benchmark-support.md](plans/slurm-benchmark-support.md)).
Their JSON sidecars and `summary.md` live under `golden/baseline/`, so
`python dev/scripts/golden_trace.py compare dev/golden/baseline <new_dir>`
works from a fresh checkout; the campaign's redacted records are under
`{after_dev}`. The traces that `golden_replay.py` compares with exactly are
the reference's replay fingerprints, one run at seed 0 of each
configuration made with the same code on the machine that
`scripts/runs/LOCAL.md` lists, gitignored under
`scripts/runs/golden/{traces}/` with the record of the port review's six
seeded gate runs, two of them with a prior object, recorded twice there
(`scripts/seeded_gate_runs.py`). The previous references,
`reference_990_20260913` of the golden suite and those before it, remain
preserved for historical comparisons.

The [promotion record]({readme_record_link}/README.md) contains the
assessment, which the PI accepted, the provenance, hashes and
verification. The {tests}-test even/odd check had no flags. The {defaults}
default cases replayed exactly under the code at promotion ({date}) in
every non-timer NPZ loop/final array, semantic final-result field and
initial design. The returned posterior's transformer is absent from the
traces and remains uncertifiable.
"""

README_REPLAY_TEMPLATE = """\
- `scripts/golden_replay.py` — the per-change trajectory gate of Stage 2:
replays configurations of the golden reference in-process with the current
code (this checkout's package), {duration} for the default set, and
compares each run with its stored trace: exact shapes and values of
every non-timer NPZ array and all semantic final-result fields, the
ELBO/live-point agreement horizons, the initial design (see below), and
final accuracy against the reference population's `Q3 + 3 IQR` envelope.
It reports "same loop, changed final" separately and applies the accuracy
fences to that case. The traces omit the returned posterior's transformer;
that state is explicitly reported as not certifiable. Sidecar counts must
agree with their NPZ arrays. An arithmetic-preserving change is expected
to part once a CMA-ES ranking flips (a change to the ELBO arithmetic parts
at iteration 0); a parted run's finals must stay inside the envelope (an
identical run is exempt: its own seed may be the outlier). The initial
design is certified from the traces: exactly where both store it
(`X_init`, written by `golden_trace.py` since commit `9d92c7f`), against
the 2026-09-03 baseline by finding a generator-drawn design point of the
new run among the reference's live rows (the start point `x0` comes from
the run seed and is identical by construction, so it does not count), and
"not certifiable" without a flag where warm-up trimming removed the whole
design (cigar). Needs the baseline `.npz` traces for the horizons (finals
only without them); `--report-only` re-renders a finished run. Flags:
`--configs` (by default {default_configs}), `--seeds` (default seed 0
only), `--baseline` (the traces directory; the default
`scripts/runs/golden/{traces}/`, the replay fingerprints of the current
reference, one run at seed 0 of each configuration, exists only on the
machine that made them, which `scripts/runs/LOCAL.md` lists), `--sidecars`
(the envelope population: a flat directory of sidecars, by default
`golden/baseline/`, the current reference's, or a population of
`population_run.py`'s array mode, a campaign directory or its tracked
copies, of which only the verified cases count; a replayed configuration
without a sidecar there is an error), `--out`, `--threads` (1, as the
baseline), `--calibration-budget` (pin all three chunk budgets to this
integer for a nondefault-profile check; omitted means historical defaults,
independent of the local cache). Replay reports retain this setting,
including on `--report-only`. Exit code 1 if anything is flagged or
nothing was compared.
"""

GOLDEN_README_TEMPLATE = """\
# Golden runs: the PyVBMC regression reference

The golden runs are saved, seeded, end-to-end PyVBMC runs on a collection
of benchmark targets. They let us check whether a code change alters the
solver's path and whether it changes inference quality across many seeds.
Each benchmark has reference evidence and posterior information, obtained
analytically or by independent numerical calculation, against which we
measure the fitted result.

There are two complementary checks:

- **Population comparison:** compare final results across seeds to detect
changes in accuracy or the number of model evaluations.
- **Trajectory replay:** rerun selected seeds and compare their initial
designs, sampled points and per-iteration ELBOs with the saved traces.

The reference records the behavior of a particular implementation. It is
not a guarantee that every fit is accurate; known bugs may be present, and
an intentional correctness fix may change its results.

## Current reference

**`{name}`: {runs} runs of the {configs} configurations of the `production`
suite, at seeds {seeds}.** {noisy_runs} of them are noisy, across
{noisy_configs} configurations. The `production` suite is the `golden`
suite at PyVBMC's production defaults: its noiseless configurations are the
golden ones, and its noisy ones, under `production`-tagged labels, run at
the package's defaults for a specified-noise target (75 (D + 2)
evaluations, VIQR) where the golden ones pin the 2020 paper's budget of
50 (D + 2). {excluded_sentence}The [results summary](baseline/summary.md)
gives the measured outcomes for every configuration.

The runs were made on a Slurm cluster by the array mode of
`population_run.py`, one case per task, from PyVBMC `{commit}` with gpyreg
`{gpyreg}` (Python {python}, NumPy {numpy}, SciPy {scipy}, cma {cma}). The
campaign's redacted records, with every case's completion record, sidecar
and boost report, are under [`{after_rel}`]({after_link}). Its traces are
in its archive, a draft release that only the repository's collaborators
see, which the [README of the release gate's records]({gate_readme_link})
names.{public_traces} {converged} of the {runs} runs converged and
{budgets}; {usable} meet the usability thresholds below.

The [promotion record]({record_link}/README.md) gives the assessment, which
compares the population seed by seed with the same runs of {before} and
which the PI accepted, and the provenance, hashes and validation. The
{tests}-test even/odd comparison of the population had no flags.

Exact replay depends on the machine and its BLAS, so the traces that
`golden_replay.py` compares with are not the cluster's: they are the
reference's replay fingerprints, one run at seed 0 of each of the
{configs} configurations, made with the same code on the machine that
`dev/scripts/runs/LOCAL.md` lists and judged against the population's
accuracy envelopes: {envelope_sentence}. {seed0_identical} of the
{configs} equal the cluster's run of seed 0 in every semantic final field.
The port review's six seeded gate runs, two of them with a prior object
(`dev/scripts/seeded_gate_runs.py`), were recorded twice on that machine
and reproduced bit for bit. The {defaults} default cases replayed with
identical non-timer NPZ arrays, semantic final results and initial designs
under the code at promotion ({date}). No trace stores the returned
posterior's transformer, so replay reports that field as uncertifiable.

The previous reference, `reference_990_20260913`, 990 runs of the `golden`
suite, is preserved with its
[promotion record](promotion_20260913/README.md): its sidecars are
`dev/golden/baseline/` at commit `{previous_commit}`, its traces are under
`dev/scripts/runs/golden/reference_990_20260913/` on the machine that
`dev/scripts/runs/LOCAL.md` lists, and this file as it stood for it, whose
links resolve from `dev/golden/`, is
[`previous_reference_README.md`]({record_link}/previous_reference_README.md).

| Target | Dimensions | What it exercises |
|---|---|---|
| `normal` | 5 | Independent Gaussian parameters with different scales |
| `corr` | 5 | Correlated Gaussian parameters |
| `halfnormal` | 2 | A posterior constrained to positive parameters |
| `rosenbrock` | 2 | A curved, narrow posterior: noiseless, and with noise SD 1 and 3 |
| `banana` | 2, 6, 10 | Nonlinear dependence across increasing dimensions |
| `cigar` | 4, 8 | A strongly elongated, rotated posterior |
| `lumpy` | 4, 10 | A Gaussian-mixture target with multiple components; D = 10 also with noise SD 3 |
| `student` | 4, 8 | Student-t likelihoods with broad Gaussian priors; D = 8 also with noise SD 3 |
| `logreg` | 5 | Logistic regression with bounded parameters: noiseless, and with noise SD 3 |
| `cigar_D15_exhaust` | 15 | The late GP regime under a fixed 750-evaluation budget |
| `multisensory_s1` | 6 | Real-data causal-inference likelihood, subject 1: noiseless, and with noise SD 1.3 |
| `multisensory_s2` | 6 | The same model on subject 2, with noise SD 1.3 |
| `timing` | 5 | Real-data Bayesian timing likelihood with noise SD 2.2 |

The real-data configurations are the Bayesian timing model and the
multisensory causal-inference model on two subjects from the 2020
noisy-VBMC paper, on exported data with spline-trapezoidal priors and the
paper's plausible boxes, defined in
[benchmark-realistic-targets.md](../plans/benchmark-realistic-targets.md),
at the paper's noise levels. Their ground truths are importance-weighted
populations under `dev/scripts/data/truths/`.

A run's seed controls the solver and separate streams for the starting point
and, where applicable, evaluation noise. Starting points are drawn uniformly
inside the plausible bounds. Target definitions, bounds, reference values
and configuration options are in
[`benchmark_targets.py`](../scripts/benchmark_targets.py).

The D15 case disables early convergence termination so it always uses its
750 evaluations, the one configuration that spends long in the regime where
the GP keeps a single hyperparameter sample.

## Reading the results

Most entries in the summary are **median [25th percentile, 75th percentile]**.
Smaller values are better for the three accuracy metrics:

| Metric | Meaning |
|---|---|
| `elbo_err` | Absolute difference between the estimated and reference log evidence |
| `gskl` | Gaussianized symmetrized KL divergence: agreement of posterior means and covariances |
| `mmtv` | Mean marginal total variation distance: agreement of the one-dimensional posterior marginals |
| `evals` | Number of target evaluations used |

`gskl` and `mmtv` capture different aspects of a posterior; neither alone
certifies its full joint shape. The summary also reports posterior-mean
RMSE, iterations, mixture size, warps and wall time. `usable` is the fraction
meeting all three accuracy thresholds: `elbo_err < 1`, `gskl < 1`, and
`mmtv < 0.2`. `failed` counts execution failures, not unconverged or
inaccurate fits.

The population check applies two-sample Kolmogorov–Smirnov tests to
`elbo_err`, `gskl`, `mmtv` and evaluation count, with a Holm correction at
alpha 0.05 across the family ({tests} tests for all {configs}
configurations). A flag means a distribution changed and needs
investigation; it does not establish that the change is a regression. No
flag is not proof of equivalence. Wall time is recorded but is not a
pass/fail metric. These runs assess inference behavior; controlled speed
comparisons need dedicated benchmarks.

## Files and checks

- [`baseline/`](baseline/) contains the {runs} JSON sidecars and the summary,
tracked in Git: the sidecars of the campaign's redacted records, copied byte
for byte. Each sidecar records the target, seed, requested and effective
options, the provenance of its run (the commits and import paths of its
code, the imported versions) and final metrics.
- The population's `.npz` traces and boost captures are in the campaign's
archive. The replay fingerprints' traces and the record of the gate runs
are local under `dev/scripts/runs/golden/{traces}/` (gitignored) on the
machine that made them.

From the repository root, compare a new population with the reference:

```console
python dev/scripts/golden_trace.py compare dev/golden/baseline <new_run_directory>
```

This needs only the JSON sidecars. Check matching configuration/seed
coverage, complete JSON/NPZ pairs and absence of error files separately:
the statistical comparison alone does not certify completeness.

On the machine that holds the replay fingerprints, run the default
{defaults}-case replay:

```console
python dev/scripts/golden_replay.py
```

The default cases cover a Gaussian, banana, bounded half-normal, cigar and
noisy Rosenbrock at the production budget, at seed 0. The report shows where
trajectories diverge and checks changed runs against the reference
population's accuracy ranges. Elsewhere, replay at the parent commit first
and pass that replay's `--out` directory as `--baseline`. Both commands
return nonzero when their checks flag a problem.

A machine gets replay fingerprints of its own by making them at the commit
the promotion record names, as they were made for the reference:
`golden_replay.py` on every configuration of the `production` suite at seed
0, with `--sidecars dev/golden/baseline` and a `--baseline` that holds no
traces, so that each run is judged against the population's envelopes, and
the gate runs with `seeded_gate_runs.py run`. That machine's
`dev/scripts/runs/LOCAL.md` then lists them, and its replays pass them as
`--baseline`.

## Code and reproducibility

The runs use single-threaded BLAS, `vectorized_target=False`,
`performance_calibration="off"` and `tol_elcbo_boost=0.1`. Reproducing a
stored trajectory requires its recorded code, seed, options, dependency
versions and numerical platform. These are part of the reference; matching
the seed alone is insufficient.

Exact environment information, file hashes and validation reports are in
the [promotion record]({record_link}/README.md). The legacy
`regenerate_baseline.sh` uses a different seed allocation and masks
comparison failures; use the recorded procedure when reproducing this
snapshot.
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fingerprints = sub.add_parser("fingerprints", help="Phase 9's runs")
    fingerprints.add_argument("--after", required=True)
    fingerprints.add_argument("--out", required=True)
    fingerprints.set_defaults(func=cmd_fingerprints)
    prepare = sub.add_parser("prepare", help="check and assemble")
    for name in (
        "--after",
        "--assessment",
        "--accepted-assessment",
        "--fingerprints",
        "--gate-runs",
        "--record",
    ):
        prepare.add_argument(name, required=True)
    prepare.add_argument("--name")
    prepare.add_argument("--rulings")
    prepare.set_defaults(func=cmd_prepare)
    replay = sub.add_parser("replay", help="replay the new defaults")
    replay.add_argument("--record", required=True)
    replay.add_argument("--out", required=True)
    replay.set_defaults(func=cmd_replay)
    publish = sub.add_parser("publish", help="publish the reference")
    publish.add_argument("--record", required=True)
    publish.add_argument("--final-replay", required=True)
    publish.add_argument(
        "--public-traces",
        help="where the population's traces are published, as the golden "
        "README names it (e.g. an asset of the release)",
    )
    publish.set_defaults(func=cmd_publish)
    return parser.parse_args(argv)


def main(argv=None, root=None):
    if sys.flags.optimize:
        raise RuntimeError("Run without -O: the checks use assertions")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    try:
        return args.func(args, Path(root) if root else REPO)
    except (
        PromotionError,
        contract.ContractError,
        AssertionError,
        RuntimeError,
        subprocess.CalledProcessError,
        KeyError,
    ) as error:
        print(
            f"reference_promote.py refused: {type(error).__name__}: {error}",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
