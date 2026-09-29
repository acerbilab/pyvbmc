"""Build the public asset of a finished campaign of the release gate.

A campaign hands back its redacted tracked copies, which enter the
repository, and its raw archive, which holds the site's details and the
operator's paths and stays in a draft release that is never published
(``dev/plans/slurm-benchmark-support.md``, "Records and hand-back"). What
the PyVBMC release publishes of a campaign is the asset this script builds,
in the account that ran the campaign, on the login node, after
``campaign_redact.sh`` (``hpc/campaign_public.sh`` runs it there): it
redacts as that command does, with the username and home of its own
process::

    python dev/scripts/campaign_public.py build --campaign DIR --copies DIR \\
        --out DIR [--path NAME=PATH ...] [--allow STRING ...] \\
        [--part-size BYTES]
    python dev/scripts/campaign_public.py check DIR

``build`` reads the campaign's verification report, which must have passed
and place no case in flight, and its tracked copies (``--copies``, what
``campaign_redact.sh`` wrote), which must be the copies of the campaign as
it stands: their files, and the SHA-256 their ``redaction.json`` records of
each copy and of its source. The asset holds, under a directory named for
the campaign:

- every tracked copy, as it is, with ``redaction.json``;
- of each case the report places as verified, the artifacts its completion
  record lists, each the file the record hashes: a ``.npz`` file as it is,
  once every array in it holds numbers or booleans (no text, no objects)
  and no array's name holds what the redaction forbids; a ``.json`` file
  redacted as the tracked copies are, or the tracked copy where it is one;
  and nothing else (the boost pickles and the ``.vbmc.pkl`` saves, which
  load only in the package that wrote them and may hold paths);
- ``public.json``: each file's SHA-256 and that of the campaign's file it
  comes from, what was left out, and the redaction's rules.

Every text file of the asset is searched as the redaction searches the
copies, and any hit, or any ``.npz`` file that is not numbers alone,
refuses and writes nothing. The asset is ``<out>/<campaign>.public.tar.gz``
in parts ``.000``, ``.001``, ... of at most ``--part-size`` bytes (by
default 1900 MiB, below GitHub's 2 GiB per release asset), with their
SHA-256 in ``<campaign>.public.tar.gz.sha256``;
``cat <campaign>.public.tar.gz.[0-9][0-9][0-9] | tar xz`` restores it.

``check DIR`` re-reads every asset in ``DIR`` (a release's directory holds
one per campaign): every part against its SHA-256, and every member of the
archive against ``public.json``, which must list them all. A part that no
listing names, which a build that failed may leave and an upload's glob
would pick up, fails the check. It prints what fails and exits 1 if
anything does, or if the directory holds no asset.
"""

import argparse
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import campaign_contract as contract  # noqa: E402

#: The largest part of an asset, below GitHub's 2 GiB per release asset.
PART_SIZE = 1900 * 2**20
#: The record the asset holds beside the campaign's files.
PUBLIC = "public.json"
#: What an array of a published ``.npz`` file may hold: booleans, integers,
#: unsigned integers, floats and complex numbers.
NUMERIC_KINDS = "biufc"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def asset_name(campaign):
    return f"{Path(campaign).name}.public.tar.gz"


def npz_problems(data, redaction):
    """Why the ``.npz`` file ``data`` may not be published as it is; empty
    when it may."""
    problems = []
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            names = list(archive.files)
            for name in names:
                try:
                    kind = archive[name].dtype.kind
                except ValueError:
                    problems.append(f"{name}: an array of objects")
                    continue
                if kind not in NUMERIC_KINDS:
                    problems.append(f"{name}: an array of kind {kind!r}")
    except (OSError, ValueError) as error:
        return [f"not a readable .npz file ({error})"]
    problems += [
        f"an array's name holds {what} {string!r}"
        for _, string, what in redaction.leaks("\n".join(names), count=False)
    ]
    return problems


def check_copies(campaign, copies, names):
    """The tracked copies are those of the campaign as it stands; returns
    their ``redaction.json``."""
    record = contract.read_redaction(copies)
    if record is None:
        raise contract.ContractError(
            f"{copies} holds no {contract.REDACTION}; --copies is what "
            "campaign_redact.sh wrote"
        )
    if record.get("campaign") != Path(campaign).name:
        raise contract.ContractError(
            f"{copies} holds the copies of {record.get('campaign')}, not of "
            f"{Path(campaign).name}"
        )
    if sorted(record["files"]) != sorted(names):
        raise contract.ContractError(
            f"{copies} holds other files than the campaign's tracked copies; "
            "redact the campaign again"
        )
    for name, entry in record["files"].items():
        for where in (campaign, copies):
            if not (Path(where) / name).is_file():
                raise contract.ContractError(
                    f"{Path(where) / name} is missing; redact the campaign "
                    "again"
                )
        if (
            contract.sha256_file(Path(campaign) / name)
            != entry["source_sha256"]
        ):
            raise contract.ContractError(
                f"{name} has changed since {copies} was written from it; "
                "redact the campaign again"
            )
        if contract.sha256_file(Path(copies) / name) != entry["sha256"]:
            raise contract.ContractError(
                f"{Path(copies) / name} is not the copy {contract.REDACTION} "
                "records"
            )
    return record


def build(
    campaign,
    copies,
    out,
    operator=None,
    environ=None,
    paths=(),
    host=None,
    allow=(),
    part_size=PART_SIZE,
    say=print,
):
    """Build the public asset of ``campaign`` (module docstring); returns
    its ``public.json`` record."""
    campaign, copies, out = (
        Path(p).resolve() for p in (campaign, copies, out)
    )
    parts = sorted(out.glob(f"{asset_name(campaign)}*"))
    if parts:
        raise contract.ContractError(
            f"{out} holds an asset of {campaign.name} already ({parts[0].name})"
        )
    read = contract._read_campaign(campaign)
    manifest, verification = read["manifest"], read["verification"]
    if verification.get("exit_code") != 0:
        raise contract.ContractError(
            f"{campaign}/verification.json did not pass; the asset is made of "
            "a finished campaign"
        )
    in_flight = [
        case["tag"]
        for case in verification.get("cases", [])
        if case.get("status") == "in_flight"
    ]
    if in_flight:
        raise contract.ContractError(
            f"{len(in_flight)} cases are in flight (the first {in_flight[0]}); "
            "finish the campaign again once they are done"
        )
    redaction_record = check_copies(campaign, copies, read["names"])
    operator = (
        contract.operator_identity(environ) if operator is None else operator
    )
    redaction = contract.redaction_rules(
        campaign,
        manifest,
        read["documents"],
        operator,
        environ,
        paths,
        host,
        allow=allow,
    )
    say(f"reading the {len(read['records'])} verified cases of {campaign}")
    # Each member: (source path or None, its bytes where held, SHA-256 of the
    # member, SHA-256 of the campaign's file, kind). Numeric files are read
    # again when written, so that the asset needs no more memory than a
    # case's files.
    members, texts, left_out, refused = {}, {}, Counter(), []
    for name, entry in redaction_record["files"].items():
        data = (copies / name).read_bytes()
        members[name] = (None, data, entry["sha256"], entry["source_sha256"])
        texts[name] = data.decode("utf-8")
    data = (copies / contract.REDACTION).read_bytes()
    members[contract.REDACTION] = (None, data, sha256(data), None)
    texts[contract.REDACTION] = data.decode("utf-8")
    kinds = Counter({"tracked copy": len(redaction_record["files"])})
    for tag, record in sorted(read["records"].items()):
        for name, entry in sorted((record.get("artifacts") or {}).items()):
            if name in members:
                continue
            path = campaign / name
            data = path.read_bytes()
            if sha256(data) != entry["sha256"]:
                raise contract.ContractError(
                    f"{path} is not the file the completion record of {tag} "
                    "hashes"
                )
            if name.endswith(".npz"):
                problems = npz_problems(data, redaction)
                if problems:
                    refused.append((name, problems))
                    continue
                members[name] = (path, None, entry["sha256"], entry["sha256"])
                kinds["numeric"] += 1
            elif name.endswith(".json"):
                text = data.decode("utf-8")
                parsed = json.loads(text)
                value = redaction.value(parsed)
                if value == parsed:
                    new = data
                else:
                    indent, newline = contract._json_layout(text)
                    new = (
                        json.dumps(value, indent=indent)
                        + ("\n" if newline else "")
                    ).encode("utf-8")
                members[name] = (None, new, sha256(new), entry["sha256"])
                texts[name] = new.decode("utf-8")
                kinds["redacted"] += 1
            else:
                left_out["".join(Path(name).suffixes) or name] += 1
    if refused:
        raise contract.ContractError(
            "numeric files that are not numbers alone, so nothing was "
            "written:\n"
            + "\n".join(
                f"  {name}: {'; '.join(problems)}"
                for name, problems in refused
            )
        )
    record = {
        "contract": contract.CONTRACT_VERSION,
        "campaign": campaign.name,
        "generated": contract.now(),
        "node_family": manifest["site"]["NODE_FEATURE"],
        "cases": len(read["records"]),
        "copies_redaction_sha256": members[contract.REDACTION][2],
        "rules": [
            "the tracked copies, as campaign_redact.sh wrote them",
            "of every verified case, its .npz artifacts as they are, each "
            "holding numbers and booleans alone, and its other .json "
            "artifacts redacted as the tracked copies are",
            "no other artifact: pickles and logs are left out",
            "each file's source_sha256 is that of the campaign's own file, "
            "which the completion records hash",
        ],
        "kinds": dict(sorted(kinds.items())),
        "left_out": dict(sorted(left_out.items())),
        "files": {
            name: {"sha256": digest, "source_sha256": source}
            for name, (_, _, digest, source) in sorted(members.items())
        },
    }
    record_text = json.dumps(record, indent=2) + "\n"
    texts[PUBLIC] = record_text
    leaks = [
        (name, line, string, what)
        for name, text in sorted(texts.items())
        for line, string, what in redaction.leaks(
            text, count=name not in (PUBLIC, contract.REDACTION)
        )
    ]
    if leaks:
        raise contract.ContractError(
            "the asset would hold what the tracked copies may not, so "
            "nothing was written:\n" + contract._leak_lines(leaks)
        )
    out.mkdir(parents=True, exist_ok=True)
    staging = out / f".{asset_name(campaign)}.building"
    try:
        with tarfile.open(staging, "w:gz") as archive:
            for name, (path, data, digest, _) in sorted(members.items()):
                if data is None:
                    data = path.read_bytes()
                    if sha256(data) != digest:
                        raise contract.ContractError(
                            f"{path} changed while the asset was built"
                        )
                add(archive, f"{campaign.name}/{name}", data)
            add(archive, f"{campaign.name}/{PUBLIC}", record_text.encode())
        written = split(staging, out / asset_name(campaign), part_size)
    finally:
        if staging.exists():
            staging.unlink()
    say(
        f"{len(members)} files of {campaign.name} ("
        + ", ".join(f"{k} {v}" for k, v in record["kinds"].items())
        + "), left out "
        + (", ".join(f"{k} {v}" for k, v in left_out.items()) or "nothing")
        + f"; {len(written)} parts in {out}"
    )
    return record


def add(archive, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def split(path, target, part_size):
    """Split ``path`` into ``<target>.000``, ... and write their SHA-256 in
    ``<target>.sha256``; returns the parts' names."""
    if part_size < 1:
        raise ValueError("the part size is at least one byte")
    written, lines = [], []
    with open(path, "rb") as stream:
        index = 0
        while True:
            block = stream.read(part_size)
            if not block and index:
                break
            part = target.with_name(f"{target.name}.{index:03d}")
            part.write_bytes(block)
            written.append(part.name)
            lines.append(f"{sha256(block)}  {part.name}\n")
            index += 1
            if len(block) < part_size:
                break
    target.with_name(f"{target.name}.sha256").write_text(
        "".join(lines), encoding="utf-8"
    )
    return written


def assets(directory):
    """The assets in ``directory``, named as :func:`asset_name` names
    them, by their listings (``<asset>.sha256``)."""
    return [
        path.name[: -len(".sha256")]
        for path in sorted(Path(directory).glob("*.public.tar.gz.sha256"))
    ]


def check(directory):
    """The problems of every asset in ``directory`` (module docstring)."""
    directory = Path(directory)
    names = assets(directory)
    if not names:
        return [f"{directory} holds no asset listing"]
    problems, listed = [], set()
    for name in names:
        problems += _check_asset(directory, name)
        for line in (
            (directory / f"{name}.sha256")
            .read_text(encoding="utf-8")
            .splitlines()
        ):
            listed.add(line.partition("  ")[2])
    problems += [
        f"{path.name} is a part that no listing names"
        for path in sorted(directory.glob("*.public.tar.gz.[0-9]*"))
        if path.name not in listed
    ]
    return problems


def _check_asset(directory, asset):
    """The problems of the asset ``asset`` in ``directory``."""
    listing = directory / f"{asset}.sha256"
    problems, parts = [], []
    for line in listing.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        path = directory / name
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif contract.sha256_file(path) != digest:
            problems.append(f"{name} is not the part its listing hashes")
        parts.append(path)
    if problems:
        return problems
    stream = io.BytesIO(b"".join(path.read_bytes() for path in parts))
    found = {}
    with tarfile.open(fileobj=stream, mode="r:gz") as archive:
        for member in archive:
            data = archive.extractfile(member).read()
            found[member.name] = data
    publics = [name for name in found if name.endswith(f"/{PUBLIC}")]
    if len(publics) != 1:
        return [f"{asset} holds {len(publics)} {PUBLIC}, not one"]
    prefix = publics[0][: -len(PUBLIC)]
    record = json.loads(found.pop(publics[0]))
    listed = {f"{prefix}{name}" for name in record["files"]}
    problems += [
        f"{name} is not in {PUBLIC}" for name in found.keys() - listed
    ]
    problems += [f"{name} is missing" for name in listed - found.keys()]
    problems += [
        f"{prefix}{name} is not the file {PUBLIC} hashes"
        for name, entry in record["files"].items()
        if f"{prefix}{name}" in found
        and sha256(found[f"{prefix}{name}"]) != entry["sha256"]
    ]
    return problems


def cmd_build(args):
    paths = []
    for item in args.path or ():
        name, equals, value = item.partition("=")
        if not equals or not value:
            print(f"--path {item!r} is not NAME=PATH", file=sys.stderr)
            return 1
        paths.append((name, value))
    try:
        build(
            args.campaign,
            args.copies,
            args.out,
            paths=paths,
            allow=list(args.allow or ()),
            part_size=args.part_size,
        )
    except (contract.ContractError, OSError, ValueError) as error:
        print(f"refusing: {error}", file=sys.stderr)
        return 1
    return 0


def cmd_check(args):
    problems = check(args.directory)
    for problem in problems:
        print(f"check: {problem}")
    if not problems:
        names = ", ".join(assets(args.directory))
        print(f"check: every asset in {args.directory} is whole ({names})")
    return 1 if problems else 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="build a campaign's asset")
    build_parser.add_argument("--campaign", required=True)
    build_parser.add_argument("--copies", required=True)
    build_parser.add_argument("--out", required=True)
    build_parser.add_argument("--path", action="append")
    build_parser.add_argument("--allow", action="append")
    build_parser.add_argument("--part-size", type=int, default=PART_SIZE)
    build_parser.set_defaults(func=cmd_build)
    check_parser = sub.add_parser("check", help="re-read an asset")
    check_parser.add_argument("directory")
    check_parser.set_defaults(func=cmd_check)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
