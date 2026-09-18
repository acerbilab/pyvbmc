"""Publish review copies of local campaign records into the tracked tree.

Raw campaign records live in the gitignored local artifact directory and
carry absolute paths of the machine that produced them. A review copy
replaces the user's home directory in every string with ``<USER_HOME>``,
adds ``publication_note`` and ``raw_artifact_sha256`` (the hash of the
original bytes, which binds the copy to the local record), and is written
next to the other tracked records with a name prefix. A publication index
lists every copy with both hashes.

    python dev/scripts/noisy_acq_publish_records.py --source-dir RAW_DIR \
        --dest TRACKED_DIR --prefix f2_ --index f2_publication_index.json \
        name1.json name2.json ...
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

PUBLICATION_NOTE = (
    "Review copy with user-directory paths redacted. The raw artifact hash"
    " binds the original local bytes; execute the local manifest."
)


def _redact(value: Any, homes: list[str]) -> Any:
    if isinstance(value, str):
        for home in homes:
            value = value.replace(home, "<USER_HOME>")
        return value
    if isinstance(value, list):
        return [_redact(item, homes) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, homes) for key, item in value.items()}
    return value


def _homes() -> list[str]:
    home = Path.home()
    return [str(home), home.as_posix(), str(home).replace("\\", "\\\\")]


def publish(
    source_dir: Path, dest: Path, prefix: str, names: list[str]
) -> dict[str, Any]:
    dest.mkdir(parents=True, exist_ok=True)
    homes = _homes()
    entries = []
    for name in names:
        raw = source_dir / name
        data = json.loads(raw.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError(f"{name} is not a JSON object")
        raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
        copy = _redact(data, homes)
        copy["publication_note"] = PUBLICATION_NOTE
        copy["raw_artifact_sha256"] = raw_sha
        target = dest / f"{prefix}{name}"
        target.write_text(
            json.dumps(copy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        entries.append(
            {
                "published": target.name,
                "raw_name": name,
                "raw_artifact_sha256": raw_sha,
                "published_sha256": hashlib.sha256(
                    target.read_bytes()
                ).hexdigest(),
            }
        )
    return {
        "kind": "publication_index",
        "schema_version": 1,
        "created_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "source_dir": _redact(str(source_dir), homes),
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--index", required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args(argv)
    index = publish(args.source_dir, args.dest, args.prefix, args.names)
    (args.dest / args.index).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for entry in index["entries"]:
        print(entry["published"], entry["raw_artifact_sha256"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
