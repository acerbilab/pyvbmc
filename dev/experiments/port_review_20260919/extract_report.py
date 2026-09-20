"""Save a reviewer's final message verbatim from its transcript.

Usage: extract_report.py <agent.jsonl> <title substring> <header file> <out file>

The report is the last string of the agent's own (assistant) output that
holds the report's title. It is written after the header, unedited.
"""

import json
import sys


def strings(node, path=""):
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from strings(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from strings(value, f"{path}[{i}]")


def main():
    jsonl, title, header_path, out_path = sys.argv[1:5]
    found = []
    with open(jsonl, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            message = obj.get("message", {})
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            for path, s in strings(message.get("content")):
                if title in s:
                    found.append((n, path, s))
    if not found:
        sys.exit(f"no assistant string holds the title {title!r}")
    n, path, report = found[-1]
    with open(header_path, encoding="utf-8") as f:
        header = f.read()
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write(report.replace("\r\n", "\n"))
        if not report.endswith("\n"):
            f.write("\n")
    lines = report.splitlines()
    print(f"candidates: {len(found)}; taken from line {n}, {path}")
    print(f"chars: {len(report)}; lines: {len(lines)}")
    print(f"first: {lines[0][:100]}")
    print(f"last:  {lines[-1][:100]}")
    print(f"starts with title: {report.lstrip().startswith(title)}")


if __name__ == "__main__":
    main()
