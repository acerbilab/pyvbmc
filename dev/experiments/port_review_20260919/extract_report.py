"""Save a reviewer's final message verbatim from its transcript.

Usage: extract_report.py <agent.jsonl> <title substring> <header file> <out file> [candidate]

The report is the last string of the agent's own (assistant) output that
holds the report's title. It is written after the header, unedited.

An agent that hands its report back through a tool call and then closes
with a short message under the same title leaves two such strings, the
report first. Every candidate is listed with its line and length; the
optional fifth argument is the index of the one to take (0 for the first,
-1, the default, for the last).
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
    pick = int(sys.argv[5]) if len(sys.argv) > 5 else -1
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
    for i, (n, path, s) in enumerate(found):
        print(f"candidate {i}: line {n}, {path}, {len(s)} chars")
    n, path, report = found[pick]
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
