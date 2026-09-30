"""Map the line citations of a tracked document onto the current code.

For every citation `pyvbmc/<path>.py:<N>` (and the bare `:<N>` tokens that
follow it on the same or the next lines of the same entry), take the commit
that last touched that line of the document (git blame), where the citation
is presumed right, and carry line N of the cited file from that commit to
the working tree through `git diff`. A citation whose line moved unchanged
is rewritten; one whose line was itself changed or removed is reported for
a reading by hand. Citations of MATLAB files and of other documents are left
alone, and so is a bare `:<N>` that follows the name of such a file, whether
or not the name carries a line number of its own: any file named in the text
ends the run of the Python path before it. A bare citation beyond the end of
the file it would be carried through is reported and left alone, and a
Python file cited without its `pyvbmc/` path is reported, since it is not
carried. The mapping is as good as the citation was at that commit: it moves
a citation with its line and does not judge whether the line was the right
one.

Run from the root of the repository, first without `--write` to read what
would move:

    python dev/experiments/port_review_20260919/refresh_citations.py DOC [--write]
"""

import re
import subprocess
import sys

DOC = sys.argv[1]
WRITE = "--write" in sys.argv


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, check=True
    ).stdout.decode("utf-8", errors="replace")


def blame_commits(doc):
    out = git("blame", "--line-porcelain", doc)
    commits = []
    for line in out.splitlines():
        if re.match(r"^[0-9a-f]{40} \d+ \d+", line):
            commits.append(line.split()[0])
    return commits


_diff_cache = {}


def line_map(commit, path):
    """Return f(old_line) -> new_line or None if the line was changed."""
    key = (commit, path)
    if key in _diff_cache:
        return _diff_cache[key]
    if commit.startswith("0000000"):
        _diff_cache[key] = lambda n: n
        return _diff_cache[key]
    diff = git("diff", "-U0", commit, "--", path)
    hunks = []
    for m in re.finditer(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff, re.M
    ):
        o, ol, n, nl = m.groups()
        hunks.append(
            (
                int(o),
                1 if ol is None else int(ol),
                int(n),
                1 if nl is None else int(nl),
            )
        )

    def f(old):
        shift = 0
        for o, ol, n, nl in hunks:
            if ol == 0:
                # pure insertion after old line o
                if old > o:
                    shift += nl
                continue
            if old < o:
                break
            if o <= old < o + ol:
                return None
            shift += nl - ol
        return old + shift

    _diff_cache[key] = f
    return f


_text_cache = {}


def old_lines(commit, path):
    """The lines of the cited file at the commit, or None when unknown."""
    key = (commit, path)
    if key not in _text_cache:
        try:
            if commit.startswith("0000000"):
                with open(path, encoding="utf-8") as f:
                    _text_cache[key] = f.read().splitlines()
            else:
                _text_cache[key] = git("show", f"{commit}:{path}").splitlines()
        except (subprocess.CalledProcessError, OSError):
            _text_cache[key] = None
    return _text_cache[key]


def old_text(commit, path, n):
    lines = old_lines(commit, path)
    if lines is None:
        return None
    return lines[n - 1] if 0 < n <= len(lines) else None


raw = open(DOC, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in raw else "\n"
lines = raw.split(nl)
commits = blame_commits(DOC)
if len(commits) != len(lines) and len(commits) != len(lines) - 1:
    print("blame/line mismatch", len(commits), len(lines))

# A file name counts with or without a line number of its own: a sentence
# that names `private/vbmc_warmup.m` and then cites `:97-102` cites that
# file, not the Python path before it.
token = re.compile(
    r"(?P<path>pyvbmc/[\w/]+\.py)\b(?::(?P<a>\d+)(?:-(?P<b>\d+))?)?"
    r"|(?P<matlab>[\w/]+\.m)\b(?::\d+(?:-\d+)?)?"
    r"|(?P<other>[\w/.]+\.(?:py|md|ini))\b(?P<oline>:\d+(?:-\d+)?)?"
    r"|(?<![\w/.])`?:(?P<c>\d+)(?:-(?P<d>\d+))?"
)

current = None
n_ok = n_same = n_manual = n_left = 0
for i, text in enumerate(lines):
    if text.startswith("#") or not text.strip():
        current = None if text.startswith("#") else current
    commit = commits[i] if i < len(commits) else commits[-1]

    def repl(m):
        global current, n_ok, n_same, n_manual, n_left
        if m.group("matlab") or m.group("other"):
            current = None
            other = m.group("other") or ""
            if other.endswith(".py") and m.group("oline"):
                n_manual += 1
                print(
                    f"MANUAL {DOC}:{i+1}: {m.group(0)}  (a Python file "
                    "cited without its pyvbmc/ path is not carried)"
                )
            return m.group(0)
        if m.group("path"):
            current = m.group("path")
            a, b = m.group("a"), m.group("b")
            if a is None:
                return m.group(0)
        else:
            if current is None:
                n_left += 1
                return m.group(0)
            a, b = m.group("c"), m.group("d")
            cited = old_lines(commit, current)
            if cited is not None and int(b or a) > len(cited):
                n_manual += 1
                print(
                    f"MANUAL {DOC}:{i+1}: :{a}"
                    + (f"-{b}" if b else "")
                    + f"  (beyond the {len(cited)} lines of {current}: it "
                    "cites another file)"
                )
                return m.group(0)
        f = line_map(commit, current)
        new = []
        for v in (a, b):
            if v is None:
                new.append(None)
                continue
            mapped = f(int(v))
            new.append(mapped)
        if new[0] is None or (b is not None and new[1] is None):
            n_manual += 1
            print(
                f"MANUAL {DOC}:{i+1}: {current}:{a}"
                + (f"-{b}" if b else "")
                + f"  (was: {old_text(commit, current, int(a))!r})"
            )
            return m.group(0)
        if new[0] == int(a) and (b is None or new[1] == int(b)):
            n_same += 1
            return m.group(0)
        n_ok += 1
        rng = f"{new[0]}" + (f"-{new[1]}" if b else "")
        print(
            f"MOVE   {DOC}:{i+1}: {current}:{a}"
            + (f"-{b}" if b else "")
            + f" -> {rng}"
        )
        prefix = m.group(0)[: m.group(0).rfind(":") + 1]
        return prefix + rng

    lines[i] = token.sub(repl, text)

print(
    f"moved {n_ok}, unchanged {n_same}, to read by hand {n_manual}, "
    f"bare citations of other files left alone {n_left}"
)
if WRITE:
    open(DOC, "w", encoding="utf-8", newline="").write(nl.join(lines))
    print("written")
