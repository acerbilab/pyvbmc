"""Transcribe the voiced lines back and list those whose words differ from the script.

    python scripts/verify_voice.py OUT [--model small.en]

Reads ../narration.json and the clips OUT/voice/<line id>.wav that
make_voice.py wrote, transcribes each clip with Whisper, and compares its
words with the line's text, ignoring case and punctuation. The spelled-out
name in the script ("Pie V B M C") and Whisper's ways of writing what it hears
("Pi VBMC", "pi-V-B-M-C") all count as "PyVBMC". Writes every transcript to
OUT/whisper.txt, prints the lines that differ, and exits with status 1 when
any does. A line can differ because Whisper mishears a short line out of its
context; listen to it before voicing it again.

Needs faster-whisper, which fetches the model into HF_HOME on first use.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
NAME = re.compile(r"\b(?:pie|pi|py)\s*v\s*b\s*m\s*c\b")


def words(text):
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("-", " "))
    return NAME.sub("pyvbmc", " ".join(text.split())).split()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="small.en")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    spec = json.loads((HERE / "narration.json").read_text(encoding="utf-8"))
    lines = [line for scene in spec["scenes"] for line in scene["lines"]]
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    heard, differ = {}, []
    for line in lines:
        segments, _ = model.transcribe(
            str(args.out / "voice" / f"{line['id']}.wav"),
            beam_size=5,
            language="en",
        )
        heard[line["id"]] = " ".join(s.text.strip() for s in segments)
        if words(heard[line["id"]]) != words(line["text"]):
            differ.append(line)
    with open(args.out / "whisper.txt", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line['id']:4s} {heard[line['id']]}\n")
    for line in differ:
        print(f"{line['id']:4s} script: {line['text']}", flush=True)
        print(f"     heard : {heard[line['id']]}", flush=True)
    print(
        f"{len(lines) - len(differ)} of {len(lines)} lines as written; "
        f"transcripts in {args.out / 'whisper.txt'}",
        flush=True,
    )
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
