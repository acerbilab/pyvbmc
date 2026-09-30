"""Put the film's sound under its recorded video, at -16 LUFS.

    python scripts/mux.py VIDEO AUDIO OUT

Two passes of ffmpeg's EBU R128 loudness normalization bring the sound to
-16 LUFS integrated and -1.5 dBTP true peak, the level that video
platforms play at. The video stream is copied, and the sound is cut to the
video's length. ffmpeg is FFMPEG, or ffmpeg on the PATH.
"""

import argparse
import json
import os
import subprocess

TARGET = "I=-16:TP=-1.5:LRA=11"


def measure(ffmpeg, path):
    p = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-i",
            path,
            "-vn",
            "-af",
            f"loudnorm={TARGET}:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(
        p.stderr[p.stderr.rindex("{") : p.stderr.rindex("}") + 1]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("audio")
    ap.add_argument("out")
    args = ap.parse_args()
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    m = measure(ffmpeg, args.audio)
    print(f"measured {m['input_i']} LUFS, {m['input_tp']} dBTP", flush=True)
    loudnorm = (
        f"loudnorm={TARGET}:measured_I={m['input_i']}:"
        f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
        f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:"
        "linear=true"
    )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            args.video,
            "-i",
            args.audio,
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-af",
            loudnorm,
            "-ar",
            "48000",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
            "-shortest",
            "-movflags",
            "+faststart",
            args.out,
        ],
        check=True,
    )
    after = measure(ffmpeg, args.out)
    print(
        f"wrote {args.out}: {after['input_i']} LUFS, "
        f"{after['input_tp']} dBTP",
        flush=True,
    )


if __name__ == "__main__":
    main()
