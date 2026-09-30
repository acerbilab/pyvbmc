"""Voice the video's narration with Kokoro and time the scenes from it.

    python scripts/make_voice.py OUT [--voice NAME] [--speed X] [--only ID,...]

Reads ../narration.json. Writes ../film_timeline.js, the timeline that
film.html plays (``window.VBMC_FILM``: each scene's start and length, and
each line's start, length and caption, in seconds), and into OUT, a folder
outside the repository:

    voice/<line id>.wav   one 48 kHz mono clip per line
    timeline.json         the same timeline
    narration.wav         the clips placed on the timeline, for the video's sound
    narration.srt         the captions

A scene lasts lead + the lines and the gaps between them + tail, or its "min"
if that is longer: the pictures of some scenes need more time than their words.
Re-voicing a line therefore re-times every scene after it. A line's "text" is
what the voice says and its "caption", when given, is what the screen shows.
A line is voiced when OUT has no clip of it, when --only names it or its
scene, or with --all; the other clips in OUT are reused. Kokoro does not
give the same take twice, so voicing a line again moves the timeline.

Needs a Python with kokoro (0.9.4 or later), soundfile and scipy. Kokoro
fetches its weights from the Hugging Face hub into HF_HOME on first use.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent.parent
SR_TTS, SR_OUT = 24000, 48000
TARGET_RMS_DB, PEAK_DB = -20.0, -1.0
PRE_ROLL, POST_ROLL = 0.05, 0.12


def db(x):
    return 10.0 ** (x / 20.0)


def trim_and_level(audio, sr):
    """Trim the silence at both ends and level the voiced part to TARGET_RMS_DB."""
    win = int(0.02 * sr)
    env = np.sqrt(np.convolve(audio**2, np.ones(win) / win, mode="same"))
    thresh = max(env.max(), 1e-9) * db(-40.0)
    voiced = np.nonzero(env > thresh)[0]
    if voiced.size == 0:
        return audio.astype(np.float32)
    a = max(0, voiced[0] - int(PRE_ROLL * sr))
    b = min(audio.size, voiced[-1] + int(POST_ROLL * sr))
    clip = audio[a:b].astype(np.float64)
    rms = np.sqrt(np.mean(clip[env[a:b] > thresh] ** 2))
    clip *= db(TARGET_RMS_DB) / max(rms, 1e-9)
    clip *= min(1.0, db(PEAK_DB) / np.abs(clip).max())
    fade = int(0.005 * sr)
    clip[:fade] *= np.linspace(0, 1, fade)
    clip[-fade:] *= np.linspace(1, 0, fade)
    return clip.astype(np.float32)


def synthesize(pipeline, text, voice, speed):
    chunks = []
    for result in pipeline(
        text, voice=voice, speed=speed, split_pattern=r"\n+"
    ):
        audio = result.audio
        if audio is not None:
            chunks.append(
                np.asarray(
                    audio.detach().cpu() if hasattr(audio, "detach") else audio
                )
            )
    if not chunks:
        raise RuntimeError(f"Kokoro returned no audio for {text!r}")
    return np.concatenate(chunks)


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--voice")
    ap.add_argument("--speed", type=float)
    ap.add_argument("--only", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    spec = json.loads((HERE / "narration.json").read_text(encoding="utf-8"))
    voice = args.voice or spec["voice"]
    speed = args.speed or spec["speed"]
    dflt = spec["defaults"]
    only = {s for s in args.only.split(",") if s}
    voice_dir = args.out / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)

    todo = []
    for scene in spec["scenes"]:
        for line in scene["lines"]:
            path = voice_dir / f"{line['id']}.wav"
            if (
                not path.exists()
                or args.all
                or line["id"] in only
                or scene["id"] in only
            ):
                todo.append((line, path))
    if todo:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")
        for line, path in todo:
            clip = trim_and_level(
                synthesize(pipeline, line["text"], voice, speed), SR_TTS
            )
            sf.write(
                path,
                resample_poly(clip, SR_OUT // SR_TTS, 1),
                SR_OUT,
                subtype="PCM_16",
            )
            print(
                f"  {line['id']:4s} {sf.info(path).duration:5.2f}s  {line['text']}",
                flush=True,
            )

    timeline = {"voice": voice, "speed": speed, "scenes": []}
    t0, placed, captions = 0.0, [], []
    for scene in spec["scenes"]:
        t = scene.get("lead", dflt["lead"])
        lines = []
        for i, line in enumerate(scene["lines"]):
            if i:
                t += line.get("gap", dflt["gap"])
            path = voice_dir / f"{line['id']}.wav"
            dur = sf.info(path).duration
            lines.append(
                {
                    "id": line["id"],
                    "caption": line.get("caption", line["text"]),
                    "start": round(t, 3),
                    "seconds": round(dur, 3),
                }
            )
            placed.append((t0 + t, path))
            captions.append(
                (t0 + t, t0 + t + dur, line.get("caption", line["text"]))
            )
            t += dur
        if scene["lines"]:
            t += scene.get("tail", dflt["tail"])
        length = max(t, scene.get("min", 0.0))
        timeline["scenes"].append(
            {
                "id": scene["id"],
                "start": round(t0, 3),
                "seconds": round(length, 3),
                "lines": lines,
            }
        )
        t0 += length
    timeline["seconds"] = round(t0, 3)
    (args.out / "timeline.json").write_text(
        json.dumps(timeline, indent=1), encoding="utf-8"
    )
    payload = json.dumps(timeline, separators=(",", ":"))
    (HERE / "film_timeline.js").write_text(
        f"window.VBMC_FILM={payload};\n", encoding="utf-8"
    )

    mix = np.zeros(int(np.ceil(t0 * SR_OUT)) + SR_OUT, np.float32)
    for start, path in placed:
        clip = sf.read(path, dtype="float32")[0]
        a = int(round(start * SR_OUT))
        mix[a : a + clip.size] += clip
    sf.write(args.out / "narration.wav", mix, SR_OUT, subtype="PCM_16")
    with open(args.out / "narration.srt", "w", encoding="utf-8") as f:
        for k, (a, b, text) in enumerate(captions, 1):
            f.write(f"{k}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n")

    print(f"\n{voice} at {speed}: {t0:.1f} s", flush=True)
    for s in timeline["scenes"]:
        print(
            f"  {s['id']:10s} {s['start']:6.1f}  +{s['seconds']:5.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
