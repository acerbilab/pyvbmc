"""Voice the video's narration and time the scenes from it.

    python scripts/make_voice.py OUT [--only ID,...] [--all]

Reads ../narration.json. Writes ../film_timeline.js, the timeline that
film.html plays (``window.VBMC_FILM``: each scene's start and length, and
each line's start, length and caption, in seconds), and into OUT, a folder
outside the repository that holds one voice's clips:

    voice/<line id>.wav   one 48 kHz mono clip per line
    timeline.json         the same timeline
    narration.wav         the clips placed on the timeline, for the video's sound
    narration.srt         the captions

A scene lasts lead + the lines and the gaps between them + tail, or its "min"
if that is longer: the pictures of some scenes need more time than their words.
Re-voicing a line therefore re-times every scene after it. A line's "text" is
what the voice says and its "caption", when given, is what the screen shows.

The "engine" of narration.json picks the voice.
- "elevenlabs" (the settings under "elevenlabs") voices each scene in one take
  (scripts/eleven.py), so that the delivery flows from line to line, and cuts
  the lines out of it at the pauses. Takes are kept in OUT/takes/<voice>/
  and reused while a scene's text is unchanged; --only re-takes the scenes it
  names, or the scenes of the lines it names, with a new seed, and --all
  re-takes every scene. A take costs ElevenLabs characters.
- "kokoro" (the "voice" and "speed" at the top) voices each line on its own,
  locally. A line is voiced when OUT has no clip of it, when --only names it
  or its scene, or with --all. Kokoro does not give the same take twice, so
  voicing a line again moves the timeline.

Needs numpy, scipy and soundfile, and kokoro (0.9.4 or later) for the Kokoro
engine, which fetches its weights from the Hugging Face hub into HF_HOME.
"""

import argparse
import json
import time
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


def write_clip(path, clip):
    sf.write(
        path,
        resample_poly(clip, SR_OUT // SR_TTS, 1).astype(np.float32),
        SR_OUT,
        subtype="PCM_16",
    )


def kokoro_lines(spec, voice_dir, only, everything):
    """Voice with Kokoro the lines that need it; the voice's name and speed."""
    voice, speed = spec["voice"], spec["speed"]
    todo = [
        (line, voice_dir / f"{line['id']}.wav")
        for scene in spec["scenes"]
        for line in scene["lines"]
        if everything
        or line["id"] in only
        or scene["id"] in only
        or not (voice_dir / f"{line['id']}.wav").exists()
    ]
    if todo:
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")
        for line, path in todo:
            raw = synthesize(pipeline, line["text"], voice, speed)
            write_clip(path, trim_and_level(raw, SR_TTS))
            print(
                f"  {line['id']:4s} {sf.info(path).duration:5.2f}s  {line['text']}",
                flush=True,
            )
    return voice, speed


def eleven_lines(spec, out, voice_dir, only, everything):
    """Voice with ElevenLabs, one take per scene; the voice's name and speed."""
    from eleven import scene_gain, scene_take, scene_text, split_scene

    cfg = spec["elevenlabs"]
    takes = out / "takes" / cfg["voice_name"].lower()
    takes.mkdir(parents=True, exist_ok=True)
    settings = {
        "stability": cfg["stability"],
        "similarity_boost": cfg["similarity"],
        "style": cfg["style"],
        "use_speaker_boost": True,
        "speed": cfg["speed"],
    }
    scenes = [s for s in spec["scenes"] if s["lines"]]
    prev = None
    for k, scene in enumerate(scenes):
        text, spans = scene_text([line["text"] for line in scene["lines"]])
        meta_path = takes / f"{scene['id']}.json"
        wav_path = takes / f"{scene['id']}.wav"
        meta = (
            json.loads(meta_path.read_text(encoding="utf-8"))
            if meta_path.exists()
            else None
        )
        named = (
            everything
            or scene["id"] in only
            or any(line["id"] in only for line in scene["lines"])
        )
        if (
            meta is None
            or meta["text"] != text
            or not wav_path.exists()
            or named
        ):
            seed = (
                cfg["seed"] + k
                if meta is None
                else meta["seed"] + 1000 * named
            )
            # The previous take's request id conditions this take on its audio,
            # but only for two hours; otherwise its last line's text does.
            fresh = (
                prev is not None
                and prev.get("request_id")
                and time.time() - prev["created"] < 7000
            )
            nxt = (
                scenes[k + 1]["lines"][0]["text"]
                if k + 1 < len(scenes)
                else None
            )
            pcm, alignment, rid = scene_take(
                text,
                cfg["voice_id"],
                cfg["model"],
                settings,
                seed,
                previous_request_ids=[prev["request_id"]] if fresh else None,
                next_text=nxt,
                previous_text=(
                    None
                    if fresh or k == 0
                    else scenes[k - 1]["lines"][-1]["text"]
                ),
                sr=SR_TTS,
            )
            sf.write(wav_path, pcm, SR_TTS, subtype="PCM_16")
            meta = {
                "text": text,
                "seed": seed,
                "request_id": rid,
                "created": time.time(),
                "alignment": alignment,
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            print(
                f"  take {scene['id']:10s} {pcm.size / SR_TTS:5.1f}s "
                f"(seed {seed}, {len(text)} characters)",
                flush=True,
            )
        pcm = sf.read(wav_path, dtype="float32")[0]
        clips = split_scene(pcm, meta["alignment"], spans, SR_TTS)
        gain = scene_gain(clips, SR_TTS, TARGET_RMS_DB, PEAK_DB)
        for line, clip in zip(scene["lines"], clips):
            write_clip(voice_dir / f"{line['id']}.wav", clip * gain)
        prev = meta
    return f"elevenlabs:{cfg['voice_name']}", cfg["speed"]


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--only", default="")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    spec = json.loads((HERE / "narration.json").read_text(encoding="utf-8"))
    dflt = spec["defaults"]
    only = {s for s in args.only.split(",") if s}
    voice_dir = args.out / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    if spec.get("engine", "kokoro") == "elevenlabs":
        voice, speed = eleven_lines(spec, args.out, voice_dir, only, args.all)
    else:
        voice, speed = kokoro_lines(spec, voice_dir, only, args.all)

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
