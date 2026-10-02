"""ElevenLabs text-to-speech for make_voice.py: one take per scene, cut at the pauses.

A scene is voiced in one request, so that its delivery flows from line to line,
with the character timings that ElevenLabs returns beside the audio; the lines
are then cut out of the take inside the pauses between them, and levelled with
one gain for the whole scene.

The API key is read from $ELEVENLABS_API_KEY, or else from
~/.config/elevenlabs/api_key. It is never printed or written anywhere else.
"""

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

API = "https://api.elevenlabs.io"


def api_key():
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        path = Path.home() / ".config" / "elevenlabs" / "api_key"
        if not path.exists():
            sys.exit(
                "No ElevenLabs API key: set ELEVENLABS_API_KEY or create "
                f"{path}"
            )
        key = path.read_text(encoding="ascii").strip()
    return key


def request(method, path, body=None, accept="application/json"):
    """The response body, and its headers."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("xi-api-key", api_key())
    req.add_header("Accept", accept)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:400]
        sys.exit(f"ElevenLabs {method} {path}: HTTP {e.code} {detail}")


def scene_text(texts):
    """A scene's lines joined into one request, and each line's span in it."""
    text, spans = "", []
    for t in texts:
        if text:
            text += " "
        spans.append((len(text), len(text) + len(t)))
        text += t
    return text, spans


def scene_take(
    text,
    voice_id,
    model_id,
    settings,
    seed,
    previous_request_ids=None,
    next_text=None,
    previous_text=None,
    sr=24000,
):
    """One take of a whole scene: samples, character timings and request id.

    ``previous_request_ids`` (valid for two hours) condition the take on the
    audio of earlier takes; without them, ``previous_text`` gives at least
    the words that come before.
    """
    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": settings,
        "seed": seed,
    }
    if previous_request_ids:
        body["previous_request_ids"] = previous_request_ids
    elif previous_text:
        body["previous_text"] = previous_text
    if next_text:
        body["next_text"] = next_text
    raw, headers = request(
        "POST",
        f"/v1/text-to-speech/{voice_id}/with-timestamps?output_format=pcm_{sr}",
        body,
    )
    res = json.loads(raw)
    pcm = np.frombuffer(base64.b64decode(res["audio_base64"]), dtype="<i2")
    rid = next(
        (v for k, v in headers.items() if k.lower() == "request-id"), None
    )
    return pcm.astype(np.float32) / 32768.0, res["alignment"], rid


def split_scene(pcm, alignment, spans, sr=24000):
    """Cut each line (a character span of the scene's text) out of its take.

    Returns, per line, the samples from just before its first word to where
    its last word has died away, with short fades. The boundary between two
    lines is the quietest point of the pause between them, searched a little
    beyond the aligned pause, because the character timings can be off by a
    few tens of milliseconds; a line keeps nothing beyond its boundaries.
    """
    hop = int(0.01 * sr)
    n = pcm.size // hop
    level = 10 * np.log10(
        np.mean(pcm[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12
    )
    level -= level.max()
    starts = np.array(alignment["character_start_times_seconds"])
    ends = np.array(alignment["character_end_times_seconds"])
    chars = alignment["characters"]
    speech = -35.0  # dB below the take's peak: clearly voiced
    t_on, t_off = [], []
    for a, b in spans:
        idx = [i for i in range(a, b) if chars[i].isalnum()]
        t_on.append(starts[idx[0]])
        t_off.append(ends[idx[-1]])
    # The noise floor: the quietest tenth of the frames in the pauses.
    pauses = [
        level[int(t_off[k] * 100) + 5 : int(t_on[k + 1] * 100) - 5]
        for k in range(len(spans) - 1)
    ]
    pause = np.concatenate(pauses) if pauses else level
    floor = float(np.percentile(pause if pause.size else level, 10))
    smooth = np.convolve(level, np.ones(3) / 3, mode="same")
    bounds = []
    for k in range(len(spans) - 1):
        lo = int(max(t_on[k], t_off[k] - 0.05) * 100)
        hi = int(min(t_off[k + 1], t_on[k + 1] + 0.05) * 100)
        bounds.append(
            (lo + int(np.argmin(smooth[lo:hi]))) / 100
            if hi > lo
            else (t_off[k] + t_on[k + 1]) / 2
        )
    out = []
    for k in range(len(spans)):
        start = bounds[k - 1] if k > 0 else 0.0
        stop = bounds[k] if k + 1 < len(spans) else pcm.size / sr
        lo = int(max(start, t_on[k] - 0.15) * 100)
        hi = int(min(t_off[k], t_on[k] + 0.15) * 100)
        voiced = np.nonzero(level[lo:hi] > speech)[0]
        onset = (lo + voiced[0]) / 100 if voiced.size else t_on[k]
        lo = int(max(t_on[k], t_off[k] - 0.15) * 100)
        hi = int(min(n, stop * 100, (t_off[k] + 0.25) * 100))
        voiced = np.nonzero(level[lo:hi] > speech)[0]
        offset = (lo + voiced[-1] + 1) / 100 if voiced.size else t_off[k]
        head = max(onset - 0.04, start)
        # The tail runs until the sound reaches the floor, at most 250 ms, and
        # never into the next line or a breath before it (energy rising again).
        limit = min(offset + 0.25, stop)
        f = best = int(offset * 100)
        while f < int(limit * 100) and f < n:
            if level[f] < level[best]:
                best = f
            if level[f] <= floor + 3 or level[f] > level[best] + 6:
                break
            f += 1
        tail = max(best, int(offset * 100)) / 100
        clip = pcm[int(head * sr) : int(tail * sr)].copy()
        fin, fout = int(0.01 * sr), int(0.045 * sr)
        clip[:fin] *= np.sin(np.linspace(0, np.pi / 2, fin)) ** 2
        clip[-fout:] *= np.cos(np.linspace(0, np.pi / 2, fout)) ** 2
        out.append(clip)
    return out


def scene_gain(clips, sr, target_db=-20.0, ceiling_db=-1.0):
    """One gain for a scene: voiced RMS to ``target_db``, peaks under ``ceiling_db``.

    Levelling the scene, not each line, keeps the take's dynamics from line
    to line.
    """
    allv = np.concatenate(clips)
    win = int(0.02 * sr)
    env = np.sqrt(np.convolve(allv**2, np.ones(win) / win, mode="same"))
    active = env > env.max() * 10 ** (-40 / 20)
    gain = 10 ** (target_db / 20) / max(
        np.sqrt(np.mean(allv[active] ** 2)), 1e-9
    )
    return min(gain, 10 ** (ceiling_db / 20) / max(np.abs(allv).max(), 1e-9))
