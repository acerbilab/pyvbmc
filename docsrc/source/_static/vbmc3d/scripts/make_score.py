"""Synthesize the film's score and mix it under the narration.

    python scripts/make_score.py OUT

Reads OUT/events.json, the times at which film.html shows things (written
by ``node scripts/record.mjs "film.html?events=1" OUT/events.json``), and
OUT/narration.wav (scripts/make_voice.py). Writes into OUT, as 48 kHz
stereo:

    score.wav   the score alone
    mix.wav     the narration with the score under it

``scripts/mux.py`` puts the mix under the recorded video and sets its
loudness.

The score follows what the film shows. Its harmony is D minor while the
film shows the cost of a posterior, and turns to D major when the true
landscape is revealed and again for the wordmark. Every evaluation on
screen sounds: the MCMC chain as clicks that thicken into a roar, the
PyBADS run as warm pings, PyVBMC's evaluations as glassy pings pitched by
the height they reveal. A high tone wobbles in pitch while the surrogate
is unsure and steadies as the time-lapse runs, the three steps of the loop
play an arpeggio that speeds up with the iterations, and the score ducks
under the voice.

Needs numpy, scipy and soundfile.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt, sosfiltfilt

SR = 48000
rng = np.random.default_rng(3)

# Chords as MIDI notes; notes below 40 are played as sines (the bass).
CHORDS = {
    "Dm": [26, 38, 45],
    "Dm9": [38, 50, 53, 57, 60, 64],
    "Bbmaj7": [34, 46, 50, 53, 57],
    "Gm9": [31, 43, 46, 50, 53, 57],
    "A7sus": [33, 45, 50, 52, 55],
    "Fmaj9": [29, 41, 45, 48, 52, 55],
    "C69": [36, 48, 52, 55, 57, 62],
    "Ebmaj7": [39, 51, 55, 58, 62],
    "Dmaj9": [38, 50, 54, 57, 61, 64],
    "Gmaj7": [31, 43, 47, 50, 54],
    "Bm7": [35, 47, 50, 54, 57],
    "Gmaj9": [31, 43, 47, 50, 54, 57],
    "A6": [33, 45, 49, 52, 54],
}
PENTA_MINOR = [0, 3, 5, 7, 10]  # D F G A C


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def lowpass(x, hz, order=2):
    return sosfilt(
        butter(order, hz, btype="low", fs=SR, output="sos"), x, axis=0
    )


def highpass(x, hz, order=2):
    return sosfilt(
        butter(order, hz, btype="high", fs=SR, output="sos"), x, axis=0
    )


def pan(mono, p):
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.stack([mono * np.cos(a), mono * np.sin(a)], axis=1)


def put(bus, t0, x):
    """Add the stereo or mono signal x into bus from time t0."""
    i = int(round(t0 * SR))
    if i >= bus.shape[0] or i + len(x) <= 0:
        return
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    a, b = max(i, 0), min(i + len(x), bus.shape[0])
    bus[a:b] += x[a - i : b - i]


def reverb(x, seconds=3.0, wet=0.3, bright=5000):
    n = int(seconds * SR)
    ir = (
        rng.standard_normal((n, 2))
        * np.exp(-np.arange(n) / (n / 6.5))[:, None]
    )
    ir = lowpass(ir, bright)
    ir /= np.sqrt((ir**2).sum(axis=0, keepdims=True))
    y = np.stack(
        [fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], axis=1
    )
    return x * (1 - wet) + y * wet


def pad(notes, dur, cutoff, attack=1.0, release=2.2):
    """Detuned sawtooth pairs through a low-pass, with a sine bass."""
    n = int((dur + release) * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for m in notes:
        f = midi(m)
        if m < 40:
            out += 0.45 * np.sin(2 * np.pi * f * t)[:, None]
            continue
        g = 0.5 / (1 + (m - 40) / 24)
        for c, cents in enumerate((-7, 7)):
            ph = f * 2 ** (cents / 1200) * t + rng.uniform()
            out[:, c] += g * (2 * (ph % 1) - 1)
    env = np.minimum(1, t / attack) * np.where(
        t > dur, np.exp(-(t - dur) / (release / 4)), 1
    )
    return lowpass(out * env[:, None], cutoff)


def ping(f, amp, tau, partial=2.76, p=0.0):
    """A glassy ping: a sine and an inharmonic partial."""
    t = np.arange(int(6 * tau * SR)) / SR
    s = np.sin(2 * np.pi * f * t) + 0.22 * np.sin(2 * np.pi * partial * f * t)
    s *= np.exp(-t / tau) * np.minimum(1, t / 0.002)
    return pan(amp * s, p)


def pluck(f, amp, tau=0.35, p=0.0):
    t = np.arange(int(5 * tau * SR)) / SR
    ph = 2 * np.pi * f * t
    s = np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)
    s *= np.exp(-t / tau) * np.minimum(1, t / 0.003)
    return pan(amp * s, p)


def boom(f, amp, tau=1.8):
    t = np.arange(int(4 * tau * SR)) / SR
    s = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t)
    return amp * s * np.exp(-t / tau) * np.minimum(1, t / 0.01)


def noise_swell(dur, amp, lo=600, hi=5000):
    """Noise that rises in level and brightness over dur seconds."""
    n = int(dur * SR)
    t = np.linspace(0, 1, n)
    x = rng.standard_normal((n, 2))
    dark, bright = lowpass(highpass(x, lo), lo * 3), highpass(x, hi / 3)
    return (
        amp
        * (dark * (1 - t)[:, None] + bright * t[:, None])
        * (t**2)[:, None]
    )


def height(z, zref):
    """The page's log view: 1 at the top of the landscape, falling below it."""
    return 1 / (1 + max(zref - z, 0) / 6.0)


def score(ev):
    L, T, S = ev["lines"], ev["t"], ev["scenes"]
    at = lambda line, off=0.0: L[line][0] + off  # noqa: E731
    total = ev["end"] + 3.0
    n = int(total * SR)
    music, fx = np.zeros((n, 2)), np.zeros((n, 2))

    # Harmony: (time, chord, gain, low-pass cutoff).
    land0 = ev["heat"][0][2]
    plan = [
        (0.0, "Dm", 0.5, 400),
        (at("p4"), "Dm9", 0.5, 1100),
        (at("p5"), "Bbmaj7", 0.5, 1200),
        (at("m1"), "Gm9", 0.5, 1400),
        (at("m2"), "A7sus", 0.55, 2200),
        (T["chain_end"] + 0.4, "A7sus", 0.22, 800),
        (at("o1"), "Fmaj9", 0.5, 1500),
        (at("o2"), "C69", 0.5, 1500),
        (at("o3"), "Bbmaj7", 0.6, 2400),
        (at("s1"), "Dm", 0.4, 350),
        (T["gp"], "Dm9", 0.45, 1200),
        (at("g3"), "Bbmaj7", 0.4, 1000),
        (at("g4"), "Gm9", 0.45, 1200),
        (at("q1"), "Fmaj9", 0.5, 1500),
        (at("q3"), "C69", 0.45, 1400),
        (at("a1"), "Gm9", 0.45, 1400),
        (at("a2"), "Ebmaj7", 0.5, 1600),
        (at("a3"), "A7sus", 0.5, 1800),
        (land0, "Dm9", 0.5, 1800),
    ]
    # The time-lapse moves through Bb F C Dm, one chord every third
    # iteration, so the harmony speeds up with the iterations.
    starts = [t for t, s in ev["steps"] if s == 0 and t >= S["cycle"][0]]
    cycle = ["Bbmaj7", "Fmaj9", "C69", "Dm9"]
    for k, t in enumerate(starts[::3]):
        u = k / max(len(starts[::3]) - 1, 1)
        plan.append((t, cycle[k % 4], 0.5 + 0.12 * u, 1800 + 1600 * u))
    plan += [
        (T["boost"], "A7sus", 0.62, 3400),
        (T["truth"], "Dmaj9", 0.72, 3000),
        (at("e1"), "Gmaj7", 0.55, 2200),
        (at("e3"), "Dmaj9", 0.55, 2200),
        (at("f1"), "Bm7", 0.5, 2000),
        (at("f2"), "Gmaj9", 0.55, 2200),
        (at("f3"), "A6", 0.55, 2400),
        (T["word"], "Dmaj9", 0.72, 3000),
        (T["card"], "Dmaj9", 0.5, 2000),
    ]
    plan.sort(key=lambda p: p[0])
    for k, (t0, ch, g, cut) in enumerate(plan):
        t1 = plan[k + 1][0] if k + 1 < len(plan) else ev["end"]
        put(music, t0, g * pad(CHORDS[ch], max(t1 - t0, 0.3), cut))

    def chord_at(t):
        return max((p for p in plan if p[0] <= t), key=lambda p: p[0])[1]

    # The surrogate's wobble: a high fifth whose pitch wobbles while the
    # surrogate is unsure, less and less during the time-lapse.
    tl0, t_on, t_off = S["timelapse"][0], T["gp"], T["truth"] + 0.5
    a, b = int(t_on * SR), int((t_off + 3.0) * SR)
    t = np.arange(a, b) / SR
    wob = np.interp(t, [t_on, t_on + 1.5, tl0, T["boost"]], [0, 1, 1, 0.12])
    vib = 0.6 * np.sin(2 * np.pi * 1.1 * t) + 0.4 * np.sin(
        2 * np.pi * 1.73 * t + 1.0
    )
    voice = np.zeros_like(t)
    for m in (74, 81):
        freq = midi(m) * 2 ** (0.45 * wob * vib / 12)
        ph = 2 * np.pi * np.cumsum(freq) / SR
        voice += np.sin(ph) + 0.15 * np.sin(2 * ph)
    voice *= 1 - 0.3 * wob * (0.5 + 0.5 * np.sin(2 * np.pi * 1.37 * t))
    voice *= np.clip((t - t_on) / 2.0, 0, 1) * np.clip(
        1 - (t - t_off) / 2.5, 0, 1
    )
    music[a:b] += 0.05 * np.stack([voice, voice], axis=1)

    # The loop's steps: evaluate, surrogate, mixture play the chord's
    # tones upwards; the time-lapse turns them into an accelerating
    # arpeggio.
    for t0, step in ev["steps"]:
        if step < 0:
            continue
        tones = sorted(m for m in CHORDS[chord_at(t0)] if m >= 45)
        m = tones[[0, 2, 4][step] % len(tones)] + 24
        put(music, t0, pluck(midi(m), 0.09, 0.3, (step - 1) * 0.35))

    # The first evaluation and PyVBMC's evaluations: glassy pings, higher
    # for higher ground, panned by x.
    zref = ev["zref"]
    put(fx, T["probe"], ping(midi(86), 0.2, 1.1))
    scale = [74 + o + d for o in (0, 12) for d in PENTA_MINOR]
    for t0, x, z in ev["lands"]:
        fast = t0 >= tl0
        m = scale[round(height(z, zref) * (len(scale) - 1))]
        amp, tau = (0.06, 0.16) if fast else (0.13, 0.3)
        put(fx, t0, ping(midi(m), amp, tau, p=0.6 * x / 7))

    # The PyBADS run: warm pings, an octave lower.
    for t0, x, z in ev["bads"]:
        m = scale[round(height(z, zref) * (len(scale) - 1))] - 12
        put(fx, t0, pluck(midi(m), 0.11, 0.25, 0.6 * x / 7))

    # The MCMC chain: one click per evaluation, and a low roar that grows
    # with their rate, cut when the chain stops.
    ch = np.array(ev["chain"])
    imp = np.zeros((n, 2))
    for (t0, c0), (t1, c1) in zip(ch[:-1], ch[1:]):
        k = int(c1 - c0)
        if k <= 0:
            continue
        rate = k / (t1 - t0)
        amp = 0.5 / np.sqrt(max(1.0, rate / 40))
        idx = (rng.uniform(t0, t1, k) * SR).astype(int)
        p = rng.uniform(-0.7, 0.7, k)
        np.add.at(imp[:, 0], idx, amp * np.cos((p + 1) * np.pi / 4))
        np.add.at(imp[:, 1], idx, amp * np.sin((p + 1) * np.pi / 4))
    kt = np.arange(int(0.004 * SR)) / SR
    kernel = highpass(
        rng.standard_normal(kt.size) * np.exp(-kt / 0.0012), 1800
    )
    clicks = np.stack(
        [fftconvolve(imp[:, c], kernel)[:n] for c in range(2)], axis=1
    )
    fx += 0.5 * clicks
    rate = np.gradient(ch[:, 1], ch[:, 0])
    tt = np.arange(n) / SR
    roar = np.clip(
        np.log10(np.maximum(np.interp(tt, ch[:, 0], rate), 1)) / 4.3, 0, 1
    )
    roar *= (tt >= ch[0, 0]) * np.clip(1 - (tt - T["chain_end"]) / 0.25, 0, 1)
    fx += (
        0.35
        * roar[:, None] ** 2
        * lowpass(highpass(rng.standard_normal((n, 2)), 60), 700)
    )

    # Where the next point goes: a swell into each acquisition shown with
    # its heat.
    for k, (t_heat, drop0, land) in enumerate(ev["heat"]):
        d = land - (drop0 - 0.6)
        put(fx, drop0 - 0.6, noise_swell(d, 0.1 if k == 0 else 0.05))

    # The reveal, the payoff's numbers and the wordmark.
    # The swell peaks, and the boom falls, just after "Here it is".
    hit = L["r2"][1] + 0.05
    put(fx, hit - 2.2, noise_swell(2.2, 0.07, 400, 8000))
    put(fx, hit, boom(midi(38), 0.2, 2.5))
    for k, m in enumerate([74, 78, 81, 85, 88, 90]):
        put(
            music,
            hit + 0.1 + 0.13 * k,
            ping(midi(m), 0.06, 1.2, 2.0, 0.5 - 0.2 * k),
        )
    for line, m in (("f1", 35), ("f2", 31), ("f3", 33)):
        put(fx, at(line), boom(midi(m + 12), 0.25))
    for k, m in enumerate([74, 78, 81, 85, 86, 90, 93, 97]):
        put(
            music,
            T["word"] + 1.6 + 0.16 * k,
            ping(midi(m), 0.07, 1.4, 2.0, -0.6 + 0.17 * k),
        )
    put(fx, T["word"] + 4.0, ping(midi(93), 0.12, 1.6))
    return music, fx


def duck(ev, n, depth):
    d = np.zeros(n)
    for a, b in ev["lines"].values():
        d[max(int((a - 0.15) * SR), 0) : int((b + 0.3) * SR)] = 1
    d = sosfiltfilt(butter(1, 2.5, fs=SR, output="sos"), d)
    return 1 - depth * np.clip(d, 0, 1)


def rms_db(x):
    x = x[np.abs(x).max(axis=1) > 1e-4]
    return 10 * np.log10(np.mean(x**2) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    ev = json.loads((args.out / "events.json").read_text(encoding="utf-8"))
    music, fx = score(ev)
    n = len(music)
    music *= 10 ** ((-28 - rms_db(music)) / 20)
    fx *= 10 ** ((-31 - rms_db(fx)) / 20)
    bus = music * duck(ev, n, 0.55)[:, None] + fx * duck(ev, n, 0.3)[:, None]
    bus = reverb(bus, 3.0, 0.3)
    tail = np.clip(
        (ev["end"] - np.arange(n) / SR) / 4.0, 0, 1
    )  # silent at the end
    bus *= np.minimum(np.arange(n) / SR / 1.0, 1)[:, None] * tail[:, None]
    sf.write(args.out / "score.wav", bus.astype(np.float32), SR)

    voice = sf.read(args.out / "narration.wav", dtype="float32")[0]
    voice = np.pad(voice, (0, max(0, n - len(voice))))[:n]
    mix = bus + voice[:, None]
    peak = np.abs(mix).max()
    mix *= min(1.0, 0.89 / peak)  # -1 dBFS at most; mux.py sets the loudness
    sf.write(args.out / "mix.wav", mix.astype(np.float32), SR)
    print(
        f"score: music {rms_db(music):.1f} dBFS, effects {rms_db(fx):.1f} "
        f"dBFS before ducking; mix peak {20 * np.log10(peak):.1f} dBFS, "
        f"{n / SR:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
