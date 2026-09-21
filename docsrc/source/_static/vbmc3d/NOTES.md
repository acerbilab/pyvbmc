# 3D animation: status, open decisions and design record

`README.md` in this folder explains how the page works and how to run,
regenerate and check it. This file records where the work stands, what is
still to be decided, and why things are the way they are, including what was
tried and set aside. The work lives on the branch `feat-3d-animation` and is
kept apart from the repository's trackers (`dev/TODO.md`, the roadmap, the
plans): nothing there refers to it.

## Status

The page, the recorded run and the exporter are complete and consistent with
each other: the page's self-check (`?debug=1`) passes on the committed trace
with the numbers quoted in the README, and exporting again on the machine
that made the trace gives a byte-identical file.

How it has been verified:

- Headless Chrome stills (software rendering) at 1280 x 720, 1600 x 900 and
  390 x 844, at many points of the timeline; the self-check; `black`,
  `isort`, `py_compile` on the exporter; `node --check` on the script.
- One independent read-only review of the whole change, which re-derived the
  target's moments, decoded the trace itself, reproduced every number in the
  README, and confirmed that the recorder does not alter the run (the
  unobserved run of the same seed has the same ELBO, iteration count and
  gsKL). Its findings were fixed afterwards; the fixes were checked with the
  self-check and stills, not reviewed again.

What nobody has verified:

- **The current version in motion.** The project owner watched earlier
  versions on a real device and steered the look, the speed of the tremble
  and the anchoring. Their last visual feedback was on the version whose
  first fit looked pinched. The remedy for that (`drawable_samples`), the
  slim trace and everything after have been seen only as stills.
- Frame rate on phones. Phones draw the full 64 x 64 mesh with half the
  segments per cell; no measurement exists.
- `capture=1` end to end: the page's hook exists, no driver has called it.
- The multisampled render target on a real GPU (it runs under software
  rendering).

## Where to resume

In this order; the first two are decisions for the project owner.

1. **Look at the first 30 seconds in motion and settle what the sheet shows
   after the first fit.** The displayed GP leaves out hyperparameter samples
   thinner than two grid cells (README, "What the sheet is"). This departs
   from the GP that VBMC averages, and was adopted without the owner's
   explicit confirmation. The alternatives considered are listed below.
2. **Decide how the documentation shows the page**: a link next to the
   corner-plot GIF on the index page, or the page embedded as the hero. Then
   publish the folder (`html_static_path` or `html_extra_path` in
   `docsrc/source/conf.py`; the Markdown files here must stay excluded from
   the sources) and link it from `docsrc/source/index.rst`. Publishing makes
   the change noticeable to users, so it is the moment for an entry in
   `CHANGELOG.md`; until then the changelog policy asks for none.
3. **Write the recording script** for video and GIF (README hero, talks,
   social): a driver that opens the page with `capture=1&hud=...`, calls
   `vbmcCapture.frame(t, dt)` for successive frames from the start, grabs
   each frame and encodes them. The development machine has Chrome and Node
   but no `ffmpeg`; the `imageio-ffmpeg` package bundles one. Formats differ
   by destination: a README GIF has to be a short, small excerpt, a talk
   wants the whole loop as video.
4. **Pacing.** The loop lasts about 95 s. The durations are the `seg(...)`
   calls and the `DP` / `DF` arrays of the timeline section.
5. Possibly **reconsider the seed** (see "Seeds").

The branch is based on `5e5fa188`; `dev-next` has moved since. Outside this
folder it touches `dev/scripts/export_animation_trace.py` (new),
`dev/README.md` (one entry appended to the scripts list) and
`docsrc/source/conf.py` (the exclusion of this folder's Markdown files).

## Decisions, and what was set aside

**A recorded real run, not a VBMC written in JavaScript.** A live in-browser
engine would let viewers reseed and switch targets, but it would be a
simplified reimplementation and most of the effort would go into numerics.
The page reads a generic trace, so a live engine could still feed it.

**three.js in the browser**, not Matplotlib 3D (no glow, no interaction),
PyVista or Manim (heavier, not interactive).

**Height is the log density**, through a compressive map, because that is
where the GP lives and where evaluations at very different values all stand
visibly on the landscape; in density space most early evaluations sit at
zero. The density view is kept for the finale, where the mixture reads as
the familiar bumps.

**See-through additive wires, no hidden-line fill.** An opaque fill hides the
floor, and the floor carries the posterior and the acquisition function. A
faint additive fill gives the sheet body.

**The posterior's sheet shows only near the top of the landscape** (it fades
out below `hLog` of about 0.4 to 0.66). An early posterior is broad, and its
sheet then covers the whole window as a second flat mesh over the GP's.

**The tremble** went through three forms. A saturating function of the SD
added in display space was wrong twice: not in log-density space, and not
zero at the data. `mean + 0.6 SD noise` in log-density space was honest but
the sheet stopped reading as the mean. It is `0.25 SD`, and 3.2 times
faster than first drawn, because slow noise was taken for features of the
mean. The `shimmer` term is the one remaining piece that is not in
log-density space (README).

**Anchoring.** Capping the SD per vertex in the shader, by the distance to
the nearest observation, made the SD right but did not make observations
look like anchors: no wire passes through them, so the nearest wires still
moved beside a still node. Hence the warped grid, baked per GP state, which
also removed a loop over all observations from the vertex shader. A mesh
with every second grid line, tried for phones, has too few crossings for
the late clusters of observations and folds.

**Needles after the first fit.** With anchoring in place, the first fit
looked pinched: every observation stood above the surface around it. The
export is correct (the exact GP mean passes through the data to 1e-3 in both
coordinate spaces). The cause is the average over hyperparameter samples:
after ten points, two of the eight samples of the committed run have a
length scale of about a thousandth of the plausible box in one coordinate,
pass through the data as needles and sit at their negative-quadratic mean
function elsewhere. The averaged mean is then a median 0.6 log units below
an observation 0.05 away from it, and 1.6 one grid cell away. Because the
rank-one updates of the next iteration keep those hyperparameters, this
lasts until the second fit, about 30 s into the animation. What was
considered:

- *A seed without such samples.* Set aside: all five seeds with an ELBO
  above -0.1 have them (table below), so such a run would not be typical.
- *The pointwise median over samples*, or *one representative sample*.
  Not tried. Both depart from VBMC's average everywhere, not only where the
  grid fails.
- *Averaging over the samples the grid can resolve* (`drawable_samples`).
  Adopted: in the committed trace it changes the first fit only (5 of 8
  samples), and the largest correction the page applies to the mean at an
  anchor falls from 2.2 log units to 0.02.

**A trimmed evaluation** (one low-density point is dropped from the training
set at the end of warm-up) had kept anchoring the sheet for an iteration and
dug a 12 log unit pit. Per-point GP states now use the training set of the
iteration they belong to.

**Trace size.** 2.2 MB of 16-bit grids in base64 became 0.5 MB: the
posterior's grids are recomputed in the page from the mixture components,
SDs and acquisition values take 8 bits, and the grids are one zlib block of
second differences.

## Seeds

`--sweep` on the machine that made the trace, PyVBMC
`1.0.5.dev156+g4d91a5e61`, default options. The target's log evidence is 0.
`min ell` is the smallest length scale of any hyperparameter sample over
the run, in the target's units (one grid cell is 0.22); it was computed for
the five seeds with an ELBO above -0.1 only. Seeds above 12 were not run.
Runs are reproducible only on one platform, so other machines give other
numbers for the same seeds.

| seed | iterations | evaluations | ELBO | ELBO SD | gsKL | min ell |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 17 | 90 | -0.066 | 0.005 | 0.066 | 0.002 |
| 1 | 15 | 80 | -0.314 | 0.002 | 0.622 | |
| 2 | 20 | 105 | -0.203 | 0.004 | 0.335 | |
| 3 | 18 | 90 | -0.135 | 0.002 | 0.173 | |
| 4 | 16 | 85 | -1.043 | 0.093 | 2.097 | |
| 5 | 15 | 80 | -0.349 | 0.002 | 0.811 | |
| 6 | 17 | 90 | -0.301 | 0.003 | 0.397 | |
| 7 | 22 | 115 | -0.050 | 0.002 | 0.071 | 0.049 |
| **8** | 26 | 135 | -0.050 | 0.003 | 0.069 | 0.010 |
| 9 | 15 | 80 | -0.236 | 0.008 | 0.214 | |
| 10 | 26 | 130 | -0.067 | 0.001 | 0.084 | 0.093 |
| 11 | 15 | 80 | -0.293 | 0.001 | 0.399 | |
| 12 | 22 | 115 | -0.053 | 0.003 | 0.061 | 0.000 |

The owner asked for an ELBO above -0.1 and a low gsKL. Seed 12 has the
lowest gsKL and a shorter run, and was the choice until its first fit
rendered as a serrated comb, which is the needle problem above at its
worst. Seed 8 replaced it before `drawable_samples` existed. Seed 12 has
not been looked at with that remedy in place, and may now be the better
run to show.

## Previews

The page can be published as a claude.ai artifact for viewing on another
device: the artifact host supplies the document skeleton, so the published
variant is `index.html` without its doctype, `html`, `head` and `body` tags
and its two `meta` tags, with `trace.js` as a supporting file. Nothing in
the repository depends on it.
