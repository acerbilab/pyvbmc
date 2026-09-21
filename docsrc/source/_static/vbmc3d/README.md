# The 3D animation of a PyVBMC run

`index.html` plays back one recorded two-dimensional PyVBMC run as a rotating
wireframe landscape of the log density: the GP surrogate, the evaluations,
the acquisition function while points are being chosen, the variational
mixture, and at the end the true target. It is a single page with no build
step. It computes no GP: every surface it draws comes from `trace.js`, which
`dev/scripts/export_animation_trace.py` writes from a real run.

`wordmark.html` is a copy of that page that plays another run and ends
differently: the camera pulls back until the final posterior, seen from
above, is the V of the PyVBMC wordmark, and the other letters appear beside
it ("The wordmark finale" below). Its run is of the banana alone, without
the lobe that `index.html`'s target has between the arms, where it would
fill the V. The rest of this file holds for both pages except where it
names one.

| File | Role |
|---|---|
| `index.html` | The page: markup, styles, and one script. It loads three.js r128 from cdnjs and two typefaces from Google Fonts, which have system fallbacks. |
| `trace.js` | The run `index.html` plays (banana with a lobe, seed 8), `window.VBMC_TRACE = {...}`, about 0.5 MB. Generated; do not edit. |
| `wordmark.html` | The wordmark page. |
| `trace_wordmark.js` | The run `wordmark.html` plays (banana alone, seed 22), about 0.4 MB. Generated; do not edit. |
| `scripts/record.mjs` | Records a page as an MP4, a GIF or PNG frames ("Recording" below). |
| `dev/scripts/export_animation_trace.py` | Runs PyVBMC and writes a trace. Its docstrings define the trace format (`build_trace`, `Encoder`). |

The Sphinx build does not publish this folder: `html_static_path` in
`docsrc/source/conf.py` lists only the stylesheet, and no page links here.
The Markdown files of this folder are excluded from the documentation
sources in the same `conf.py`.

## Working on it

View it through a local server:

```console
cd docsrc/source/_static/vbmc3d
python -m http.server 8765        # then http://127.0.0.1:8765/ or /wordmark.html
```

Regenerate a run, from the repository root (about a minute; set
`PYTHONPATH` to the checkout if the package is installed from another one):

```console
python -u dev/scripts/export_animation_trace.py                              # trace.js
python -u dev/scripts/export_animation_trace.py --target banana --seed 22    # trace_wordmark.js
python -u dev/scripts/export_animation_trace.py --target banana --sweep 0:60 # score seeds, write nothing
```

`--sweep` prints each seed's ELBO (the target's log evidence is 0), gsKL
against the target's exact moments, the smallest length scale of any GP
hyperparameter sample, and the arm coverage: the share of points near the
banana's arm tips that have an evaluation within about three grid cells,
for the worse arm (`arm_coverage`). A run can score well on ELBO and gsKL
and still leave the arm tips bare, and the surrogate then only guesses at
the ends of the arms. The traces are committed because a run is
reproducible only on the platform that made it: another machine's BLAS
gives the same seed a different trajectory. `trace_wordmark.js` was
exported with BLAS single-threaded (`OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`), as were the sweeps that chose
its seed.

Query parameters, for checks and captures:

| Parameter | Effect |
|---|---|
| `t=SECONDS`, `paused=1`, `speed=2` | Start time, start paused, playback rate. |
| `hud=0` | No text or controls (also the `h` key). |
| `debug=1` | Runs the anchoring self-check and reports it in the page title. |
| `capture=1` | Nothing moves by itself; a driver calls `vbmcCapture.frame(t, dt)`. No frame reads the wall clock, but the camera, the morph and the caption typing carry over between frames, so a sequence is reproducible when stepped in order from the start with one `dt`, not when it seeks. |
| `tremble=0.25`, `wobble=3.2` | Amplitude and rate of the uncertainty motion. |
| `msaa=0` | No multisampled render target. |
| `cam=AZ,EL,R` | `wordmark.html` only: pins the camera (azimuth and elevation in radians, distance in scene units), for stills. |
| `wmview=density` | `wordmark.html` only: the V in the density view that the story ends in, instead of the log view. |

A still frame without a GPU, which is how the page is checked from a
terminal. Use a fresh `--user-data-dir` after regenerating a trace:
headless Chrome otherwise serves the cached one. On Windows headless Chrome
lays the page out at least about 520 px wide whatever `--window-size`
says, and crops the screenshot, so a phone check at 390 x 844 shows a
cropped wide page; use the phone's aspect at a larger size (540 x 1168).

```console
chrome --headless=new --enable-unsafe-swiftshader --window-size=1600,900 \
  --virtual-time-budget=4000 --user-data-dir=/tmp/fresh-profile \
  --screenshot=frame.png "http://127.0.0.1:8765/index.html?paused=1&t=30"
chrome --headless=new --enable-unsafe-swiftshader --virtual-time-budget=5000 \
  --user-data-dir=/tmp/fresh-profile --dump-dom \
  "http://127.0.0.1:8765/index.html?debug=1" | grep -o "<title>[^<]*"
```

On `trace.js` the self-check reports 3042 anchors in 66 GP states, no
unanchored observation, a position error below 1e-7 cells, SD zero at the
anchors, no folded cells, and a largest correction of the mean of 0.02 log
units. On `trace_wordmark.js` it reports 2047 anchors in 58 states, a
largest correction of 0.01 and otherwise the same, except for 4 unanchored
observations of 95: the four that lie outside the display window, two past
the arm tips and two below the vertex. `wordmark.html` adds to the title
the typeface its letters got (STIX Two Text, or a fallback when the font
did not arrive within 2.5 s) and when the finale starts. A larger
correction means the grid mean misses the data, which is a defect of the
trace (see "What the sheet is"), not something to hide.

Run the self-check after every regeneration: nothing else gates the page.
A trace from another seed, a longer run or another `--detail-iters` can
reach paths that the committed ones do not, and they announce themselves
only there and in the browser console: an evaluation whose neighbouring
crossings are all taken does not anchor, and a displacement weight above
1.5 cells is limited, which leaves its crossing short of the observation.
The pace per acquired point is set for eight detailed iterations (`DP`,
`DF`; later ones reuse the last entry).

### Recording

`scripts/record.mjs` records a page through `capture=1`, frame by frame, as
an MP4, a GIF or a folder of PNG frames. It needs Node 22 or later and
Chrome, and ffmpeg for the two encoded formats (`FFMPEG`, or `ffmpeg` on
the `PATH`; `pip install --target DIR imageio-ffmpeg` puts one under
`DIR/imageio_ffmpeg/binaries/` without touching an environment). It serves
this folder itself, so no server needs to run. From this folder, with
`OUT` a folder outside the repository (recordings are not tracked):

```console
node scripts/record.mjs wordmark.html OUT/loop.mp4                    # the whole loop, 1280 x 720, 30 fps
node scripts/record.mjs "wordmark.html?hud=0" OUT/finale.gif \
  --from 83.5 --to 94.45 --size 960x540                              # the finale as a 640 px GIF, 15 fps
node scripts/record.mjs "wordmark.html?hud=0" OUT/frames --from 80 --to 82  # PNG frames, for checks
```

The header of the script lists its options. The playback controls are left
out; `hud=0` leaves out the captions and telemetry too, and at 640 x 360
the captions overlap the wordmark. Only a recording that starts at 0 shows
the page as it plays: one that starts later begins with the camera where
that segment wants it. A GIF of an excerpt cuts from its last frame to its
first where it loops; a recording of the whole loop starts and ends on the
wordmark.

Under software rendering a frame takes about 0.18 s at 960 x 540 and
0.45 s at 1280 x 720, so the whole loop at 30 fps (2744 frames) takes about
20 minutes. The grain and scanlines of the post-processing change every
pixel of every frame and dominate the size: the loop at 1280 x 720 is about
118 MB at the default CRF 18 and about 37 MB with `--denoise 2:2:5:5
--crf 24`. The finale GIF above is about 9 MB (21 MB without the
denoising that GIFs get by default); `--gif-width 480 --fps 12 --colors 64
--bayer 2` gives about 3 MB.

## How the page works

The script is one `async` function in sections, in this order: renderer,
trace, anchored GP grids, timeline, scene, post, camera, HUD, update, loop.
`wordmark.html` has one more, wordmark, between scene and post.

**Everything drawn is a function of the timeline time.** The timeline
section turns the trace into lists built once: segments (`segs`, each with a
caption), and the events that the update section looks up by time: when each
evaluation drops and lands (`pointEv`), transitions between GP states
(`gpTr`) and between posteriors (`vpTr`), and the intervals during which an
acquisition surface shows (`acqEv`). `update(tau, dt, clock)` reads them and
sets uniforms; it keeps no state except the smoothed camera and the smoothed
morph and truth levels, and those levels start at the new time's values
whenever story time goes back (the story starting after the title card, a
seek), so that nothing of the finished state eases into the opening. That
is what makes the timeline scrubbable and
`capture=1` reproducible. The loop starts on the finished state (the title
card), fades it out and tells the story, so the end runs into the start.
A caption about what the algorithm did in an iteration (rotating and
rescaling the space, say) comes from what the run logged for it
(`actions`), never from the iteration's shape: an iteration without new
evaluations is not a warp, and a warp still acquires points. In the run of
`trace.js` the one rotoscale, at iteration 16, was undone, so that caption
does not play.

**Coordinates.** Data `x` maps to scene `X` in [-1, 1] and data `y` to `-Z`;
the display window is square so the floor is not stretched. Height is
`LIFT + HGT * h(z)`, where `z` is a log density and `h` is `hLog` in the log
view, `1 / (1 + (zref - z) / S_SCALE)` with `zref` the true maximum,
saturating smoothly above it, and `hDen`, `exp(z - zref)`, in the density
view; `morph` blends them. `hLog` and `hDen` exist twice, in GLSL for the
sheets and in JavaScript for the evaluation nodes, and must stay identical.

**Sheets are displaced on the GPU.** One vertex shader (`SHEET_VS`) serves
the GP, the posterior and the truth. A sheet's geometry is grid coordinates
only; the shader reads a state from a 64 x 64 float texture (channels: mean,
SD as `s / (s + 1)`, and the x and y displacement of the wire in cells),
interpolates along the wire with Catmull-Rom, mixes two states for a
transition, and maps to height. Wires are line segments along grid rows and
columns, not triangle edges. Everything is drawn additively with the depth
test off, so draw order does not matter and overlapping sheets add up:
where the posterior (magenta) locks onto the surrogate (cyan) the ridge
turns white.

**The floor** is one fragment shader: grid, plausible box, posterior density
with contours, acquisition heat with contours, landing rings, the beacon's
foot. The mixture components are 1-sigma and 2-sigma ellipses rebuilt on the
CPU; between two posteriors, matched components morph, new ones split off
their nearest neighbor and pruned ones shrink into theirs (`matchTracks`).
The posterior's grid is evaluated in the page from the components, which is
exact because the problem is unbounded and the exporter checks it against
`vp.pdf`.

**Post-processing** is a four-level separable blur added back as bloom,
then a soft exposure curve, vignette, scanlines and grain. It is written out
here, as is the orbit camera, because cdnjs hosts only the core file of
three.js, not its add-ons, and the page takes its scripts from cdnjs alone
so that it also runs under a content security policy that admits no other
script host. r128 has a classic (UMD) core build there, and nothing in the
page needs a newer version.

## What the sheet is, and where it departs from the algorithm

These are display decisions. Each has one home, named here.

- **Observations anchor the sheet** (section "anchored GP grids"). An
  observation falls between the wires of a 64 x 64 grid, and the predictive SD
  recovers within a fraction of a cell, so a plain grid shows wires trembling
  beside a still node. Each observation therefore owns the nearest free wire
  crossing (assigned once, in evaluation order, so anchors hold through
  transitions). Every GP state is baked at load time into a warped grid: a
  smooth local displacement carries the crossing onto the observation, the
  mean there is corrected to the observed value, and the SD is capped by `sf *
  sqrt(1 - exp(-rho2))`, what conditioning on that one observation alone
  leaves: zero at the observation, and for one hyperparameter sample an upper
  bound on the SD, since more observations only lower it (the spread between
  samples, which it does not bound, vanishes at an observation too). The
  displacement and the correction are sums of narrow Gaussian bumps (`BUMP`,
  0.75 cells) whose weights solve the linear system that makes them exact at
  every anchor of the state. Every grid line is a wire, on phones too: a
  sparser mesh has too few crossings for the late clusters of observations and
  folds.
- **An evaluation trimmed at the end of warm-up** leaves the training set in
  the iteration whose record no longer lists it (`active`); from then on it
  does not anchor and its node is dimmed.
- **A node may float for under a second after landing.** Within an iteration
  the GP is updated after each point except the last, which waits for the
  refit. The page shows that as it happens.
- **The motion is a fraction of the uncertainty, in log-density space.** The
  sheet is drawn at `mean + TREMBLE * SD * noise`, mapped to height like the
  mean, with `TREMBLE = 0.25` and `noise` a sum of four moving sines in
  [-1, 1] (RMS about 0.35): at most a quarter of an SD, so that the sheet
  still reads as the mean, and fast (`WOBBLE`) so that it cannot be taken
  for the mean. It is not a posterior draw. SDs are drawn up to 249 log
  units, the largest that their 8 bits resolve. One small term is not in
  log-density space: `shimmer`, 0.025 in height units times the saturating
  `s / (s + 1)`, which keeps the flat uncertain outskirts from looking
  frozen where the height map compresses the honest motion to nothing.
- **The displayed GP leaves out hyperparameter samples the grid cannot
  draw** (`drawable_samples` in the exporter). VBMC's GP is the average over
  its hyperparameter samples. After the first fit, on ten points, a few
  samples typically have a length scale of a thousandth of the plausible
  box: they pass through the data as needles and sit at their mean function
  elsewhere, so the averaged mean has a needle at every observation. A grid
  can only draw that as a surrogate that misses its data. The displayed
  average keeps the samples whose smallest length scale is at least two
  cells (at least the better-resolved half); each state records how many
  (`gp_shown`). In `trace.js` only the first fit is affected, 5 of 8
  samples; in `trace_wordmark.js` no sample is left out. The acquisition
  function, the posterior and every number on the page are the run's own.
- **Heights above the true maximum saturate**, so that a spurious early peak
  of the surrogate stays in frame.
- **The acquisition heat** is `exp(a / 2.4)` of the log-valued acquisition
  relative to its maximum; the contours are every 4 log units.
- **In the wordmark finale** (`wordmark.html`) the surrogate's sheet keeps
  only its upper part, where the V is, and the V is shown in the log view
  even though the story before it ends in the density view. The letters
  other than the V are drawn, not data.

## The wordmark finale

`wordmark.html` ends on a segment of its own (`wordmark`, 6 s, after
`density`). The camera eases, as it does between all segments, to face the
V, nearly overhead (elevation `WM.el`, 1.38 rad), far enough back for the
whole word to fit the frame and aimed at the word's center. Over the first
2.5 s the view morphs back to log, the surrogate's flat outskirts fade (its
`fadeLow`, which the posterior's and the truth's sheets always have) and
the window's frame and the plausible box go. The letters fade in from
1.6 s. The title card holds the same view, so the loop opens on the
wordmark, and the reset brings the camera down into the story. On a frame
with `dt = 0` (the first one, and `capture=1`'s first) the camera starts
where the segment wants it instead of easing there.

The V is not drawn for the wordmark: it is the final state's sheets and
nodes seen from above. The rest of the word is two flat meshes lying on the
floor, "Py" to the left of the V and "BMC" to the right, set in STIX Two
Text Regular (loaded from Google Fonts, with a system serif if it has not
arrived after 2.5 s). Each is a canvas texture holding the glyphs filled
(red channel) and outlined (green), and its shader draws the outline, a
faint fill and, inside the glyphs, the sheets' wire grid continued across
the floor, in the surrogate's amber. The far edge of the display window is
the cap line of the letters; the baseline and the width of the V at the cap
line are read from the final posterior's grid (where it is within 2.3 and 5
log units of its maximum), so another run gives letters that fit its V.
The camera's distance fits the word's width, or its height on a narrow
screen, with a margin; on a phone the word is small.

## Status

`TODO.md` in this folder lists what is left to do, in order, and the
constraints on the work. `NOTES.md` records where the work stands and why
the page is the way it is, including what was tried and set aside.
