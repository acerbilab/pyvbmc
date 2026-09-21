# The 3D animation of a PyVBMC run

`index.html` plays back one recorded two-dimensional PyVBMC run as a rotating
wireframe landscape of the log density: the GP surrogate, the evaluations,
the acquisition function while points are being chosen, the variational
mixture, and at the end the true target. It is a single page with no build
step. It computes no GP: every surface it draws comes from `trace.js`, which
`dev/scripts/export_animation_trace.py` writes from a real run.

| File | Role |
|---|---|
| `index.html` | The page: markup, styles, and one script. It loads three.js r128 from cdnjs and two typefaces from Google Fonts, which have system fallbacks. |
| `trace.js` | The recorded run, `window.VBMC_TRACE = {...}`, about 0.5 MB. Generated; do not edit. |
| `dev/scripts/export_animation_trace.py` | Runs PyVBMC and writes `trace.js`. Its docstrings define the trace format (`build_trace`, `Encoder`). |

The Sphinx build does not publish this folder: `html_static_path` in
`docsrc/source/conf.py` lists only the stylesheet, and no page links here.
The Markdown files of this folder are excluded from the documentation
sources in the same `conf.py`.

## Working on it

View it through a local server:

```console
cd docsrc/source/_static/vbmc3d
python -m http.server 8765        # then http://127.0.0.1:8765/
```

Regenerate the run, from the repository root (about a minute; set
`PYTHONPATH` to the checkout if the package is installed from another one):

```console
python -u dev/scripts/export_animation_trace.py              # the committed seed
python -u dev/scripts/export_animation_trace.py --sweep 0:24 # score seeds, write nothing
```

`--sweep` prints each seed's ELBO (the target's log evidence is 0), gsKL
against the target's exact moments, and the smallest length scale of any GP
hyperparameter sample. The trace is committed because a run is reproducible
only on the platform that made it: another machine's BLAS gives the same
seed a different trajectory.

Query parameters, for checks and captures:

| Parameter | Effect |
|---|---|
| `t=SECONDS`, `paused=1`, `speed=2` | Start time, start paused, playback rate. |
| `hud=0` | No text or controls (also the `h` key). |
| `debug=1` | Runs the anchoring self-check and reports it in the page title. |
| `capture=1` | Nothing moves by itself; a driver calls `vbmcCapture.frame(t, dt)`. No frame reads the wall clock, but the camera, the morph and the caption typing carry over between frames, so a sequence is reproducible when stepped in order from the start with one `dt`, not when it seeks. |
| `tremble=0.25`, `wobble=3.2` | Amplitude and rate of the uncertainty motion. |
| `msaa=0` | No multisampled render target. |

A still frame without a GPU, which is how the page is checked from a
terminal. Use a fresh `--user-data-dir` after regenerating `trace.js`:
headless Chrome otherwise serves the cached one.

```console
chrome --headless=new --enable-unsafe-swiftshader --window-size=1600,900 \
  --virtual-time-budget=4000 --user-data-dir=/tmp/fresh-profile \
  --screenshot=frame.png "http://127.0.0.1:8765/index.html?paused=1&t=30"
chrome --headless=new --enable-unsafe-swiftshader --virtual-time-budget=5000 \
  --user-data-dir=/tmp/fresh-profile --dump-dom \
  "http://127.0.0.1:8765/index.html?debug=1" | grep -o "<title>[^<]*"
```

On the committed trace the self-check reports 3042 anchors in 66 GP states,
no unanchored observation, a position error below 1e-7 cells, SD zero at the
anchors, no folded cells, and a largest correction of the mean of 0.02 log
units. A larger correction means the grid mean misses the data, which is a
defect of the trace (see "What the sheet is"), not something to hide.

Run the self-check after every regeneration: nothing else gates the page.
A trace from another seed, a longer run or another `--detail-iters` can
reach paths that the committed one does not, and they announce themselves
only there and in the browser console: an evaluation whose neighbouring
crossings are all taken does not anchor, and a displacement weight above
1.5 cells is limited, which leaves its crossing short of the observation.
The pace per acquired point is set for eight detailed iterations (`DP`,
`DF`; later ones reuse the last entry).

## How the page works

The script is one `async` function in sections, in this order: renderer,
trace, anchored GP grids, timeline, scene, post, camera, HUD, update, loop.

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
evaluations is not a warp, and a warp still acquires points. In the
committed run the one rotoscale, at iteration 16, was undone, so that
caption does not play.

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
  (`gp_shown`). In the committed trace only the first fit is affected, 5 of
  8 samples. The acquisition function, the posterior and every number on
  the page are the run's own.
- **Heights above the true maximum saturate**, so that a spurious early peak
  of the surrogate stays in frame.
- **The acquisition heat** is `exp(a / 2.4)` of the log-valued acquisition
  relative to its maximum; the contours are every 4 log units.

## Status

`TODO.md` in this folder lists what is left to do, in order, and the
constraints on the work. `NOTES.md` records where the work stands and why
the page is the way it is, including what was tried and set aside.
