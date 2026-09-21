# 3D animation: status and design record

`README.md` in this folder explains how the pages work and how to run,
regenerate and check them; `TODO.md` lists what is left to do, in order. This
file records where the work stands and why things are the way they are,
including what was tried and set aside.

## Status

The page, the recorded run and the exporter are complete and consistent with
each other: the page's self-check (`?debug=1`) passes on the committed trace
with the numbers quoted in the README, and exporting again on the machine
that made the trace gives a byte-identical file.

`wordmark.html` and `trace_wordmark.js` are a working first version of the
wordmark finale: the self-check passes on the trace with the numbers quoted
in the README, and exporting seed 22 again reproduces the sweep's run
(ELBO, iterations and evaluations). The owner has watched it in a desktop
browser on the development machine; their feedback so far (a flash of the
target's sheet as the story starts, and the finale's caption) has been
acted on, and the rest of the finale's look is still theirs to steer.

How it has been verified:

- Headless Chrome stills (software rendering) at 1280 x 720 and 1600 x 900,
  at many points of the timeline; the self-check; `black`, `isort`,
  `py_compile` on the exporter; `node --check` on the script. Stills
  requested at 390 x 844 show a page laid out about 520 px wide and cropped
  (README), so of the phone layout only `wordmark.html`'s title card has
  been seen, at 540 x 1168.
- `wordmark.html`'s finale and the loop back into the story, stepped
  through `capture=1` at 10 frames per second by a throwaway driver over the
  DevTools protocol (Node, no dependencies) and looked at as a contact
  sheet.
- One independent read-only review of the whole change, which re-derived the
  target's moments, decoded the trace itself, reproduced every number in the
  README, and confirmed that the recorder does not alter the run (the
  unobserved run of the same seed has the same ELBO, iteration count and
  gsKL). Its findings were fixed afterwards; the fixes were checked with the
  self-check and stills, not reviewed again.

What nobody has verified:

- **`index.html` in motion.** The project owner watched earlier versions on
  a real device and steered the look, the speed of the tremble and the
  anchoring. Their last visual feedback on this page was on the version
  whose first fit looked pinched. The remedy for that (`drawable_samples`)
  has been seen only as stills. (The rest of the page is shared with
  `wordmark.html`, which the owner has watched.)
- Frame rate on phones. Phones draw the full 64 x 64 mesh with half the
  segments per cell; no measurement exists.
- `capture=1` as a recording, in the repository: a throwaway driver
  outside it recorded `wordmark.html`'s whole loop to video and its finale
  to GIF (`TODO.md`, item 3, has the recipe and the numbers); no script in
  the repository does it yet.
- The multisampled render target on a real GPU (it runs under software
  rendering).

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

**The wordmark.** Seen from above, the banana is a V, so the finale pulls
back until it is the V of PyVBMC. The repository's logo (`logo.svg`) was the
starting point, not a template: in it "Py" and "MC" are orange Times, the V
is a one-dimensional GP with its band and observations, and the B is a
two-dimensional posterior drawn as filled contours. The finale keeps one
graphic letter, the V, made by the run; a B made of a posterior was judged
weak by the owner. It keeps the warm letters against the cool graphic. What
was decided:

- *A separate page.* `index.html` and its run stay as they are;
  `wordmark.html` started as a copy of it.
- *The banana alone.* The lobe of `index.html`'s target sits between the
  arms, in the V's counter. The banana's round bottom was a concern (a U,
  not a V); the owner judged that it reads as a V, and the stills agree.
  A target built to be a V (straight arms, even stroke) was sketched and not
  pursued.
- *The run fills the arms.* Many runs spend their evaluations near the
  vertex and leave the arm tips bare, and the surrogate then only guesses
  at the ends of the V. The owner asked for a run of 80 to 100 evaluations
  that fills the tails; `arm_coverage` measures that (Seeds, below).
- *The letters lie on the floor and are made of the mesh.* Drawn on the
  floor, they go through the same bloom, grain and perspective as the V and
  can be seen from the tilted camera. Filled with the sheets' wire grid,
  they read as part of the scene rather than a caption. A flat overlay in
  screen space would miss the post-processing; it was not tried in the
  page. Flat mock-ups with the V as filled contours, a glow or a light
  background, made while choosing the layout, were not kept.
- *The log view for the V.* In the density view the arms are low and thin
  next to the letters; in the log view the V has a letter's body.
  `?wmview=density` shows the other.
- *STIX Two Text Regular.* A Times-like face as in the logo, and the page
  already loads STIX Two for its captions. The medium weight made the
  letters outweigh the V.

## Seeds

### `trace_wordmark.js`

Seeds 0 to 59 on the banana alone, PyVBMC `1.0.5.dev156+g4d91a5e61`,
default options, BLAS single-threaded, on the machine that made the trace:
the runs of `--target banana --sweep 0:60` (the sweep itself was checked on
seeds 22 and 2, which it reproduces). Here, evaluations are the ones still
in the training set at the end; `--sweep` also counts the ones trimmed at
the end of warm-up (95 for seed 22). `arms` is `arm_coverage`. The ELBO SD
is 0.001 for every run. 25 runs ended with 80 to 100 evaluations; the table
has the eight of them with the best arm coverage, and seeds 3 and 2, which
leave the arm tips bare.

| seed | iterations | evaluations | ELBO | gsKL | arms |
|---:|---:|---:|---:|---:|---:|
| **22** | 18 | 93 | -0.013 | 0.058 | 1.00 |
| 50 | 18 | 92 | -0.025 | 0.084 | 0.97 |
| 4 | 17 | 87 | -0.029 | 0.087 | 0.96 |
| 6 | 16 | 84 | -0.026 | 0.105 | 0.96 |
| 14 | 16 | 83 | -0.022 | 0.076 | 0.94 |
| 47 | 16 | 82 | -0.025 | 0.099 | 0.92 |
| 48 | 16 | 83 | -0.027 | 0.088 | 0.91 |
| 41 | 18 | 93 | -0.032 | 0.096 | 0.91 |
| 3 | 15 | 80 | -0.048 | 0.191 | 0.63 |
| 2 | 15 | 80 | -0.050 | 0.168 | 0.57 |

Within the range of evaluations, seed 22 has the best ELBO, gsKL and arm
coverage; its ELBO is the best of all 60 runs. Its smallest hyperparameter
length scale is 1.05 (seed 8 of `trace.js`: 0.010), so no GP state of its
trace leaves a sample out (`drawable_samples`).

### `trace.js`

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
