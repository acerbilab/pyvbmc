# 3D animation: status and design record

`README.md` in this folder explains how the pages work and how to run,
regenerate and check them; `TODO.md` lists what is left to do, in order. This
file records where the work stands and why things are the way they are,
including what was tried and set aside.

## Status

The page, the recorded run and the exporter are complete and consistent with
each other: the page's self-check (`?debug=1`) passes on the committed trace
with the numbers quoted in the README. `trace.js` is a run of the PyVBMC of
`5e5fa188`, which exporting again at that revision on the machine that made
it reproduced byte for byte; the PyVBMC of `254dd6cc` gives its seed
another run (Seeds, below).

The film (`film.html`) is finished: its seventh draft, narrated by Travis
(ElevenLabs) with every line passing `scripts/verify_voice.py`, was
approved by the director and a postdoc of the lab on 2026-10-05, and is
mastered at 1920 x 1080 and -16 LUFS, with captions and without, beside
the `.srt` of its captions. Outside the captions the two masters differ by
no more than two encodings of the same frames do (at least 40.3 dB PSNR on
every frame, where the sixth draft's pair reached 39.3 dB), and each `.srt`
cue matches the timeline that the page plays. The title from the first
frame (below) came after that approval and was checked in stills of the
opening and in the masters. The sixth draft had been mastered before the
director changed the film's framing (*The framing*, below); its masters and
takes are kept (`renders/draft6/`, `voice-travis-draft6/`). Publishing it
remains (`TODO.md`, "The video"). The masters, drafts and voice takes are
local media (`README.md`, "The film").

`wordmark.html` and `trace_wordmark.js` are a working first version of the
wordmark finale: the self-check passes on the trace with the numbers quoted
in the README, and exporting seed 42 again reproduces the trace. The page
has been watched in a desktop browser while it played the run of seed 22
that the trace held before (Seeds, below). The feedback so far (a flash of
the target's sheet as the story starts, and the finale's caption) has been
acted on. The rest of the finale's look is open.

How it has been verified:

- Headless Chrome stills (software rendering) at 1280 x 720 and 1600 x 900,
  at many points of the timeline; the self-check; `black`, `isort`,
  `py_compile` on the exporter; `node --check` on the script. Stills
  requested at 390 x 844 show a page laid out about 520 px wide and cropped
  (README), so of the phone layout only `wordmark.html`'s title card has
  been seen, at 540 x 1168.
- `wordmark.html`'s finale and the loop back into the story, stepped
  through `capture=1` at 10 frames per second and looked at as a contact
  sheet; the whole loop recorded to video and the finale to GIF.
  `scripts/record.mjs`, a tidied version of the driver that made those
  recordings, has since recorded PNG frames, a short MP4 and the finale GIF
  (8.8 MB), not the whole loop.
- One independent read-only review of the whole change, which re-derived the
  target's moments, decoded the trace itself, reproduced every number that
  the README then quoted (on the runs of seeds 8 and 22), and confirmed
  that the recorder does not alter the run (the
  unobserved run of the same seed has the same ELBO, iteration count and
  gsKL). Its findings were fixed afterwards; the fixes were checked with the
  self-check and stills, not reviewed again.

What nobody has verified:

- **`index.html` in motion.** Earlier versions were watched on a real
  device. The look, the speed of the tremble and the anchoring were tuned
  from that feedback. The last visual feedback on this page was on the
  version whose first fit looked pinched. The remedy for that
  (`drawable_samples`) has been seen only as stills. (The rest of the page
  is shared with `wordmark.html`, which has been watched in motion.)
- **`wordmark.html` playing the run of seed 42.** Seen only in headless
  stills; the recordings were made from the run of seed 22.
- Frame rate on phones. Phones draw the full 64 x 64 mesh with half the
  segments per cell; no measurement exists.
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
weak. It keeps the warm letters against the cool graphic. What
was decided:

- *A separate page.* `index.html` and its run stay as they are;
  `wordmark.html` started as a copy of it.
- *The banana alone.* The lobe of `index.html`'s target sits between the
  arms, in the V's counter. The banana's round bottom was a concern (a U,
  not a V). In motion and in the stills it reads as a V.
  A target built to be a V (straight arms, even stroke) was sketched and not
  pursued.
- *The run fills the arms.* Many runs spend their evaluations near the vertex
  and leave the arm tips bare, and the surrogate then only guesses at the ends
  of the V. The run should use 80 to 100 evaluations, or slightly more, and
  fill the tails; `arm_coverage` measures that (Seeds, below).
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

**A narrated video beside the page.** The page introduces every layer of
the algorithm within fifteen seconds and captions each evaluation it
acquires for a second or two. A viewer who does not know VBMC sees a lot
happening without learning what any of it is. So the page stays an
exhibit, and a separate video explains the algorithm. The same page draws
the video in the same style. The video's audience is scientists who fit
models, most of whom do not know Gaussian processes or variational
inference. Its
angle is Bayesian optimization applied to the whole posterior. Its
drafts were voiced with Kokoro, an open model, while the script changed;
the final voice is Travis, from the ElevenLabs Voice Library.
`STORYBOARD.md` has the rest.

**The film page.** `film.html` is a copy of `wordmark.html` with its own
timeline, HUD and opening, not a mode of that page. Almost everything that
moves differs between the two (the pacing, the captions, the camera, the
readouts), and a mode would have put a branch into most of the exhibit's
update code. The price is two copies of the renderer and the sheets, which
the README asks to keep in step. Other decisions:

- *The opening explores the same target from the same starting point* as
  the PyVBMC run (`x0` of the exporter's `run`). The MCMC run is emcee with
  32 walkers, the fastest black-box sampler of the matched-budget
  measurement (`dev/results/2026-09-30-matched-mcmc-budget.md` on
  `dev-next`), stopped at its median matched budget, so that the count on
  screen is the measured number. The first draft had a random-walk
  Metropolis chain of 20,000 evaluations, a length chosen by hand; the
  measurement and the footnote replaced it. The Bayesian
  optimization is a real PyBADS run with the same plausible box, which ends
  at the true maximum.
- *A cost readout, not a speed factor.* The storyboard first had the
  time-lapse show its playback speed. Counting every evaluation at
  3 minutes, the *3 MIN* of the first evaluation, puts the three methods
  on one scale and leads to the payoff. The `timey-wimey` nod moved there.
- *The framing* (the director, 2026-10-04). Through the sixth draft the
  film's first line was "Your model's likelihood takes minutes to
  compute.", and with the 3-minute readout and the payoff's "Weeks of
  waiting became an afternoon" it anchored PyVBMC on very expensive
  likelihoods, while the README recommends it from about half a second
  per evaluation. A viewer whose likelihood takes a second could conclude
  that the tool is not for them. The first line now names the range
  ("seconds, or minutes"), the 3 minutes stay as the worked example from
  the first evaluation to the payoff, and the payoff adds a line and a
  second row of costs at 1 s per evaluation, where the gap between hours
  and minutes is as plain as that between weeks and an afternoon.
  "Waiting" stays. At 3 minutes an evaluation can occupy a whole machine,
  so evaluating in parallel, which shortens the wait of a sampler like
  emcee, is not free, and it never reduces the compute. The MCMC footnote
  therefore ends "Times are total compute cost.", which states what the
  times are; a sentence about emcee's parallel evaluation was set aside as
  defensive about a point many viewers would not raise. PyVBMC's own
  computing (18.6 s here) is counted in the payoff's times but not shown,
  since it changes neither rounded time.
- *The title from the first frame* (the director, 2026-10-05). The
  top-left mark first appeared with scene 4, after the opening on MCMC
  and Bayesian optimization, so for the first 42 s nothing said whose film
  it was: a gap that a feed, playing the film muted from wherever a viewer
  meets it, makes costly. A student of the lab pointed it out after the
  seventh draft was approved. The mark now fades in with the picture, as
  in the PyBADS film; the bottom-left readout still names the method on
  screen, so the mark reads as the film's title over the opening's other
  methods.
- *The number of mixture components and the evidence stay hidden* until the
  narration introduces them, because a number nobody has explained is noise.
- *The voice.* The film wants a light American accent and the delivery of a
  technical scientist, unlike the voice of the lab's earlier film. The
  Voice Library's tags matched mostly performed reads (social media,
  customer support, meditation), so the choice came from auditions of six
  lines of this script. Travis was chosen over Sebastian and Tess, both
  British. The library's most cloned voices, heard in many other videos,
  were left out. After the sixth draft, a second round on the same lines
  tried Tamsin (British), two voices named Adam (ElevenLabs' own "Dominant,
  Firm" and a British narrator) and two named Thaddeus (American), and
  Travis was kept. His notice period is two years, the longest an owner
  can set: if he is withdrawn from the library, he can still voice new
  lines for that long, and the takes already made stay usable for good.
- *Both methods explore.* Lines `o2` and `a3` use the word that scientists
  and the literature on active learning use. Bayesian optimization "decides
  where to explore next" and PyVBMC "explores places that are plausible and
  still uncertain", so the contrast with the optimizer is in what each
  explores for, the peak or the whole posterior, not in whether it explores.
  The label *active learning* names the idea for viewers who know it.
- *Why a mixture, said aloud.* Right after "PyVBMC fits a mixture of
  Gaussians", a viewer who fits models asks why it does not use the
  surrogate itself, so the mixture scene answers. The surrogate gives a
  height at every point, and the mixture can be sampled and averaged over
  exactly. The footnote credits Bayesian quadrature (O'Hagan 1991;
  Rasmussen and Ghahramani's "Bayesian Monte Carlo", 2003), where the
  method's name comes from, and the evidence scene ties back to the same
  calculation.

## Seeds

### `trace_wordmark.js`

Seeds 0 to 59 on the banana alone, the PyVBMC of `254dd6cc` (unchanged at
`dc900e12`, the revision the trace records) with gpyreg 1.4.0, default options,
BLAS single-threaded, on the machine that made the trace: the runs of `--target
banana --sweep 0:60`. Exporting a seed reproduces its sweep line (checked on
the 28 seeds exported for this table). Here, evaluations are the ones still in
the training set at the end; `--sweep` also counts the ones trimmed at the end
of warm-up (105 for seed 42). `arms` is `arm_coverage`. The ELBO SD is 0.001
for every run. The table has the runs that end with 80 to 105 evaluations and
an arm coverage above 0.9, the three that end with more and cover both arms,
seed 3, which leaves the arm tips bare, and seed 22.

| seed | iterations | evaluations | ELBO | gsKL | arms |
|---:|---:|---:|---:|---:|---:|
| **42** | 20 | 103 | -0.019 | 0.056 | 1.00 |
| 50 | 20 | 102 | -0.021 | 0.059 | 1.00 |
| 57 | 18 | 94 | -0.033 | 0.090 | 1.00 |
| 51 | 20 | 104 | -0.026 | 0.087 | 0.99 |
| 10 | 20 | 103 | -0.032 | 0.106 | 0.97 |
| 26 | 20 | 102 | -0.028 | 0.073 | 0.93 |
| 49 | 18 | 94 | -0.036 | 0.088 | 0.93 |
| 4 | 17 | 87 | -0.033 | 0.120 | 0.92 |
| 16 | 16 | 84 | -0.031 | 0.116 | 0.92 |
| 18 | 20 | 103 | -0.043 | 0.152 | 0.92 |
| 19 | 21 | 107 | -0.026 | 0.087 | 1.00 |
| 41 | 21 | 108 | -0.029 | 0.055 | 1.00 |
| 21 | 22 | 112 | -0.012 | 0.042 | 1.00 |
| 3 | 15 | 80 | -0.066 | 0.263 | 0.26 |
| 22 | 13 | 68 | -0.057 | 0.259 | 0.71 |

Among the runs that end with 80 to 105 evaluations, seed 42 has the best ELBO
and the best gsKL and covers both arms. Its smallest hyperparameter length
scale is 1.10 (seed 8 of `trace.js`: 0.010), so no GP state of its trace leaves
a sample out (`drawable_samples`); seed 50 comes close to it and leaves one of
eight samples out of its first fit. Seed 57 is the best run that keeps at most
100. Of the runs that keep more than 105, seed 41's gsKL is 0.001 lower than
seed 42's and seed 21's ELBO is the best of the 60 runs.

Before this sweep, the trace was the run of seed 22 with the PyVBMC of
`5e5fa188`, chosen the same way from a sweep of that code: 18 iterations,
93 evaluations, ELBO -0.013, gsKL 0.058, arm coverage 1.00. The code of
this table gives seed 22 the run of its last row.

### `trace.js`

`--sweep` on the machine that made the trace, with the PyVBMC of
`5e5fa188`, default options. The target's log evidence is 0.
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

The criteria were an ELBO above -0.1 and a low gsKL. Seed 12 has the
lowest gsKL and a shorter run, and was the choice until its first fit
rendered as a serrated comb, which is the needle problem above at its
worst. Seed 8 replaced it before `drawable_samples` existed.

The PyVBMC of `254dd6cc` gives seed 8 another run: 27 iterations, 135
evaluations, ELBO -0.234, gsKL 0.234. This table describes the code of
`5e5fa188`; if `index.html` stays, its seed is chosen again from a sweep
of the current code.
