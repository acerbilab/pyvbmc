# 3D animation: to do

Current actions, constraints and links for the work in this folder. An action
that is done is removed from this file; what was decided and why goes to
`NOTES.md`. How the pages work, and how to run, regenerate and check them,
is in `README.md`.

## Next actions

In this order. The first two are decisions for the PI.

1. **Steer `wordmark.html`'s finale.** It has been watched, with the
   recordings of item 3, on the run of seed 22; the page now plays the
   run of seed 42 (`NOTES.md`, "Seeds"), which only headless stills have
   shown, so the first thing to settle is whether its V is the one to keep.
   What is open after that is the camera's pull-back, how long the wordmark
   holds, the letters' color, weight and brightness against the V, and the
   V in the log view against `?wmview=density`.
2. **Decide which page the documentation shows, and how**: `wordmark.html`,
   `index.html` or both; a link next to the corner-plot GIF on the index
   page, or a page embedded as the hero. Then publish the folder
   (`html_static_path` or `html_extra_path` in `docsrc/source/conf.py`,
   keeping this folder's Markdown files excluded from the sources and
   `scripts/` out of the published output), link it from
   `docsrc/source/index.rst`, and add the entry to `CHANGELOG.md`:
   publishing is what makes the change noticeable to users, and until then
   the changelog policy asks for none.
3. **Make the recordings for their uses** with `scripts/record.mjs`
   (README, "Recording"): a README GIF has to be a short, small excerpt
   (the wordmark finale is one), a talk wants the whole loop as video. The
   grain dominates the file sizes; a query parameter that turns it off for
   recordings would shrink them. A GIF of an excerpt cuts where it loops,
   from the held wordmark back to the orbiting landscape; a fade at both
   ends would hide the cut.
4. **Pacing.** A loop lasts about 94 s on `wordmark.html` and 95 s on
   `index.html`. The durations are the `seg(...)` calls and the `DP` / `DF`
   arrays of the timeline section.
5. **Phones.** On a portrait screen the wordmark is small and the V's dense
   wires add up to white. The frame rate on phones, which draw the full
   64 x 64 mesh, is unmeasured.

If `index.html` stays, its trace needs a seed chosen again: `trace.js` is a
run of the PyVBMC of `5e5fa188`, and the PyVBMC of `254dd6cc` gives seed 8 a
worse run (`NOTES.md`, "Seeds"). Sweep the default target
(`--sweep 0:60`) and choose as before; then, if the chosen run's first fit
leaves samples out, settle whether `drawable_samples` is acceptable
there (the alternatives are in `NOTES.md`, "Needles after the first fit").

### The video

`STORYBOARD.md` has the script and the scenes, and README.md ("The film")
the commands. In this order:

1. **Review the draft** (1280 x 720, Kokoro's `af_heart`, the synthesized
   score) for the pictures, the pacing, the script and the sound. Before the
   drafts, the score was checked only as levels and a spectrogram.
2. **Choose the final voice**, then voice and time the film again.
3. **Render at 1920 x 1080**, with and without captions, and write the
   `.srt` of the final timeline.

## Constraints

- The repository's trackers name this work in two places: an item of
  `dev/TODO.md`, and the plan `dev/plans/matched-mcmc-budget.md`, whose
  number the film quotes. The animation's design, status and next actions
  stay in this folder.
- After regenerating a trace, run its page's self-check (`?debug=1`, recipe
  in `README.md`) with a fresh browser profile. Nothing else gates the
  pages.
- A run is reproducible only on the platform that made it. On another
  machine the committed seeds give other trajectories: choose seeds again
  with `--sweep`, and expect the numbers quoted in `README.md` and
  `NOTES.md` to change.
- `film.html` shares its sections from "renderer" to "post" with
  `wordmark.html` (README). A fix to one of them goes into both pages.
- Outside this folder the branch `feat-3d-animation` touches
  `docsrc/source/conf.py` (the exclusion of this folder's Markdown files).
  `dev/scripts/export_animation_trace.py` and its entry in the scripts list
  of `dev/README.md` are on `dev-next` as well since `e723609f`, so a change
  to either is made on `dev-next`. A change to `pyvbmc/` or gpyreg can move a
  noiseless run, so after the branch takes in `dev-next`, export the
  committed seeds again and compare with the traces, which record the
  revisions that made them (`meta.pyvbmc`, `meta.gpyreg`).
