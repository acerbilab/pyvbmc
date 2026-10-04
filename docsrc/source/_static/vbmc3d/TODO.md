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
the commands. The film is mastered (`NOTES.md`, Status), with its final
end card. What is left is to publish it, and first to settle one
question about its framing:

1. **Decide whether to fix the film's framing (open).** The film anchors
   PyVBMC on very expensive likelihoods: the first line is "Your model's
   likelihood takes minutes to compute", the cost readout counts 3 min per
   evaluation, and the payoff says "Weeks of waiting became an afternoon"
   (its card: "AT 3 MIN PER EVALUATION · A 2-PARAMETER EXAMPLE"). The
   README says PyVBMC is effective from about half a second per
   evaluation. On the film's own counts, 27,828 evaluations of emcee
   against PyVBMC's 105 (plus about a minute of its own computing, an
   estimate in `STORYBOARD.md`), one second per evaluation still turns
   about 7.7 hours into about 3 minutes, which the film never shows; a
   viewer whose likelihood takes a second may conclude the tool is not for
   them. Each line is honest as an example; together they narrow the use.
   - *Leave the film* and carry the range in the YouTube title and
     description, the posts and the documentation.
   - *Fix it*: re-voice `p1` (for instance "Each evaluation of your
     model's likelihood takes time.") and perhaps `f3`, so that the opening
     states the general case and the 3-minute readout becomes the worked
     example. Each re-voiced scene is a new Travis take (ElevenLabs
     characters; no take comes out the same) and re-times every scene
     after it; then the events and the score again, and both 1080p masters
     again (about 3 hours of recording).
   A public YouTube upload cannot be replaced, only uploaded again under a
   new address, so the upload stays unlisted until this is decided.
2. **Upload the clean master to YouTube** with its `.srt`, as the English
   subtitles. The title, the description, the chapters, the thumbnail and
   the social posts are drafted with the lab's media notes
   (`dev/scripts/runs/LOCAL.md` says where); the title avoids "minutes".
3. **Post the film**: the captioned master natively on Bluesky and
   LinkedIn (2:44 and 174 MB are inside both platforms' limits), and the
   YouTube link on X, which caps free accounts at 2:20.
4. **Link the film** from the documentation and the README. Where it goes
   is part of the second of the next actions above, which page the
   documentation shows and how.
5. **Back up `dev/media/vbmc3d-film/`** (the Travis takes, the masters,
   the thumbnail): it exists only in this worktree, and removing the
   worktree deletes it.

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
