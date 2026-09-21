# 3D animation: to do

Current actions, constraints and links for the work in this folder. An action
that is done is removed from this file; what was decided and why goes to
`NOTES.md`. How the pages work, and how to run, regenerate and check them,
is in `README.md`.

## Next actions

In this order. The first two are decisions for the project owner.

1. **Steer `wordmark.html`'s finale.** The owner has watched it and the
   recordings of item 3; what is still open is the camera's pull-back, how
   long the wordmark holds, the letters' color, weight and brightness
   against the V, and the V in the log view against `?wmview=density`.
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
4. **Pacing.** A loop lasts about 91 s on `wordmark.html` and 95 s on
   `index.html`. The durations are the `seg(...)` calls and the `DP` / `DF`
   arrays of the timeline section.
5. **Phones.** On a portrait screen the wordmark is small and the V's dense
   wires add up to white. The frame rate on phones, which draw the full
   64 x 64 mesh, is unmeasured.

If `index.html` stays, two of its questions are open: whether the owner
accepts `drawable_samples` on its first fit (5 of 8 samples shown; the
alternatives are in `NOTES.md`, "Needles after the first fit"), and whether
seed 12, set aside before `drawable_samples` existed, is now the better run
to show (`NOTES.md`, "Seeds").

## Constraints

- This work is kept apart from the repository's trackers (`dev/TODO.md`, the
  roadmap, the plans, `AGENTS.md`): nothing there refers to it, and its
  notes stay in this folder.
- After regenerating a trace, run its page's self-check (`?debug=1`, recipe
  in `README.md`) with a fresh browser profile. Nothing else gates the
  pages.
- A run is reproducible only on the platform that made it. On another
  machine the committed seeds give other trajectories: choose seeds again
  with `--sweep`, and expect the numbers quoted in `README.md` and
  `NOTES.md` to change.
- The branch `feat-3d-animation` is based on `5e5fa188`, behind `dev-next`.
  Outside this folder it touches `dev/scripts/export_animation_trace.py`
  (new), `dev/README.md` (one entry in the scripts list, a merge conflict
  with `dev-next`) and `docsrc/source/conf.py` (the exclusion of this
  folder's Markdown files). `dev-next` also changes `pyvbmc/` in ways that
  can move a noiseless run (among them the carried-over GP bounds and the
  tie-breaking of the warm-up trim), so after merging, export both seeds
  again and compare with the committed traces before trusting them.
