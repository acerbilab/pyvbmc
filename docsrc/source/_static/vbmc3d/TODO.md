# 3D animation: to do

Current actions, constraints and links for the work in this folder. An action
that is done is removed from this file; what was decided and why goes to
`NOTES.md`. How the page works, and how to run, regenerate and check it, is
in `README.md`.

## Next actions

In this order. The first two are decisions for the project owner.

1. **Settle what the sheet shows after the first fit.** Watch the first 30
   seconds of the page in motion. The displayed GP leaves out hyperparameter
   samples thinner than two grid cells (`drawable_samples` in
   `dev/scripts/export_animation_trace.py`), which departs from the GP that
   VBMC averages and has not been confirmed by the owner. Nobody has seen
   the page in motion since that remedy went in. The alternatives considered
   are in `NOTES.md` ("Needles after the first fit").
2. **Decide how the documentation shows the page**: a link next to the
   corner-plot GIF on the index page, or the page embedded as the hero. Then
   publish the folder (`html_static_path` or `html_extra_path` in
   `docsrc/source/conf.py`, keeping this folder's Markdown files excluded
   from the sources), link it from `docsrc/source/index.rst`, and add the
   entry to `CHANGELOG.md`: publishing is what makes the change noticeable
   to users, and until then the changelog policy asks for none.
3. **Write the recording script** for video and GIF (README hero, talks,
   social): a driver that opens the page with `capture=1`, calls
   `vbmcCapture.frame(t, dt)` for successive frames from the start with one
   `dt`, grabs each frame and encodes them. The development machine has
   Chrome and Node but no `ffmpeg`; the `imageio-ffmpeg` package bundles
   one. A README GIF has to be a short, small excerpt; a talk wants the
   whole loop as video.
4. **Pacing.** The loop lasts about 95 s. The durations are the `seg(...)`
   calls and the `DP` / `DF` arrays of the page's timeline section.
5. **Possibly reconsider the seed.** Seed 12 has the lowest gsKL and a
   shorter run than the committed seed 8, and was set aside for a rendering
   problem that `drawable_samples` has since addressed (`NOTES.md`,
   "Seeds"). Export it and look.

Also unmeasured: the frame rate on phones, which draw the full 64 x 64 mesh.

## Constraints

- This work is kept apart from the repository's trackers (`dev/TODO.md`, the
  roadmap, the plans, `AGENTS.md`): nothing there refers to it, and its
  notes stay in this folder.
- After regenerating `trace.js`, run the page's self-check (`?debug=1`,
  recipe in `README.md`) with a fresh browser profile. Nothing else gates
  the page.
- A run is reproducible only on the platform that made it. On another
  machine the committed seed gives another trajectory: choose a seed again
  with `--sweep`, and expect the numbers quoted in `README.md` and
  `NOTES.md` to change.
- The branch `feat-3d-animation` is based on `5e5fa188`, behind `dev-next`.
  Outside this folder it touches `dev/scripts/export_animation_trace.py`
  (new), `dev/README.md` (one entry appended to the scripts list, a likely
  merge conflict) and `docsrc/source/conf.py` (the exclusion of this
  folder's Markdown files).
