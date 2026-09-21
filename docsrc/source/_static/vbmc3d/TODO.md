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
   keeping this folder's Markdown files excluded from the sources), link it
   from `docsrc/source/index.rst`, and add the entry to `CHANGELOG.md`:
   publishing is what makes the change noticeable to users, and until then
   the changelog policy asks for none.
3. **Put the recording script in the repository** (video and GIF for the
   README hero, talks, social). A throwaway one, kept outside the
   repository, recorded `wordmark.html`; the recipe, to write it again:
   - Plain Node (24 has `fetch` and `WebSocket`, so no dependencies).
     Start headless Chrome (`--headless=new --enable-unsafe-swiftshader
     --remote-debugging-port=P`), take the page target from
     `http://127.0.0.1:P/json/list`, and over its WebSocket:
     `Emulation.setDeviceMetricsOverride` for the exact viewport,
     `Page.navigate` to the page with `capture=1`, wait until
     `window.vbmcCapture` exists and 1.5 s more for the typeface, then
     `vbmcCapture.frame(t0, 0)` and `vbmcCapture.frame(t, 1 / fps)` for each
     later frame, each followed by `Page.captureScreenshot` (PNG).
   - Pipe the PNGs to ffmpeg (`-f image2pipe -framerate 30 -c:v png -i -
     -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow -movflags
     +faststart`). The machine has no `ffmpeg`; `pip install --target DIR
     imageio-ffmpeg` bundles one without touching the venv.
   - For a video with captions and without the controls, inject a style
     `#transport, #chips { display: none !important; }`; `hud=0` hides the
     captions too.
   - Software rendering takes about 0.18 s a frame at 960 x 540 and 0.45 s
     at 1280 x 720: the whole 91.45 s loop at 30 fps (2744 frames) took 20
     minutes.
   - The post-processing's grain and scanlines change every pixel of every
     frame, which is what makes files large: the loop at 1280 x 720 is 118
     MB at CRF 18, 37 MB after `hqdn3d=2:2:5:5` at CRF 24, and 17 MB at
     960 x 540 after `hqdn3d=3:3:6:6` at CRF 25. A GIF of the finale (from
     83.5 s to 3 s into the title card, 11 s) was 21 MB at 640 px encoded
     directly; with `hqdn3d=6:5:10:10`, `palettegen=max_colors=96
     (or 64):stats_mode=diff` and `paletteuse=dither=bayer:bayer_scale=3
     (or 2):diff_mode=rectangle` it is 8.3 MB at 640 px and 15 fps, 3.1 MB
     at 480 px and 12 fps. A query parameter that turns the grain off would
     shrink recordings further.
   - The GIF cuts from the held wordmark back to the orbiting landscape
     where it loops; the loop video starts and ends on the wordmark.
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
