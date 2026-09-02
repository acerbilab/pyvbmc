# Developer notes

Dated developer logs for PyVBMC. One file per working session or decision,
named `YYYY-MM-DD-short-slug.md`, newest at the bottom of the list below.

These are working notes for maintainers, not user documentation. They record
what was investigated, what was decided and why, and what was deliberately
left open, so that a future contributor (or the same person six months later)
can reconstruct the reasoning without re-deriving it.

Why `dev/` and not `docs/dev/`: `docs/` is the Sphinx HTML build output
directory. It is gitignored and its contents are copied verbatim to the
`gh-pages` branch by `.github/workflows/docs.yml`, so nothing placed there
would be committed, and anything that was would be published to the docs site.

## Index

- `2026-09-02-modernization-discussion.md` — Assessment of porting the
  numerical core from NumPy to a tensor backend. Hot-path analysis, gradient
  inventory, gpyreg audit, backend decision (PyTorch), staged plan, repo
  process decisions, list of latent bugs found along the way.
- `2026-09-02-user-agent-skill.md` — Deferred idea: ship an Agent Skills
  folder so users' coding agents set up and troubleshoot PyVBMC correctly.
  What BayesFlow did, the spec, and a design that lives in-repo and ships in
  the wheel. Build after the 1.5 API settles.
## Plans, worklogs and task files

Devlogs (above) are the record of discussions and decisions that involved a
person, most often the PI. Everything an agent produces on its own along the
way lives in `plans/`: plans, checklists and status trackers, worklogs of a
work session, measurement reports, task files. They are updated in place
while the work is open and kept afterwards as the record of what was done;
status and next steps never go into the devlogs.

- `plans/modernization-roadmap.md` — living tracker of the staged plan in
  `2026-09-02-modernization-discussion.md` §10: stage status, pickup point.
- `plans/profile-and-gradient-checks.md` — dev environment, baseline test
  run, first measured profile (D=5, D=10) and the first Stage 0
  finite-difference gradient checks, which found the reshape-order bug in
  `_vp_bound_loss`.
- `plans/stage1-rng-generator.md` — Stage 1 worklog: `VBMC(seed=)`,
  `vbmc.rng`, the gpyreg/cma global-state seam, the random-state save
  format, what the tests had to change, review findings, follow-ups.

Naming: `plans/` files are named by slug only, never by date (the date is in
the file header), so that they cannot be mistaken for copies of the dated
devlogs.

## Scripts

`scripts/` holds developer tooling that is not part of the package or the
test suite. Output directories under it (e.g. `scripts/runs/`) are gitignored;
results that matter get summarized in a dated devlog entry, not committed raw.

- `scripts/profile_run.py` — run VBMC on a cheap synthetic target under a
  fixed seed and report per-stage timers and, with `--cprofile`, a cProfile
  attribution of the hot paths. See its docstring.
