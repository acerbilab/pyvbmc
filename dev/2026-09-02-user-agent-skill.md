# 2026-09-02 — A PyVBMC skill for users' coding agents

**Status:** idea recorded, deferred. Build after the 1.5 work in
`2026-09-02-modernization-discussion.md` has stabilised the user-facing API
(seed argument, batched evaluation, `to_torch`), so the skill documents the
API we want rather than the one we are about to change.

## The idea

Many PyVBMC users will be writing their inference scripts with a coding agent,
if they aren't already. A skill in the open Agent Skills format (a folder with
`SKILL.md` plus optional `references/` and `scripts/`, loaded by Claude Code,
Codex, Gemini CLI, Cursor and others) lets that agent set up and troubleshoot
a PyVBMC run using our own guidance instead of guessing from docstrings.

A good chunk of the content already exists. The VBMC wiki FAQ
(https://github.com/acerbilab/vbmc/wiki, ~8,500 words, ~40 questions,
algorithm-general but MATLAB-flavoured) is mostly about setup and
interpretation: hard bounds used where plausible bounds were meant, plausible
bounds far too wide, a target returning `-inf`/NaN, noisy targets without
`specify_target_noise` and an SD estimate, `elbo_sd` read as the gap to the
true evidence, a non-monotone ELBO trace read as failure, a single run trusted
without diagnostics. An agent won't apply these rules unless told, which is
also the case BayesFlow makes for its skill.

## What BayesFlow did (checked 2026-09-02)

- Not in the library repo. The BayesFlow README has a one-line "Agentic AI
  Workflows" section pointing to a separate repo,
  https://github.com/Learning-Bayesian-Statistics/baygent-skills (Alexandre
  Andorra, with Stefan Radev as co-author; MIT). Three skills:
  `bayesian-workflow` (PyMC/ArviZ), `causal-inference`, `amortized-workflow`
  (BayesFlow).
- Installation is copying the folder into `~/.claude/skills/` (or the
  equivalent for other agents).
- `amortized-workflow/SKILL.md` is an opinionated workflow, not documentation:
  a ten-step pipeline; eleven MUST/NEVER rules described as "critical
  guardrails that agents will usually not apply unprompted"; a complete
  runnable template (~150 lines); anti-patterns; verification gates
  (diagnostics before proceeding); a troubleshooting table mapping failure
  modes to causes and fixes; seven `references/*.md` files loaded on demand;
  two `scripts/` (`inspect_training.py`, `check_diagnostics.py`) the agent
  runs. The `description` field is long and keyword-dense, so the skill
  triggers on mentions of the library or of SBI concepts.
- Weakness: `SKILL.md` is ~850 lines. The spec recommends under 500 lines /
  ~5,000 tokens for the always-loaded part, with detail in `references/`.
  No documented mechanism keeps the skill in sync with library versions.

Spec essentials (https://agentskills.io/specification): frontmatter `name`
(lowercase, hyphens, must match the folder), `description` (≤1024 chars, say
what and when), optional `license`, `compatibility`, `metadata`,
`allowed-tools`. Progressive disclosure: metadata always loaded (~100 tokens),
body on activation, references on demand. Validate with `skills-ref validate`.

## Design for PyVBMC

Differences from the BayesFlow approach:

- **In this repo, under `skills/pyvbmc/`, and shipped inside the wheel**, with
  a one-line installer (e.g. `python -m pyvbmc install-skill`) that copies it
  to the agent's skills directory. The skill version then always matches the
  installed library; CI validates the frontmatter and smoke-tests the template
  with a tiny budget; the options reference is generated from the `.ini` files
  rather than maintained by hand. A README "Agentic AI Workflows" section
  points to it, as BayesFlow's does.
- **Short, opinionated `SKILL.md` (<500 lines):**
  - when to use PyVBMC vs MCMC/PyMC, PyBADS, or SBI, and the dimension/cost
    regime it is designed for;
  - the four-step workflow with a runnable template;
  - hard rules: target always finite; prior included in the target; hard
    bounds = support, plausible bounds ≈ the 68% prior interval, and the two
    must differ; no integer parameters; noisy targets need
    `specify_target_noise=True` and an approximately unbiased SD estimate
    (ideally SD ≈ 1, at most ~3); run more than once; check convergence
    diagnostics before trusting anything;
  - a table for reading `results` (`elbo`, `elbo_sd` is GP uncertainty, not
    the variational gap; `success_flag`; `r_index`; `sKL`; `K`);
  - a troubleshooting table (non-convergence warning, wild ELBO oscillation,
    non-finite values, run-to-run variability).
- **`references/`:** the FAQ ported to Python names and PyVBMC options;
  noisy targets; priors (`pyvbmc.priors` and `convert_to_prior`);
  diagnostics; the `VariationalPosterior` API with array shapes and
  `orig_flag`; the generated options reference; pointers to the example
  notebooks.
- **`scripts/`:** a pre-flight check that validates shapes and bounds and
  probes the target at a few points inside the plausible box (finite? noise
  tuple shape right?); a results summariser that reads `results` and the
  iteration history and states in plain words whether the run converged and
  what to try next.

## Interaction with the 1.5 plan

- The skill is one more reason to settle the user-facing API surface before
  release; every API change means a skill change.
- `seed=` becomes a hard rule in the skill ("always pass a seed") rather than
  a workaround, and batched evaluation of the initial design becomes a
  recommendation for GPU/vectorised targets.
- The FAQ port is independent of the code and could start earlier if someone
  has time; it is also useful as a plain documentation page.

## Sources

- https://github.com/bayesflow-org/bayesflow (README, "Agentic AI Workflows")
- https://github.com/Learning-Bayesian-Statistics/baygent-skills
- https://agentskills.io/specification
- https://github.com/acerbilab/vbmc/wiki
