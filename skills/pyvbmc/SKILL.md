---
name: pyvbmc
description: Find and apply PyVBMC documentation when setting up Bayesian inference, troubleshooting a run, or interpreting posterior and model-evidence estimates.
---

# PyVBMC

Use the documentation below to answer the user's question or work on their
analysis. Read the sections relevant to the task; follow links for details
as needed.

Check the installed PyVBMC version before using version-specific features.
This skill accompanies PyVBMC 1.5. Prefer documentation from the user's
checkout when available. The published [documentation](https://acerbilab.github.io/pyvbmc/)
may lag development; the source links below point to the
[`dev-next` branch](https://github.com/acerbilab/pyvbmc/tree/dev-next).
For another version, use the corresponding Git tag or branch and check API
signatures and docstrings in that version.

## What to read

Paths are relative to the PyVBMC repository root. The links also work when
this skill folder has been copied elsewhere.

| Task | Read |
| --- | --- |
| Decide whether PyVBMC fits the problem; install it | [README.md](https://github.com/acerbilab/pyvbmc/blob/dev-next/README.md): “When should I use PyVBMC?” and “Installation”. |
| Set up or adapt an analysis | [docsrc/source/quickstart.rst](https://github.com/acerbilab/pyvbmc/blob/dev-next/docsrc/source/quickstart.rst); consult the FAQ's “Input arguments” sections for target functions, priors, starting points and bounds. |
| Handle noisy likelihoods | The FAQ's “Noisy target function” section and [Example 6](https://github.com/acerbilab/pyvbmc/blob/dev-next/examples/pyvbmc_example_6_noisy_likelihoods.ipynb). |
| Interpret results, compare runs, or troubleshoot | The FAQ's “Output arguments”, “Display” and “Troubleshooting” sections; [Example 4](https://github.com/acerbilab/pyvbmc/blob/dev-next/examples/pyvbmc_example_4_validation.ipynb) for validation. |
| Save or resume; combine posteriors | The FAQ's “How do I save and continue a run?” and “Can I combine the posteriors of several runs?”; [Example 7](https://github.com/acerbilab/pyvbmc/blob/dev-next/examples/pyvbmc_example_7_stacking.ipynb) for S-VBMC. |
| Look up exact arguments, options or posterior methods | The [API reference](https://acerbilab.github.io/pyvbmc/documentation.html), with sources under `docsrc/source/api/` and implementation docstrings under `pyvbmc/`. The quickstart also covers vectorized targets and optional exports; the FAQ's “Troubleshooting” section covers reproducibility. |

The FAQ is [docsrc/source/faq.md](https://github.com/acerbilab/pyvbmc/blob/dev-next/docsrc/source/faq.md).

Before executing model evaluations or additional fits, establish the user's
evaluation budget and account for any diagnostic calls within it. When
working on an existing analysis, inspect its setup and saved results before
deciding whether another run is needed. Link the relevant documentation in
your explanation so the user can check the reasoning.
