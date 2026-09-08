# S-VBMC compatibility with current PyVBMC

The shipped-corpus compatibility check passes. No PyVBMC production change or
posterior-file regeneration was needed. This checks the current S-VBMC
package against the integrated latent fixes; the form of future integration
remains a separate decision (roadmap pickup 10).

## Sources and environment

- PyVBMC: `feadf30`, branch `dev-svbmc-compat` created from `dev-next`.
- S-VBMC: `13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01`, version 0.1.1,
  isolated checkout from `acerbilab/svbmc`.
- Gpyreg: `a2f8ddce867f502e29717959cf0ff3529f598618`, matching the CI pin.
- Windows, Python 3.12.6; NumPy 2.5.2, SciPy 1.18.1, Matplotlib 3.11.1,
  pytest 9.1.1, CPU Torch 2.14.0+cpu. BLAS and Torch use one thread.

Imports were verified: PyVBMC resolves to this repository's `pyvbmc/`,
gpyreg to the sibling pinned checkout, and S-VBMC to the isolated checkout's
`src/svbmc/`. Torch and its dependencies were installed with `pip --target`
under the ignored run directory; the existing PyVBMC environment was not
upgraded. S-VBMC was used directly from source via `PYTHONPATH`.

S-VBMC declares Python >=3.11, NumPy >=2.3, SciPy >=1.16, Matplotlib >=3.10
and Torch >=2.7, a higher minimum stack than core PyVBMC. This observation
does not change PyVBMC's Python floor or the agreed PyTorch feasibility plan.

## Checks and results

The unchanged upstream suite passes: **32 passed in 3.53 s**, with no reruns
or skips. The suite's posterior objects are mocks; it does not load the
shipped posteriors, so that result alone is not a saved-object compatibility
gate. Its plotting tests also use a corner stub.

All **30 shipped files** were loaded through plain `pickle.load`, as in
S-VBMC's README. Each resolves to the current `VariationalPosterior` class
and passes component shape/finite-value checks, positive scale and normalized
weight checks, and the required `stable`, `elbo`, `I_sk`, `J_sjk` presence
checks. `I_sk` and `J_sjk` shapes and finite values pass. Sampling 32 points
and evaluating their original-space densities pass for every file.

Each group was then passed to `SVBMC` with its default filtering and ordinary
sampling mode (`testing=False`). The bounded check used seed 1701 for NumPy
and Torch, entropy samples per component 5, and 3 optimization steps.

| Shipped group | Loaded | Retained by default filters | Stacking checks |
| --- | ---: | ---: | --- |
| GMM | 10 | 10 | Passed |
| GMM_noisy | 10 | 10 | Passed |
| Ring | 10 | 10 | Passed |

For each group, construction, finite entropy/Jacobian corrections and
weight gradients, short weight optimization, normalized nonnegative finite
output weights, finite ELBO/entropy, and sampling pass. Input component
parameters remain bit-identical; transformer identity and stored numerical
state are unchanged. The real-posterior checks completed in about 7.5 s.

An additional bounded probe confirms an upstream S-VBMC shape defect for
one-dimensional inputs. SciPy's component draws are squeezed to `(5,)`
instead of `(5, 1)`, and `stacked_entropy` passes them to the transform's
Jacobian method, which interprets a vector as one point. Two actual D=1 VPs
therefore raise an indexing error with `n_samples=5`; the identical D=2
control passes with a finite gradient. The affected transformer/decorator
methods are unchanged from PyVBMC `origin/main`; this is not introduced by
the latent fixes. All shipped posteriors and the suite's VP mocks use D=2.
Correct the component-draw shape and add a real D=1 regression when bringing
S-VBMC into PyVBMC. No upstream code was changed in this compatibility check.

Other behavior details should inform subsequent integration:

- `SVBMC.sample(64)` returned 65 points for each tested group because it
  rounds each posterior's allocated count independently. This is existing
  S-VBMC behavior, not a PyVBMC compatibility regression.
- Current VP deep copies share their random generator. S-VBMC samples from
  copied VPs, so sampling can advance the input VPs' generators while their
  posterior parameters remain unchanged. This follows PyVBMC's documented
  RNG contract; it is not isolated sampling with respect to RNG state.
- S-VBMC assigns copied component weights as a one-dimensional array.
  Current VP sampling tolerates that by flattening weights, but integration
  should preserve the canonical `(1, K)` shape.

This is a bounded API/save-compatibility check on Windows/Python 3.12, not
a convergence or population-quality experiment, a GPU check, or S-VBMC's
full OS/Python release matrix. Fresh VBMC target optimization was not run.

## Reproduction and evidence

Local artifacts are under the ignored
`dev/scripts/runs/svbmc_compat_20260908/`: `source/` (pinned checkout),
`deps/` (Torch overlay), `environment.json`, `torch-install.json`,
`suite.log`, `suite.xml`, `check_posteriors.py`, `posteriors.log`,
`posteriors.json`, and `dimension_probe.log`. The one-off checker is local
diagnostic material; the durable contracts and outcomes are recorded above.

On a clean checkout, fetch S-VBMC from `https://github.com/acerbilab/svbmc`
and check out the full source revision above. Its 30 tracked pickle files
come with that checkout. Recreate the optional dependency overlay with
`python -m pip install --target <deps> torch==2.14.0+cpu --index-url
https://download.pytorch.org/whl/cpu` using the recorded Python 3.12 stack,
or create a fresh environment with the recorded versions. Point PyVBMC and
gpyreg imports at the recorded checkouts and verify them before running.
The upstream suite is reproducible from those repositories. Historical
local logs and the one-off checker are not part of a fresh clone; if they
are unavailable, regenerate a diagnostic from the exact operations, seeds,
shapes and budgets listed above. They are not prerequisites for Stage 4.

With the repository root, `source/src` and `deps` on `PYTHONPATH`, run the
repository's `.venv/Scripts/python.exe` with:

```text
-m pytest dev/scripts/runs/svbmc_compat_20260908/source/tests -q -p no:cacheprovider
dev/scripts/runs/svbmc_compat_20260908/check_posteriors.py
```

The initial restricted-account import attempt could not read the Torch
installation created by the package installer; its apparent missing
`torch.optim` error was a local permission issue. The subsequent import
verification and both completed checks ran with access to that directory.

Next: use these compatibility results to decide the integration approach
and downstream release coordination. No S-VBMC source or pickle was changed.

Independent Sol review verified the artifacts, checker and stated limits,
with no findings. Repository documentation formatting hooks pass. The
compatibility check is complete; the D=1 repair and interface improvements
belong to the subsequent integration implementation.
