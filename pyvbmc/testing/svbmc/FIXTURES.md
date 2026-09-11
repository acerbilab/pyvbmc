# S-VBMC fixtures

Fitted posteriors and regression references for the tests of
`pyvbmc.svbmc`, stored as plain arrays. Every posterior is one
`VariationalPosterior` written with the oracle snapshot codec of
`pyvbmc/testing/oracles/_state.py`: `<name>.npz` holds the arrays (loaded
with `np.load(path, allow_pickle=False)`), `<name>.json` the tree, the
scalars and a `meta` block with the provenance. `_fixtures.py` next to
this file rebuilds them through the public constructors (`load_vp`,
`load_group`), so no file depends on the layout of any class. Everything
here is written by `dev/scripts/make_svbmc_fixtures.py`; regenerate only to
set a new baseline on purpose.

Each posterior snapshot stores the transformer (`lb_orig`, `ub_orig`, `mu`,
`delta`, `type`, `R_mat`, `scale`) and the posterior (`w`, `eta`, `mu`,
`sigma`, `lambd`, the `optimize_*` flags, `bounds` and the complete `stats`
dictionary, including the `stable`, `elbo`, `elbo_sd`, `I_sk` and `J_sjk`
entries that stacking reads). The plausible bounds in the `pt` block are
placeholders that only make the rebuilt transformer's constructor valid;
the stored `mu`, `delta` and `type` overwrite what the constructor derives
from them.

## Upstream corpus (`upstream_*`, 30 files)

The thirty posteriors shipped with S-VBMC 0.1.1 (`acerbilab/svbmc` at
`13a78f6c4a3ffe9c4557fee4f4b8f67c98b29e01`, `vbmc_runs/`), converted from
their pickles on 2026-09-11 (`make_svbmc_fixtures.py convert`). Every
rebuilt posterior matches its pickle exactly: component parameters,
statistics, transform outputs and `pdf` values. The `meta` block of each
file records the source path and the SHA-256 of the pickle.

| group | files | D | K | transform | target (upstream description) |
| --- | ---: | ---: | ---: | --- | --- |
| `upstream_GMM` | `upstream_GMM_00` to `_09` | 2 | 50 | unbounded (identity) | `svbmc.targets.GMM`, noiseless |
| `upstream_GMM_noisy` | `upstream_GMM_noisy_00` to `_09` | 2 | 50 | unbounded (identity) | `svbmc.targets.GMM`, noisy log-likelihood |
| `upstream_Ring` | `upstream_Ring_00` to `_09` | 2 | 50 | unbounded (identity) | `svbmc.targets.Ring` |

All thirty runs are `stable`. The corpus is entirely two-dimensional,
unbounded and unwarped, which is why the supplementary set below exists.

## Supplementary posteriors (9 files)

Short seeded VBMC runs on inline targets (`make_svbmc_fixtures.py generate`,
2026-09-11, PyVBMC `8c2412d`, options `{"display": "off"}`, otherwise the
defaults; the starting point is drawn uniformly in the run's plausible box
from `np.random.default_rng(seed)`). Three runs per group, seeds 100, 101
and 102; every run ended `stable` with finite statistics and K = 50 after
the final boost. The `meta` block records the seed, the plausible box, the
ELBO and its standard deviation, the number of function evaluations and
whether a warp was in place.

| group | D | bounds | plausible boxes (per run) | warp | target |
| --- | ---: | --- | --- | --- | --- |
| `normal_D1` | 1 | none | `[-3, 4]` for all runs | no | `log N(x; 0.5, 1.2^2)` |
| `bounded_D2` | 2 | `[-3, 3]^2` (probit) | `[-2, 2]^2`; `[-1.5, 1.5]^2`; `[-1, 2.5] x [-2.5, 1]` | no | `log N(x; (0.5, -0.5), diag(0.8^2, 0.6^2))` on the box |
| `corr_D3` | 3 | none | `[-4, 4]^3` for all runs | seed 100 no; seeds 101 and 102 yes | `log N(x; 0, S R S)`, sd `(1, 2, 0.5)`, all correlations 0.85 |

`bounded_D2` gives three runs of one problem with three different
transforms (different `mu` and `delta`); `corr_D3` mixes an unwarped run
with two warped ones (`R_mat` and `scale` set, `delta` reset to one).

## Regression references (`references.npz` / `references.json`)

`make_svbmc_fixtures.py references` (2026-09-11, Torch 2.14.0 CPU with one
thread, NumPy 2.5.2, PyVBMC `8c2412d`): for every group above and every
optimization mode, `SVBMC(vps, seed=0).optimize(max_steps=3, version=mode)`
with `vps = load_group(group, rng=0)[0]`. The tree holds, under
`groups/<group>/<mode>`, the optimized weights `w` (an array of shape
`(1, K_total)`), the three `elbo` values (`estimated`,
`debiased_I_median`, `debiased_E_median`), the `entropy`, `M` and `K`;
`meta` records the environment. The references pin the numerics of the
integrated class (the Monte Carlo draws come from the object's generator,
so they cannot be compared with the upstream package draw for draw) and
are the gate for later performance changes to the entropy computation.

## Tests

| test module | reads | tolerance |
| --- | --- | --- |
| `test_svbmc.py` | `upstream_GMM`, `normal_D1`, `bounded_D2`, `corr_D3` via `load_group(group, rng=0)` | shapes, dtypes and reproducibility exact; sample statistics loose (means within 0.25 or 0.5, standard deviations at `rtol=0.25`) |
| `test_svbmc_filters.py` | synthetic posteriors and `upstream_GMM` | exact |
| `test_svbmc_references.py` | every group via `load_group(group, rng=0)` and `references` | `rtol=1e-8`, `atol=1e-10` on `w`, the three `elbo` values and `entropy`; `M` and `K` exact |
| `test_svbmc_imports.py`, `test_svbmc_utils.py` | none | run without torch |

New fixture files or directories must be added to `MANIFEST.in`.
