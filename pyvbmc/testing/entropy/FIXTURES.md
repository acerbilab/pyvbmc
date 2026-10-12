# Entropy fixtures

Reference values produced by the MATLAB VBMC toolbox and stored as plain
NumPy arrays (`np.load(path, allow_pickle=False)`). MATLAB is needed to make
them again, and the script that produced this one is not in the repository,
so treat the numbers as fixed.

## `entropy-test.npz`

One variational posterior with `D = 4` and `K = 3`, together with the entropy
of that mixture and its gradient as MATLAB computes them, both by the Jensen
lower bound (`Hl`, `dHl`) and by Monte Carlo over `Ns` samples (`H`, `dH`).
In the repository since 2022-11-23.

| key | shape | contents |
| --- | --- | --- |
| `D` | (1, 1) int | number of dimensions, 4 |
| `K` | (1, 1) int | number of mixture components, 3 |
| `Ns` | (1, 1) int | Monte Carlo sample count behind `H` and `dH`, 100000 |
| `jacobian_flag` | (1, 1) bool | True: the gradients are with respect to the transformed parameters (log sigma, log lambda, softmax w) |
| `vp_mu` | (4, 3) | component means |
| `vp_sigma` | (3, 1) | component scales, one per component; the entropy functions ravel this, so the column orientation carries no meaning |
| `vp_lambd` | (4, 1) | per-dimension length scales |
| `vp_w` | (1, 3) | mixture weights |
| `vp_eta` | (1, 3) | unnormalized weights, the softmax pre-image of `vp_w` |
| `Hl` | (1, 1) | Jensen lower bound on the entropy |
| `dHl` | (22, 1) | its gradient |
| `H` | (1, 1) | Monte Carlo estimate of the entropy |
| `dH` | (22, 1) | its gradient |

The 22 gradient entries are ordered `mu` (12, column-major over the `(D, K)`
array), then `sigma` (3), `lambda` (4), `w` (3).

`test_entlb_vbmc.py::test_entlb_vbmc_matlab` checks `Hl` and `dHl` at NumPy's
default `isclose`/`allclose` tolerance; the Jensen bound is analytic, so the
two implementations agree closely. `test_entmc_vbmc.py::test_entmc_vbmc_matlab`
checks `H` at `rtol=0.01` and `dH` at `rtol=atol=0.01`: PyVBMC draws its own
`Ns` samples (`rng=42`), so only the Monte Carlo estimate is being compared,
not the sample path.
