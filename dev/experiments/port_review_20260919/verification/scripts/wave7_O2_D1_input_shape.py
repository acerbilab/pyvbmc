"""The D = 1 input shape of vp.pdf / vp.log_pdf.

A flat array of N > 1 points for a one-dimensional posterior: what each
call returns or raises, against a column of the same points. Also a
transcription of vbmc_pdf.m:41-66 on a 1-by-N row for D = 1, to see what
MATLAB does with the same mistake.
"""
import os
import sys
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wave7_O2_common import banner, build_vp

from pyvbmc.parameter_transformer import ParameterTransformer

banner()
warnings.simplefilter("ignore")
xs = np.array([0.1, 0.5, 0.9])
vp_u = build_vp([[0.0, 1.0]], [1.0, 0.5], [1.0], [0.5, 0.5])
pt = ParameterTransformer(
    1,
    np.array([[0.0]]),
    np.array([[1.0]]),
    np.array([[0.1]]),
    np.array([[0.9]]),
)
vp_b = build_vp([[0.0, 1.0]], [1.0, 0.5], [1.0], [0.5, 0.5], transformer=pt)

calls = [
    ("pdf(x) unbounded", lambda x: vp_u.pdf(x)),
    ("pdf(x) bounded", lambda x: vp_b.pdf(x)),
    ("pdf(x, orig_flag=False)", lambda x: vp_u.pdf(x, orig_flag=False)),
    ("log_pdf(x)", lambda x: vp_u.log_pdf(x)),
    (
        "pdf(x, orig_flag=False, grad_flag=True)",
        lambda x: vp_u.pdf(x, orig_flag=False, grad_flag=True),
    ),
    ("pdf(x=..., keyword)", lambda x: vp_u.pdf(x=x)),
    ("pdf(x, df=3)", lambda x: vp_u.pdf(x, df=3)),
    (
        "pdf(x, orig_flag=False, df=3)",
        lambda x: vp_u.pdf(x, orig_flag=False, df=3),
    ),
    (
        "pdf(x, orig_flag=False, df=-3)",
        lambda x: vp_u.pdf(x, orig_flag=False, df=-3),
    ),
]
for name, fn in calls:
    col = fn(xs[:, None])
    colv = col[0] if isinstance(col, tuple) else col
    try:
        flat = fn(xs)
        flatv = flat[0] if isinstance(flat, tuple) else flat
        print(
            f"{name:42s} flat -> {np.ravel(flatv)} (column {np.ravel(colv)})",
            flush=True,
        )
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)[-1]
        print(
            f"{name:42s} flat raises {type(e).__name__}: {e} "
            f"[{os.path.basename(tb.filename)}:{tb.lineno}] "
            f"(column {np.ravel(colv)})",
            flush=True,
        )

# MATLAB transcription, a 1-by-3 row X with D = 1 (vbmc_pdf.m:41-66)
X = xs[None, :]
N, D = X.shape
mu_t = np.array([[0.0], [1.0]])
sigma = np.array([1.0, 0.5])
lam = np.array([1.0])
w = np.array([0.5, 0.5])
nf = 1 / (2 * np.pi) ** (D / 2) / np.prod(lam)
y = np.zeros((N, 1))
for k in range(2):
    d2 = np.sum(((X - mu_t[k]) / (sigma[k] * lam)) ** 2, axis=1)[:, None]
    y = y + nf * w[k] / sigma[k] ** D * np.exp(-0.5 * d2)
print(
    "MATLAB transcription, 1-by-3 row with D=1: one value",
    y.ravel(),
    "; product of the three mixture densities",
    np.prod(vp_u.pdf(xs[:, None], orig_flag=False)),
    flush=True,
)
