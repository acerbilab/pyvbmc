"""O3 F1: what MATLAB's 'g' action of shared/warpvars_vbmc.m returns,
against the gradient of the log Jacobian, on the Python transformer.

My transcription of the 'g' path (warpvars_vbmc.m:463-770, read in this
session): y is rescaled and un-rotated (:466-470); logpdf_flag is false for
'g' (:472), so p starts at ones; each coordinate type WRITES its log term
(type 0: log(delta), :486-489; type 3: log(b-a) - y + 2 z + log(delta),
:496-503; type 12: log(b-a) - log(2 pi)/2 - y^2/2 + log(delta), :741-747;
type 13: log(b-a) + log(3/8) - 5/2 log1p(y^2/4) + log(delta), :750-756);
log(scale) is skipped (:763), there is no sum over coordinates (:767),
and p = exp(p) (:768). So 'g' returns the N x D matrix of exp(per-
coordinate log-Jacobian terms), at the pre-rotation coordinates.

Checked here, on a D=3 transformer with a rotation and a rescaling:
(1) that matrix equals the per-coordinate derivative dx_i/dv_i of the
    coordinate-wise inverse map, v the pre-rotation coordinate (finite
    differences of the Python inverse with R and scale removed);
(2) it is not the gradient of log|J| with respect to u (finite
    differences of log_abs_det_jacobian);
(3) the gradient of log|J| equals the per-coordinate closed forms of the
    report, (1-2 sigma(y)) delta, -y delta, -(5/4) y delta / (1+y^2/4),
    carried through the rotation and the scale, in configurations the FD
    tests leave out (a point near a bound, |y| ~ 6-8).
"""
import copy

import gpyreg as gpr
import numpy as np
from scipy.special import expit

import pyvbmc
from pyvbmc.parameter_transformer import ParameterTransformer

print("pyvbmc:", pyvbmc.__file__)
print("gpyreg:", gpr.__file__)

D = 3
lb = np.array([[-2.0, 0.0, -np.inf]])
ub = np.array([[4.0, 5.0, np.inf]])
plb = np.array([[-1.0, 1.0, -2.0]])
pub = np.array([[3.0, 4.0, 3.0]])
th = 0.7
R = np.array(
    [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]]
)
R = R @ np.array(
    [[1, 0, 0], [0, np.cos(0.4), -np.sin(0.4)], [0, np.sin(0.4), np.cos(0.4)]]
)
scale = np.array([0.7, 1.9, 1.3])


def matlab_g(pt, u):
    """Transcription of warpvars_vbmc(u,'g',trinfo)."""
    y = u * pt.scale if pt.scale is not None else u.copy()
    if pt.R_mat is not None:
        y = y @ pt.R_mat.T
    p = np.ones_like(y)
    a, b = pt.lb_orig[0], pt.ub_orig[0]
    for i in range(y.shape[1]):
        t = pt.type[i]
        if t == 0:
            p[:, i] = np.log(pt.delta[i])
            continue
        yy = y[:, i] * pt.delta[i] + pt.mu[i]
        if t == 3:
            z = -np.log1p(np.exp(-yy))
            p[:, i] = np.log(b[i] - a[i]) - yy + 2 * z + np.log(pt.delta[i])
        elif t == 12:
            p[:, i] = (
                np.log(b[i] - a[i])
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * yy**2
                + np.log(pt.delta[i])
            )
        elif t == 13:
            p[:, i] = (
                np.log(b[i] - a[i])
                + np.log(3 / 8)
                - 2.5 * np.log1p(yy**2 / 4)
                + np.log(pt.delta[i])
            )
    return np.exp(p)


def fd_grad(f, u, h=1e-6):
    g = np.zeros_like(u)
    for j in range(u.shape[1]):
        e = np.zeros(u.shape[1])
        e[j] = h
        g[:, j] = (f(u + e) - f(u - e)) / (2 * h)
    return g


def closed_form_grad(pt, u):
    v = u * pt.scale
    v = v @ pt.R_mat.T  # pre-rotation, centred coordinates
    gv = np.zeros_like(v)
    for i in range(v.shape[1]):
        t = pt.type[i]
        yy = v[:, i] * pt.delta[i] + pt.mu[i]
        if t == 3:
            gv[:, i] = (1 - 2 * expit(yy)) * pt.delta[i]
        elif t == 12:
            gv[:, i] = -yy * pt.delta[i]
        elif t == 13:
            gv[:, i] = -1.25 * yy * pt.delta[i] / (1 + yy**2 / 4)
    # u -> v is v = (u * s) R^T, so d/du = s * (d/dv R)
    return (gv @ pt.R_mat) * pt.scale


rng = np.random.default_rng(0)
for kind in ("logit", "probit", "student4"):
    pt = ParameterTransformer(D, lb, ub, plb, pub, transform_type=kind)
    ptw = copy.deepcopy(pt)
    ptw.R_mat, ptw.scale = R, scale
    U = rng.standard_normal((6, D))
    # a point whose first coordinate sits close to the upper bound
    x_near = np.array([[4.0 - 1e-6, 2.5, 0.3]])
    U = np.vstack([U, ptw(x_near)])
    G = matlab_g(ptw, U)
    # (1) per-coordinate derivative of the coordinate-wise inverse
    pt0 = copy.deepcopy(
        pt
    )  # no R, no scale: inverse acts coordinate-wise on v
    V = (U * scale) @ R.T
    dxdv = np.zeros_like(V)
    for i in range(D):
        h = 1e-6
        e = np.zeros(D)
        e[i] = h
        dxdv[:, i] = (pt0.inverse(V + e)[:, i] - pt0.inverse(V - e)[:, i]) / (
            2 * h
        )
    # interior rows only: at the near-bound row x carries few digits of b - x (F4)
    rel1 = np.max(np.abs(G[:-1] / dxdv[:-1] - 1))
    # (2) gradient of log|J| wrt u
    g_fd = fd_grad(lambda uu: ptw.log_abs_det_jacobian(uu), U)
    g_cf = closed_form_grad(ptw, U)
    print(
        f"{kind:8s}: 'g' vs FD dx_i/dv_i (interior rows): max rel diff {rel1:.1e} | "
        f"'g' vs FD grad log|J|: max abs diff {np.max(np.abs(G - g_fd)):.2f} | "
        f"closed-form grad log|J| vs FD: max abs diff {np.max(np.abs(g_cf - g_fd)):.1e}"
    )
    print(
        f"          e.g. row 0: 'g' = {np.round(G[0], 4)}, grad log|J| = {np.round(g_fd[0], 4)}"
    )
