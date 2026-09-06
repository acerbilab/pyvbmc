"""Optional torch snapshot of a NumPy variational posterior.

Imported only by ``VariationalPosterior.to_torch``. None of these objects
are installed on the source posterior or its parameter transformer.
"""

import math

import numpy as np
import torch
from torch.distributions import (
    Categorical,
    Independent,
    MixtureSameFamily,
    Normal,
    Transform,
    TransformedDistribution,
    constraints,
)
from torch.nn.functional import softplus


def _student4_icdf(lower, upper):
    """Student-t(4) quantile, smooth at 1/2 and stable in either tail."""
    center = (lower >= 0.1) & (upper >= 0.1)
    result = torch.empty_like(lower)
    # With r = t / sqrt(t**2 + 4), 2*p-1 = (3*r-r**3)/2.
    # This root avoids sign(p-.5)*sqrt(q-1), singular at the center.
    r = 2 * torch.sin(torch.asin(lower[center] - upper[center]) / 3)
    result[center] = 2 * r / torch.sqrt((1 - r) * (1 + r))
    tail = ~center
    p = torch.minimum(lower[tail], upper[tail])
    a = 2 * torch.sqrt(p) * torch.sqrt(1 - p)
    q = torch.cos(torch.acos(a) / 3) / a
    magnitude = 2 * torch.sqrt(q - 1)
    result[tail] = torch.where(
        lower[tail] < upper[tail], -magnitude, magnitude
    )
    return result


class _StrictBounds(constraints.Constraint):
    """Finite vector values strictly inside each coordinate's bounds."""

    event_dim = 1

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        super().__init__()

    def check(self, value):
        return (
            torch.isfinite(value) & (value > self.lower) & (value < self.upper)
        ).all(dim=-1)


class _InverseParameterTransform(Transform):
    """Vector-event inverse of the core's bounded and whitening maps."""

    domain = constraints.real_vector
    bijective = True

    def __init__(self, pt, D, tensor):
        super().__init__(cache_size=0)
        types = np.asarray(pt.type)
        if types.shape != (D,) or not np.isin(types, (0, 3, 12, 13)).all():
            raise ValueError("Unsupported parameter-transform types.")
        self.types = tuple(int(t) for t in types)
        self.mu = tensor(pt.mu).reshape(D)
        self.delta = tensor(pt.delta).reshape(D)
        self.lower = tensor(pt.lb_orig).reshape(D)
        self.upper = tensor(pt.ub_orig).reshape(D)
        self.scale = tensor(
            np.ones(D) if pt.scale is None else pt.scale
        ).reshape(D)
        rotation = np.eye(D) if pt.R_mat is None else np.asarray(pt.R_mat)
        if (
            rotation.shape != (D, D)
            or not np.isfinite(rotation).all()
            or not np.allclose(
                rotation.T @ rotation, np.eye(D), rtol=1e-10, atol=1e-10
            )
        ):
            raise ValueError(
                "Parameter-transform rotation must be orthogonal."
            )
        self.rotation = tensor(rotation)
        if not (
            torch.isfinite(self.mu).all()
            and torch.isfinite(self.delta).all()
            and (self.delta > 0).all()
            and torch.isfinite(self.scale).all()
            and (self.scale > 0).all()
        ):
            raise ValueError(
                "Parameter-transform centers must be finite and scales "
                "(delta and scale) finite and positive in the export dtype."
            )
        for d, kind in enumerate(self.types):
            lb, ub = self.lower[d], self.upper[d]
            if kind == 0:
                if not (torch.isneginf(lb) and torch.isposinf(ub)):
                    raise ValueError(
                        "One-sided parameter bounds are unsupported."
                    )
            elif not (
                torch.isfinite(lb)
                and torch.isfinite(ub)
                and torch.isfinite(ub - lb)
                and torch.nextafter(lb, ub) < ub
            ):
                raise ValueError(
                    "Bounded parameters need finite ordered bounds with a "
                    "representable interior and width in the export dtype."
                )
        self.codomain = _StrictBounds(self.lower, self.upper)

    def _uncenter(self, u):
        return ((u * self.scale) @ self.rotation.T) * self.delta + self.mu

    def _call(self, u):
        z = self._uncenter(u)
        columns = []
        for d, kind in enumerate(self.types):
            v = z[..., d]
            if kind == 0:
                columns.append(v)
                continue
            if kind == 3:
                p = torch.sigmoid(v)
            elif kind == 12:
                p = 0.5 * torch.erfc(-v / math.sqrt(2))
            else:
                # Factor the t4 tail to avoid subtracting its CDF from 1/2.
                h = torch.hypot(v, v.new_tensor(2))
                r = v.abs() / h
                one_minus_r = (4 / h) / (h + v.abs())
                tail = one_minus_r.square() * (2 + r) / 4
                tail_cdf = torch.where(v < 0, tail, 1 - tail)
                signed_r = v / h
                center_cdf = 0.5 + signed_r * (3 - signed_r.square()) / 4
                p = torch.where(v.abs() < 2, center_cdf, tail_cdf)
            columns.append(self.lower[d] + (self.upper[d] - self.lower[d]) * p)
        return torch.stack(columns, dim=-1)

    def _inverse(self, x):
        columns = []
        for d, kind in enumerate(self.types):
            v = x[..., d]
            if kind != 0:
                width = self.upper[d] - self.lower[d]
                lower_distance = v - self.lower[d]
                upper_distance = self.upper[d] - v
                lower = lower_distance / width
                upper = upper_distance / width
                # A strict-interior distance can be subnormal and divide to
                # zero. Preserve a representable tail, just as the NumPy
                # transform's safe unit-interval conversion does. This is
                # only an underflow correction, not a clamp of the bijector.
                tiny = torch.nextafter(
                    torch.zeros_like(lower), torch.ones_like(lower)
                )
                lower = torch.where(
                    (lower == 0) & (lower_distance > 0), tiny, lower
                )
                upper = torch.where(
                    (upper == 0) & (upper_distance > 0), tiny, upper
                )
                if kind == 3:
                    # The width cancels; log distances avoid division loss.
                    v = torch.log(lower_distance) - torch.log(upper_distance)
                elif kind == 12:
                    # Compute the small tail directly; 2*p-1 loses tiny p.
                    q = torch.special.ndtri(
                        torch.where(lower < upper, lower, upper)
                    )
                    v = torch.where(lower < upper, q, -q)
                else:
                    v = _student4_icdf(lower, upper)
            columns.append((v - self.mu[d]) / self.delta[d])
        return (torch.stack(columns, dim=-1) @ self.rotation) / self.scale

    def log_abs_det_jacobian(self, u, x):
        z = self._uncenter(u)
        terms = []
        for d, kind in enumerate(self.types):
            v = z[..., d]
            term = torch.zeros_like(v) + self.delta[d].log()
            if kind != 0:
                term = term + (self.upper[d] - self.lower[d]).log()
                if kind == 3:
                    term = term - softplus(v) - softplus(-v)
                elif kind == 12:
                    term = term - 0.5 * (math.log(2 * math.pi) + v.square())
                else:
                    term = (
                        term
                        + math.log(3 / 8)
                        - 2.5 * torch.log1p((v / 2).square())
                    )
            terms.append(term)
        return torch.stack(terms, dim=-1).sum(-1) + self.scale.log().sum()

    def interior_sample(self, value):
        """Correct only sampled endpoints caused by floating-point saturation."""
        columns = []
        for d, kind in enumerate(self.types):
            column = value[..., d]
            if kind != 0:
                column = column.clamp(
                    min=torch.nextafter(self.lower[d], self.upper[d]),
                    max=torch.nextafter(self.upper[d], self.lower[d]),
                )
            columns.append(column)
        return torch.stack(columns, dim=-1)


class _OriginalDistribution(TransformedDistribution):
    def sample(self, sample_shape=torch.Size()):
        # The analytic transform is deliberately unclamped. Only draws that
        # round to a hard bound are moved one representable step inside it.
        with torch.no_grad():
            value = super().sample(sample_shape)
            return self.transforms[0].interior_sample(value)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(_OriginalDistribution, _instance)
        return super().expand(batch_shape, _instance=new)


def to_torch(vp, orig_flag=True, *, dtype=None, device=None):
    """Build an independent, explicitly typed torch distribution snapshot."""
    dtype = torch.float64 if dtype is None else dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "Torch export dtype must be torch.float32 or float64."
        )
    device = torch.device("cpu") if device is None else torch.device(device)

    def tensor(value):
        # torch.tensor copies even CPU float64 NumPy input; as_tensor does not.
        return torch.tensor(
            np.asarray(value).copy(), dtype=dtype, device=device
        )

    parameters = {}
    for name, shape in (
        ("w", (1, vp.K)),
        ("mu", (vp.D, vp.K)),
        ("sigma", (1, vp.K)),
        ("lambd", (vp.D, 1)),
    ):
        value = np.asarray(getattr(vp, name))
        if value.shape != shape:
            raise ValueError(f"VP {name} must have shape {shape}.")
        parameters[name] = tensor(value)
        if not torch.isfinite(parameters[name]).all():
            raise ValueError(f"VP {name} must be finite in the export dtype.")
    for name in ("sigma", "lambd"):
        if not (parameters[name] > 0).all():
            raise ValueError(
                f"VP {name} must be positive in the export dtype."
            )
    weights = parameters["w"].reshape(vp.K)
    if not (
        torch.isfinite(weights).all()
        and (weights >= 0).all()
        and torch.isfinite(weights.sum())
        and weights.sum() > 0
    ):
        raise ValueError(
            "Mixture weights must be finite, nonnegative and nonzero."
        )
    # Categorical(probs=...) clamps zero probabilities when computing logits.
    # Explicit log weights retain exactly absent mixture components.
    mixture = Categorical(logits=weights.log(), validate_args=True)
    means = parameters["mu"].T
    scales = parameters["sigma"].T * parameters["lambd"].T
    if not (torch.isfinite(scales).all() and (scales > 0).all()):
        raise ValueError(
            "Component scales must be finite and positive in the export dtype."
        )
    components = Independent(Normal(means, scales, validate_args=True), 1)
    base = MixtureSameFamily(mixture, components, validate_args=True)
    if not orig_flag:
        return base
    transform = _InverseParameterTransform(
        vp.parameter_transformer, vp.D, tensor
    )
    return _OriginalDistribution(base, [transform], validate_args=True)
