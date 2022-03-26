"""
PyVBMC-PyMC interface
"""

import arviz as az
import numpy as np
import pymc as pm
from pymc.initial_point import make_initial_point_fn
from pymc.backends.arviz import dict_to_dataset, to_inference_data
from pymc.backends.base import MultiTrace
from pymc.backends.ndarray import NDArray
from pymc.blocking import DictToArrayBijection
from pymc.aesaraf import (
    compile_pymc,
    inputvars,
    join_nonshared_inputs,
    make_shared_replacements,
)


from .vbmc import VBMC


def pymc_fit(
    model=None,
    autobound=True,
    draws=1000,
    init_method="moment",
    random_seed=None,
    lower_bounds=None,
    upper_bounds=None,
    plausible_lower_bounds=None,
    plausible_upper_bounds=None,
    user_options=None,
):
    """
    Use VBMC with a model built with PyMC.

    Parameters
    ----------
    model : PyMC Model (optional if in ``with`` context)
        Model to sample from. The model needs to have free random variables.
    autobound : bool
        Whether to compute hard and plausible bounds automatically. Defaults to True.
    draws : int
        The number of samples to draw from the posterior approximation and store in the returned
        InferenceData. Defaults to 1000.
    init_method : str
        Method to define the initial points. If ``"moments"`` use the moments from the PyMC
        distributions if ``map`` use PyMC to compute the maximum a posteriori.
    random_seed : int
        random seed used for the automatic computations of bounds from prior samples.
    lower_bounds, upper_bounds : np.ndarray, optional
        See ``vbmc`` for details. Ignored if autobound True
    plausible_lower_bounds, plausible_upper_bounds : np.ndarray, optional
        See ``vbmc`` for details. Ignored if autobound True
    user_options : dict, optional
        Additional options can be passed as a dict. Please refer to the
        VBMC options page for the default options. If no `user_options` are
        passed, the default options are used.

    Returns
    -------
    vp : VariationalPosterior
        The ``VariationalPosterior`` computed by VBMC.
    elbo : float
        An estimate of the ELBO for the returned `vp`.
    elbo_sd : float
        The standard deviation of the estimate of the ELBO. Note that this
        standard deviation is *not* representative of the error between the
        `elbo` and the true log marginal likelihood.
    idata : InferenceData
    """

    model = pm.modelcontext(model)

    if user_options is None:
        user_options = {}

    if random_seed is None:
        rng_seeder = np.random.RandomState()
    else:
        rng_seeder = np.random.RandomState(random_seed)
    seed = rng_seeder.randint(2**30)

    variables = inputvars(model.value_vars)
    variable_names = []
    for v in variables:
        name = v.name
        if pm.util.is_transformed_name(v.name):
            name = pm.util.get_untransformed_name(name)

        variable_names.append(name)

    target = initialize_target(model, variables, joint=True)

    initial_points, var_info = compute_initial_points(
        model, variables, variable_names, init_method, seed
    )

    if autobound:
        plausible_lower_bounds, plausible_upper_bounds, lower_bounds, upper_bounds = compute_bounds(
            model, variable_names
        )

    vbmc = VBMC(
        target,
        x0=initial_points,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        plausible_lower_bounds=plausible_lower_bounds,
        plausible_upper_bounds=plausible_upper_bounds,
        user_options=user_options,
    )
    vp, elbo, elbo_sd, _, _ = vbmc.optimize()
    idata = get_idata(model, vp, variables, var_info, draws)

    return vp, elbo, elbo_sd, idata


def initialize_target(model, variables, joint=True):
    """
    Compute an aesara compiled function with the logp from the model

    Parameters
    ----------
    model : PyMC Model
        Model to sample from. The model needs to have free random variables.
    variables : List
        Unobserved random variables excluding deterministics.
    joint : bool
        Whether to return the joint model logp, i.e. prior + likelihood or return them separated.
    """

    initial_point = model.compute_initial_point()
    shared = make_shared_replacements(initial_point, variables, model)

    if joint:
        target = _logp_forw(initial_point, [model.varlogpt + model.datalogpt], variables, shared)
    else:
        prior_target = _logp_forw(initial_point, [model.varlogpt], variables, shared)
        likelihood_target = _logp_forw(initial_point, [model.datalogpt], variables, shared)
        target = (prior_target, likelihood_target)

    return target


def compute_initial_points(model, variables, variable_names, method, seed):
    """
    Get a sensible initial point from a PyMC model.

    Parameters
    ----------
    model : PyMC Model
        Model to sample from. The model needs to have free random variables.
    variables_names : List
        List of the names of the unobserved random variables excluding deterministics.
    method : str
        Method to define the initial points. If ``"moments"`` use the moments from the PyMC
        distributions if ``map`` use PyMC to compute the maximum a posteriori.
    """
    if method == "moment":
        fn = make_initial_point_fn(model=model, return_transformed=False, default_strategy=method)
        init = fn(seed)
        initial_point = pm.Point({v: init[v] for v in variable_names}, model=model)
    elif method == "map":
        init = pm.find_MAP(model=model, include_transformed=False, progressbar=False)
        initial_point = pm.Point({v: init[v] for v in variable_names}, model=model)

    init = model.compute_initial_point(0)
    var_info = {v.name: (init[v.name].shape, init[v.name].size) for v in variables}

    return DictToArrayBijection.map(initial_point).data[None, :], var_info


def compute_bounds(model, variable_names, draws=5000):
    """
    Use samples from the prior distribution to compute bounds.

    The plausible bounds are computed using the HDI of the prior samples and the "hard" bounds using
    the min and max values.

    Parameters
    ----------
    model : PyMC Model
        Model to sample from. The model needs to have free random variables.
    variables_names : List
        List of the names of the unobserved random variables excluding deterministics.
    draws : int
        Number of draws from the prior distribution.
    """
    result = pm.sample_prior_predictive(
        draws,
        model=model,
        return_inferencedata=True,
    )
    hdi = az.hdi(result, group="prior", var_names=variable_names, hdi_prob=0.75)
    stacked = hdi.to_stacked_array("__stacked__", sample_dims=["hdi"])
    plb = stacked.sel(hdi="lower").values
    pub = stacked.sel(hdi="higher").values

    min_ = result["prior"].min(["chain", "draw"])
    max_ = result["prior"].max(["chain", "draw"])
    lb = np.hstack([min_[v] for v in variable_names])
    ub = np.hstack([max_[v] for v in variable_names])

    return plb[None, :], pub[None, :], lb[None, :], ub[None, :]


def get_idata(model, vp, variables, var_info, draws):
    """Save results into an InferenceData.


    We save to a trace as intermediate object so the trace automatically takes care of the
    deterministic variables.

    Parameters
    ----------
    model : PyMC Model
        Model to sample from. The model needs to have free random variables.
    vp : VariationalPosterior
        The ``VariationalPosterior`` computed by VBMC.
    variables : List
        Unobserved random variables excluding deterministics.
    var_info : dict
        dictionary with information about model variables shape and size.
    draws : int
        The number of samples to draw from the posterior approximation
    """
    samples = vp.sample(draws)[0]
    lenght_pos = len(samples)
    varnames = [v.name for v in variables]
    with model:
        strace = NDArray(name=model.name)
        strace.setup(lenght_pos, chain=0)
    for i in range(lenght_pos):
        value = []
        size = 0
        for varname in varnames:
            shape, new_size = var_info[varname]
            var_samples = samples[i][size : size + new_size]
            value.append(var_samples.reshape(shape))
            size += new_size
        strace.record(point=dict(zip(varnames, value)))

    sample_stats = dict_to_dataset(
        vp.stats,
        attrs={
            "inference_library": "PyVBMC",
            "inference_library_version": "0.0.1",
            "modeling_interface": "PyMC",
            "modeling_interface_version": pm.__version__,
        },
    )

    trace = MultiTrace([strace])
    idata = to_inference_data(trace, model=model)
    idata = az.InferenceData(**idata, sample_stats=sample_stats)

    return idata


def _logp_forw(point, out_vars, in_vars, shared):
    """Compile Aesara function of the model and the input and output variables.

    Parameters
    ----------
    out_vars : List
        containing :class:`pymc.Distribution` for the output variables
    in_vars : List
        containing :class:`pymc.Distribution` for the input variables
    shared : List
        containing :class:`aesara.tensor.Tensor` for depended shared data
    """
    out_list, inarray0 = join_nonshared_inputs(point, out_vars, in_vars, shared)
    f = compile_pymc([inarray0], out_list[0])
    f.trust_input = True
    return f
