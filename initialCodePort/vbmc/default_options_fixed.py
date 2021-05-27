def get_default_options_fixed():
    """
    Retrieve unsupported/untested features options of VBMC which you
    should *never* modify.

    Returns
    -------
    default_options_fixed : dict
        These are the unsupported/untested features options of VBMC which you
        should *never* modify.
    """
    default_options_fixed = {
        # Nonlinear input warping
        "WarpNonlinear": False,
        # Uncertainty weight during ELCBO optimization
        "ELCBOWeight": 0,
        # Check variational posteriors back to these previous iterations'
        "VarParamsBack": 0,
        # Use alternative Monte Carlo computation for the entropy
        "AltMCEntropy": False,
        # Variational active sampling
        "VarActiveSample": False,
        # Test a new experimental feature
        "FeatureTest": False,
        # Bayesian-optimization-like warmup stage'
        "BOWarmup": False,
        # GP default output warping function
        "gpOutwarpFun": [],
    }
    return default_options_fixed