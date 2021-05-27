# Advanced options for unsupported/untested features (do *not* modify)
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