# Advanced options for unsupported/untested features (do *not* modify)
default_options_fixed = {
    # Nonlinear input warping
    "defopts.WarpNonlinear": False,
    # Uncertainty weight during ELCBO optimization
    "defopts.ELCBOWeight": 0,
    # Check variational posteriors back to these previous iterations'
    "defopts.VarParamsBack": 0,
    # Use alternative Monte Carlo computation for the entropy
    "defopts.AltMCEntropy": False,
    # Variational active sampling
    "defopts.VarActiveSample": False,
    # Test a new experimental feature
    "defopts.FeatureTest": False,
    # Bayesian-optimization-like warmup stage'
    "defopts.BOWarmup": False,
    # GP default output warping function
    "defopts.gpOutwarpFun": [],
}