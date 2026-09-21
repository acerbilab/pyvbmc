# VP subpackage developing notes

These notes are used for keeping track of ToDos and porting information.

## Porting status
- The function `vbmc_power`: [vbmc_power.m](https://github.com/acerbilab/vbmc/blob/master/vbmc_power.m) has not been ported, but it is very low priority and we might not need it at all (it was part of an experimental feature).
- The function `initFromVP` in [vbmc.m](https://github.com/acerbilab/vbmc/blob/master/vbmc.m) has not been ported yet.
- `mode()` does not follow `vbmc_mode.m`: it starts its optimizations at the best of many draws of the posterior, where MATLAB starts one at each component mean. The search was rewritten in 2022 after a bug met in the example 2 notebook. The mode is not invariant to the space it is computed in, so `orig_flag` changes the answer.
- `mode()` stores the mode of the original space in the attribute `_mode`, which a call without `n_opts` returns. `set_parameters()` and `get_parameters()` clear it, as `rescale_params.m` removes `vp.mode`; an assignment to `mu`, `sigma`, `lambd` or `w` from outside the class does not.
- The `kl_div` and `mtv` methods have a bit of a clunky interface, in which the second posterior could be passed as a `VariationalPosterior` or as a set of samples, using two *separate* inputs. Not sure why we are doing this in the end. I think we should use a simple form of overloading (i.e., a single input, and the program reacts differently based on the type of input, whether it's a `VariationalPosterior` or a `numpy.ndarray`).
- `get_bounds()`: This method should not be in the `VariationalPosterior` class, or at least not in this form. It does a bunch of stuff which is clearly linked to things done during the variational optimization part. This method is likely a private method of the variational optimization module.


## Matlab references
- get_parameters(): [get_vptheta.m](https://github.com/acerbilab/vbmc/blob/master/misc/get_vptheta.m)
- kl_div(): [vbmc_kldiv.m](https://github.com/acerbilab/vbmc/blob/master/vbmc_kldiv.m)
- mode(): [vbmc_mode.m](https://github.com/acerbilab/vbmc/blob/master/vbmc_mode.m)
- moments(): [vbmc_moments.m](https://github.com/acerbilab/vbmc/blob/master/vbmc_moments.m)
- mtv(): [vbmc_mtv.m](https://github.com/acerbilab/vbmc/blob/master/vbmc_mtv.m)
- pdf(): [vbmc_pdf.m](https://github.com/acerbilab/vbmc/blob/master/vbmc_pdf.m)
- sample(): [vbmc_rnd.m](https://github.com/acerbilab/vbmc/blob/master/vbmc_rnd.m)
   - gp_sample is missing
- set_parameters(): [rescale_params.m](https://github.com/acerbilab/vbmc/blob/master/misc/rescale_params.m)
- get_bounds(): [vpbounds.m](https://github.com/acerbilab/vbmc/blob/master/misc/vpbounds.m)
- plot() [vbmc_plot.m](https://github.com/acerbilab/vbmc/blob/master/vbmc_plot.m)
