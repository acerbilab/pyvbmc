# VP subpackage developing notes

These notes are used for keeping track of ToDos and porting information.

## Porting status
- The function vbmc_power: [vbmc_power.m](https://github.com/lacerbi/vbmc/blob/master/vbmc_power.m) (according to Luigi this function is not as important for now)
- The function in [vbmc.m](https://github.com/lacerbi/vbmc/blob/master/vbmc.m) has not been ported yet.
- There is a bug with the mode() function: Luigi suspects it is related to the computation of the mode being invariant to different spaces. We have not investigated the bug in depth, but it occured in the example2 notebook.
- In the mode() function we set the attribute \_mode. We should talk about this and double-check that it is being cleared correctly when the mode changes.

## Matlab references
- get_parameters(): [get_vptheta.m](https://github.com/lacerbi/vbmc/blob/master/misc/get_vptheta.m) 
- kldiv(): [vbmc_kldiv.m](https://github.com/lacerbi/vbmc/blob/master/vbmc_kldiv.m) 
- mode(): [vbmc_mode.m](https://github.com/lacerbi/vbmc/blob/master/vbmc_mode.m) 
- moments(): [vbmc_moments.m](https://github.com/lacerbi/vbmc/blob/master/vbmc_moments.m) 
- mtv(): [vbmc_mtv.m](https://github.com/lacerbi/vbmc/blob/master/vbmc_mtv.m)  
- pdf(): [vbmc_pdf.m](https://github.com/lacerbi/vbmc/blob/master/vbmc_pdf.m)  
- sample(): [vbmc_rnd.m](https://github.com/lacerbi/vbmc/blob/master/vbmc_rnd.m)
   - gp_sample is missing
- set_parameters(): [rescale_params.m](https://github.com/lacerbi/vbmc/blob/master/misc/rescale_params.m) 
- get_bounds(): [vpbounds.m](https://github.com/lacerbi/vbmc/blob/master/misc/vpbounds.m) 
- plot() [vbmc_plot.m](https://github.com/lacerbi/vbmc/blob/master/vbmc_plot.m)
