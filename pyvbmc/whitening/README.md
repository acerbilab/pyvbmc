# Whitening subpackage developing notes

These notes are used for keeping track of ToDos and porting information.

## Porting status
- Rotating and rescaling has been implemented, but nonlinear warping has not.
- `warp_gp_and_vp` warps the length scales of the GP whatever its mean function, and re-expresses the hyperparameters of a constant and a negative quadratic mean; a zero mean has none. `warp_gpandvp_vbmc.m` numbers its mean functions and its `case 0`, commented "Warp constant mean", catches the zero mean and reads a hyperparameter past its end, while the constant mean falls to its `otherwise` and errors. For a zero mean, the constant shift of the stored log joint is left to the refit that follows the warp (port review, wave 7, row W7-1).

## Matlab references
- Corresponding MATLAB functions:
    - [unscent_warp.m](https://github.com/acerbilab/vbmc/blob/master/utils/unscent_warp.m)
    - [warp_input_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/warp_input_vbmc.m)
    - [warp_gpandvp_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/misc/warp_gpandvp_vbmc.m)
