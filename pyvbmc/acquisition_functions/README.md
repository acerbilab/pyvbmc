# acquisition_functions subpackage developing notes

These notes are used for keeping track of ToDos and porting information.

## Porting status
- discuss which other functions from [vbmc/acq](https://github.com/acerbilab/vbmc/blob/master/acq) to port
- discuss the naming of the AcqFcns, because AcqFcn and AcqFcnVanilla seem a bit arbitrary

## Matlab references
- AbstractAcqFcn: [acqwrapper_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/acq/acqwrapper_vbmc.m)
- AcqFcn: [acqf_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/acq/acqf_vbmc.m)
- AcqFcnLog: [acqflog_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/acq/acqflog_vbmc.m)
- AcqFcnNoisy: [acqfsn2_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/acq/acqfsn2_vbmc.m)
- AcqFcnVanila: [acqus_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/acq/acqus_vbmc.m)
- AcqFcnVIQR: [acqviqr_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/acq/acqviqr_vbmc.m)
- AcqFcnIMIQR: [acqimiqr_vbmc.m](https://github.com/acerbilab/vbmc/blob/master/acq/acqimiqr_vbmc.m)
