% test_direct_transform_type3_within
NVARS = 3;
LB = -10*ones(1,NVARS);
UB = 10*ones(1,NVARS);
TRINFO = warpvars_vbmc_trimmed(NVARS, LB, UB);
X = 3*ones(10,NVARS);
% WARPVARS(X,'dir',TRINFO) performs direct transform of constrained 
Y = warpvars_vbmc_trimmed(X, 'dir', TRINFO)

% test_direct_transform_type3_within_negative
NVARS = 3;
LB = -10*ones(1,NVARS);
UB = 10*ones(1,NVARS);
TRINFO = warpvars_vbmc_trimmed(NVARS, LB, UB);
X = -4*ones(10,NVARS);
% WARPVARS(X,'dir',TRINFO) performs direct transform of constrained 
Y = warpvars_vbmc_trimmed(X, 'dir', TRINFO)

% test_direct_transform_type1_within
NVARS = 3;
LB = -10*ones(1,NVARS);
UB = 10*ones(1,NVARS);
TRINFO = warpvars_vbmc_trimmed(NVARS, [], []);
TRINFO.type
X = 3*ones(10,NVARS);
% WARPVARS(X,'dir',TRINFO) performs direct transform of constrained 
Y = warpvars_vbmc_trimmed(X, 'dir', TRINFO)

% test_direct_transform_type1_within_negative
NVARS = 3;
LB = -10*ones(1,NVARS);
UB = 10*ones(1,NVARS);
TRINFO = warpvars_vbmc_trimmed(NVARS, [], []);
X = -4*ones(10,NVARS);
% WARPVARS(X,'dir',TRINFO) performs direct transform of constrained 
Y = warpvars_vbmc_trimmed(X, 'dir', TRINFO)