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

% test_inverse_transform_type3_within
NVARS = 3;
LB = -10*ones(1,NVARS);
UB = 10*ones(1,NVARS);
TRINFO = warpvars_vbmc_trimmed(NVARS, LB, UB);
X = 3*ones(10,NVARS);
% WARPVARS(X,'dir',TRINFO) performs direct transform of constrained 
Y = warpvars_vbmc_trimmed(X, 'inv', TRINFO)

% test_inverse_transform_type3_within_negative
NVARS = 3;
LB = -10*ones(1,NVARS);
UB = 10*ones(1,NVARS);
TRINFO = warpvars_vbmc_trimmed(NVARS, LB, UB);
X = -4*ones(10,NVARS);
% WARPVARS(X,'dir',TRINFO) performs direct transform of constrained 
Y = warpvars_vbmc_trimmed(X, 'inv', TRINFO)

% check divide by zero error
a = trinfo.lb_orig;
b = trinfo.ub_orig;
mu = trinfo.mu;
delta = trinfo.delta;
x = 10*ones(1,NVARS)
y = x
% Lower and upper bounded scalars
idx = trinfo.type == 3;
if any(idx)
    z = bsxfun(@rdivide, bsxfun(@minus, x(:,idx), a(idx)), b(idx) - a(idx)); 
    y(:,idx) = log(z./(1-z));
end   
y