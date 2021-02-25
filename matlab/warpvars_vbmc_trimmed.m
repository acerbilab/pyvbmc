function varargout = warpvars_vbmc_trimmed(varargin)
%WARPVARS Linear and nonlinear transformation of variables.
%
%  TRINFO = WARPVARS(NVARS,LB,UB,PLB,PUB) returns the transformation structure 
%  TRINFO for a problem with NVARS dimensions and lower/upper bounds
%  respectively LB and UB. LB and UB are either scalars or row arrays that 
%  can contain real numbers and Inf's
%  The ordering LB <= UB needs to hold coordinate-wise.
%
%  Variables with lower or upper bounds are transformed via a log transform.
%  Variables with both lower and upper bounds are transformed via a logit
%  transform. 
%
%  Y = WARPVARS(X,'dir',TRINFO) performs direct transform of constrained 
%  variables X into unconstrained variables Y according to transformation 
%  encoded in structure TRINFO. X must be a N x NVARS array, where N is the 
%  number of input data and NVARS is the number of dimensions.
%
%  X = WARPVARS(Y,'inv',TRINFO) performs inverse transform of unconstrained 
%  variables Y into constrained variables X.
%
%  P = WARPVARS(Y,'prob',TRINFO) returns probability multiplier for the 
%  original pdf evaluated at f^{-1}(Y), that is | df^{-1}(y) / dy |.
%
%  LP = WARPVARS(Y,'logprob',TRINFO) returns log probability term for the 
%  original log pdf evaluated at f^{-1}(Y).

%  Author: Luigi Acerbi
%  e-mail: luigi.acerbi@gmail.com

if nargin < 3
    error('WARPVARS requires a minimum of three input arguments.');
end

%% Transform variables
if nargin == 3 && (isstruct(varargin{3}) || ischar(varargin{2}))
    
    Tol = sqrt(eps);    % Small number
    
    action = varargin{2};
    trinfo = varargin{3};

    if isempty(action)
        error('The transformation direction cannot be empty. Allowed values are direct (''dir'' or ''d'') and inverse (''inv'' or ''i'').');
    end

    if isempty(trinfo)
        % Empty TRINFO - consider as identity transformation

        x = varargin{1};
        
        switch lower(action(1))
            case {'d','i'}
                varargout{1} = x;
            case 'p'
                varargout{1} = ones(size(x,1),1);
            case 'l'
                varargout{1} = zeros(size(x,1),1);
            case {'m','f','g'}
                error('TRINFO is empty.');
            otherwise
                error(['Unkwnown transformation direction ''' action '''. Allowed values are direct (''dir'' or ''d'') and inverse (''inv'' or ''i'').']);
        end
    else
                
        scale = [];
        if isfield(trinfo,'scale') && ~isempty(trinfo.scale) && any(trinfo.scale ~= 1)
            scale = trinfo.scale;
        end

        if ~isfield(trinfo,'	'); trinfo.R_mat = []; end
        
        nvars = numel(trinfo.lb_orig);  % Number of variables
        
        switch lower(action(1))
        %% DIRECT TRANSFORM
            case 'd'    % Direct transform
                x = varargin{1};            
                y = x;
                a = trinfo.lb_orig;
                b = trinfo.ub_orig;
                mu = trinfo.mu;
                delta = trinfo.delta;
                
                % Unbounded scalars (possibly center and rescale)
                idx = trinfo.type == 0;
                if any(idx)
                    y(:,idx) = bsxfun(@rdivide,bsxfun(@minus,x(:,idx),mu(idx)),delta(idx));
                end

                % Lower and upper bounded scalars
                idx = trinfo.type == 3;
                if any(idx)
                    z = bsxfun(@rdivide, bsxfun(@minus, x(:,idx), a(idx)), ...
                        b(idx) - a(idx)); 
                    y(:,idx) = log(z./(1-z));
                    y(:,idx) = bsxfun(@rdivide,bsxfun(@minus,y(:,idx),mu(idx)),delta(idx));
                end
                                
                % Rotate output
                if ~isempty(trinfo.R_mat); y = y*trinfo.R_mat; end
                
                % Rescale output
                if ~isempty(scale); y = bsxfun(@rdivide,y,scale); end
                
                varargout{1} = y;
                
            %% INVERSE TRANSFORM
            case 'i'    % Inverse transform
                y = varargin{1};                
                % Rescale input
                if ~isempty(scale); y = bsxfun(@times,y,scale); end
                
                % Rotate input
                if ~isempty(trinfo.R_mat); y = y*trinfo.R_mat'; end        
                                
                x = y;
                a = trinfo.lb_orig;
                b = trinfo.ub_orig;
                mu = trinfo.mu;
                delta = trinfo.delta;                

                % Unbounded scalars (possibly unscale and uncenter)
                idx = trinfo.type == 0;
                if any(idx)
                    x(:,idx) = bsxfun(@plus,bsxfun(@times,y(:,idx),delta(idx)),mu(idx));
                end
                
                % Lower and upper bounded scalars
                idx = trinfo.type == 3;
                if any(idx)
                    x(:,idx) = bsxfun(@plus,bsxfun(@times,y(:,idx),delta(idx)),mu(idx));
                    x(:,idx) = bsxfun(@plus, a(:,idx), bsxfun(@times, ...
                        b(idx)-a(idx), 1./(1+exp(-x(:,idx)))));
                end
                                
                % Force to stay within bounds
                a(isfinite(a)) = a(isfinite(a)) + eps(a(isfinite(a)));
                b(isfinite(b)) = b(isfinite(b)) - eps(b(isfinite(b)));
                x = bsxfun(@min,bsxfun(@max,x,a),b);
                varargout{1} = x;
                
            %% PDF (OR LOG PDF) CORRECTION           
            case {'p','l','g'}  % pdf (or log pdf) correction
                y = varargin{1};
                % Rescale input
                if ~isempty(scale); y = bsxfun(@times,y,scale); end

                % Rotate input
                if ~isempty(trinfo.R_mat); y = y*trinfo.R_mat'; end        
                
                logpdf_flag = strcmpi(action(1),'l');
                if logpdf_flag
                    p = zeros(size(y,1),nvars);
                else
                    p = ones(size(y,1),nvars);
                end
                grad_flag = strcmpi(action(1),'g');
                
                a = trinfo.lb_orig;
                b = trinfo.ub_orig;
                mu = trinfo.mu;
                delta = trinfo.delta;                
                
                % Unbounded scalars
                idx = trinfo.type == 0;
                if any(idx)
                    p(:,idx) = repmat(log(delta(idx)),[size(p,1),1]);
                end

                % Lower and upper bounded scalars
                idx = trinfo.type == 3;
                if any(idx)
                    y(:,idx) = bsxfun(@plus,bsxfun(@times,y(:,idx),delta(idx)),mu(idx));
                    z = -log1p(exp(-y(:,idx)));
                    p(:,idx) = bsxfun(@plus, log(b(idx)-a(idx)), -y(:,idx) + 2*z);
                    p(:,idx) = bsxfun(@plus, p(:,idx), log(delta(idx)));
                end
                                
                % Scale transform
                if ~isempty(scale) && ~grad_flag
                    p = bsxfun(@plus,p,log(scale));
                end
                
                if ~grad_flag; p = sum(p,2); end
                if ~logpdf_flag; p = exp(p); end
                
                varargout{1} = p;
                                
            otherwise
                error(['Unkwnown transformation direction ''' action '''. Allowed values are direct (''dir'' or ''d'') and inverse (''inv'' or ''i'').']);
        end
    end
    
else
%% Create transform

    nvars = varargin{1};
    lb = varargin{2}(:)';
    ub = varargin{3}(:)';
    if nargin > 3
        plb = varargin{4}(:)';
        pub = varargin{5}(:)';
    else
        plb = []; pub = [];
    end
            
    % Empty LB and UB are Infs
    if isempty(lb); lb = -Inf; end
    if isempty(ub); ub = Inf; end

    % Empty plausible bounds equal hard bounds
    if isempty(plb); plb = lb; end
    if isempty(pub); pub = ub; end
    
    % Convert scalar inputs to row vectors
    if isscalar(lb); lb = lb*ones(1,nvars); end
    if isscalar(ub); ub = ub*ones(1,nvars); end
    if isscalar(plb); plb = plb*ones(1,nvars); end
    if isscalar(pub); pub = pub*ones(1,nvars); end
    
    % Check that the order of bounds is respected
    assert(all(lb <= plb & plb < pub & pub <= ub), ...
        'Variable bounds should be LB <= PLB < PUB <= UB for all variables.');
    
    % Transform to log coordinates
    trinfo.lb_orig = lb;
    trinfo.ub_orig = ub;
    
    trinfo.type = zeros(1,nvars);    
    for i = 1:nvars
        if isfinite(lb(i)) && isinf(ub(i)); trinfo.type(i) = 1; end
        if isinf(lb(i)) && isfinite(ub(i)); trinfo.type(i) = 2; end
        if isfinite(lb(i)) && isfinite(ub(i)) && lb(i) < ub(i); trinfo.type(i) = 3; end
    end
    
    % Centering (at the end of the transform)
    trinfo.mu = zeros(1,nvars);
    trinfo.delta = ones(1,nvars);
    
    % Get transformed PLB and PUB
    plb = warpvars_vbmc(plb,'d',trinfo);
    pub = warpvars_vbmc(pub,'d',trinfo);
    
    % Center in transformed space
    for i = 1:nvars
        if isfinite(plb(i)) && isfinite(pub(i))
            trinfo.mu(i) = 0.5*(plb(i)+pub(i));
            trinfo.delta(i) = (pub(i)-plb(i));
        end
    end
        
    varargout{1} = trinfo;
    
end

end