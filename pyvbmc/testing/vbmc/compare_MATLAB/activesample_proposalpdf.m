D = 3;
vp.mu = [-1, -2, -3; 3, 2, 1]';
vp.w = [0.7, 0.3];
vp.K = 2;
vp.sigma = [1.0, 1.0];
vp.lambda = [1.0, 1.0, 1.0]';
X = reshape(-7:7, 5, 3);
for ii = 1:5
    y(ii) = mvnormlpdf(X(ii, :), zeros(D,1), eye(D));
end
y = y(:);
hyp = [-2.0, -3.0, -4.0, 1.0, 0.0, -(D / 2) * log(2 * pi), 0.0, 0.25, 0.5, -0.5, 0.0, 0.5]';
hyp = [hyp, 2 * hyp];
gp = gplite_post(hyp, X, y, 1, 4);

Xa = 2 * reshape(-4:4, 3, 3) / pi;
w_vp = 0.5;
rect_delta = 2 * std(gp.X);
[ln_weights_viqr, f_s2_viqr] = activcesample_proposalpdf(Xa, gp, vp, w_vp, rect_delta, @acqviqr_vbmc, vp, true)
[ln_weights_imiqr, f_s2_imiqr] = activcesample_proposalpdf(Xa, gp, vp, w_vp, rect_delta, @acqimiqr_vbmc, vp, true)
dirpath = fileparts(matlab.desktop.editor.getActiveFilename);
save(dirpath + "/activcesample_proposalpdf.mat", "ln_weights_viqr", "f_s2_viqr", "ln_weights_imiqr", "f_s2_imiqr")

%%
function ll = mvnormlpdf(theta, mu, Sigma)
    theta = theta(:);
    mu = mu(:);
    D = length(mu);
    L = chol(Sigma);
    halflogdet = sum(log(diag(L)));
    z = L \ (theta - mu);
    ll = -(D/2) * log(2 * pi) - halflogdet - (1/2) * (z' * z);
end
%--------------------------------------------------------------------------
function [lnw,fs2] = activcesample_proposalpdf(Xa,gp,vp_is,w_vp,rect_delta,acqfun,vp,isamplevp_flag)
%ACTIVCESAMPLE_PROPOSALPDF Compute importance weights for proposal pdf

[N,D] = size(gp.X);
Na = size(Xa,1);

[~,~,fmu,fs2] = gplite_pred(gp,Xa,[],[],1,0);
Ntot = 1 + N; % Total number of mixture elements

if w_vp < 1; templpdf = zeros(Na,Ntot); end

% Mixture of variational posteriors
if w_vp > 0
    log_flag = true;
    templpdf(:,1) = vbmc_pdf(vp_is,Xa,0,log_flag) + log(w_vp);
else
    templpdf(:,1) = -Inf;
end

% Fixed log weight for importance sampling (log fixed integrand)
if isamplevp_flag
    vlnpdf = max(vbmc_pdf(vp,Xa,0,1),log(realmin));
    lny = acqfun('islogf1',vlnpdf,[],[],fmu,fs2);
else
    lny = acqfun('islogf1',[],[],[],fmu,fs2);
end
% lny = lny - warpvars_vbmc(Xa,'logp',vp.trinfo);

% Mixture of box-uniforms
if w_vp < 1
    VV = prod(2*rect_delta);

    for ii = 1:N
%         if any(all(abs(bsxfun(@minus,Xa,gp.X(ii,:))) < rect_delta,2) / VV / N * (1 - w_vp) == 0)
%             warning('Divide by zero in log.')
%         end
        templpdf(:,ii+1) = log(all(abs(bsxfun(@minus,Xa,gp.X(ii,:))) < rect_delta,2) / VV / N * (1 - w_vp));
    end

    mmax = max(templpdf,[],2);
    lpdf = log(sum(exp(templpdf - mmax),2));
    lnw = bsxfun(@minus,lny,lpdf + mmax);
else
    lnw = bsxfun(@minus,lny,templpdf);
end

end
