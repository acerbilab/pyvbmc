D = 3;
vp.mu = [-1.5, -1.0, -0.5; 0, 1, 2]';
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

y_viqr = log_isbasefun(Xa, @acqviqr_vbmc, gp, vp)
y_imiqr = log_isbasefun(Xa, @acqimiqr_vbmc, gp, vp)
dirpath = fileparts(matlab.desktop.editor.getActiveFilename);
save(dirpath + "/log_isbasefun.mat", "y_viqr", "y_imiqr");

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
function y = log_isbasefun(x,acqfun,gp,vp)
%LOG_ISBASEFUN Base importance sampling proposal log pdf

[fmu,fs2] = gplite_pred(gp,x);
if isempty(vp)
    y = acqfun('islogf',[],[],[],fmu,fs2);
else
    vlnpdf = max(vbmc_pdf(vp,x,0,1),log(realmin));
    y = acqfun('islogf',vlnpdf,[],[],fmu,fs2);
end

end
