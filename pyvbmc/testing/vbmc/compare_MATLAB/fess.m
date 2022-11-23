D = 3;
vp.mu = [-1, -1, -1; 1, 1, 1]';
vp.w = [0.7, 0.3];
vp.K = 2;
vp.sigma = [1.0, 1.0] * 1e-3;
vp.lambda = [1.0, 1.0, 1.0];
gp_means = reshape(-5:4, 5, 2) * pi;
fess_means = fess_vbmc(vp, gp_means, reshape(-7:7, 5, 3))

X = reshape(-7:7, 5, 3);
for ii = 1:5
    y(ii) = mvnormlpdf(X(ii, :), zeros(D,1), eye(D));
end
y = y(:);
hyp = [-2.0, -3.0, -4.0, 1.0, 0.0, -(D / 2) * log(2 * pi), 0.0, 0.25, 0.5, -0.5, 0.0, 0.5]';
hyp = [hyp, 2 * hyp];
gp = gplite_post(hyp, X, y, 1, 4);

Xa = 2 * reshape(-4:4, 3, 3) / pi;

fess_gp = fess_vbmc(vp, gp, Xa)
dirpath = fileparts(matlab.desktop.editor.getActiveFilename);
save(dirpath + "/fess.mat", "fess_means", "fess_gp");

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
