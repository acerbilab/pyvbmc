D = 3;
vp.mu = [-1, -1, -1; 1, 1, 1]';
vp.w = [0.7, 0.3];
vp.K = 2;
vp.sigma = [1.0, 1.0] * 1e-3;
vp.lambda = [1.0, 1.0, 1.0];
gp_means = reshape(-5:4, 5, 2) * pi;
fess_MATLAB = fess_vbmc(vp, gp_means, reshape(-7:7, 5, 3))
dirpath = fileparts(matlab.desktop.editor.getActiveFilename);
save(dirpath + "/fess.mat", "fess_MATLAB")