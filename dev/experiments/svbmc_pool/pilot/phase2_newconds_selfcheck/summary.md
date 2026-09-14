# Honest (cross-run) expected log joint of stacked posteriors

Generated 2026-09-14 14:02:31 from the artifacts of the pool alone (`--self-check`, no cells), 100 draws per component, seed 0. Every estimate of the stack's expected log joint is scored by its bias against the cell's `e_log_joint_mc` (the mean of the target's noiseless log density over the arm's own draws of the stacked posterior); adding the cell's `entropy_ref` to every estimate gives the ELBOs and leaves every bias unchanged. `raw` is `sum w I_corr`, `capped_I` and `capped_E` the class's component-median and run-median caps of it; an honest value replaces every component's expected log joint by the covering other runs' GP estimates under a coverage rule (`ratioR`: a run's mean predictive SD on the component at most `R` times the component's own run's or below 0.1 nats, and at most 2.236 nats; `cap_only`: the cap alone; `none`: every other run) combined by the `median`, `precision` weighting or the `mean`, the component's own value when no other run covers it. The headline is `ratio2` with the `median`. `truth_strat` is the bias of the stratified Monte Carlo truth (`sum w E_true`) against `e_log_joint_mc`, a consistency check between the two Monte Carlo routes, and `pooled` the headline rule with the component's own run included (not honest). Medians over the cells of a condition and `M` with a 10,000-resample bootstrap 95 % interval. `covered` is the stacked weight on components at least one other run covers. `mc_se` is the Monte Carlo standard error of the honest value from the draws (the covering runs average the same draws), `gp_sd` its GP-side standard deviation under independence across components and under full correlation within each evaluating run. `within 2 SE` is the fraction of cells whose headline bias is within twice the combined Monte Carlo standard error of the estimate and of the reference.

## multisensory_s1_D6_svbmc (noiseless)

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| multisensory_s1_D6_svbmc_seed1000 | -0.017 | 0.031 | -0.37 | 0.99 | 0.066 | 0.198 | 2.96 (+0.34) | 3.26 (-0.23) |

## student_D8_noise3_svbmc (noisy)

| Run | own error | se | own z median | own abs z median | BQ sd median | pred sd median | own-run max abs z (offset z) | Jacobian max abs z (offset z) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| student_D8_noise3_svbmc_seed1000 | 0.316 | 0.057 | 0.44 | 0.45 | 0.442 | 0.841 | 2.65 (-1.41) | 0.00 (+0.00) |
