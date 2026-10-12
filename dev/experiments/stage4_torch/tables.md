# Stage 4 Torch campaign evidence tables

Instrumented tables use the complete variational-fit total (`complete_run.timing.total`) as their primary timing. Clean performance comparisons use the separately measured trace-disabled totals and paired clean Torch/NumPy ratios. External host wall, setup, GP extraction/transfer, optimizer, fine scoring/pruning, and the source-validation/report-construction residual remain separate. Both lanes are prototype measurements with frozen GP factors, outside an end-to-end solver or production backend.

## Complete-fit timing

| Case | Arm | Primary seconds, median [range] | External host seconds, median [range] |
| --- | --- | ---: | ---: |
| warmup | numpy_cpu | 0.0295687 [0.0282591, 0.0385934] (n=3) | 0.095345 [0.0944674, 0.106821] (n=3) |
| warmup | torch_cpu | 0.166592 [0.117133, 0.20952] (n=3) | 0.245004 [0.182644, 0.28875] (n=3) |
| warmup | torch_cuda | 0.709328 [0.631704, 0.87437] (n=3) | 0.777409 [0.727994, 0.991846] (n=3) |
| deterministic | numpy_cpu | 0.0153331 [0.014246, 0.0154612] (n=3) | 0.0848522 [0.084297, 0.0913331] (n=3) |
| deterministic | torch_cpu | 0.0324049 [0.0316378, 0.0354679] (n=3) | 0.109836 [0.0981449, 0.13226] (n=3) |
| deterministic | torch_cuda | 0.270125 [0.246318, 0.286221] (n=3) | 0.352361 [0.31646, 0.357855] (n=3) |
| bounded | numpy_cpu | 0.0863082 [0.0831593, 0.166748] (n=3) | 0.159851 [0.149743, 0.240853] (n=3) |
| bounded | torch_cpu | 0.305225 [0.238453, 0.560443] (n=3) | 0.38182 [0.312669, 0.637976] (n=3) |
| bounded | torch_cuda | 0.765656 [0.751876, 2.20201] (n=3) | 0.846868 [0.821089, 2.31288] (n=3) |
| warped | numpy_cpu | 0.379991 [0.274821, 0.426291] (n=3) | 0.453117 [0.344784, 0.51173] (n=3) |
| warped | torch_cpu | 0.870329 [0.843357, 1.0293] (n=3) | 0.974969 [0.913673, 1.12544] (n=3) |
| warped | torch_cuda | 2.209 [1.65737, 3.05574] (n=3) | 2.28911 [1.76781, 3.18775] (n=3) |
| noisy | numpy_cpu | 0.030121 [0.0278286, 0.0332064] (n=3) | 0.0964211 [0.0940539, 0.102915] (n=3) |
| noisy | torch_cpu | 0.123735 [0.119537, 0.143028] (n=3) | 0.193546 [0.189389, 0.214511] (n=3) |
| noisy | torch_cuda | 0.624023 [0.534182, 0.775849] (n=3) | 0.694256 [0.61327, 0.849194] (n=3) |
| boost_sampled | numpy_cpu | 2.10042 [2.03247, 4.92316] (n=3) | 2.16883 [2.09809, 4.98778] (n=3) |
| boost_sampled | torch_cpu | 5.49712 [5.46766, 15.9313] (n=3) | 5.5944 [5.54587, 16.0395] (n=3) |
| boost_sampled | torch_cuda | 7.91504 [6.96054, 22.4664] (n=3) | 8.00387 [7.04076, 22.5361] (n=3) |
| medium_synthetic | numpy_cpu | 5.38044 [5.03804, 6.03624] (n=3) | 5.4411 [5.09982, 6.1137] (n=3) |
| medium_synthetic | torch_cpu | 12.9029 [11.9633, 13.1003] (n=3) | 12.969 [12.026, 13.1635] (n=3) |
| medium_synthetic | torch_cuda | 17.4983 [16.4732, 18.8723] (n=3) | 17.5915 [16.5397, 18.9397] (n=3) |
| boost_single | numpy_cpu | 8.4386 [8.40205, 8.46409] (n=3) | 8.49831 [8.46131, 8.52651] (n=3) |
| boost_single | torch_cpu | 25.4239 [25.3086, 27.9888] (n=3) | 25.4838 [25.3688, 28.048] (n=3) |
| boost_single | torch_cuda | 40.0721 [36.7217, 40.3838] (n=3) | 40.1369 [36.7842, 40.4564] (n=3) |

## Paired complete-fit ratios

| Case | Torch arm | Torch / NumPy, median [range] |
| --- | --- | ---: |
| warmup | torch_cpu | 4.3166 [3.96138, 7.41426] (n=3) |
| warmup | torch_cuda | 22.354 [18.3795, 29.5708] (n=3) |
| deterministic | torch_cpu | 2.22082 [2.1134, 2.29399] (n=3) |
| deterministic | torch_cuda | 17.6171 [17.2903, 18.5122] (n=3) |
| bounded | torch_cpu | 3.36101 [2.86743, 3.53646] (n=3) |
| bounded | torch_cuda | 9.2071 [8.71152, 13.2056] (n=3) |
| warped | torch_cpu | 2.70874 [2.70874, 2.70874] (n=1) |
| warped | torch_cuda | 8.0416 [8.0416, 8.0416] (n=1) |
| noisy | torch_cpu | 4.30725 [3.96855, 4.44632] (n=3) |
| noisy | torch_cuda | 22.4238 [17.7345, 23.3644] (n=3) |
| boost_sampled | torch_cpu | 2.70466 [2.60313, 3.23599] (n=3) |
| boost_sampled | torch_cuda | 3.8943 [3.31389, 4.5634] (n=3) |
| medium_synthetic | torch_cpu | 2.37458 [2.13757, 2.4348] (n=3) |
| medium_synthetic | torch_cuda | 3.12651 [3.06167, 3.47324] (n=3) |
| boost_single | torch_cpu | 3.01281 [2.99011, 3.33119] (n=3) |
| boost_single | torch_cuda | 4.73437 [4.35163, 4.80643] (n=3) |

The instrumented ratio summaries exclude 4 paired observations whose original raw artifacts failed a kernel gate. Those failures remain recorded in the raw-failure sections; corrected diagnostics do not overwrite them.

## Separately measured clean controls

| Case | Arm | Clean primary seconds, median [range] | Clean / instrumented, median [range] |
| --- | --- | ---: | ---: |
| warmup | numpy_cpu | 0.0154143 [0.0154143, 0.0154143] (n=1) | 0.521305 [0.521305, 0.521305] (n=1) |
| warmup | torch_cpu | 0.0890938 [0.0890938, 0.0890938] (n=1) | 0.760622 [0.760622, 0.760622] (n=1) |
| warmup | torch_cuda | 0.359792 [0.359792, 0.359792] (n=1) | 0.411488 [0.411488, 0.411488] (n=1) |
| deterministic | numpy_cpu | 0.0050537 [0.0050537, 0.0050537] (n=1) | 0.329594 [0.329594, 0.329594] (n=1) |
| deterministic | torch_cpu | 0.0207641 [0.0207641, 0.0207641] (n=1) | 0.64077 [0.64077, 0.64077] (n=1) |
| deterministic | torch_cuda | 0.092748 [0.092748, 0.092748] (n=1) | 0.343352 [0.343352, 0.343352] (n=1) |
| bounded | numpy_cpu | 0.121052 [0.121052, 0.121052] (n=1) | 0.725957 [0.725957, 0.725957] (n=1) |
| bounded | torch_cpu | 0.480445 [0.480445, 0.480445] (n=1) | 0.857259 [0.857259, 0.857259] (n=1) |
| bounded | torch_cuda | 1.7953 [1.7953, 1.7953] (n=1) | 0.815301 [0.815301, 0.815301] (n=1) |
| warped | numpy_cpu | 0.234851 [0.234851, 0.234851] (n=1) | 0.854561 [0.854561, 0.854561] (n=1) |
| warped | torch_cpu | 0.497719 [0.497719, 0.497719] (n=1) | 0.571874 [0.571874, 0.571874] (n=1) |
| warped | torch_cuda | 1.31033 [1.31033, 1.31033] (n=1) | 0.790603 [0.790603, 0.790603] (n=1) |
| noisy | numpy_cpu | 0.0174942 [0.0174942, 0.0174942] (n=1) | 0.580797 [0.580797, 0.580797] (n=1) |
| noisy | torch_cpu | 0.109463 [0.109463, 0.109463] (n=1) | 0.915725 [0.915725, 0.915725] (n=1) |
| noisy | torch_cuda | 0.321828 [0.321828, 0.321828] (n=1) | 0.602469 [0.602469, 0.602469] (n=1) |
| boost_sampled | numpy_cpu | 1.50803 [1.50803, 1.50803] (n=1) | 0.741969 [0.741969, 0.741969] (n=1) |
| boost_sampled | torch_cpu | 3.9318 [3.9318, 3.9318] (n=1) | 0.715247 [0.715247, 0.715247] (n=1) |
| boost_sampled | torch_cuda | 5.05881 [5.05881, 5.05881] (n=1) | 0.639139 [0.639139, 0.639139] (n=1) |
| medium_synthetic | numpy_cpu | 4.72971 [4.72971, 4.72971] (n=1) | 0.783553 [0.783553, 0.783553] (n=1) |
| medium_synthetic | torch_cpu | 11.7632 [11.7632, 11.7632] (n=1) | 0.911673 [0.911673, 0.911673] (n=1) |
| medium_synthetic | torch_cuda | 20.9367 [20.9367, 20.9367] (n=1) | 1.10938 [1.10938, 1.10938] (n=1) |
| boost_single | numpy_cpu | 8.37731 [8.37731, 8.37731] (n=1) | 0.997055 [0.997055, 0.997055] (n=1) |
| boost_single | torch_cpu | 29.4457 [29.4457, 29.4457] (n=1) | 1.05205 [1.05205, 1.05205] (n=1) |
| boost_single | torch_cuda | 40.5281 [40.5281, 40.5281] (n=1) | 1.00357 [1.00357, 1.00357] (n=1) |

Clean-control values are independent trace-off measurements; the report does not subtract traced and clean timings to estimate an untraced runtime.

### Clean-control paired backend ratios

| Case | Torch arm | Clean Torch / clean NumPy | Replicates |
| --- | --- | ---: | ---: |
| warmup | torch_cpu | 5.77994 [5.77994, 5.77994] (n=1) | 1 |
| warmup | torch_cuda | 23.3415 [23.3415, 23.3415] (n=1) | 1 |
| deterministic | torch_cpu | 4.10869 [4.10869, 4.10869] (n=1) | 1 |
| deterministic | torch_cuda | 18.3525 [18.3525, 18.3525] (n=1) | 1 |
| bounded | torch_cpu | 3.96891 [3.96891, 3.96891] (n=1) | 1 |
| bounded | torch_cuda | 14.8308 [14.8308, 14.8308] (n=1) | 1 |
| warped | torch_cpu | 2.11929 [2.11929, 2.11929] (n=1) | 1 |
| warped | torch_cuda | 5.57938 [5.57938, 5.57938] (n=1) | 1 |
| noisy | torch_cpu | 6.25709 [6.25709, 6.25709] (n=1) | 1 |
| noisy | torch_cuda | 18.3963 [18.3963, 18.3963] (n=1) | 1 |
| boost_sampled | torch_cpu | 2.60725 [2.60725, 2.60725] (n=1) | 1 |
| boost_sampled | torch_cuda | 3.35459 [3.35459, 3.35459] (n=1) | 1 |
| medium_synthetic | torch_cpu | 2.48709 [2.48709, 2.48709] (n=1) | 1 |
| medium_synthetic | torch_cuda | 4.42663 [4.42663, 4.42663] (n=1) | 1 |
| boost_single | torch_cpu | 3.51494 [3.51494, 3.51494] (n=1) | 1 |
| boost_single | torch_cuda | 4.83785 [4.83785, 4.83785] (n=1) | 1 |

Each clean backend ratio has one paired seed (`n=1`); it has no three-seed spread.

### Clean-control parity with the same frozen fit

| Clean run | Frozen run | Status | Posterior mismatches | Statistic mismatches | Structure mismatches |
| --- | --- | --- | --- | --- | --- |
| clean_control_numpy_cpu_boost_sampled_seed1701 | numpy_cpu_boost_sampled_seed1701 | matched |  |  |  |
| clean_control_numpy_cpu_boost_single_seed1701 | numpy_cpu_boost_single_seed1701 | matched |  |  |  |
| clean_control_numpy_cpu_bounded_seed1701 | numpy_cpu_bounded_seed1701 | matched |  |  |  |
| clean_control_numpy_cpu_deterministic_seed1701 | numpy_cpu_deterministic_seed1701 | matched |  |  |  |
| clean_control_numpy_cpu_medium_synthetic_seed1701 | numpy_cpu_medium_synthetic_seed1701 | matched |  |  |  |
| clean_control_numpy_cpu_noisy_seed1701 | numpy_cpu_noisy_seed1701 | matched |  |  |  |
| clean_control_numpy_cpu_warmup_seed1701 | numpy_cpu_warmup_seed1701 | matched |  |  |  |
| clean_control_numpy_cpu_warped_seed1701 | numpy_cpu_warped_seed1701 | matched |  |  |  |
| clean_control_torch_cpu_boost_sampled_seed1701 | torch_cpu_boost_sampled_seed1701 | matched |  |  |  |
| clean_control_torch_cpu_boost_single_seed1701 | torch_cpu_boost_single_seed1701 | matched |  |  |  |
| clean_control_torch_cpu_bounded_seed1701 | torch_cpu_bounded_seed1701 | matched |  |  |  |
| clean_control_torch_cpu_deterministic_seed1701 | torch_cpu_deterministic_seed1701 | matched |  |  |  |
| clean_control_torch_cpu_medium_synthetic_seed1701 | torch_cpu_medium_synthetic_seed1701 | matched |  |  |  |
| clean_control_torch_cpu_noisy_seed1701 | torch_cpu_noisy_seed1701 | matched |  |  |  |
| clean_control_torch_cpu_warmup_seed1701 | torch_cpu_warmup_seed1701 | matched |  |  |  |
| clean_control_torch_cpu_warped_seed1701 | torch_cpu_warped_seed1701 | matched |  |  |  |
| clean_control_torch_cuda_boost_sampled_seed1701 | torch_cuda_boost_sampled_seed1701 | matched |  |  |  |
| clean_control_torch_cuda_boost_single_seed1701 | torch_cuda_boost_single_seed1701 | matched |  |  |  |
| clean_control_torch_cuda_bounded_seed1701 | torch_cuda_bounded_seed1701 | matched |  |  |  |
| clean_control_torch_cuda_deterministic_seed1701 | torch_cuda_deterministic_seed1701 | matched |  |  |  |
| clean_control_torch_cuda_medium_synthetic_seed1701 | torch_cuda_medium_synthetic_seed1701 | matched |  |  |  |
| clean_control_torch_cuda_noisy_seed1701 | torch_cuda_noisy_seed1701 | matched |  |  |  |
| clean_control_torch_cuda_warmup_seed1701 | torch_cuda_warmup_seed1701 | matched |  |  |  |
| clean_control_torch_cuda_warped_seed1701 | torch_cuda_warped_seed1701 | matched |  |  |  |

Trace events, retained candidates, optimizer traces, fixed replay, and fine rescoring are intentionally absent from clean controls and are recorded as `not_compared` rather than inferred.

### Clean-control warmup and memory

| Run | Warmup calls | Warmup walls s | Post-fit peak RSS bytes | CUDA allocated/reserved bytes |
| --- | ---: | --- | ---: | --- |
| clean_control_numpy_cpu_boost_sampled_seed1701 | 3 | [0.0012070999946445227, 0.0009929999941959977, 0.0009740999666973948] | 159420416 | None/None |
| clean_control_numpy_cpu_boost_single_seed1701 | 3 | [0.004024500027298927, 0.003721000044606626, 0.0037682000547647476] | 210370560 | None/None |
| clean_control_numpy_cpu_bounded_seed1701 | 3 | [0.0008184999460354447, 0.0006435000104829669, 0.0005896000657230616] | 143179776 | None/None |
| clean_control_numpy_cpu_deterministic_seed1701 | 3 | [0.00041800003964453936, 0.0002524999435991049, 0.0002234000712633133] | 142651392 | None/None |
| clean_control_numpy_cpu_medium_synthetic_seed1701 | 3 | [0.004411399946548045, 0.003346799989230931, 0.0037131999852135777] | 170061824 | None/None |
| clean_control_numpy_cpu_noisy_seed1701 | 3 | [0.00046060001477599144, 0.0003076000139117241, 0.0002668999368324876] | 141877248 | None/None |
| clean_control_numpy_cpu_warmup_seed1701 | 3 | [0.0004545000847429037, 0.00028359994757920504, 0.00026000000070780516] | 142131200 | None/None |
| clean_control_numpy_cpu_warped_seed1701 | 3 | [0.0014959999825805426, 0.0010776999406516552, 0.0010757000418379903] | 148692992 | None/None |
| clean_control_torch_cpu_boost_sampled_seed1701 | 3 | [0.00553830002900213, 0.004080400103703141, 0.004095700103789568] | 359845888 | None/None |
| clean_control_torch_cpu_boost_single_seed1701 | 3 | [0.015643200022168458, 0.012364199967123568, 0.013057800009846687] | 459452416 | None/None |
| clean_control_torch_cpu_bounded_seed1701 | 3 | [0.004668199922889471, 0.0026186000322923064, 0.002563699963502586] | 316530688 | None/None |
| clean_control_torch_cpu_deterministic_seed1701 | 3 | [0.002722000004723668, 0.0015070999506860971, 0.0014214999973773956] | 309768192 | None/None |
| clean_control_torch_cpu_medium_synthetic_seed1701 | 3 | [0.01386469998396933, 0.022039899951778352, 0.016578800044953823] | 362692608 | None/None |
| clean_control_torch_cpu_noisy_seed1701 | 3 | [0.0035810000263154507, 0.0020871999440714717, 0.0019018000457435846] | 311730176 | None/None |
| clean_control_torch_cpu_warmup_seed1701 | 3 | [0.0029897999484091997, 0.0017450000159442425, 0.0016657999949529767] | 311533568 | None/None |
| clean_control_torch_cpu_warped_seed1701 | 3 | [0.006578700034879148, 0.004853000049479306, 0.004807200049981475] | 323801088 | None/None |
| clean_control_torch_cuda_boost_sampled_seed1701 | 3 | [0.12073309998959303, 0.012450700043700635, 0.00970180006697774] | 1180217344 | 57917952/77594624 |
| clean_control_torch_cuda_boost_single_seed1701 | 3 | [0.14397039997857064, 0.0251460000872612, 0.017015199991874397] | 1217789952 | 143196160/192937984 |
| clean_control_torch_cuda_bounded_seed1701 | 3 | [0.1482407000148669, 0.02003389992751181, 0.016141299973241985] | 1172140032 | 19954688/25165824 |
| clean_control_torch_cuda_deterministic_seed1701 | 3 | [0.1186041000764817, 0.0057712000561878085, 0.004745400045067072] | 1151832064 | 17453056/23068672 |
| clean_control_torch_cuda_medium_synthetic_seed1701 | 3 | [0.12013900000602007, 0.012637900072149932, 0.01601299992762506] | 1186336768 | 44653568/73400320 |
| clean_control_torch_cuda_noisy_seed1701 | 3 | [0.11279160005506128, 0.01487690000794828, 0.014120200066827238] | 1169977344 | 18130432/23068672 |
| clean_control_torch_cuda_warmup_seed1701 | 3 | [0.10834330006036907, 0.006698200013488531, 0.006872599944472313] | 1168965632 | 18115072/23068672 |
| clean_control_torch_cuda_warped_seed1701 | 3 | [0.11812409991398454, 0.01908569992519915, 0.0077559000346809626] | 1174118400 | 24253952/48234496 |

## Paired complete-fit final-output parity

| NumPy reference | Torch run | Status | Posterior mismatches | Statistic mismatches | Structure mismatches |
| --- | --- | --- | --- | --- | --- |
| numpy_cpu_warmup_seed1701 | torch_cpu_warmup_seed1701 | matched |  |  |  |
| numpy_cpu_warmup_seed1701 | torch_cuda_warmup_seed1701 | matched |  |  |  |
| numpy_cpu_warmup_seed1702 | torch_cpu_warmup_seed1702 | matched |  |  |  |
| numpy_cpu_warmup_seed1702 | torch_cuda_warmup_seed1702 | matched |  |  |  |
| numpy_cpu_warmup_seed1703 | torch_cpu_warmup_seed1703 | matched |  |  |  |
| numpy_cpu_warmup_seed1703 | torch_cuda_warmup_seed1703 | matched |  |  |  |
| numpy_cpu_deterministic_seed1701 | torch_cpu_deterministic_seed1701 | matched |  |  |  |
| numpy_cpu_deterministic_seed1701 | torch_cuda_deterministic_seed1701 | matched |  |  |  |
| numpy_cpu_deterministic_seed1702 | torch_cpu_deterministic_seed1702 | matched |  |  |  |
| numpy_cpu_deterministic_seed1702 | torch_cuda_deterministic_seed1702 | matched |  |  |  |
| numpy_cpu_deterministic_seed1703 | torch_cpu_deterministic_seed1703 | matched |  |  |  |
| numpy_cpu_deterministic_seed1703 | torch_cuda_deterministic_seed1703 | matched |  |  |  |
| numpy_cpu_bounded_seed1701 | torch_cpu_bounded_seed1701 | mismatched | mu, w, eta |  |  |
| numpy_cpu_bounded_seed1701 | torch_cuda_bounded_seed1701 | mismatched | mu, sigma, w, eta |  |  |
| numpy_cpu_bounded_seed1702 | torch_cpu_bounded_seed1702 | mismatched | mu, w, eta |  |  |
| numpy_cpu_bounded_seed1702 | torch_cuda_bounded_seed1702 | mismatched | mu, w, eta |  |  |
| numpy_cpu_bounded_seed1703 | torch_cpu_bounded_seed1703 | mismatched | mu, w, eta |  |  |
| numpy_cpu_bounded_seed1703 | torch_cuda_bounded_seed1703 | mismatched | mu, w, eta |  |  |
| numpy_cpu_warped_seed1701 | torch_cpu_warped_seed1701 | mismatched | mu, w, eta |  |  |
| numpy_cpu_warped_seed1701 | torch_cuda_warped_seed1701 | mismatched | mu, w, eta |  |  |
| numpy_cpu_warped_seed1702 | torch_cpu_warped_seed1702 | mismatched | mu, w, eta |  |  |
| numpy_cpu_warped_seed1702 | torch_cuda_warped_seed1702 | mismatched | mu, w, eta |  |  |
| numpy_cpu_warped_seed1703 | torch_cpu_warped_seed1703 | mismatched | mu, w, eta |  |  |
| numpy_cpu_warped_seed1703 | torch_cuda_warped_seed1703 | mismatched | mu, w, eta |  |  |
| numpy_cpu_noisy_seed1701 | torch_cpu_noisy_seed1701 | matched |  |  |  |
| numpy_cpu_noisy_seed1701 | torch_cuda_noisy_seed1701 | matched |  |  |  |
| numpy_cpu_noisy_seed1702 | torch_cpu_noisy_seed1702 | matched |  |  |  |
| numpy_cpu_noisy_seed1702 | torch_cuda_noisy_seed1702 | matched |  |  |  |
| numpy_cpu_noisy_seed1703 | torch_cpu_noisy_seed1703 | matched |  |  |  |
| numpy_cpu_noisy_seed1703 | torch_cuda_noisy_seed1703 | matched |  |  |  |
| numpy_cpu_boost_sampled_seed1701 | torch_cpu_boost_sampled_seed1701 | mismatched | mu, w, eta |  |  |
| numpy_cpu_boost_sampled_seed1701 | torch_cuda_boost_sampled_seed1701 | mismatched | mu, w, eta |  |  |
| numpy_cpu_boost_sampled_seed1702 | torch_cpu_boost_sampled_seed1702 | mismatched | mu, w, eta |  |  |
| numpy_cpu_boost_sampled_seed1702 | torch_cuda_boost_sampled_seed1702 | mismatched | mu, w, eta |  |  |
| numpy_cpu_boost_sampled_seed1703 | torch_cpu_boost_sampled_seed1703 | mismatched | mu, sigma, lambd, w, eta | entropy, I_sk, J_sjk |  |
| numpy_cpu_boost_sampled_seed1703 | torch_cuda_boost_sampled_seed1703 | mismatched | mu, sigma, lambd, w, eta | e_log_joint, entropy, elbo, I_sk, J_sjk |  |
| numpy_cpu_medium_synthetic_seed1701 | torch_cpu_medium_synthetic_seed1701 | mismatched | mu, eta |  |  |
| numpy_cpu_medium_synthetic_seed1701 | torch_cuda_medium_synthetic_seed1701 | mismatched | mu, eta |  |  |
| numpy_cpu_medium_synthetic_seed1702 | torch_cpu_medium_synthetic_seed1702 | mismatched | mu, sigma, lambd, w, eta | entropy, I_sk |  |
| numpy_cpu_medium_synthetic_seed1702 | torch_cuda_medium_synthetic_seed1702 | mismatched | mu, sigma, lambd, w, eta | entropy, I_sk |  |
| numpy_cpu_medium_synthetic_seed1703 | torch_cpu_medium_synthetic_seed1703 | mismatched | mu, eta |  |  |
| numpy_cpu_medium_synthetic_seed1703 | torch_cuda_medium_synthetic_seed1703 | mismatched | mu, eta |  |  |
| numpy_cpu_boost_single_seed1701 | torch_cpu_boost_single_seed1701 | mismatched | eta |  |  |
| numpy_cpu_boost_single_seed1701 | torch_cuda_boost_single_seed1701 | mismatched | eta |  |  |
| numpy_cpu_boost_single_seed1702 | torch_cpu_boost_single_seed1702 | mismatched | eta |  |  |
| numpy_cpu_boost_single_seed1702 | torch_cuda_boost_single_seed1702 | mismatched | eta |  |  |
| numpy_cpu_boost_single_seed1703 | torch_cpu_boost_single_seed1703 | mismatched | eta |  |  |
| numpy_cpu_boost_single_seed1703 | torch_cuda_boost_single_seed1703 | mismatched | eta |  |  |

## Complete-fit structure and quality

| Run | Status | Iterations | K | Pruned | Selected | Boost accepted | ELBO | Weights/scales valid |
| --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |
| numpy_cpu_warmup_seed1701 | ok | [40] | 2 | 0 | endpoint:1 | None | -0.00836822 | True/True |
| torch_cpu_warmup_seed1701 | ok | [40] | 2 | 0 | endpoint:1 | None | -0.00836822 | True/True |
| torch_cuda_warmup_seed1701 | ok | [40] | 2 | 0 | endpoint:1 | None | -0.00836822 | True/True |
| numpy_cpu_warmup_seed1702 | ok | [60] | 2 | 0 | endpoint:1 | None | 0.0146028 | True/True |
| torch_cpu_warmup_seed1702 | ok | [60] | 2 | 0 | endpoint:1 | None | 0.0146028 | True/True |
| torch_cuda_warmup_seed1702 | ok | [60] | 2 | 0 | endpoint:1 | None | 0.0146028 | True/True |
| numpy_cpu_warmup_seed1703 | ok | [40] | 2 | 0 | midpoint:0 | None | -0.00761881 | True/True |
| torch_cpu_warmup_seed1703 | ok | [40] | 2 | 0 | midpoint:0 | None | -0.00761881 | True/True |
| torch_cuda_warmup_seed1703 | ok | [40] | 2 | 0 | midpoint:0 | None | -0.00761881 | True/True |
| numpy_cpu_deterministic_seed1701 | ok | [] | 1 | 0 | endpoint:0 | None | 0.0259422 | True/True |
| torch_cpu_deterministic_seed1701 | ok | [] | 1 | 0 | endpoint:0 | None | 0.0259422 | True/True |
| torch_cuda_deterministic_seed1701 | ok | [] | 1 | 0 | endpoint:0 | None | 0.0259422 | True/True |
| numpy_cpu_deterministic_seed1702 | ok | [] | 1 | 0 | endpoint:0 | None | 0.0123431 | True/True |
| torch_cpu_deterministic_seed1702 | ok | [] | 1 | 0 | endpoint:0 | None | 0.0123431 | True/True |
| torch_cuda_deterministic_seed1702 | ok | [] | 1 | 0 | endpoint:0 | None | 0.0123431 | True/True |
| numpy_cpu_deterministic_seed1703 | ok | [] | 1 | 0 | endpoint:0 | None | -0.0128786 | True/True |
| torch_cpu_deterministic_seed1703 | ok | [] | 1 | 0 | endpoint:0 | None | -0.0128786 | True/True |
| torch_cuda_deterministic_seed1703 | ok | [] | 1 | 0 | endpoint:0 | None | -0.0128786 | True/True |
| numpy_cpu_bounded_seed1701 | ok | [140] | 9 | 0 | endpoint:1 | None | -1.39305 | True/True |
| torch_cpu_bounded_seed1701 | ok | [140] | 9 | 0 | endpoint:1 | None | -1.39305 | True/True |
| torch_cuda_bounded_seed1701 | ok | [140] | 9 | 0 | endpoint:1 | None | -1.39305 | True/True |
| numpy_cpu_bounded_seed1702 | ok | [40] | 9 | 0 | midpoint:0 | None | -1.39114 | True/True |
| torch_cpu_bounded_seed1702 | ok | [40] | 9 | 0 | midpoint:0 | None | -1.39114 | True/True |
| torch_cuda_bounded_seed1702 | ok | [40] | 9 | 0 | midpoint:0 | None | -1.39114 | True/True |
| numpy_cpu_bounded_seed1703 | ok | [40] | 9 | 0 | endpoint:1 | None | -1.38532 | True/True |
| torch_cpu_bounded_seed1703 | ok | [40] | 9 | 0 | endpoint:1 | None | -1.38532 | True/True |
| torch_cuda_bounded_seed1703 | ok | [40] | 9 | 0 | endpoint:1 | None | -1.38532 | True/True |
| numpy_cpu_warped_seed1701 | ok | [40] | 17 | 0 | endpoint:1 | None | -0.0154616 | True/True |
| torch_cpu_warped_seed1701 | ok | [40] | 17 | 0 | endpoint:1 | None | -0.0154616 | True/True |
| torch_cuda_warped_seed1701 | ok | [40] | 17 | 0 | endpoint:1 | None | -0.0154616 | True/True |
| numpy_cpu_warped_seed1702 | ok | [80] | 17 | 0 | midpoint:0 | None | -0.0129098 | True/True |
| torch_cpu_warped_seed1702 | ok | [80] | 17 | 0 | midpoint:0 | None | -0.0129098 | True/True |
| torch_cuda_warped_seed1702 | ok | [80] | 17 | 0 | midpoint:0 | None | -0.0129098 | True/True |
| numpy_cpu_warped_seed1703 | failed_after_complete | [80] | 16 | 1 | endpoint:1 | None | -0.0133423 | True/True |
| torch_cpu_warped_seed1703 | failed_after_complete | [80] | 16 | 1 | endpoint:1 | None | -0.0133423 | True/True |
| torch_cuda_warped_seed1703 | failed_after_complete | [80] | 16 | 1 | endpoint:1 | None | -0.0133423 | True/True |
| numpy_cpu_noisy_seed1701 | ok | [40] | 2 | 0 | endpoint:1 | None | -3.07619 | True/True |
| torch_cpu_noisy_seed1701 | ok | [40] | 2 | 0 | endpoint:1 | None | -3.07619 | True/True |
| torch_cuda_noisy_seed1701 | ok | [40] | 2 | 0 | endpoint:1 | None | -3.07619 | True/True |
| numpy_cpu_noisy_seed1702 | ok | [40] | 2 | 0 | endpoint:1 | None | -3.0502 | True/True |
| torch_cpu_noisy_seed1702 | ok | [40] | 2 | 0 | endpoint:1 | None | -3.0502 | True/True |
| torch_cuda_noisy_seed1702 | ok | [40] | 2 | 0 | endpoint:1 | None | -3.0502 | True/True |
| numpy_cpu_noisy_seed1703 | ok | [40] | 2 | 0 | endpoint:1 | None | -3.07016 | True/True |
| torch_cpu_noisy_seed1703 | ok | [40] | 2 | 0 | endpoint:1 | None | -3.07016 | True/True |
| torch_cuda_noisy_seed1703 | ok | [40] | 2 | 0 | endpoint:1 | None | -3.07016 | True/True |
| numpy_cpu_boost_sampled_seed1701 | ok | [120] | 50 | 0 | endpoint:1 | True | -3.50855 | True/True |
| torch_cpu_boost_sampled_seed1701 | ok | [120] | 50 | 0 | endpoint:1 | True | -3.50855 | True/True |
| torch_cuda_boost_sampled_seed1701 | ok | [120] | 50 | 0 | endpoint:1 | True | -3.50855 | True/True |
| numpy_cpu_boost_sampled_seed1702 | ok | [120] | 50 | 0 | endpoint:1 | True | -3.51651 | True/True |
| torch_cpu_boost_sampled_seed1702 | ok | [120] | 50 | 0 | endpoint:1 | True | -3.51651 | True/True |
| torch_cuda_boost_sampled_seed1702 | ok | [120] | 50 | 0 | endpoint:1 | True | -3.51651 | True/True |
| numpy_cpu_boost_sampled_seed1703 | ok | [640] | 50 | 0 | midpoint:0 | True | -3.51014 | True/True |
| torch_cpu_boost_sampled_seed1703 | ok | [640] | 50 | 0 | midpoint:0 | True | -3.51014 | True/True |
| torch_cuda_boost_sampled_seed1703 | ok | [640] | 50 | 0 | midpoint:0 | True | -3.51014 | True/True |
| numpy_cpu_medium_synthetic_seed1701 | ok | [340, 400] | 25 | 0 | endpoint:1 | None | 19.5679 | True/True |
| torch_cpu_medium_synthetic_seed1701 | ok | [340, 400] | 25 | 0 | endpoint:1 | None | 19.5679 | True/True |
| torch_cuda_medium_synthetic_seed1701 | ok | [340, 400] | 25 | 0 | endpoint:1 | None | 19.5679 | True/True |
| numpy_cpu_medium_synthetic_seed1702 | ok | [380, 360] | 25 | 0 | endpoint:1 | None | 19.4301 | True/True |
| torch_cpu_medium_synthetic_seed1702 | ok | [380, 360] | 25 | 0 | endpoint:1 | None | 19.4301 | True/True |
| torch_cuda_medium_synthetic_seed1702 | ok | [380, 360] | 25 | 0 | endpoint:1 | None | 19.4301 | True/True |
| numpy_cpu_medium_synthetic_seed1703 | ok | [360, 440] | 25 | 0 | midpoint:0 | None | 19.6949 | True/True |
| torch_cpu_medium_synthetic_seed1703 | ok | [360, 440] | 25 | 0 | midpoint:0 | None | 19.6949 | True/True |
| torch_cuda_medium_synthetic_seed1703 | ok | [360, 440] | 25 | 0 | midpoint:0 | None | 19.6949 | True/True |
| numpy_cpu_boost_single_seed1701 | ok | [460] | 50 | 0 | endpoint:1 | True | 29.4895 | True/True |
| torch_cpu_boost_single_seed1701 | ok | [460] | 50 | 0 | endpoint:1 | True | 29.4895 | True/True |
| torch_cuda_boost_single_seed1701 | ok | [460] | 50 | 0 | endpoint:1 | True | 29.4895 | True/True |
| numpy_cpu_boost_single_seed1702 | ok | [420] | 50 | 0 | midpoint:0 | True | 29.0791 | True/True |
| torch_cpu_boost_single_seed1702 | ok | [420] | 50 | 0 | midpoint:0 | True | 29.0791 | True/True |
| torch_cuda_boost_single_seed1702 | ok | [420] | 50 | 0 | midpoint:0 | True | 29.0791 | True/True |
| numpy_cpu_boost_single_seed1703 | ok | [420] | 50 | 0 | midpoint:0 | True | 29.1762 | True/True |
| torch_cpu_boost_single_seed1703 | ok | [420] | 50 | 0 | midpoint:0 | True | 29.1762 | True/True |
| torch_cuda_boost_single_seed1703 | ok | [420] | 50 | 0 | midpoint:0 | True | 29.1762 | True/True |

## Fixed 100-update replay

| Case/seed | Torch arm | Starts | Same epsilon hashes/shapes | Max objective abs | Objective RMS | Max gradient abs | Gradient RMS | Initial theta max abs | Exact post-theta hashes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| warmup/1701 | torch_cpu | 1/1 | True | 3.10862e-15 | 7.36208e-16 | 3.0892e-14 | 5.42743e-15 | 0 | 1/100 |
| warmup/1701 | torch_cuda | 1/1 | True | 3.77476e-15 | 8.92764e-16 | 3.75533e-14 | 6.52953e-15 | 0 | 0/100 |
| warmup/1702 | torch_cpu | 1/1 | True | 1.66533e-15 | 3.48661e-16 | 1.113e-14 | 1.45364e-15 | 0 | 0/100 |
| warmup/1702 | torch_cuda | 1/1 | True | 1.9984e-15 | 3.97864e-16 | 1.47382e-14 | 1.91679e-15 | 0 | 0/100 |
| warmup/1703 | torch_cpu | 1/1 | True | 4.44089e-16 | 1.61651e-16 | 3.9968e-15 | 6.78314e-16 | 0 | 0/100 |
| warmup/1703 | torch_cuda | 1/1 | True | 5.55112e-16 | 1.7876e-16 | 4.21885e-15 | 6.99848e-16 | 0 | 0/100 |
| deterministic/1701 | torch_cpu | 1/0 | None |  |  |  |  |  | None/None |
| deterministic/1701 | torch_cuda | 1/0 | None |  |  |  |  |  | None/None |
| deterministic/1702 | torch_cpu | 1/0 | None |  |  |  |  |  | None/None |
| deterministic/1702 | torch_cuda | 1/0 | None |  |  |  |  |  | None/None |
| deterministic/1703 | torch_cpu | 1/0 | None |  |  |  |  |  | None/None |
| deterministic/1703 | torch_cuda | 1/0 | None |  |  |  |  |  | None/None |
| bounded/1701 | torch_cpu | 1/1 | True | 1.52757e-09 | 5.16202e-10 | 7.30642e-10 | 7.47241e-11 | 0 | 0/100 |
| bounded/1701 | torch_cuda | 1/1 | True | 1.36587e-09 | 4.70416e-10 | 7.12969e-10 | 7.34684e-11 | 0 | 0/100 |
| bounded/1702 | torch_cpu | 1/1 | True | 1.35392e-09 | 5.34348e-10 | 9.22587e-10 | 7.35991e-11 | 0 | 0/100 |
| bounded/1702 | torch_cuda | 1/1 | True | 1.34019e-09 | 4.85745e-10 | 8.07187e-10 | 7.2813e-11 | 0 | 0/100 |
| bounded/1703 | torch_cpu | 1/1 | True | 1.21021e-09 | 4.69806e-10 | 8.17654e-10 | 7.62405e-11 | 0 | 0/100 |
| bounded/1703 | torch_cuda | 1/1 | True | 1.52362e-09 | 5.70845e-10 | 7.73651e-10 | 7.46288e-11 | 0 | 0/100 |
| warped/1701 | torch_cpu | 1/1 | True | 5.75273e-09 | 2.23349e-09 | 2.43275e-09 | 1.68812e-10 | 2.22045e-16 | 0/100 |
| warped/1701 | torch_cuda | 1/1 | True | 5.97358e-09 | 1.90278e-09 | 2.47478e-09 | 1.66336e-10 | 2.22045e-16 | 0/100 |
| warped/1702 | torch_cpu | 1/1 | True | 6.86181e-09 | 2.16771e-09 | 2.7326e-09 | 1.62249e-10 | 2.22045e-16 | 0/100 |
| warped/1702 | torch_cuda | 1/1 | True | 6.39726e-09 | 2.02864e-09 | 2.51817e-09 | 1.69724e-10 | 2.22045e-16 | 0/100 |
| warped/1703 | torch_cpu | 1/1 | True | 5.70926e-09 | 2.10461e-09 | 3.29679e-09 | 1.78204e-10 | 2.22045e-16 | 0/100 |
| warped/1703 | torch_cuda | 1/1 | True | 4.73084e-09 | 1.97382e-09 | 2.98429e-09 | 1.72441e-10 | 2.22045e-16 | 0/100 |
| noisy/1701 | torch_cpu | 1/1 | True | 2.22489e-13 | 8.92794e-14 | 2.08722e-12 | 4.28079e-13 | 0 | 0/100 |
| noisy/1701 | torch_cuda | 1/1 | True | 2.77556e-13 | 8.74449e-14 | 1.25547e-12 | 2.68998e-13 | 0 | 0/100 |
| noisy/1702 | torch_cpu | 1/1 | True | 2.62013e-13 | 9.29561e-14 | 1.43746e-12 | 2.68682e-13 | 0 | 0/100 |
| noisy/1702 | torch_cuda | 1/1 | True | 2.64233e-13 | 9.48093e-14 | 1.99418e-12 | 2.93903e-13 | 0 | 0/100 |
| noisy/1703 | torch_cpu | 1/1 | True | 2.25597e-13 | 9.24809e-14 | 1.31573e-12 | 2.40857e-13 | 0 | 1/100 |
| noisy/1703 | torch_cuda | 1/1 | True | 2.3892e-13 | 9.97854e-14 | 1.39744e-12 | 2.72669e-13 | 0 | 0/100 |
| boost_sampled/1701 | torch_cpu | 1/1 | True | 4.74579e-10 | 1.24924e-10 | 1.27941e-09 | 1.28251e-11 | 0 | 0/100 |
| boost_sampled/1701 | torch_cuda | 1/1 | True | 3.73696e-10 | 1.19648e-10 | 1.81168e-09 | 1.85335e-11 | 0 | 0/100 |
| boost_sampled/1702 | torch_cpu | 1/1 | True | 3.75834e-10 | 1.1036e-10 | 1.54771e-10 | 7.04569e-12 | 0 | 0/100 |
| boost_sampled/1702 | torch_cuda | 1/1 | True | 3.70414e-10 | 1.17474e-10 | 1.52713e-10 | 6.93487e-12 | 0 | 0/100 |
| boost_sampled/1703 | torch_cpu | 1/1 | True | 3.12191e-10 | 1.2827e-10 | 1.3161e-10 | 7.16333e-12 | 0 | 0/100 |
| boost_sampled/1703 | torch_cuda | 1/1 | True | 3.66633e-10 | 1.3146e-10 | 7.66377e-10 | 8.31849e-12 | 0 | 0/100 |
| medium_synthetic/1701 | torch_cpu | 1/2 | True | 7.10543e-15 | 1.90803e-15 | 2.69229e-15 | 8.74751e-17 | 0 | 0/100 |
| medium_synthetic/1701 | torch_cpu | 2/2 | True | 3.55271e-15 | 1.11554e-15 | 1.4988e-15 | 7.40275e-17 | 4.44089e-16 | 0/100 |
| medium_synthetic/1701 | torch_cuda | 1/2 | True | 5.32907e-15 | 1.7585e-15 | 2.22045e-15 | 8.40522e-17 | 0 | 0/100 |
| medium_synthetic/1701 | torch_cuda | 2/2 | True | 3.55271e-15 | 1.34204e-15 | 1.52656e-15 | 7.20727e-17 | 4.44089e-16 | 0/100 |
| medium_synthetic/1702 | torch_cpu | 1/2 | True | 7.10543e-15 | 2.01364e-15 | 4.18093e-11 | 3.28127e-13 | 0 | 0/100 |
| medium_synthetic/1702 | torch_cpu | 2/2 | True | 4.44089e-15 | 1.54029e-15 | 1.94289e-15 | 8.20298e-17 | 4.44089e-16 | 0/100 |
| medium_synthetic/1702 | torch_cuda | 1/2 | True | 4.44089e-15 | 2.00235e-15 | 3.98265e-13 | 2.38577e-15 | 0 | 0/100 |
| medium_synthetic/1702 | torch_cuda | 2/2 | True | 5.32907e-15 | 1.37165e-15 | 1.88738e-15 | 8.15954e-17 | 4.44089e-16 | 0/100 |
| medium_synthetic/1703 | torch_cpu | 1/2 | True | 7.10543e-15 | 2.02048e-15 | 3.01842e-15 | 9.33555e-17 | 4.44089e-16 | 0/100 |
| medium_synthetic/1703 | torch_cpu | 2/2 | True | 5.32907e-15 | 1.39305e-15 | 1.88738e-15 | 7.98999e-17 | 0 | 0/100 |
| medium_synthetic/1703 | torch_cuda | 1/2 | True | 5.32907e-15 | 1.93217e-15 | 1.85962e-15 | 8.53271e-17 | 4.44089e-16 | 0/100 |
| medium_synthetic/1703 | torch_cuda | 2/2 | True | 3.55271e-15 | 1.31963e-15 | 1.33227e-15 | 7.5056e-17 | 0 | 0/100 |
| boost_single/1701 | torch_cpu | 1/1 | True | 1.06581e-14 | 3.20607e-15 | 4.01762e-15 | 8.43818e-17 | 0 | 0/100 |
| boost_single/1701 | torch_cuda | 1/1 | True | 1.06581e-14 | 2.90123e-15 | 3.86496e-15 | 1.00374e-16 | 0 | 0/100 |
| boost_single/1702 | torch_cpu | 1/1 | True | 7.10543e-15 | 2.68224e-15 | 4.88498e-15 | 7.83078e-17 | 2.22045e-16 | 0/100 |
| boost_single/1702 | torch_cuda | 1/1 | True | 7.10543e-15 | 2.93099e-15 | 4.57967e-15 | 7.84458e-17 | 2.22045e-16 | 0/100 |
| boost_single/1703 | torch_cpu | 1/1 | True | 7.10543e-15 | 2.60465e-15 | 4.02456e-15 | 8.23376e-17 |  | 0/100 |
| boost_single/1703 | torch_cuda | 1/1 | True | 1.06581e-14 | 2.95378e-15 | 3.76088e-15 | 8.23033e-17 |  | 0/100 |

Exact theta-hash mismatches are not classified as structural branch divergence. Structural differences are limited to iteration counts, selected candidate, pruning, returned K, and boost acceptance.

## Common-stream final-candidate rescoring

| Case/seed | Torch arm | NumPy ELBO mean ± sd | Torch ELBO mean ± sd | Torch - NumPy mean ± sd | Max abs delta | Streams |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| warmup/1701 | torch_cpu | -0.000639467 ± 0.0178703 | -0.000639467 ± 0.0178703 | 2.66454e-16 ± 6.08094e-17 | 3.33067e-16 | 5 |
| warmup/1701 | torch_cuda | -0.000639467 ± 0.0178703 | -0.000639467 ± 0.0178703 | 0 ± 7.85046e-17 | 1.11022e-16 | 5 |
| warmup/1702 | torch_cpu | 0.0197545 ± 0.025115 | 0.0197545 ± 0.025115 | -3.9968e-16 ± 6.08094e-17 | 4.44089e-16 | 5 |
| warmup/1702 | torch_cuda | 0.0197545 ± 0.025115 | 0.0197545 ± 0.025115 | -2.22045e-16 ± 0 | 2.22045e-16 | 5 |
| warmup/1703 | torch_cpu | -0.00857739 ± 0.0142262 | -0.00857739 ± 0.0142262 | 0 ± 0 | 0 | 5 |
| warmup/1703 | torch_cuda | -0.00857739 ± 0.0142262 | -0.00857739 ± 0.0142262 | 1.55431e-16 ± 9.93014e-17 | 2.22045e-16 | 5 |
| deterministic/1701 | torch_cpu | -0.000961322 ± 0.00520283 | -0.000961322 ± 0.00520283 | 1.9984e-16 ± 4.96507e-17 | 2.22045e-16 | 5 |
| deterministic/1701 | torch_cuda | -0.000961322 ± 0.00520283 | -0.000961322 ± 0.00520283 | 2.88658e-16 ± 6.08094e-17 | 3.33067e-16 | 5 |
| deterministic/1702 | torch_cpu | 0.0278479 ± 0.0308472 | 0.0278479 ± 0.0308472 | 1.33227e-16 ± 4.96507e-17 | 2.22045e-16 | 5 |
| deterministic/1702 | torch_cuda | 0.0278479 ± 0.0308472 | 0.0278479 ± 0.0308472 | 1.77636e-16 ± 6.08094e-17 | 2.22045e-16 | 5 |
| deterministic/1703 | torch_cpu | 0.00207047 ± 0.0185949 | 0.00207047 ± 0.0185949 | 1.33227e-16 ± 4.96507e-17 | 2.22045e-16 | 5 |
| deterministic/1703 | torch_cuda | 0.00207047 ± 0.0185949 | 0.00207047 ± 0.0185949 | 1.55431e-16 ± 6.08094e-17 | 2.22045e-16 | 5 |
| bounded/1701 | torch_cpu | -1.39282 ± 0.0105173 | -1.39282 ± 0.0105173 | -6.48529e-11 ± 5.06427e-12 | 7.11549e-11 | 5 |
| bounded/1701 | torch_cuda | -1.39282 ± 0.0105173 | -1.39282 ± 0.0105173 | 5.4568e-11 ± 5.45761e-12 | 6.09963e-11 | 5 |
| bounded/1702 | torch_cpu | -1.39264 ± 0.00964098 | -1.39264 ± 0.00964098 | -4.61387e-10 ± 3.29318e-13 | 4.61646e-10 | 5 |
| bounded/1702 | torch_cuda | -1.39264 ± 0.00964098 | -1.39264 ± 0.00964098 | -2.06608e-10 ± 4.81978e-13 | 2.07117e-10 | 5 |
| bounded/1703 | torch_cpu | -1.39549 ± 0.00520894 | -1.39549 ± 0.00520894 | 1.99199e-10 ± 1.04418e-12 | 1.99855e-10 | 5 |
| bounded/1703 | torch_cuda | -1.39549 ± 0.00520894 | -1.39549 ± 0.00520894 | -1.57289e-10 ± 8.63455e-13 | 1.58739e-10 | 5 |
| warped/1701 | torch_cpu | -0.0113213 ± 0.00496164 | -0.0113213 ± 0.00496164 | -9.63477e-10 ± 2.50491e-12 | 9.66783e-10 | 5 |
| warped/1701 | torch_cuda | -0.0113213 ± 0.00496164 | -0.0113213 ± 0.00496164 | -7.50685e-10 ± 3.11894e-12 | 7.55967e-10 | 5 |
| warped/1702 | torch_cpu | -0.0206932 ± 0.0116386 | -0.0206932 ± 0.0116386 | 2.20444e-09 ± 5.55894e-12 | 2.2116e-09 | 5 |
| warped/1702 | torch_cuda | -0.0206932 ± 0.0116386 | -0.0206932 ± 0.0116386 | 4.68128e-09 ± 7.1622e-12 | 4.69196e-09 | 5 |
| warped/1703 | torch_cpu | -0.0172136 ± 0.00706097 | -0.0172136 ± 0.00706097 | -1.05584e-09 ± 8.82785e-12 | 1.06894e-09 | 5 |
| warped/1703 | torch_cuda | -0.0172136 ± 0.00706097 | -0.0172136 ± 0.00706097 | 2.98429e-09 ± 5.63347e-12 | 2.99088e-09 | 5 |
| noisy/1701 | torch_cpu | -3.06368 ± 0.0162693 | -3.06368 ± 0.0162693 | 2.27196e-13 ± 8.06728e-16 | 2.28262e-13 | 5 |
| noisy/1701 | torch_cuda | -3.06368 ± 0.0162693 | -3.06368 ± 0.0162693 | 8.23341e-14 ± 6.73495e-16 | 8.30447e-14 | 5 |
| noisy/1702 | torch_cpu | -3.05106 ± 0.0237825 | -3.05106 ± 0.0237825 | -8.6775e-14 ± 2.43238e-16 | 8.70415e-14 | 5 |
| noisy/1702 | torch_cuda | -3.05106 ± 0.0237825 | -3.05106 ± 0.0237825 | 7.37188e-14 ± 5.43896e-16 | 7.41629e-14 | 5 |
| noisy/1703 | torch_cpu | -3.06355 ± 0.0151088 | -3.06355 ± 0.0151088 | 3.72147e-14 ± 3.71552e-16 | 3.77476e-14 | 5 |
| noisy/1703 | torch_cuda | -3.06355 ± 0.0151088 | -3.06355 ± 0.0151088 | 2.78e-14 ± 9.72951e-16 | 2.93099e-14 | 5 |
| boost_sampled/1701 | torch_cpu | -3.51532 ± 0.00232133 | -3.51532 ± 0.00232133 | -9.0375e-11 ± 3.10809e-13 | 9.07363e-11 | 5 |
| boost_sampled/1701 | torch_cuda | -3.51532 ± 0.00232133 | -3.51532 ± 0.00232133 | -1.51305e-10 ± 7.92851e-13 | 1.52441e-10 | 5 |
| boost_sampled/1702 | torch_cpu | -3.5169 ± 0.00279391 | -3.5169 ± 0.00279391 | 2.50431e-11 ± 8.12614e-13 | 2.61027e-11 | 5 |
| boost_sampled/1702 | torch_cuda | -3.5169 ± 0.00279391 | -3.5169 ± 0.00279391 | 5.31211e-11 ± 9.51716e-13 | 5.44143e-11 | 5 |
| boost_sampled/1703 | torch_cpu | -3.51457 ± 0.00276156 | -3.51457 ± 0.00276113 | -1.07167e-06 ± 9.60613e-07 | 1.86772e-06 | 5 |
| boost_sampled/1703 | torch_cuda | -3.51457 ± 0.00276156 | -3.51458 ± 0.0027613 | -4.76009e-06 ± 4.21194e-07 | 5.17911e-06 | 5 |
| medium_synthetic/1701 | torch_cpu | 19.5806 ± 0.0164959 | 19.5806 ± 0.0164959 | 3.0596e-12 ± 1.39563e-13 | 3.25073e-12 | 5 |
| medium_synthetic/1701 | torch_cuda | 19.5806 ± 0.0164959 | 19.5806 ± 0.0164959 | -1.7792e-12 ± 8.12244e-14 | 1.89004e-12 | 5 |
| medium_synthetic/1702 | torch_cpu | 19.4213 ± 0.00739412 | 19.4213 ± 0.00739412 | 2.57774e-08 ± 2.70922e-08 | 5.48564e-08 | 5 |
| medium_synthetic/1702 | torch_cuda | 19.4213 ± 0.00739412 | 19.4213 ± 0.00739412 | -4.31117e-08 ± 2.56951e-08 | 8.44677e-08 | 5 |
| medium_synthetic/1703 | torch_cpu | 19.6983 ± 0.00980846 | 19.6983 ± 0.00980846 | 8.86189e-12 ± 5.31802e-13 | 9.47153e-12 | 5 |
| medium_synthetic/1703 | torch_cuda | 19.6983 ± 0.00980846 | 19.6983 ± 0.00980846 | 8.86047e-12 ± 5.31704e-13 | 9.46798e-12 | 5 |
| boost_single/1701 | torch_cpu | 29.4898 ± 0.0146343 | 29.4898 ± 0.0146343 | 1.42109e-15 ± 3.17764e-15 | 7.10543e-15 | 5 |
| boost_single/1701 | torch_cuda | 29.4898 ± 0.0146343 | 29.4898 ± 0.0146343 | 4.26326e-15 ± 3.8918e-15 | 7.10543e-15 | 5 |
| boost_single/1702 | torch_cpu | 29.0892 ± 0.00557094 | 29.0892 ± 0.00557094 | 2.84217e-15 ± 3.8918e-15 | 7.10543e-15 | 5 |
| boost_single/1702 | torch_cuda | 29.0892 ± 0.00557094 | 29.0892 ± 0.00557094 | 1.06581e-14 ± 0 | 1.06581e-14 | 5 |
| boost_single/1703 | torch_cpu | 29.1555 ± 0.0113568 | 29.1555 ± 0.0113568 | 0 ± 0 | 0 | 5 |
| boost_single/1703 | torch_cuda | 29.1555 ± 0.0113568 | 29.1555 ± 0.0113568 | 1.42109e-15 ± 3.17764e-15 | 7.10543e-15 | 5 |

## Kernel gates and failures

| Run | Kernel gate | Failed quantities | Failure log |
| --- | --- | --- | --- |
| numpy_cpu_warmup_seed1701 | True |  |  |
| torch_cpu_warmup_seed1701 | True |  |  |
| torch_cuda_warmup_seed1701 | True |  |  |
| numpy_cpu_warmup_seed1702 | True |  |  |
| torch_cpu_warmup_seed1702 | True |  |  |
| torch_cuda_warmup_seed1702 | True |  |  |
| numpy_cpu_warmup_seed1703 | True |  |  |
| torch_cpu_warmup_seed1703 | True |  |  |
| torch_cuda_warmup_seed1703 | True |  |  |
| numpy_cpu_deterministic_seed1701 | True |  |  |
| torch_cpu_deterministic_seed1701 | True |  |  |
| torch_cuda_deterministic_seed1701 | True |  |  |
| numpy_cpu_deterministic_seed1702 | True |  |  |
| torch_cpu_deterministic_seed1702 | True |  |  |
| torch_cuda_deterministic_seed1702 | True |  |  |
| numpy_cpu_deterministic_seed1703 | True |  |  |
| torch_cpu_deterministic_seed1703 | True |  |  |
| torch_cuda_deterministic_seed1703 | True |  |  |
| numpy_cpu_bounded_seed1701 | True |  |  |
| torch_cpu_bounded_seed1701 | True |  |  |
| torch_cuda_bounded_seed1701 | True |  |  |
| numpy_cpu_bounded_seed1702 | True |  |  |
| torch_cpu_bounded_seed1702 | True |  |  |
| torch_cuda_bounded_seed1702 | True |  |  |
| numpy_cpu_bounded_seed1703 | True |  |  |
| torch_cpu_bounded_seed1703 | True |  |  |
| torch_cuda_bounded_seed1703 | True |  |  |
| numpy_cpu_warped_seed1701 | True |  |  |
| torch_cpu_warped_seed1701 | False | dG, dH_det, dH_mc |  |
| torch_cuda_warped_seed1701 | False | dG, dH_det, dH_mc |  |
| numpy_cpu_warped_seed1702 | True |  |  |
| torch_cpu_warped_seed1702 | False | dG, dH_det, dH_mc |  |
| torch_cuda_warped_seed1702 | False | dG, dH_det, dH_mc |  |
| numpy_cpu_warped_seed1703 | None |  | operands could not be broadcast together with shapes (117,) (124,)  |
| torch_cpu_warped_seed1703 | None |  | operands could not be broadcast together with shapes (117,) (124,)  |
| torch_cuda_warped_seed1703 | None |  | operands could not be broadcast together with shapes (117,) (124,)  |
| numpy_cpu_noisy_seed1701 | True |  |  |
| torch_cpu_noisy_seed1701 | True |  |  |
| torch_cuda_noisy_seed1701 | True |  |  |
| numpy_cpu_noisy_seed1702 | True |  |  |
| torch_cpu_noisy_seed1702 | True |  |  |
| torch_cuda_noisy_seed1702 | True |  |  |
| numpy_cpu_noisy_seed1703 | True |  |  |
| torch_cpu_noisy_seed1703 | True |  |  |
| torch_cuda_noisy_seed1703 | True |  |  |
| numpy_cpu_boost_sampled_seed1701 | True |  |  |
| torch_cpu_boost_sampled_seed1701 | True |  |  |
| torch_cuda_boost_sampled_seed1701 | True |  |  |
| numpy_cpu_boost_sampled_seed1702 | True |  |  |
| torch_cpu_boost_sampled_seed1702 | True |  |  |
| torch_cuda_boost_sampled_seed1702 | True |  |  |
| numpy_cpu_boost_sampled_seed1703 | True |  |  |
| torch_cpu_boost_sampled_seed1703 | True |  |  |
| torch_cuda_boost_sampled_seed1703 | True |  |  |
| numpy_cpu_medium_synthetic_seed1701 | True |  |  |
| torch_cpu_medium_synthetic_seed1701 | True |  |  |
| torch_cuda_medium_synthetic_seed1701 | True |  |  |
| numpy_cpu_medium_synthetic_seed1702 | True |  |  |
| torch_cpu_medium_synthetic_seed1702 | True |  |  |
| torch_cuda_medium_synthetic_seed1702 | True |  |  |
| numpy_cpu_medium_synthetic_seed1703 | True |  |  |
| torch_cpu_medium_synthetic_seed1703 | True |  |  |
| torch_cuda_medium_synthetic_seed1703 | True |  |  |
| numpy_cpu_boost_single_seed1701 | True |  |  |
| torch_cpu_boost_single_seed1701 | True |  |  |
| torch_cuda_boost_single_seed1701 | True |  |  |
| numpy_cpu_boost_single_seed1702 | True |  |  |
| torch_cpu_boost_single_seed1702 | True |  |  |
| torch_cuda_boost_single_seed1702 | True |  |  |
| numpy_cpu_boost_single_seed1703 | True |  |  |
| torch_cpu_boost_single_seed1703 | True |  |  |
| torch_cuda_boost_single_seed1703 | True |  |  |

### Corrected kernel-only diagnostics

| Supplemental run | Status | Kernel gate | Failed quantities | Frozen source link |
| --- | --- | --- | --- | --- |
| kernel_only_torch_cpu_boost_single_seed1701 | ok | True |  | torch_cpu_boost_single_seed1701 |
| kernel_only_torch_cpu_kernel_cigar_D4_boosted_seed1701 | ok | True |  | None |
| kernel_only_torch_cpu_kernel_cigar_D4_largeK_seed1701 | ok | True |  | None |
| kernel_only_torch_cpu_kernel_corr_D5_warped_seed1701 | ok | True |  | None |
| kernel_only_torch_cpu_kernel_halfnormal_D2_bounded_seed1701 | ok | True |  | None |
| kernel_only_torch_cpu_kernel_normal_D2_K1_seed1701 | ok | True |  | None |
| kernel_only_torch_cpu_kernel_normal_D2_singlesample_seed1701 | ok | True |  | None |
| kernel_only_torch_cpu_kernel_normal_D2_warmup_seed1701 | ok | True |  | None |
| kernel_only_torch_cpu_kernel_rosenbrock_D2_noise1_viqr_seed1701 | ok | True |  | None |
| kernel_only_torch_cpu_kernel_stress_seed1701 | ok | True |  | None |
| kernel_only_torch_cpu_medium_synthetic_seed1701 | ok | True |  | torch_cpu_medium_synthetic_seed1701 |
| kernel_only_torch_cpu_warped_seed1701 | ok | True |  | torch_cpu_warped_seed1701 |
| kernel_only_torch_cpu_warped_seed1702 | ok | True |  | torch_cpu_warped_seed1702 |
| kernel_only_torch_cpu_warped_seed1703 | ok | True |  | torch_cpu_warped_seed1703 |
| kernel_only_torch_cuda_boost_single_seed1701 | ok | True |  | torch_cuda_boost_single_seed1701 |
| kernel_only_torch_cuda_kernel_cigar_D4_boosted_seed1701 | ok | True |  | None |
| kernel_only_torch_cuda_kernel_cigar_D4_largeK_seed1701 | ok | True |  | None |
| kernel_only_torch_cuda_kernel_corr_D5_warped_seed1701 | ok | True |  | None |
| kernel_only_torch_cuda_kernel_halfnormal_D2_bounded_seed1701 | ok | True |  | None |
| kernel_only_torch_cuda_kernel_normal_D2_K1_seed1701 | ok | True |  | None |
| kernel_only_torch_cuda_kernel_normal_D2_singlesample_seed1701 | ok | True |  | None |
| kernel_only_torch_cuda_kernel_normal_D2_warmup_seed1701 | ok | True |  | None |
| kernel_only_torch_cuda_kernel_rosenbrock_D2_noise1_viqr_seed1701 | ok | True |  | None |
| kernel_only_torch_cuda_kernel_stress_seed1701 | ok | True |  | None |
| kernel_only_torch_cuda_medium_synthetic_seed1701 | ok | True |  | torch_cuda_medium_synthetic_seed1701 |
| kernel_only_torch_cuda_warped_seed1701 | ok | True |  | torch_cuda_warped_seed1701 |
| kernel_only_torch_cuda_warped_seed1702 | ok | True |  | torch_cuda_warped_seed1702 |
| kernel_only_torch_cuda_warped_seed1703 | ok | True |  | torch_cuda_warped_seed1703 |

Corrected diagnostics are supplemental. Original gate failures and their raw errors remain in the frozen campaign evidence.

### Recovered auxiliary diagnostics

| Recovery run | Case/seed | Arm | Fixed 100 present | Fine rescores present |
| --- | --- | --- | --- | --- |
| aux_numpy_cpu_warped_seed1703 | warped/1703 | numpy_cpu | True | True |
| aux_torch_cpu_warped_seed1703 | warped/1703 | torch_cpu | True | True |
| aux_torch_cuda_warped_seed1703 | warped/1703 | torch_cuda | True | True |

Kernel failures retain their raw error records in `results.json`; a later kernel-only correction does not silently turn an earlier failed artifact green.

## Missing auxiliary data

- `{'run_id': 'numpy_cpu_warped_seed1703', 'missing': 'auxiliary diagnostics after saved primary fit', 'failure_message': 'operands could not be broadcast together with shapes (117,) (124,) '}`
- `{'run_id': 'torch_cpu_warped_seed1703', 'missing': 'auxiliary diagnostics after saved primary fit', 'failure_message': 'operands could not be broadcast together with shapes (117,) (124,) '}`
- `{'run_id': 'torch_cuda_warped_seed1703', 'missing': 'auxiliary diagnostics after saved primary fit', 'failure_message': 'operands could not be broadcast together with shapes (117,) (124,) '}`

No feasibility recommendation is made by this artifact reconciler.
