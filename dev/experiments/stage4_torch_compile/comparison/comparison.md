# Stage 4 compiled-Torch comparison

ordinary optimization objective and backward are compiled; exact full-variance/per-component fine scoring is deliberately eager

Original harness status: **completed_with_failures** with 10 recorded failure(s).
The reporter reclassified 10 worker failure(s) caused solely by the documented final-boost transformer-identity check. The archived outputs establish the expected false identity and boost return policy, but do not retain transformer arrays for a direct value comparison.
Corrected measurement integrity: **pass**. Strict cross-arm trajectory matching: **fail**. Trajectory drift is retained as scientific evidence and does not suppress timing summaries.

Warm-only summaries exclude subsequent fits that generated a new graph or kernel.
Recorded cold total sums imports, compiler/MSVC setup, CUDA context, state rebuild, and the externally timed first complete fit. It excludes worker environment and manifest checks outside the API call, plus unrecorded process startup.

| Case | Arm | Cold fit s | Recorded cold total s | All subsequent median [range] | Warm-only median [range] |
| --- | --- | ---: | ---: | --- | --- |
| deterministic | numpy_cpu | 0.00482 | 0.179553 | 0.0043443 [0.0042983, 0.0046025] (n=3) | 0.0043443 [0.0042983, 0.0046025] (n=3) |
| deterministic | eager_cpu | 0.0207619 | 1.55775 | 0.0187597 [0.0186095, 0.0248918] (n=3) | 0.0187597 [0.0186095, 0.0248918] (n=3) |
| deterministic | compiled_cpu | 22.9865 | 27.6752 | 0.0086692 [0.0080982, 0.0107501] (n=3) | 0.0086692 [0.0080982, 0.0107501] (n=3) |
| deterministic | eager_cuda | 0.380377 | 2.3055 | 0.0755637 [0.0690495, 0.0759736] (n=3) | 0.0755637 [0.0690495, 0.0759736] (n=3) |
| deterministic | compiled_cuda | 11.9625 | 17.8435 | 0.0235976 [0.022984, 0.024896] (n=3) | 0.0235976 [0.022984, 0.024896] (n=3) |
| warped | numpy_cpu | 0.229055 | 0.425213 | 0.274071 [0.221169, 0.278263] (n=3) | 0.274071 [0.221169, 0.278263] (n=3) |
| warped | eager_cpu | 0.677427 | 2.44662 | 0.722153 [0.575667, 0.775601] (n=3) | 0.722153 [0.575667, 0.775601] (n=3) |
| warped | compiled_cpu | 85.5942 | 90.8347 | 0.573675 [0.416907, 0.575112] (n=3) | 0.573675 [0.416907, 0.575112] (n=3) |
| warped | eager_cuda | 0.99585 | 2.97638 | 1.15822 [0.779091, 1.17889] (n=3) | 1.15822 [0.779091, 1.17889] (n=3) |
| warped | compiled_cuda | 46.7793 | 52.82 | 0.40604 [0.386113, 0.585055] (n=3) | 0.40604 [0.386113, 0.585055] (n=3) |
| boost_sampled | numpy_cpu | 1.46693 | 1.66829 | 1.47976 [1.47921, 4.41416] (n=3) | 1.47976 [1.47921, 4.41416] (n=3) |
| boost_sampled | eager_cpu | 3.61827 | 5.31253 | 3.94396 [3.62755, 13.0312] (n=3) | 3.94396 [3.62755, 13.0312] (n=3) |
| boost_sampled | compiled_cpu | 64.5444 | 69.723 | 3.15828 [3.0578, 10.997] (n=3) | 3.15828 [3.0578, 10.997] (n=3) |
| boost_sampled | eager_cuda | 4.76137 | 6.79231 | 4.47835 [4.19501, 15.237] (n=3) | 4.47835 [4.19501, 15.237] (n=3) |
| boost_sampled | compiled_cuda | 105.62 | 111.817 | 1.19201 [1.16038, 2.83732] (n=3) | 1.19201 [1.16038, 2.83732] (n=3) |
| boost_single | numpy_cpu | 8.73724 | 8.94048 | 7.90248 [7.86969, 8.51795] (n=3) | 7.90248 [7.86969, 8.51795] (n=3) |
| boost_single | eager_cpu | 27.5072 | 29.1713 | 24.7178 [24.2787, 26.4456] (n=3) | 24.7178 [24.2787, 26.4456] (n=3) |
| boost_single | compiled_cpu | 444.355 | 449.315 | 30.3725 [21.2738, 31.7874] (n=3) | 30.3725 [21.2738, 31.7874] (n=3) |
| boost_single | eager_cuda | 32.6533 | 34.629 | 34.1214 [29.9715, 39.2768] (n=3) | 34.1214 [29.9715, 39.2768] (n=3) |
| boost_single | compiled_cuda | 121.538 | 128.029 | 5.13221 [5.11692, 5.24426] (n=3) | 5.13221 [5.11692, 5.24426] (n=3) |

## Paired hot-fit speedups

Speedup is reference wall time divided by compiled wall time for the same seed. The table retains all three subsequent fits, including recompilations.

| Case | Compiled arm | Reference | All subsequent median [range] | Warm-only median [range] |
| --- | --- | --- | --- | --- |
| deterministic | compiled_cpu | numpy_cpu | 0.530903 [0.399838, 0.536453] (n=3) | 0.530903 [0.399838, 0.536453] (n=3) |
| deterministic | compiled_cpu | eager_cpu | 2.14662 [1.74507, 3.07374] (n=3) | 2.14662 [1.74507, 3.07374] (n=3) |
| deterministic | compiled_cuda | numpy_cpu | 0.189014 [0.17265, 0.195041] (n=3) | 0.189014 [0.17265, 0.195041] (n=3) |
| deterministic | compiled_cuda | eager_cuda | 3.21955 [2.77352, 3.28767] (n=3) | 3.21955 [2.77352, 3.28767] (n=3) |
| warped | compiled_cpu | numpy_cpu | 0.483842 [0.477746, 0.5305] (n=3) | 0.483842 [0.477746, 0.5305] (n=3) |
| warped | compiled_cpu | eager_cpu | 1.34861 [1.25882, 1.3808] (n=3) | 1.34861 [1.25882, 1.3808] (n=3) |
| warped | compiled_cuda | numpy_cpu | 0.674985 [0.378032, 0.720679] (n=3) | 0.674985 [0.378032, 0.720679] (n=3) |
| warped | compiled_cuda | eager_cuda | 2.85247 [1.33165, 3.05323] (n=3) | 2.85247 [1.33165, 3.05323] (n=3) |
| boost_sampled | compiled_cpu | numpy_cpu | 0.468358 [0.401398, 0.483929] (n=3) | 0.468358 [0.401398, 0.483929] (n=3) |
| boost_sampled | compiled_cpu | eager_cpu | 1.18633 [1.18498, 1.24877] (n=3) | 1.18633 [1.18498, 1.24877] (n=3) |
| boost_sampled | compiled_cuda | numpy_cpu | 1.27523 [1.24093, 1.55575] (n=3) | 1.27523 [1.24093, 1.55575] (n=3) |
| boost_sampled | compiled_cuda | eager_cuda | 3.85938 [3.51927, 5.37021] (n=3) | 3.85938 [3.51927, 5.37021] (n=3) |
| boost_single | compiled_cpu | numpy_cpu | 0.260186 [0.247573, 0.400397] (n=3) | 0.260186 [0.247573, 0.400397] (n=3) |
| boost_single | compiled_cpu | eager_cpu | 0.813823 [0.763784, 1.24311] (n=3) | 0.813823 [0.763784, 1.24311] (n=3) |
| boost_single | compiled_cuda | numpy_cpu | 1.54438 [1.53339, 1.62424] (n=3) | 1.54438 [1.53339, 1.62424] (n=3) |
| boost_single | compiled_cuda | eager_cuda | 6.50642 [5.83988, 7.67587] (n=3) | 6.50642 [5.83988, 7.67587] (n=3) |

Numerical endpoint, decision, RNG, rescore, compiler, and amortization details are retained in `comparison.json`.
