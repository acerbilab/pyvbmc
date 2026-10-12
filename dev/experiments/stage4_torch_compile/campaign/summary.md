# Compiled Torch Stage 4 campaign status

Status: **completed_with_failures**

| Case | Arm | Status | Cold fit s | Hot fit s | Recompilations |
| --- | --- | --- | ---: | ---: | ---: |
| deterministic | numpy_cpu | ok | 0.004820000031031668 | 0.0046025, 0.0043443, 0.0042983 | 0 |
| deterministic | eager_cpu | ok | 0.02076189999934286 | 0.0186095, 0.0248918, 0.0187597 | 0 |
| deterministic | compiled_cpu | ok | 22.986549000022933 | 0.0086692, 0.0080982, 0.0107501 | 0 |
| deterministic | eager_cuda | ok | 0.3803774000843987 | 0.0759736, 0.0755637, 0.0690495 | 0 |
| deterministic | compiled_cuda | ok | 11.962511499994434 | 0.0235976, 0.022984, 0.024896 | 0 |
| warped | eager_cpu | ok | 0.6774269000161439 | 0.575667, 0.722153, 0.775601 | 0 |
| warped | compiled_cpu | ok | 85.59419610002078 | 0.416907, 0.573675, 0.575112 | 0 |
| warped | eager_cuda | ok | 0.9958495999453589 | 0.779091, 1.15822, 1.17889 | 0 |
| warped | compiled_cuda | ok | 46.77929249999579 | 0.585055, 0.40604, 0.386113 | 0 |
| warped | numpy_cpu | ok | 0.22905460000038147 | 0.221169, 0.274071, 0.278263 | 0 |
| boost_sampled | compiled_cpu | gate_failed | 64.54436629998963 | 3.0578, 3.15828, 10.997 | 0 |
| boost_sampled | eager_cuda | gate_failed | 4.761369499959983 | 4.47835, 4.19501, 15.237 | 0 |
| boost_sampled | compiled_cuda | gate_failed | 105.61980300000869 | 1.16038, 1.19201, 2.83732 | 0 |
| boost_sampled | numpy_cpu | gate_failed | 1.4669300999958068 | 1.47976, 1.47921, 4.41416 | 0 |
| boost_sampled | eager_cpu | gate_failed | 3.6182680000783876 | 3.62755, 3.94396, 13.0312 | 0 |
| boost_single | eager_cuda | gate_failed | 32.653283399995416 | 34.1214, 29.9715, 39.2768 | 0 |
| boost_single | compiled_cuda | gate_failed | 121.53786529996432 | 5.24426, 5.13221, 5.11692 | 0 |
| boost_single | numpy_cpu | gate_failed | 8.737242699950002 | 8.51795, 7.86969, 7.90248 | 0 |
| boost_single | eager_cpu | gate_failed | 27.507163700065576 | 26.4456, 24.2787, 24.7178 | 0 |
| boost_single | compiled_cpu | gate_failed | 444.35537829995155 | 21.2738, 31.7874, 30.3725 | 0 |

Failures are retained in `campaign.json` and worker artifacts.
