# Machine-local PDF and entropy chunk calibration

## Package integration

The approved facility is implemented on `dev-machine-calibration` for PyVBMC
1.5. `pyvbmc.calibrate()` runs a fresh campaign on every explicit call, prints
progress and a summary, and saves a compatible machine/environment profile.
Normal runs only load cached settings or historical defaults. Profiles remain
fixed through early posterior use, optimization, whitening, final boost and
save/resume. Detailed reports stay in the cache; saved runs retain compact
settings and provenance. See the [public API](../../docsrc/source/api/functions/calibrate.rst)
and [integration artifacts](../experiments/machine_calibration/integrated/README.md).

Two corrected package campaigns retained **65,536 for PDF, entropy gradients
and entropy values**. Neither found a repeatable gain meeting the discovery
and control gates, so there is no measured tuning benefit or break-even point
on this machine. Keeping the default was a complete, persisted result in both
campaigns. Each call finished its prescribed work beyond the 30-second estimate.

| Seconds | Fresh campaign | Explicit rerun |
|---|---:|---:|
| Public API, including validation/persistence | 42.39 | 43.53 |
| Package import, separately measured | 1.41 | 1.28 |
| Workload setup | 0.047 | 0.032 |
| Numerical diagnostics | 3.92 | 3.38 |
| Discovery | 28.48 | 29.92 |
| Held-out measurements | 9.75 | 10.05 |

A fresh process read the cached profile in 0.141 s including lazy cache
imports; twelve further lookups had median 0.00633 s. Already-resolved kernels
perform no cache reads. The measured environment was Python 3.12.6, NumPy
2.5.2, SciPy 1.18.1, gpyreg 1.1.0 and single-threaded OpenBLAS on the Windows
laptop described below. Reports preserve the installed editable version string;
the retained file hashes and `f7f0ce1` baseline identify the actual source.

All PDF checks were exact. Maximum absolute entropy differences were
3.55e-15 for gradients and 7.11e-15 for values; every tested budget preserved
RNG advancement, inputs, VP state and float64. Global NumPy RNG state was
unchanged. Incremental allocation tracing peaked at 25,567,912 bytes
(24.38 MiB); the largest analytical estimate was 43,545,832 bytes (41.53 MiB),
below the 64 MiB campaign safeguard. These are incremental allocations and
estimates, not process RSS measurements. Review corrected the estimate to
include overlapping temporaries, reduction workspace and allocation metadata.

Paired baseline/current public-kernel probes found roughly 1 to 3 microseconds of
added cost for a tiny PDF call (about 25 to 28 microseconds total). Larger cases
were variable; several same-version controls differed by more than 5%, so
these probes do not establish a whole-fit performance gain or regression.
The raw twelve-round four-arm timings and controls are retained. Allocation
tracing ran separately from performance measurements.

Verification: 82 calibration tests pass; all 11 exact oracle fixtures
pass; the five default-profile seed-0 replays match stored loops, initial
designs and final results exactly. With all budgets fixed at 16,384, initial
designs remain exact and all finals finite, while entropy rounding changes
trajectories as expected. Four cases satisfy every historical accuracy fence.
The fifth, normal_D5, already exceeds the old gsKL fence in the prechange
baseline (0.000491 versus 0.000429); calibration changes that gsKL by only
2.22e-16. The raw report retains its flag rather than relabeling it a pass.
Historical traces omit returned transformer state, which remains explicitly
uncertified by these replay files.

Independent Sol cross-component review has no remaining findings. It corrected
timing order, cache validation consistency and fallback feedback, and added
focused whitening/undo and boost coverage. Preliminary 36.80/46.78-second
campaigns are retained with an exclusion label because their discovery/control
and held-out pair orders were unbalanced. A failed initial coarse-clock probe
never persisted; kernel timings now use `perf_counter` separately from the
monotonic watchdog. No 870-case benchmark or S-VBMC work was launched.

The full suite passed 1,140 tests with 35 skips in 337.94 s (three existing
log-PDF derivative warnings, no reruns). After the final metadata-only manual
profile provenance correction, all 93 calibration and existing seeded-run
checks passed. Wheel/sdist builds and isolated target wheel import/PDF/entropy
checks pass. The packaging check also corrected existing package-data rules
so test files ship only in the sdist. Full docs build with one existing Plotly
MIME warning; the final incremental build has no warnings. A second
CPU/stack is unavailable in this workspace; other-machine performance and the
3-OS CI matrix remain release validation, with no claimed portable speedup.

## Initial developer experiment

The bounded first experiment is complete. **Neither of two balanced sweeps
found a repeatable improvement large enough to replace the current
65,536-element budget on this machine.** All nine selections kept the
default. PDF comparisons were exact; entropy changed at rounding level
with identical RNG advancement. The full synthetic grid costs substantially
more than a few seconds to calibrate.

These initial experiments predate package integration. The approved package
implementation now provides the explicit calibration API and cache, and preserves
fixed settings through the run lifecycle. The integration plan records the PI's explicit-campaign, feedback and fixed-per-run decisions;
about 30 seconds is an acceptable estimate for occasional calibration, not
a deadline. Only excessive runtime warrants a generous watchdog. Its API/cache
details are implemented and locally verified for 1.5.
Astra orchestrated; Sol
implemented and independently reviewed. The [plan](../plans/machine-local-calibration.md)
and [artifact instructions](../experiments/machine_calibration/README.md)
retain scope and reproduction details.

## Comparison and selection rule

The developer runner derives private functions from current source, changing
only the chunk budget. PDF timings include the public input adapter. Entropy
uses copied globals, never a production monkeypatch. Source guards/hashes
identify the measured functions, and the entropy guard checks the production
module-level default. All synthetic VPs and entropy calls use private RNGs.

The candidates are 2^14, 2^15, 2^16, 2^17 and 2^18. Equivalent chunk layouts
are measured once, preferring the default as representative. Inputs, sample
counts, gradients and paired entropy draws are fixed within each workload.
Held-out timings use different synthetic inputs. These are coverage cases,
not a solver call-frequency model.

The predeclared exploratory rule requires median paired speedup >=1.10 and
wins in >=80% of selection rounds; otherwise it keeps the default. Fresh
held-out comparisons use selected/default arms and a second copy of each
setting as timing controls. **No nondefault candidate passed selection in
either balanced sweep.** All held-out arms are consequently same-setting
controls; their fluctuations are not tuning gains.

The primary uses five selection/four held-out rounds. The repeat reverses
the candidate list and uses ten/eight rounds. Each complete cycle places
every arm in every position once; direction reverses between full cycles.
First calls and additional warmup are separate from selection rounds;
held-out arms also receive warmup.

An initial run used rotation plus reversal within each cycle, which review
found was position-imbalanced. Its source/JSON remain as
`initial_unbalanced*`, excluded from timing conclusions. It also kept every
default. Windows formatter workers stalled during setup; they were stopped,
and absence of Python helpers was verified before the balanced primary.
Numerical measurements ran sequentially in the main thread. No full optimizer,
S-VBMC job or 870-case benchmark was launched.

## Timings and benefit

Python 3.12.6, NumPy 2.5.2, SciPy 1.18.1, GPyReg 1.1.0; Windows x86-64.
CPU identifier: `Intel64 Family 6 Model 170 Stepping 4, GenuineIntel`.
Both OpenBLAS libraries report one thread and Haswell kernels: NumPy
0.3.34.0.0, SciPy 0.3.31.dev. Raw JSON retains fuller metadata.

The table shows median milliseconds for the held-out **default** arm.
PDF performance is value-only; gradient correctness is checked separately.
Entropy counts are per component, rounded up to even.

| Workload | D / K | Rows or samples/component | Primary ms | Repeat ms |
| --- | ---: | ---: | ---: | ---: |
| PDF acquisition batch | 4 / 10 | 8 rows | 0.0238 | 0.0234 |
| PDF sieve | 4 / 20 | 8,192 rows | 4.07 | 4.11 |
| PDF large density call | 15 / 26 | 100,000 rows | 135.17 | 139.35 |
| Entropy Adam, all gradients | 4 / 20 | 38 | 0.369 | 0.373 |
| Entropy boost, all gradients | 4 / 50 | 56 | 2.60 | 2.42 |
| Entropy boost, all gradients | 15 / 50 | 56 | 6.76 | 7.12 |
| Entropy fine, value-only | 4 / 20 | 4,096 | 24.25 | 35.57 |
| Entropy fine, value-only | 15 / 26 | 4,096 | 103.26 | 193.31 |
| Entropy active, all gradients | 4 / 20 | 200 | 1.71 | 1.75 |

All nine settings stayed at 65,536 in both sweeps. Adam requests
`ceil(100 K^(-1/3))` = 37 at K=20, then uses 38; boost requests
`ceil(200 K^(-1/3))` = 55 at K=50, then uses 56. The historical plan's
approximate K=50 count of 54 is not the measured count.

The PDF sieve's 32,768 candidate is only about 1.5-3.7% faster by the ratio
of selection medians. At the large PDF shape, primary selection medians
are 136.93 ms at 65,536, 140.70 ms at 32,768 and 326.86 ms at 131,072.
High-D fine entropy similarly takes 105.09 ms at 65,536 versus 238.36 ms
at 131,072. Large chunks remain a clear loss on this machine. Tiny PDF
calls have one chunk at every candidate budget, hence no choice to improve.

Absolute variability is material: the repeat's fine-entropy controls are
much slower than the primary's, with no established cause. Same-budget
paired median ratios span about 0.92-1.12 in the primary and 0.93-1.02 in
the repeat; individual small-call ratios reach 0.42-2.43. Every observation
is retained. These controls limit small-improvement claims. They do not
establish a universal optimum, and no solver-wide speedup is inferred.

## Startup cost and fallback

| Cost, seconds | Primary | Repeat |
| --- | ---: | ---: |
| Fresh-process imports measured inside Python | 1.237 | 1.227 |
| Initial source inspection/hashing and clone setup | 0.030 | 0.030 |
| Discovery: synthetic setup, first calls, warmups, selection rounds | 15.009 | 27.144 |
| Numerical diagnostics, held-out validation, allocation tracing | 8.616 | 17.233 |
| Total after imports, including metadata/report overhead | 23.739 | 44.489 |

Totals exclude interpreter launch/teardown and final output writing. Fresh
process does not imply an empty OS file cache. Clone compilation is prototype
infrastructure, not a measured future production startup cost. Selection
input construction costs 0.020/0.022 s, included in discovery; warmups plus
selection rounds account for 12.837/24.993 s. Raw artifacts retain first
calls per candidate: primary default first calls include 0.0656 ms for the
PDF batch, 140.14 ms for large PDF, 3.18 ms for D4 boost entropy and
107.52 ms for D15 fine entropy.

Most primary discovery cost comes from large PDF (7.56 s) and high-D fine
entropy (5.65 s); the other seven cases together cost about 1.80 s. This
full grid does not fit a few-second first-use procedure. With no nondefault
selection, calibration has **no demonstrated repayment**. The summary
script returns null amortization for same-setting controls even when timing
noise gives one copy an apparent saving.

A separate **2-second discovery allowance**, with a 5-second total allowance,
returned `partial_defaulted` after **3.20 s after imports**, plus 1.27 s of
imports. It began three PDF cases, did not reach entropy, and retained all
nine defaults. Timing loops check limits between calls; grouped numerical
diagnostics can overrun discovery allowances. This is a soft experimental
limit, not a strict 2-second startup guarantee. Both full sweeps completed
inside their 90-second discovery/180-second overall allowances. Production
deadline accounting must check individual diagnostics as well. The subsequent
explicit ~30-second campaign design does not require narrower proxy workloads
to satisfy the earlier two-second hypothesis.

## Numerical behavior and memory

Each balanced sweep passes **67 numerical comparisons**: 45 actual-workload
candidate checks plus 22 dedicated default-clone, partial-chunk, gradient/
Jacobian and original-coordinate PDF comparisons. All checked outputs are
finite float64. PDF values, log values and input gradients are exact where
compared. Default entropy clones are exact; across budgets the largest
actual-workload entropy difference is **7.11e-15**, and the largest
actual-workload gradient difference is **1.33e-15**. Dedicated adverse-mixture
checks differ by at most 3.56e-15 over their outputs. Entropy uses
`rtol=1e-12, atol=1e-13`; the absolute floor covers near-zero gradient
cancellation. No relative-accuracy claim is made for near-zero entries;
per-block conditioning is not exhaustively characterized.

Every tested entropy candidate consumes exactly the production baseline's
private RNG draws. VP parameters, checked input arrays and VP generators
remain unchanged; the whole campaign also leaves NumPy's global RNG
unchanged. Rounding-level fixed-input agreement does not imply identical
optimizer trajectories. Neither changed trajectories nor PDF-gradient
performance was measured.

At the unchanged budget, separate allocation traces observe peaks of
**13.87 MiB for large PDF**, **24.38 MiB for D15 fine entropy**, **1.68 MiB
for D4 boost entropy**, and **1.98 MiB for active entropy**. The 0.5 MiB
distance tensor is only one allocation. Entropy constructs full antithetic
samples before chunk processing; gradient expressions also retain multiple
arrays. These are Python/NumPy allocator traces, not RSS or hard caps. The
runner's analytic workspace formulas are partial tallies: some gradient
cases undercount the trace by nearly twofold. Use traced observations for
this report's memory evidence; do not use the formulas as memory limits.

Validation also includes **11 focused runner tests** and **8 existing
entropy/PDF-gradient tests**, including finite differences and MATLAB entropy
comparison. Independent Sol review identified the timing-order repair and
reviewed retained evidence. Production code/oracle fixtures were not modified
or re-baselined.

## Integration follow-up

The PI clarified that this work is to plan calibration as a PyVBMC feature.
An occasional campaign taking roughly 30 seconds is acceptable as an estimate,
not a limit: complete the work, with a generous watchdog only for excessive
runtime (e.g. five minutes). Give feedback and allow explicit initiation/reruns;
a two-second proxy is not a prerequisite. The
[integration plan](../plans/machine-local-calibration.md) specifies proposed
API, cache location/schema/invalidation, bounded campaign checks, fallback
and lifecycle handling. Normal optimization uses a valid profile or defaults
without starting a campaign. Settings remain fixed per run, including final
boost and save/resume. Entropy rounding differences are checked and recorded;
different profiles need not produce identical seeded trajectories. There is
no separate unresolved entropy policy gate. A different CPU/stack remains
useful validation; this machine's default retention is a valid outcome.

S-VBMC integration remains parked for its main developer's review; the final
integrated benchmark remains unlaunched.
