# Wall time of a run: PyVBMC 1.0.4 against the release code

Written 2026-10-06. Arm 0's speed measurement
([plan](../plans/arm-1.0.4-comparison.md), "Speed"): the release code
against PyVBMC 1.0.4 as released, each on its own defaults, on the
`production` suite, run on the developer's machine one process at a time.
The evidence is in
[experiments/speed_v104_20261006/](../experiments/speed_v104_20261006/README.md).

## What was measured

Three campaigns of the population harness (`dev/scripts/population_run.py`)
on the `production` suite, prepared at seeds 0–2:

- **release**: the harness checkout's own package at `b196402b`, the commit
  that runs Arm 0 on the cluster, whose package is the after arm's, with
  gpyreg `v1.4.0`;
- **1.0.4**: the same harness checkout, its legacy profile running
  `v1.0.4` with gpyreg `v1.0.4` (and the profile's compatibility patch for
  NumPy 2.4 and later, two lines of indexing that cost no time);
- **Arm 3**: the harness checkout of Arm 3's branch at `ad63dd5a`, the
  release code with the end of warm-up that the port had before two of the
  port review's fixes moved it to MATLAB's
  ([Arm 3's plan](../plans/arm-3-warmup-comparison.md),
  [the warm-up note](2026-10-05-noisy-rosenbrock-warmup.md)), with gpyreg
  `v1.4.0`, on the eight noisy configurations only.

Each version runs on its own defaults with the harness's options: no
display, and for the release code and Arm 3 `performance_calibration="off"`,
which gives the settings that the package's default gives on a machine that
was never calibrated. Every case is one VBMC run in a fresh process of the
same Python 3.12.6 environment (NumPy 2.5.2, SciPy 1.18.1), one BLAS
thread. The time compared is each run's `wall_s`, the wall time of
`VBMC.optimize()`, which leaves out the harness's scoring.

A group is one configuration at one seed: its arms ran back to back, in an
order rotating from group to group, so that a change in the machine's speed
falls on every arm alike. Seed 0 ran every configuration,
`cigar_D15_exhaust` last; seeds 1 and 2 every configuration but
`cigar_D15_exhaust`, until a deadline of 6.9 hours, the time the machine
was left free, before which a group started only if 1.3 times its
estimated duration fitted. A fixed NumPy workload timed before each group
stayed between 0.78 and 0.98 seconds over the 64 groups, median 0.84: the
machine's speed did not drift. The machine is a laptop on mains power, with
a hybrid processor (Intel Core Ultra 7 155H) under the Balanced power plan.

All 146 runs completed, and the harness verified every one. The pairs:
every configuration at seeds 0 and 1, the noiseless ones and the two noisy
Rosenbrock targets also at seed 2, `cigar_D15_exhaust` at seed 0 alone (64
pairs); Arm 3 at the noisy configurations' 18 seeds. Six noisy groups of
seed 2 did not fit before the deadline.

## Results

The ratio is 1.0.4's wall time over the release code's for one
configuration and seed, averaged geometrically over a configuration's
seeds; above 1 the release code is faster.

| group | configurations | pairs | geometric mean of the configurations' ratios | ratio of total times | hours, 1.0.4 / release |
|---|---:|---:|---:|---:|---|
| noiseless | 16 | 46 | 3.70 | 3.69 | 2.40 / 0.65 |
| noiseless, 4 to 15 variables | 13 | 37 | 3.67 | 3.68 | 2.29 / 0.62 |
| noisy | 8 | 18 | 2.19 | 2.08 | 1.73 / 0.83 |

- **Noiseless targets**: the release code's runs take a fifth to a third
  of 1.0.4's time on every configuration but one (ratios 3.2 to 5.1),
  and half on `student_D4` (2.1), where the release code makes more
  evaluations (130 to 150 against 90 to 110). With 10 variables, a run of
  `banana_D10` takes 42 seconds against 172, and one of `lumpy_D10` 120
  against 464 (geometric means over the three seeds). The 750 evaluations
  of `cigar_D15_exhaust` take 589 seconds against 2044.
- **Noisy targets**: the release code's runs take a third to a half of
  1.0.4's time (ratios 1.96 to 3.01), but for `lumpy_D10_noise3_production`
  (1.25), where 1.0.4's run at seed 1 stopped after 180 evaluations and the
  release code's made 435. Per evaluation, the release code is 2.1 to 2.5
  times faster on every noisy configuration.
- **Arm 3**: on the noisy configurations its runs took 4 % longer than
  the release code's (geometric mean over the configurations; 0.92 to 1.18
  by configuration; 2.5 % on the total time), with as many iterations and
  evaluations in all (835 against 834 iterations and 4180 against 4160
  evaluations over the 18 runs): a difference within the spread of two or
  three seeds. Against 1.0.4, Arm 3's noisy runs take 2.1 times less time.
  Arm 3 ran the noisy configurations alone, so the noiseless figure holds
  if MATLAB's end of warm-up stays for noiseless targets, which Arm 3's
  decision 5 leaves open.

The figures differ from those of the changelog's current entry (two to
three times less time on noiseless targets, about 20 per cent less on
noisy ones), which were measured against another baseline and before the
port review's corrections; this note does not reconcile the two.

The table of every configuration is
[analysis.md](../experiments/speed_v104_20261006/analysis.md).

## For the changelog

The changelog's "Runs are faster" gave timings measured on 2026-09-03 to
09-05, before the port review's corrections. From these runs it reads, in
the PI's wording (2026-10-06):

> **Runs are faster.** On the 24 problems of our benchmark suite, with 2 to
> 15 variables, a run took on average 3.7 times less time than with the
> previous version of PyVBMC (v1.0.4) on noiseless targets and 2.2 times
> less on noisy ones, each version with its default options (one machine
> with one BLAS thread).

The "What's new" blocks of `README.md` and `docsrc/source/index.rst` give
the same two figures. The noisy figure is that of the release code's end of
warm-up, MATLAB's; if the port's end is adopted for noisy targets (Arm 3),
it becomes 2.1 in all three places.

MATLAB's end of warm-up stays for noisy and noiseless targets (PI,
2026-10-06; [the reading](2026-10-06-arm-3-reading.md)), so both figures
stand.

## Limits

- Two or three seeds per configuration (one for `cigar_D15_exhaust`): a
  configuration's ratio is uncertain, since a run's length varies with its
  seed (`lumpy_D10`'s release runs took 79 to 175 seconds); the group
  aggregates, over 16 and 8 configurations, are the figures to quote.
- Each version runs on its own defaults, so their runs differ in length as
  well as in speed per evaluation; both ratios are in the table.
- The order of the arms rotates from group to group. Seed 0 has 24 groups,
  a multiple of two and of three, so a configuration ran its arms in the
  same order at seeds 0 and 1: the rotation balances the order over the
  configurations, not within one, which suffices for the factors of 3.7
  and 2.2 but not for a difference of a few per cent.
- One machine. A hybrid processor schedules a process on cores of two
  speeds; the rotating order and the steady probe make a bias between the
  arms unlikely, but the absolute seconds are those of the laptop described
  above.
- 1.0.4 ran with the release's NumPy and SciPy, not with the versions of
  its own time.
