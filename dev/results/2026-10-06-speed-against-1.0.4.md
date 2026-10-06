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
  NumPy 2.4 and later);
- **Arm 3**: the harness checkout of Arm 3's branch at `ad63dd5a` (the
  release code with the port's end of warm-up), with gpyreg `v1.4.0`, on
  the eight noisy configurations only.

Every case is one VBMC run in a fresh process of the same Python 3.12.6
environment (NumPy 2.5.2, SciPy 1.18.1), one BLAS thread. The time compared
is each run's `wall_s`, the wall time of `VBMC.optimize()`, which leaves out
the harness's scoring. A group is one configuration at one seed: its arms
ran back to back, in an order rotating from group to group, so that a
change in the machine's speed falls on every arm alike. Seed 0 ran every
configuration, `cigar_D15_exhaust` last; seeds 1 and 2 every configuration
but `cigar_D15_exhaust`, until a deadline of 6.9 hours, before which a
group started only if its estimated duration fitted. A fixed NumPy workload
timed before each group stayed between 0.78 and 0.98 seconds over the 64
groups, median 0.84: the machine's speed did not drift. The machine is a
laptop on mains power, with a hybrid processor (Intel Core Ultra 7 155H)
under the Balanced power plan.

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
  of 1.0.4's time on every configuration but one (ratios 3.2 to 5.2),
  and half on `student_D4` (2.1), where the release code makes more
  evaluations (130 to 150 against 90 to 110). With 10 variables, a run of
  `banana_D10` takes 42 seconds against 172, and one of `lumpy_D10` 120
  against 464. The 750 evaluations of `cigar_D15_exhaust` take 589 seconds
  against 2044.
- **Noisy targets**: the release code's runs take a third to a half of
  1.0.4's time (ratios 1.96 to 3.01), but for `lumpy_D10_noise3_production`
  (1.25), where 1.0.4's run at seed 1 stopped after 180 evaluations and the
  release code's made 435. Per evaluation, the release code is 2.1 to 2.5
  times faster on every noisy configuration.
- **Arm 3**: on the noisy configurations its runs take 4 % longer than the
  release code's (geometric mean; 0.92 to 1.18 by configuration), in line
  with the few more iterations that the port's end of warm-up gives a
  noisy run. If the port's end is adopted for noisy targets, their speed-up
  against 1.0.4 is about 2.1 rather than 2.2.

The table of every configuration is
[analysis.md](../experiments/speed_v104_20261006/analysis.md).

## For the changelog

The changelog's "Runs are faster" gives timings measured on 2026-09-03 to
09-05, before the port review's corrections (`dev/TODO.md`, "Changelog for
1.5"). Against 1.0.4, from these runs, the entry could read:

> **Runs are faster.** On our benchmark problems a run took about a quarter
> of the time of PyVBMC 1.0.4 on noiseless targets (42 seconds against 172
> with 10 variables) and about half on noisy targets, each version with its
> default options, on one machine with one BLAS thread.

The wording is the PI's to settle, with the "What's new" blocks of
`README.md` and `docsrc/source/index.rst` that move with it; the noisy
figure moves to about 2.1 if the port's end of warm-up is adopted.

## Limits

- Two or three seeds per configuration (one for `cigar_D15_exhaust`): a
  configuration's ratio is uncertain, since a run's length varies with its
  seed (`lumpy_D10`'s release runs took 79 to 175 seconds); the group
  aggregates, over 16 and 8 configurations, are the figures to quote.
- Each version runs on its own defaults, so their runs differ in length as
  well as in speed per evaluation; both ratios are in the table.
- One machine. A hybrid processor schedules a process on cores of two
  speeds; the rotating order and the steady probe make a bias between the
  arms unlikely, but the absolute seconds are this machine's.
