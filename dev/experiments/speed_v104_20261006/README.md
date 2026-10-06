# Wall time of PyVBMC 1.0.4 against the release code (2026-10-05/06)

The evidence of Arm 0's speed measurement
([plan](../../plans/arm-1.0.4-comparison.md), "Speed"); the reading is in
[results/2026-10-06-speed-against-1.0.4.md](../../results/2026-10-06-speed-against-1.0.4.md).

- `analysis.md`: the tables, per configuration and per group.
- `analysis.json`: the same, with each pair's wall times, evaluation counts
  and ratios (`rows`), the group aggregates (`groups`) and the machine
  probe (`probe`).
- `events.jsonl`: the driver's events in order: each case's arm,
  configuration, seed, exit code, seconds including the harness's worker
  and `wall_s` of its sidecar; the probe before each group; the groups
  skipped at the deadline.

The three campaigns of the population harness (`release`, `v104`, `arm3`),
their verification reports, the driver and the analysis script are raw
runs, kept on the machine that `dev/scripts/runs/LOCAL.md` lists
(`speed_20261005/`).
