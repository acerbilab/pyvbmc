# Hand-off: generating the S-VBMC run pools on the cluster

Written 2026-09-14 for the developer (and their coding agents) who will
generate the pools on the HPC cluster. Everything referenced is on
`dev-next` from commit `1903506`.

## What this is and why

S-VBMC stacks the posteriors of several independent VBMC runs. For PyVBMC
1.5 we need, for eight target conditions, a **pool** of such runs, each
saved with its final posterior *and* the Gaussian process behind it, so
that two things can be done on a laptop afterwards: a matched comparison
of the integrated `pyvbmc.svbmc` against the original standalone package,
and an experiment on an "honest" cross-run estimate of the stacked ELBO
that needs the runs' GPs. No retained artifact in the repository carries a
posterior together with its GP, so the pools are generated fresh. They are
a once-in-a-while golden fixture, expensive enough (about 1100 VBMC runs,
one to six minutes each, roughly 45 CPU-hours) to belong on the cluster;
the analyses that consume them stay local.

The design, decisions and status live in
[plans/svbmc-benchmark-campaign.md](plans/svbmc-benchmark-campaign.md);
the section **"Cluster generation"** there is the specification of this
job, and the module docstring of `dev/scripts/svbmc_pool_run.py` carries
the sbatch sketch. The campaign is authorized by the PI (decision 8 of the
plan); you do not need a further go-ahead to run it.

## What we need from you

1. A clone of this repository at `dev-next` (commit `1903506` or later),
   with the package installed, `psutil` and `filelock` available, and a
   **gpyreg checkout at tag `v1.2.1`** (commit `9e70e6b`) in a sibling
   directory. No Torch and no original-svbmc checkout are needed on the
   cluster. Both checkouts must be real git repositories with clean
   trees; the harness records their commits and refuses to mix code
   states. Confirm the environment with
   `python -m pytest dev/scripts/test_svbmc_pool_run.py -q` (about a
   minute; it runs two tiny VBMC fits).
2. **On the login node**, prepare the campaign in a directory under
   `dev/scripts/runs/` (gitignored):
   ```
   python dev/scripts/svbmc_pool_run.py prepare --out dev/scripts/runs/svbmc_pool_<date> \
       --suite svbmc_pool --gpyreg-source <path to the v1.2.1 checkout> \
       --ready --authorized-by "Luigi Acerbi (PI, 2026-09-14)"
   ```
   The defaults are the approved allocation (100 filtered runs per noisy
   condition, 50 per noiseless control, seed caps 150, 200 for the ring,
   75 for the controls). `prepare` must run on the cluster: the manifest
   stores the gpyreg path as given.
3. Enumerate the cases and submit them as a Slurm array, one case per
   task, one core and under 2 GB each, following the sketch in the
   runner's docstring: `cases` writes the `(label, seed)` list (1100
   lines; Slurm's default array limit is 1001, so submit in two chunks or
   with a throttle), and each task runs
   `svbmc_pool_run.py worker --out DIR --label L --seed S` with
   `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`,
   `MPLBACKEND=Agg` and `PYVBMC_GPYREG_SOURCE` set. Do **not** use the
   `run` subcommand on the cluster; it is the sequential laptop
   supervisor. A failed task leaves `<tag>.error.txt` and exits
   non-zero; rerun it or leave it out.
4. When the array is done, `select --out DIR` writes the filtered pool
   (`selection.json`, the lowest-seed runs that pass the filters up to
   each condition's target) and `summarize --out DIR` the summary. Then
   verify every artifact post hoc:
   ```
   PYVBMC_GPYREG_SOURCE=<gpyreg path> python -c "
   import sys, glob; sys.path.insert(0, 'dev/scripts')
   import svbmc_pool_run
   from svbmc_pool_io import verify_run
   for p in sorted(x[:-4] for x in glob.glob('dev/scripts/runs/svbmc_pool_<date>/*.npz')):
       verify_run(p)
   print('all artifacts verify')"
   ```
   (A `verify` subcommand wrapping this would be a welcome addition.)
5. Hand back **two things**. First, the whole campaign directory
   (artifacts, `records/`, `manifest.json`, `selection.json`, summaries,
   logs) as one archive (`tar.zst` or zip, about 100–150 MB). The PI has
   no direct access to the cluster, so upload it as an asset of a
   **draft release** of this repository (`gh release create
   svbmc-pool-<date> --draft`, then `gh release upload svbmc-pool-<date>
   <archive>`; a draft is not public, and a single asset may be up to
   2 GiB) and record the asset's name, size and SHA-256 in the README of
   your PR (below). It is downloaded here into `dev/scripts/runs/`,
   where it stays gitignored; every artifact's hash is already in its
   completion record, so the copy is verified before anything runs on
   it. When the pool is promoted the draft is published, as the golden
   traces are. Second, a **pull request to
   `dev-next`** from a branch you create off `dev-next` (for example
   `dev-svbmc-pool-hpc`), containing: your sbatch scripts (under
   `dev/scripts/hpc/` or a similar new directory, documented in
   `dev/README.md`), the tracked copies of `manifest.json`,
   `selection.json`, `selection.md`, `summary.json` and `summary.md` under
   `dev/experiments/svbmc_pool/pool_<date>/`, a short README there in the
   style of `dev/experiments/svbmc_pool/README.md`, and any fix to the
   harness you needed, each with a test. We review the PR here. Do not
   commit raw artifacts, and do not touch `pyvbmc/` or
   `dev/scripts/benchmark_targets.py` while the campaign runs (the
   identity check would refuse the later cases).

## Things to know

- The two conditions `student_D8_noise3_svbmc` and
  `multisensory_s1_D6_svbmc` were added last; a one-seed end-to-end check
  of each was run on 2026-09-14 (see the plan's worklog for the result).
  The other six ran in a three-seed pilot on the laptop.
- Runs on Linux with the cluster's BLAS will not reproduce laptop runs bit
  for bit. That is expected for a pool; the records carry the platform.
- Working rules for this repository are in `AGENTS.md` and the "Scripts"
  section of [dev/README.md](README.md): conventional commits, formatting
  by the pre-commit hook, nothing written into `docs/`, results
  summarized in the plan's worklog rather than committed raw.
- Questions about the design go to the plan; questions about the
  harness to the module docstrings of `svbmc_pool_run.py`,
  `svbmc_pool_io.py` and `svbmc_pool_stack.py` (the last one is the local
  comparison and is not needed on the cluster).
