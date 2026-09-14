# Generating the S-VBMC run pools on a Slurm cluster

Scripts for the cluster half of the S-VBMC run-pool campaign
([plans/svbmc-benchmark-campaign.md](../../plans/svbmc-benchmark-campaign.md),
section "Cluster generation"; the brief is
[2026-09-14-svbmc-pool-handoff.md](../../2026-09-14-svbmc-pool-handoff.md)).
They implement the sbatch sketch in the module docstring of
[`svbmc_pool_run.py`](../svbmc_pool_run.py): `prepare` once on the login
node, one `worker` per Slurm array task, then `verify`, `select` and
`summarize`. Written for the University of Helsinki `kale` cluster (login
node `turso02`); every site-specific value is an environment variable.

## Prerequisites

- A clone of this repository at the commit the pool is to be generated
  with, **clean**, with the package installed editable in a conda
  environment together with the `test` extra, `gpyreg==1.2.1` and
  `psutil` (`pip install -e ".[test]" "gpyreg==1.2.1" psutil`), and with
  `zstd` for the archive (`conda install -n pyvbmc-pool zstd`; activating
  the environment drops the base environment's binaries from the PATH).
  The environment's name goes in `POOL_CONDA_ENV` (default `pyvbmc-pool`)
  and conda's hook in `CONDA_SH` (default
  `~/miniconda3/etc/profile.d/conda.sh`).
- A gpyreg checkout at the pinned tag (`v1.2.1`, commit `9e70e6b`) that is
  a real, clean git repository. Put it at the harness default
  `dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1` (gitignored): the
  test module `test_svbmc_pool_run.py` skips every test when that path is
  absent, so a checkout elsewhere makes the environment check vacuous.
  ```
  git clone https://github.com/acerbilab/gpyreg dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1
  git -C dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1 checkout --detach v1.2.1
  ```
- The environment check, under `srun` rather than on the login node
  (it fits two small VBMC runs):
  ```
  srun -p short -c 1 --mem=2G -t 00:20:00 bash -c \
    'REPO=$PWD; source dev/scripts/hpc/svbmc_pool_env.sh; python -m pytest dev/scripts/test_svbmc_pool_run.py -q'
  ```
  It must report tests passed and none skipped.

## Scripts

- `svbmc_pool_env.sh` — sourced by the others with `REPO` set: activates
  the conda environment, exports single-threaded BLAS and `MPLBACKEND=Agg`,
  and, when `POOL_DIR` holds a manifest, `PYVBMC_GPYREG_SOURCE` from it.
- `svbmc_pool_submit.sh POOL_DIR [prepare flags]` — refuses a dirty
  checkout, runs `prepare` when the directory holds no manifest (the
  flags go to `prepare`, whose suite defaults to `svbmc_pool`; a prepared
  campaign is never revised here), writes `cases.txt` once and refuses to
  overwrite it with a different list, and submits the array in chunks
  that stay below the site's `MaxArraySize` (the largest valid array
  index is one less, so chunk `k` carries `INDEX_OFFSET = k × (MaxArraySize − 1)`
  and array indices from 1). `ARRAY` names case indices, the lines of
  `cases.txt`, as an sbatch-style list of numbers and ranges and is
  mapped onto those chunks, one submission per chunk touched, so
  `ARRAY=1` is a canary and `ARRAY=17,233` a resubmission on any site.
  A task whose case already has a completion record exits at once, so
  resubmitting a range that includes finished cases is safe. `THROTTLE`
  (200), `TIME` (00:30:00), `MEM` (2G), `PARTITION` (short) and
  `SBATCH_EXTRA` tune the submission. Job ids go to
  `POOL_DIR/slurm/jobs.txt`.
- `svbmc_pool_task.sbatch` — the array task: line
  `INDEX_OFFSET + SLURM_ARRAY_TASK_ID` of `cases.txt`, one `worker` call
  with the required environment. Its output is
  `POOL_DIR/slurm/<jobid>_<array index>.out`; in a chunk with an offset
  the array index is the case index minus the offset, and the task's
  first line names its case.
- `svbmc_pool_finish.sh POOL_DIR [--no-archive] [--allow-missing]` —
  `sacct` for every recorded job into `slurm/sacct.txt`, `verify` under
  `srun` (`verification.json`; `VERIFY_TIME`, default 01:00:00, and
  `VERIFY_MEM`, default 2G, size that step), then `select`, `summarize`
  and the archive `<parent>/<name>.tar.zst` with its SHA-256 and size. It
  stops while tasks of the recorded jobs are still queued or running,
  when verification fails, or when cases are missing, printing the
  indices to resubmit; `--allow-missing` lets it go on past the queue
  check and the missing cases.

## A campaign, step by step

```
REPO=~/francesco_projects/pyvbmc          # a clean checkout at the campaign's commit
GPYREG=$REPO/dev/scripts/runs/svbmc_pool_20260913/gpyreg_1.2.1
POOL=$REPO/dev/scripts/runs/svbmc_pool_20260914
cd $REPO

# smoke test of the machinery (three short cases, a scratch directory)
SMOKE=$REPO/dev/scripts/runs/svbmc_pool_smoke
dev/scripts/hpc/svbmc_pool_submit.sh $SMOKE --gpyreg-source $GPYREG \
    --suite smoke --only normal_D2 --target 2 --max-seeds 3 --seed-start 4000
# ... wait for the three tasks ...
dev/scripts/hpc/svbmc_pool_finish.sh $SMOKE --no-archive
rm -rf $SMOKE

# the campaign: prepare, one canary task, then the rest
ARRAY=1 dev/scripts/hpc/svbmc_pool_submit.sh $POOL --gpyreg-source $GPYREG
# ... the canary's record appears under $POOL/records/ ...
ARRAY=2-1100 dev/scripts/hpc/svbmc_pool_submit.sh $POOL
squeue -u $USER -n svbmc_pool

# when the queue is empty
dev/scripts/hpc/svbmc_pool_finish.sh $POOL --no-archive --allow-missing
# resubmit what verify lists as missing (raise TIME/MEM if Slurm killed them)
ARRAY=<indices> TIME=01:00:00 dev/scripts/hpc/svbmc_pool_submit.sh $POOL
dev/scripts/hpc/svbmc_pool_finish.sh $POOL
```

## Rules that protect the data

- **Do not move the checkout while the array runs.** Every worker
  compares the identity fixed by `prepare`: the checkout's HEAD, the
  hashes of the suite module and both pool scripts, the gpyreg commit and
  the Python, NumPy and SciPy versions. A commit checked out, a pull, or
  an edit to those files makes every later task refuse; a harness fix
  therefore means a new commit and a fresh campaign directory (`prepare`
  refuses to change the identity of an existing one).
- **Only light commands on the login node**: `prepare`, `cases`,
  `select`, `summarize`, `sbatch`, `sacct`, `tar`. The tests and `verify`
  run under `srun`.
- **A failed case** leaves `<tag>.error.txt` and no artifact, exits
  non-zero, and is skipped by `select`; rerun it (`ARRAY=<index>`; a
  success clears the error file) or leave it out. **A task Slurm kills**
  (time limit, memory) leaves nothing; `verify` lists it as missing with
  its index. `select` and `summarize` cannot see such a case, so run
  `verify` before them.
- `--save-vbmc` is deliberately off: the comparison reads the `.npz` and
  `.json` artifacts only, and the pickles would multiply the archive.
- The machine holding the raw directory lists it in the gitignored
  `dev/scripts/runs/LOCAL.md`, with the two commands above that recreate
  the gpyreg checkout.

## Hand-back

The archive is uploaded as an asset of a draft release of this repository
and its name, size and SHA-256 recorded in the README under
`dev/experiments/svbmc_pool/pool_<date>/`, together with the tracked
copies of `manifest.json`, `selection.json`, `selection.md`,
`summary.json`, `summary.md` and `verification.json`. The manifest stores
the cluster's absolute gpyreg path; on another machine
`svbmc_pool_run.py verify --gpyreg-source PATH` overrides it, while the
stacking comparison (`svbmc_pool_stack.py --pool`) reads the manifest's
path and needs that one field edited to a local 1.2.1 checkout.
