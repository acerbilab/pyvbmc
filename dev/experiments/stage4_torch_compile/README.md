# Stage 4 compiled-Torch experiment artifacts

This directory preserves the bounded follow-up that compared the completed
NumPy and eager-Torch Stage 4 prototype with a scoped `torch.compile` adapter.
The compiled arms replace only the ordinary optimization objective and its
backward pass. Full-variance and per-component fine scoring deliberately use
the unchanged eager Torch implementation because compiling the largest static
fine-score loop would unroll thousands of Python iterations.

## Layout

- `setup/` contains the installation manifest and all five CPU/CUDA compiler
  smoke attempts, including failed attempts. The successful smoke JSON files
  are the preflight inputs referenced by the campaign.
- `campaign/` contains the incremental coordinator files (`campaign.json`,
  `workers.jsonl`, and `summary.md`). Its `runs/` children preserve each fresh
  process's log, summary, JSON/NPZ artifact pair, and incremental NPZ
  checkpoints. Arrays referenced by a worker's `result.json` are stored in the
  named NPZ archive and carry shape, dtype, and SHA-256 metadata.
- `campaign/source/manifest.json` hashes the exact source files used by the
  measured workers. `measured-source.zip` contains those bytes.
- `comparison/` contains the post-campaign machine-readable comparison and its
  compact Markdown rendering. The comparison reads only the archived worker
  artifacts and verifies every JSON/NPZ array reference before analysis.

Each case/backend worker is a fresh process with a unique compiler cache. It
records one cold seed-1701 complete fit followed by same-process fits for seeds
1701, 1702, and 1703. `fit.timing.total` is the primary clean complete-fit wall
time; the separately retained API wall also includes source checking around
`run_step`. Compiler counters are sampled outside measured fit walls, and a
subsequent fit that generates a graph or kernel is labeled `recompiled` rather
than warm. Fixed-input value/gradient gates and independent rescoring occur
only after all measured fits.

Compiler caches are intentionally absent. They lived under the ignored
`dev/scripts/runs/tcc/<worker-token>/` paths during execution. Reproduction
therefore starts from fresh caches and should use a new output directory.

## Measured source and the identity-gate correction

The source ZIP is authoritative for the campaign. The runner archived there
required `result.transformer_identity` to be true for every workload. That is
correct for ordinary variational fits, but not for final boost: production
intentionally optimizes and returns an independent deep copy of the pre-boost
posterior, while the outer `VBMC` caller restores live-state ownership. The
completed Stage 4 controls already record false identity for both final-boost
workloads and both numerical backends.

Consequently, the raw campaign records final-boost workers as `gate_failed`
even when completion, mode, policy, finite-state, kernel, and compiled-backward
checks all passed. `comparison/comparison.json` preserves those original
statuses and reclassifies only this single known harness predicate. The
reclassification is allowed only when every retained fit is a complete
production final boost, identity is false, a boost result is present, all
other fit predicates pass, and the kernel and compiled-backward gates pass.
No measured value, timing, decision, RNG state, or raw failure record is
changed. The post-run runner fix changes only the future expectation to false
for final boost and true otherwise; it is not represented as measured source.

## Reproduction

For byte-level reproduction, extract `campaign/source/measured-source.zip`
over a clean compatible checkout and use the pinned fixture/dependency inputs
recorded by the manifests. Run the coordinator from the repository root with
fresh output and cache paths:

```console
python dev/scripts/torch_vi_compile_benchmark.py --phase campaign --out <fresh-output> --cpu-overlay <cpu-torch-overlay> --cuda-overlay <cuda-torch-overlay> --compiler-overlay <triton-overlay> --cpu-smoke <cpu-smoke.json> --cuda-smoke <cuda-smoke.json>
python dev/scripts/torch_vi_compile_report.py --campaign <fresh-output>/campaign.json --out <fresh-comparison-output>
```

The coordinator enforces 30 minutes per case/backend worker and three hours for
the campaign, uses one BLAS/Torch thread, executes backend arms in rotated
order, checkpoints each outcome, and does not retry failures. CUDA reproduction
requires the compatible Triton-Windows overlay recorded by setup; its bundled
CUDA directory is exposed through process-local `CUDA_PATH`.

The successful smoke artifacts were produced before two parent-launcher-only
hardening edits: refusal to reuse a partial cache directory and process-tree
termination after timeout. The child compile test, MSVC activation, Torch
overlay, and compiler path did not change. Smoke JSON records and hashes are
preserved, but those files do not contain the toolchain script SHA-256, so they
do not establish byte-identical provenance for the launcher script itself.
