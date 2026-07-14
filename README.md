# nilpotent_butterfly_custom_kernels

Self-contained copy of the kernels needed to profile the **two_stage_swz** path (the fastest
two-stage generalized Givens butterfly variant from `benchmarks/profile_givens_twostage.py`):

```
x -> blkdiag1 (swz_s1) -> Givens G1 (rot_fixed) -> blkdiag2 (swz_s2) -> Givens G2 (rot_fixed) -> y
```

## Contents

- `readonce_swizzle/` — CUTLASS-based read-once swizzle block-diagonal GEMM (`readonce_swz_ext`:
  `swz_s1`, `swz_s2`). Batched high-occupancy structure with a batch-fast threadblock swizzle so
  the 64-col window overlap hits in L2 (x read ~1x from DRAM). Pipeline depth defaults to 4 stages
  (override at build time with `SWZ_STAGES`).
- `givens_colmajor/` — column-major per-cycle Givens rotation (`givens_colmajor_ext`; the profiled
  entry point is `rot_fixed`, register-resident with compile-time pair indices, ~memory floor).
- `fused_stage_gemm_rot/` — the FUSED stage kernel (`fused_stage_gemm_rot_ext`): all 32 blocks of
  the block-diagonal GEMM + both Givens layers (tau+rho) applied in a full-width 32x2049 fp16
  shared-memory buffer, so the (2048, M) intermediate never round-trips through HBM. Correct but
  ~10x slower than the two-kernel path: the full-row buffer caps occupancy at 1 CTA/SM with a
  single warp. `kernel_cfg.cu` is the config-swept variant (`fused_stage_gemm_rot_cfg`).
- `parallel_stage/` — the CO-RESIDENT rotation (`parallel_stage_ext`: `rot_persist`): a
  persistent-CTA Givens rotation launched first with a fixed grid (~1 CTA/SM, register budget
  sized to fit alongside a BDMM CTA) so a concurrently launched BDMM backfills the same SMs.
  Not a fusion — two kernels genuinely co-scheduled per SM. `RESULTS.md` has the experiment log.
  Known result: under the A100-PCIe 250 W cap the race never meaningfully beats sequential.
- `bdmm_only/` — the CUTLASS block-diagonal GEMM (`bdmm_ext`: `bdmm`, `bdmm_cfg`, `*_into`
  variants with swept compile-time tiles); the producer that `rot_persist` races against.
- `givens_perm.py`, `colmajor_setup.py` — pure-Python helpers: the Monarch permutation (`perm1`)
  and construction of the rotation tensors in both row (plane) and col (cycle) forms
  (`build_stage`).
- `profile_two_stage_swz.py` — batch-sweep timing of the full two-stage layer vs dense
  `nn.Linear`. Timing only; correctness vs the row-major reference was verified in the original
  `benchmarks/profile_givens_twostage.py`.
- `profile_fused_stage.py` — verify + batch-sweep the fused stage-1 kernel against the fastest
  two-kernel stage 1 (`swz_s1 + rot_fixed`) and the free-read ceiling (`rot_fixed_fr`).
- `profile_parallel_stage.py` — race BDMM ∥ rot_persist combos round-robin against the sequential
  baseline (shared clock state); `--check` verifies rot_persist is bit-exact vs `rot_fixed`.
  Uses the 8x512x256 blocking (not the two-stage 32x128x64).

## Install

Run everything inside the official PyTorch docker image (from the repo root, so the repo —
including the `cutlass/` checkout — is mounted at `/workspace`):

```bash
docker run -it \
  --gpus all \
  -v "$(pwd):/workspace" \
  pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel \
  bash
```

Then, inside the container, install the Python dependencies:

```bash
cd /workspace
pip install -r requirements.txt
```

## Build

`readonce_swizzle` needs the CUTLASS headers; point `CUTLASS_DIR` at a CUTLASS checkout
(in the usual container the repo is mounted at `/workspace` and CUTLASS is at
`/workspace/cutlass`). Kernels target sm_80 (A100).

```bash
cd readonce_swizzle      && CUTLASS_DIR=/workspace/cutlass python setup.py build_ext --inplace && cd ..
cd givens_colmajor       && python setup.py build_ext --inplace && cd ..
cd fused_stage_gemm_rot  && CUTLASS_DIR=/workspace/cutlass python setup.py build_ext --inplace && cd ..
cd bdmm_only             && CUTLASS_DIR=/workspace/cutlass python setup.py build_ext --inplace && cd ..
cd parallel_stage        && python setup.py build_ext --inplace && cd ..
```

## Run

```bash
CUDA_VISIBLE_DEVICES=<idle GPU> python profile_two_stage_swz.py   # two-stage swz layer vs dense
CUDA_VISIBLE_DEVICES=<idle GPU> python profile_fused_stage.py     # fused stage-1 vs two-kernel
CUDA_VISIBLE_DEVICES=<idle GPU> python profile_parallel_stage.py --check  # co-resident BDMM ∥ rot race
```

Defaults: features=2048, 32 blocks of 128x64, stride 64, seq-len 4096, batch sweep 1..256.
