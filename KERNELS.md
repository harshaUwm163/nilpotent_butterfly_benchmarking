# Two-stage butterfly layer on A100 — kernel summary

**Goal.** Replace a dense 2048x2048 fp16 layer (`nn.Linear`) with a two-stage generalized
butterfly: `x -> blockdiag GEMM -> Givens rotation pair -> blockdiag GEMM -> Givens rotation
pair -> y`. Each stage's GEMM is 32 blocks of 128x64 with column stride 64 (adjacent blocks
re-read half their input window), so total MACs are ~1/8 of dense; the rotations are two
sparse permutation-structured layers (5 concurrent 2x2 plane rotations per 11-cycle, all 2048
features touched per row) with negligible FLOPs but full-matrix memory traffic.

**Hardware.** A100-PCIe 40 GB, **250 W power cap** (this matters — see kernel 3), fp16 in/out,
fp32 accumulate, CUTLASS 2.x-style kernels, sm_80. Timings at M = 131072 rows (batch 32 x
seq 4096) unless noted.

---

## 1. `readonce_swizzle` + `givens_colmajor` — the baseline (fastest known)

Four launches: GEMM, rotation, GEMM, rotation; intermediate (2048 x M fp16) round-trips HBM
between each pair.

- **GEMM (`swz_s1/swz_s2`)**: batched block-diagonal CUTLASS GEMM, feature-major (column-major)
  output, 4-stage cp.async pipeline. A custom batch-fastest threadblock swizzle makes the
  overlapping input windows L2 hits, cutting X's DRAM reads from ~1.94x to ~1x at full
  occupancy. Standalone BDMM study (larger 8x512x256 blocking): best tile reaches ~65% of
  tensor peak / 42% DRAM — compute-side is reasonably tuned.
- **Rotation (`rot_fixed`)**: per-cycle kernel, compile-time pair positions, whole cycle in
  registers, one read + one write per element. Measured at **~86% of DRAM peak**, ~0% tensor
  pipe — it is at the memory floor; only removing the traffic can improve it.

**Status:** ~**1.9x faster than dense** at batch >= 8 for the standalone layer (e.g. 3.59 ms vs
6.72 ms at M=131072). Two caveats: (a) a "free-read" ceiling (rotation reads nothing) shows
only ~25-30% more is available from eliminating the intermediate traffic; (b) when composed
into a full attention block, the feature-major output convention costs a transpose-shaped tax
that erases the win (0.72-0.96x vs dense end-to-end).

## 2. `fused_stage_gemm_rot` — vertical fusion in shared memory (failed: occupancy)

Attempt to delete the intermediate round-trip. One CTA owns a 32-row strip, loops a CUTLASS
warp-level MMA over all 32 diagonal blocks into a full-width fp16 smem buffer (32 x 2049,
~131 KB; +1 padding makes the row-parallel rotation bank-conflict-free), applies both rotation
layers in smem (fp32 register math), writes the stage output once.

The buffer must be full-width because the rotation pairs span all 2048 features (the
permutation's 11-cycles are global) — the stage cannot be tiled feature-wise.

**Status: correct but 10-16x slower** than the two-kernel path, 14-17x above the free-read
ceiling. The 131 KB buffer caps occupancy at **1 CTA/SM with a single warp** — no latency
hiding, tensor pipes idle. A tile/stage/buffer config sweep does not escape: smem capacity vs.
the global rotation footprint is the binding constraint on Ampere. (Saving ~1 GB of round-trip
traffic can never repay losing GEMM parallelism at these sizes.)

## 3. `parallel_stage` + `bdmm_only` — co-resident persistent-CTA overlap (failed: power cap)

Attempt to hide the rotation under the GEMM instead of fusing. Naive two-stream racing never
overlaps (the work distributor drains the first grid before dispatching a comparable-size
second grid), so the rotation is rebuilt as a **persistent-CTA kernel** (`rot_persist`): a
fixed grid (~2 CTAs/SM) launched first so it is resident from t=0, register footprint sized
via `__launch_bounds__` to fit beside a BDMM CTA in the 64K register file, optional evict-first
(`__ldcs/__stcs`) loads so its streaming does not evict the GEMM's L2 working set. The BDMM
(here 8 blocks of 512x256) backfills the same SMs.

**Status: the mechanism works, the economics do not.** Co-residency is real and bit-exact;
best race is ~1.2x vs sequential with a 1.34x ceiling on independent buffers. The dependent
(producer->consumer, chunked/double-buffered/event-synchronized) version verifiably deletes the
intermediate's HBM round-trip (2.22 -> 0.98 GB, ncu-confirmed) and is **still 0.87x** — because
at the 250 W cap wall-clock ~= energy/250 W, and DRAM traffic is only ~10% of the energy
budget. Overlap reshuffles the same joules. (L2-residency pinning of the intermediate was also
tried: no benefit, net harm at large M.)

---

## Where things stand / questions

Best known: 1.9x over dense standalone (vs ~8x FLOP reduction); both component kernels are
individually near their roofs (rotation ~86% DRAM peak, GEMM ~65% tensor peak); the two fusion/
overlap attempts fail for structural reasons (smem capacity vs. global rotation connectivity;
power-cap energy economics).

1. Is there an Ampere-feasible way to fuse or forward the (2048 x M) intermediate that we
   missed, given the rotation needs all 2048 features of a row?
2. Does Hopper/Blackwell change the answer — TMA, threadblock clusters + distributed shared
   memory for producer->consumer forwarding, or the higher power envelope?
3. The feature-major output tax when composing into attention (item 1b): is there a known
   pattern for keeping a batched narrow-N GEMM fast with row-major output?

Reproduce: `README.md` in this directory; each kernel has a self-contained
`profile_*.py` (correctness check + batch sweep).
