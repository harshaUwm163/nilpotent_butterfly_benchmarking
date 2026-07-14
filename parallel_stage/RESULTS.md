# Co-scheduling BDMM ∥ Givens on A100-PCIe: it works, and it still doesn't pay

**Hardware:** NVIDIA A100-PCIE-40GB (CC 8.0, 108 SMs, 65,536 regs/SM, 250 W hard cap).
**Op:** stage-1 `Y = Givens(τ,ρ) ∘ (X · W_blockdiag)`, fp16, block `(nb,K,N) = (8,512,256)`, seq 4096.
**Code:** `kernel.cu` (persistent-CTA rotation), `bench.py`, `sweep.py`, `timeline.py`, `final.py`,
`powerprobe.py`, `steady.py`. Reproduced on GPUs 5 and 7.

---

## TL;DR

Two independent grids *can* be made genuinely co-resident on every SM — the register-file
argument is right, and the persistent-CTA rotation achieves it (both kernels start at t≈0.02 ms).
But **sustained throughput does not improve**: `seq` 2.461 ms vs `race` 2.492 ms (0.99×).

The reason is not the memory system. It is the **250 W power cap**. Every arm that contains the
GEMM runs pinned at the cap, so wall-clock ≈ *energy* / 250 W. Concurrency rearranges work; it does
not remove joules. The race in fact costs ~2-4% *more* energy (lower DRAM efficiency when the two
access streams interleave), so it is slightly slower.

**Corollary for this project:** on a power-capped part, only schemes that *delete work* can win.
That is exactly the DRAM round-trip. Overlap, L2 pinning, spatial pipelines and racing all
rearrange; none of them reduce joules. This retro-explains the four previous failures.

---

## 1. A measurement bug invalidated the prior "14% overlap" result

`benchmarks/race_winner.py` times `t_prod` *before* `t_race` for each cfg. The A100 idles at
765 MHz and boosts under load, so the first arm measured is systematically slow. For cfg 5 this
produced `t_prod = 1.095 ms`, `t_race = 0.871 ms` and the impossible `eff = 2.01` (a race faster
than the producer alone).

Timing all arms **round-robin in one loop** (shared clock state) collapses it: cfg 5's `t_prod`
drops to 0.646 ms and the overlap goes to 0.7%. Across all 11 tile cfgs on block `(8,512,256)` at
batch 32, measured overlap is **−0.6% … +0.8%**, i.e. zero, with no dependence on arithmetic
intensity. The prior claim that "making the BDMM compute-bound gives ~14% overlap" is an artifact.

## 2. The register file explains why two grids never co-reside

A100 SM = 65,536 registers. Register allocation is per-warp in 256-register units.

| cfg | tile | thr | occ | reg/thr | reg/CTA | used | **free** |
|----:|------|----:|----:|--------:|--------:|-----:|---------:|
| 0 | 64x128x32 S4 | 128 | 3 | 142 | 18,432 | 55,296 | 10,240 |
| 1 | 64x256x32 S4 | 128 | 2 | 238 | 30,720 | 61,440 | 4,096 |
| 2 | 128x128x32 S3 | 128 | 2 | 226 | 29,696 | 59,392 | 6,144 |
| 5 | 256x128x32 S3 | 256 | 1 | 220 | 57,344 | 57,344 | 8,192 |
| **6** | 128x128x64 S3 | 128 | 1 | 246 | 31,744 | 31,744 | **33,792** |
| 8 | 64x64x32 S4 | 128 | 4 | 96 | 12,288 | 49,152 | 16,384 |
| **11** | 64x128x64 S3 | 128 | 2 | 154 | 20,480 | 40,960 | **24,576** |

`rot_cycle_fixed` is 74 regs × 256 threads = **20,480 regs/CTA**. Only cfgs 6 and 11 (both
*shared-memory*-limited, so registers go unused) leave room for it. Everywhere else a Givens CTA
must evict a BDMM CTA.

But the register hole is **necessary, not sufficient**: cfg 11 still measured −0.6% overlap. With a
23,808-CTA rotation grid launched alongside a 28,672-CTA GEMM grid, the work distributor simply
never dispatches the rotation until the GEMM drains.

## 3. Persistent CTAs do achieve co-residency

`rot_persist<THREADS,VEC,MINB,STREAM>` (in `kernel.cu`) is a grid-stride rotation with a **fixed**
grid. `MINB` caps regs/thread via `__launch_bounds__`, so the CTA can be sized to fit a chosen
BDMM cfg's free registers. Verified against `cudaOccupancyMaxActiveBlocksPerMultiprocessor`:

| threads | vec | minb | reg/thr | reg/CTA | spill | fits |
|--------:|----:|-----:|--------:|--------:|-------|------|
| 128 | 1 | 8 | 63 | **8,192** | none | cfg 5 (exactly), 0, 6, 8, 9, 11 |
| 128 | 2 | 8 | 64 | 8,192 | **64 B** | — (avoid) |
| 128 | 1 | 1 | 72 | 9,216 | none | 0, 6, 8, 9, 11 |
| 256 | 2 | 3 | 80 | 20,480 | none | 6, 11 |

Output is bit-exact vs `rot_fixed` (`max_abs_err = 0`).

`timeline.py` records events on both streams against a common t0. In **every** configuration both
kernels start at ≈0.02 ms — they really are co-resident, not tail-overlapped. Prior "racing" work
was measuring tail overlap only.

## 4. …and it still doesn't help, because the GPU is power-capped

`powerprobe.py`, sustained loops, batch 32:

| arm | SM clock | power | sw_power_cap active |
|-----|---------:|------:|--------------------:|
| idle | 765 MHz | 44 W | 0% |
| BDMM alone | 1035 MHz | 248 W | **87%** |
| rot_fixed alone | 1410 MHz | 215 W | 4% |
| seq | 1125 MHz | 250 W | **96%** |
| race | 1125 MHz | 251 W | **100%** |

`power.max_limit = 250 W` — a hard ceiling, `nvidia-smi -pl 300` is rejected. The GEMM alone already
loses 27% of its clock (1035 of 1410 MHz) to the cap.

`steady.py`, sustained interleaved blocks (batch 32, GPU 7):

| arm | ms/iter | power | J/iter |
|-----|--------:|------:|-------:|
| prod (BDMM) | 1.793 | 249.8 W | 0.4480 |
| cons (rot_fixed) | 0.810 | 121.2 W | 0.0981 |
| cons_p (rot_persist, 216 CTAs) | 0.875 | 125.9 W | 0.1101 |
| **seq** | **2.461** | 247.2 W | 0.6084 |
| race (persistent) | 2.492 | 253.6 W | 0.6321 |
| race_fx (stock rot, 2 streams) | 2.457 | 252.9 W | 0.6216 |

- `race_fx / seq = 0.998` — overlapping the stock rotation buys **exactly nothing**.
- `race / seq = 0.987` — the persistent version is *worse*, and its energy is 3.9% higher.
- Wall-clock tracks energy: both arms sit at ~250 W, so `t ≈ J / 250 W`.

Batch 64: seq 4.863 ms / race 4.964 ms (0.98×), prod at 249.7 W. Batch 8: seq 0.611 / race 0.641
(0.95×) — here the cap does *not* bind (prod 172 W), and the loss is co-residency tax instead.
(Batch-8 power readings are unreliable: nvidia-smi samples at ~30 ms against 0.6 ms kernels.)

Reproduced on GPU 5: seq 2.432, race 2.458, race_fx 2.433.

## 5. Where the headroom actually is

Ignoring power, the resource floor is real and generous. ncu (`--cache-control none`, batch 32):

- BDMM cfg 5: **1.149 GB** DRAM (1.016 main + 0.133 tail), 30–39% of peak
- rot_fixed: **1.071 GB**, 84.5% of peak

Combined 2.220 GB; at the 1332 GB/s the rotation actually sustains, the DRAM floor is 1.667 ms —
*below* BDMM's own 1.721 ms. So a perfect overlap would take 1.72 ms vs 2.46 ms sequential = **1.43×**,
and DRAM is not the constraint. What eats it is the 250 W budget: at an unthrottled 1410 MHz the
GEMM would take ≈1.34 ms, and delivering both kernels' work in that window needs ≈350 W.

**On a 400 W A100-SXM4 or an H100 this machinery should deliver most of that 1.4×.** The kernel is
parameterised (`THREADS`, `VEC`, `MINB`, grid size, stream priority) precisely so the CTA can be
resized to whatever GEMM tile is used there. On A100-PCIe it cannot.

Things that were tried and did not help:
- **Streaming / evict-first L2 hints** (`__ldcs`/`__stcs`, `STREAM=1`): *worse* (2.500 vs 2.324 ms
  event-timed). The L2-pollution hypothesis is not supported.
- **Launch order**: matters a lot for the balance (rotation-first starves the GEMM; GEMM-first
  starves the rotation) but never changes the total.
- **Stream priority**: same — it only moves the see-saw, total is flat.
- **Grid size** 54 … 864 CTAs: the rotation is *latency*-limited at 1 CTA/SM (267 GB/s), and
  bandwidth-limited above ~4 CTAs/SM. No setting makes both kernels fast at once.

---

## 6. The dependent (synchronised producer→consumer) version — `dependent.py`

Everything above raced BDMM and the rotation over **two separate buffers**, so there was no
dependency. Two things had to be checked.

**6a. The unsynchronised in-place race computes garbage.** Running both kernels concurrently on the
*same* intermediate (which is what `race_winner.py` and `l2_race_probe.py` time) gives, vs the
sequential reference: `max_abs_err = 179.03`, **99.9% of elements wrong**. Every "race" timing in
this repo was timing an incorrect computation.

**6b. A properly synchronised pipeline is exact — and slower.** `pipe(Ms)`: chunk the tokens,
double-buffer a `(2048, Ms)` slab, `bdmm(chunk t) → slab[t%2]` on `s_prod`, `rot_fixed_io(slab[t%2])
→ out[:, t*Ms:]` on `s_cons`, ordered by CUDA events. Bit-exact (`max_abs_err = 0`) at every `Ms`,
also after CUDA-graph capture.

It does both of the things it is supposed to do:

- **It deletes the DRAM round-trip.** The rotation reads the slab the GEMM just wrote, out of L2.
  ncu (`--cache-control none`, batch 32, exactly one pipeline iteration):

  | | seq_full | Ms=4096 | Ms=2048 | Ms=1024 | Ms=512 |
  |---|---|---|---|---|---|
  | slab (x2) | — | 16 MB | 8 MB | 4 MB | 2 MB |
  | DRAM bytes | 2.218 GB | 1.840 | 1.598 | 1.264 | **0.979** |

- **It overlaps.** `pipeG2048` = 3.834 ms vs `bdmmC2048` 3.324 + `rotC2048` 0.908 → 44% of the
  rotation is hidden behind the next chunk's GEMM.

But chunking wrecks the GEMM, and that dwarfs both gains (sustained, graph-captured, batch 32):

| arm | ms/iter | J/iter |
|---|---:|---:|
| bdmm_full | 1.767 | 0.465 |
| rot_full | 0.810 | — |
| **seq_full** | **2.492** | **0.625** |
| bdmmC8192 (chunked GEMM alone) | 2.062 | |
| bdmmC4096 | 2.489 | |
| bdmmC2048 | **3.324** | |
| pipeG8192 | 2.798 | 0.733 |
| pipeG4096 | 3.077 | 0.768 |
| pipeG2048 | 3.834 | 0.968 |
| pipeG512 | 8.829 | 1.327 |

`bdmmC2048` **alone** (3.324 ms) already exceeds the entire sequential path. cfg 5's grid is
`(nb−1, Ms/TBN)` for the main batched GEMM **plus a separate `(1, Ms/TBN)` tail GEMM**; at Ms=2048
that is 112 + 16 CTAs on 108 SMs at occ=1 — wave quantisation plus a nearly-empty tail launch, per
chunk. At Ms=512 the GEMM grid is 28 CTAs and the GPU idles at 150 W / 1410 MHz.

**No tile rescues it** (`chunkcfg.py`, batch 32; best seq_full 2.51 ms). The best chunked GEMMs are
cfg 3 @ Ms=4096 (2.129) and cfg 9 @ Ms=2048 (2.476), but the full pipelines still lose:

| pipeline cfg | Ms=2048 | Ms=4096 | vs seq (2.506) |
|---|---:|---:|---|
| 3 | 3.188 | **2.877** | **0.87×** |
| 9 | 3.204 | 3.025 | 0.83× |
| 2 | 3.271 | 2.906 | 0.86× |

And they burn *more* energy (0.72 J vs 0.62 J) despite moving less DRAM.

**Why the round-trip was never the prize.** Cutting DRAM traffic 2.27× (2.218 → 0.979 GB, at
Ms=512) costs +0.70 J of GEMM inefficiency while saving at most ~0.05 J of HBM energy. On this
workload **DRAM traffic is worth ~10% of the energy budget; SM utilisation is the rest.**

**The structural obstacle.** Under `perm1(2048, 8, 256)` the Givens cycles have length 11 and span
*all* 2048 features. So no unit smaller than the full feature dimension is a self-contained rotation.
Fusing the rotation into the GEMM epilogue therefore needs a `(2048 × TBN)` output tile — which is
exactly why `fused_stage_gemm_rot/` had to give up N-tiling and came out ~10× slower. Chunking the
tokens (this experiment) is the same constraint pushed into the token dimension: the slab must hold
all 2048 features, so it is large, so it must be short, so the GEMM starves.

The one lever that changes the structure is the **permutation**, not the kernel: if the rotation's
cycles were block-local (contained within one 256-feature block-diagonal block), the epilogue fusion
would be free and the 1.41× (`seq_full` → `bdmm_full`) would be available. That is a modelling
change, not a scheduling one.

---

## Reproduce

```bash
# register/occupancy fit table
python regtable.py

# correctness + first race
python bench.py --cfgs 5 --variants 128,1,8 --ctas-per-sm 1

# per-stream timeline: proves co-residency
python timeline.py --cfg 5 --variant 128,1,8 --ctas 108 216 432 864 --prio 0 -1

# the verdict: sustained throughput + energy
python powerprobe.py --cfg 5 --ctas 216
python steady.py --cfg 5 --ctas 216 --iters 150 --reps 4
```

**Methodology note:** never time one arm after another on this GPU. It idles at 765 MHz and boosts
under sustained load, so the first-measured arm is penalised by up to 40%. Time arms round-robin in
a single loop (`timeit_roundrobin`), and prefer `steady.py`'s sustained-block measurement for any
claim about throughput.
