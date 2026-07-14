#!/usr/bin/env python3
"""Time the two-stage generalized (Givens) butterfly layer, swizzle read-once variant, vs dense.

Self-contained extract of benchmarks/profile_givens_twostage.py: only the fastest variant
(two_stage_swz) is profiled here. The layer computes

    x -> blkdiag1 (swz_s1) -> G1(tau1,rho1) (rot_fixed) -> blkdiag2 (swz_s2) -> G2 (rot_fixed) -> y

swz_s1/swz_s2 are the read-once swizzle GEMMs (col_s1's batched high-occupancy structure + a
batch-fast threadblock swizzle so the 64-col window overlap is an L2 hit; x read ~1x from DRAM).
rot_fixed is the column-major per-cycle rotation with compile-time pair indices. Output is
column-major (features x M). Correctness vs the row-major reference was verified in the original
benchmark; this copy only times.

Build the two extensions in-place first (see README.md), then run on an idle GPU:
    CUDA_VISIBLE_DEVICES=<idle> python profile_two_stage_swz.py
"""
import argparse, statistics, sys
from pathlib import Path
import torch
import torch.nn as nn

_here = Path(__file__).parent
sys.path.insert(0, str(_here))                      # givens_perm.py, colmajor_setup.py
sys.path.insert(0, str(_here / "readonce_swizzle"))
sys.path.insert(0, str(_here / "givens_colmajor"))
import readonce_swz_ext as SW                       # read-once swizzle GEMMs (swz_s1, swz_s2)
import givens_colmajor_ext as RotCM                 # column-major per-cycle rotation (rot_fixed)
from givens_perm import perm1 as monarch_perm
from colmajor_setup import build_stage              # builds rotation tensors with shared angles


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); torch.cuda.synchronize()
        s.append(e0.elapsed_time(e1))
    return statistics.median(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--features", type=int, default=2048)
    ap.add_argument("--nblocks", type=int, default=32)
    ap.add_argument("--block-h", type=int, default=128)
    ap.add_argument("--block-w", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"
    dev = "cuda"; dt = torch.float16
    n = args.features; st = args.stride

    p1 = monarch_perm(n, args.nblocks, n // args.nblocks)
    inv = [0] * n
    for f in range(n):
        inv[p1[f]] = f                                       # perm2 = perm1^{-1}
    gen = torch.Generator().manual_seed(0)
    _, c1 = build_stage(p1, gen, dev)                        # stage 1 (perm1); row tensors unused
    _, c2 = build_stage(inv, gen, dev)                       # stage 2 (perm2)

    W1 = torch.randn(args.nblocks, args.block_h, args.block_w, device=dev, dtype=dt)
    W2 = torch.randn(args.nblocks, args.block_h, args.block_w, device=dev, dtype=dt)

    def two_stage_swz(x):
        i1 = SW.swz_s1(x, W1, st)
        g1 = RotCM.rot_fixed(i1, c1['cyc_feat'], c1['fx_cos'], c1['fx_sin'])
        i2 = SW.swz_s2(g1, W2, st)
        return RotCM.rot_fixed(i2, c2['cyc_feat'], c2['fx_cos'], c2['fx_sin'])

    print(f"features={n} blocks=({args.nblocks},{args.block_h},{args.block_w}) stride={st}  "
          f"two-stage Givens swz vs dense; warmup={args.warmup} iters={args.iters}\n")
    hdr = ["batch", "M", "swz", "dense", "d/swz"]
    w = [6, 9, 8, 8, 7]
    print("  ".join(h.rjust(x) for h, x in zip(hdr, w)))
    print("  ".join("-" * x for x in w))
    for bs in args.batch_sizes:
        M = bs * args.seq_len
        x = torch.randn(M, n, device=dev, dtype=dt)
        dense = nn.Linear(n, n, bias=False, device=dev, dtype=dt).requires_grad_(False)
        with torch.inference_mode():
            tswz = time_fn(lambda: two_stage_swz(x), args.warmup, args.iters)
            dm   = time_fn(lambda: dense(x), args.warmup, args.iters)
        print("  ".join(str(v).rjust(x) for v, x in zip(
            [bs, M, f"{tswz:.3f}", f"{dm:.3f}", f"{dm/tswz:.2f}x"], w)))


if __name__ == "__main__":
    main()
