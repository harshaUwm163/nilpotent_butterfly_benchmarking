#!/usr/bin/env python3
"""Verify + benchmark the FULLY FUSED stage-1 kernel (all-block block-diagonal GEMM + both Givens
rotation layers in shared memory, NO (2048, M) intermediate round-trip through HBM) against the
fastest two-kernel stage-1 it replaces.

Self-contained extract of benchmarks/profile_fused_all_blocks.py, using only the extensions in
this directory:
  swz2k  : swz_s1 + rot_fixed     — the fastest two-kernel stage 1 (col-major output)
  fused  : fused_stage_gemm_rot   — one launch, no round-trip (row-major output)
  fr-ceil: swz_s1 + rot_fixed_fr  — free-read ceiling (rotation reads m=0 only, i.e. the
                                    intermediate read is ~L2-free)

Correctness: fused output is checked against swz2k transposed (same op, row- vs col-major) and
against a pure-torch fp32 reference on 8 rows (isolates the algorithm from fp16 rounding).

Known result: the fusion is correct but ~10x SLOWER — the full-row 32x2049 fp16 smem buffer caps
occupancy at 1 CTA/SM with a single warp, which costs far more than the round-trip saves.

Build both extensions in-place first (see README.md), then run on an idle GPU:
    CUDA_VISIBLE_DEVICES=<idle> python profile_fused_stage.py
"""
import argparse, statistics, sys
from pathlib import Path
import torch

_here = Path(__file__).parent
sys.path.insert(0, str(_here))                      # givens_perm.py, colmajor_setup.py
sys.path.insert(0, str(_here / "readonce_swizzle"))
sys.path.insert(0, str(_here / "givens_colmajor"))
sys.path.insert(0, str(_here / "fused_stage_gemm_rot"))
import readonce_swz_ext as SW                       # read-once swizzle GEMM (swz_s1)
import givens_colmajor_ext as RotCM                 # column-major rotation (rot_fixed, rot_fixed_fr)
import fused_stage_gemm_rot_ext as FS               # fused GEMM+rotation kernel
from givens_perm import perm1 as monarch_perm
from colmajor_setup import build_stage              # row (plane) + col (cycle) forms, shared angles


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
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
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
    gen = torch.Generator().manual_seed(0)
    r1, c1 = build_stage(p1, gen, dev)   # fused kernel uses the row form, rot_fixed the col form
    W1 = torch.randn(args.nblocks, args.block_h, args.block_w, device=dev, dtype=dt)

    def fused(x):
        return FS.fused_stage_gemm_rot(x, W1, st,
                                       r1['I1'], r1['J1'], r1['C1'], r1['S1'],
                                       r1['I2'], r1['J2'], r1['C2'], r1['S2'])

    def swz2k(x):
        i1 = SW.swz_s1(x, W1, st)                                   # (2048, M) col-major
        return RotCM.rot_fixed(i1, c1['cyc_feat'], c1['fx_cos'], c1['fx_sin'])

    def fr_ceil(x):
        i1 = SW.swz_s1(x, W1, st)
        return RotCM.rot_fixed_fr(i1, c1['cyc_feat'], c1['fx_cos'], c1['fx_sin'])

    # ---- correctness: fused (M,2048 row-major) vs swz2k transposed; and vs fp32 torch math ----
    torch.manual_seed(0)
    xchk = torch.randn(256, n, device=dev, dtype=dt)
    with torch.inference_mode():
        got = fused(xchk).float()
        ref = swz2k(xchk).t().float()
    rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)

    # tighter check: fp32 math reference on a few rows (isolates the algorithm from fp16 rounding)
    xg = xchk[:8].float()
    i1 = torch.zeros(8, n, device=dev)
    for b in range(args.nblocks):
        Kb = args.block_h if b < args.nblocks - 1 else n - (args.nblocks - 1) * st
        i1[:, b*args.block_w:(b+1)*args.block_w] = xg[:, b*st:b*st+Kb] @ W1[b, :Kb].float()
    def rot(Y, I, J, C, S):
        a = Y[:, I.long()].clone(); b = Y[:, J.long()].clone()
        Y[:, I.long()] = C * a - S * b; Y[:, J.long()] = S * a + C * b
    rot(i1, r1['I1'], r1['J1'], r1['C1'], r1['S1'])
    rot(i1, r1['I2'], r1['J2'], r1['C2'], r1['S2'])
    rel_fp32 = (fused(xchk)[:8].float() - i1).abs().max().item() / (i1.abs().max().item() + 1e-9)

    print(f"features={n} blocks=({args.nblocks},{args.block_h},{args.block_w}) stride={st}")
    print(f"correctness: fused-vs-swz2k.T rel={rel:.2e}   fused-vs-fp32(8 rows) rel={rel_fp32:.2e}\n")

    hdr = ["batch", "M", "swz2k", "fused", "fr-ceil", "swz/fus", "fus/ceil"]
    w = [6, 9, 9, 9, 9, 9, 9]
    print("  ".join(h.rjust(x) for h, x in zip(hdr, w)))
    print("  ".join("-" * x for x in w))
    for bs in args.batch_sizes:
        M = bs * args.seq_len
        x = torch.randn(M, n, device=dev, dtype=dt)
        with torch.inference_mode():
            tswz = time_fn(lambda: swz2k(x),   args.warmup, args.iters)
            tfus = time_fn(lambda: fused(x),   args.warmup, args.iters)
            tcei = time_fn(lambda: fr_ceil(x), args.warmup, args.iters)
        print("  ".join(str(v).rjust(x) for v, x in zip(
            [bs, M, f"{tswz:.3f}", f"{tfus:.3f}", f"{tcei:.3f}",
             f"{tswz/tfus:.2f}x", f"{tfus/tcei:.2f}x"], w)))


if __name__ == "__main__":
    main()
