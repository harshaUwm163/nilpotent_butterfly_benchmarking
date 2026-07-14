#!/usr/bin/env python3
"""Profile the CO-RESIDENT (persistent-CTA) Givens rotation racing a BDMM GEMM.

Self-contained extract of custom_kernels/parallel_stage/final.py, using only the extensions in
this directory. rot_persist is launched FIRST with a fixed grid (typically <=1 CTA/SM, register
budget sized to fit alongside a BDMM CTA), so its CTAs are resident from t=0 and the BDMM
backfills around them — genuine same-SM co-scheduling, unlike two-stream "racing" of stock grids
where the work distributor serializes comparable-size grids.

Every race combo is timed round-robin against the SAME sequential baseline (`seq` = bdmm cfg +
stock rot_fixed) so GPU clock state is shared between arms (this A100 idles at 765 MHz; timing
arms back-to-back instead fakes a large win for whichever runs second).

NOTE: the race arm computes on INDEPENDENT buffers (interA / interB) — it measures co-residency
overlap, not the dependent producer->consumer stage. Known result: under the 250 W power cap,
wall-clock tracks energy, so the race never beats sequential by much (best ~1.0x); judge schemes
by DRAM traffic / joules, not overlap.

combo format: cfg:threads,vec,minb:ctas:prod_prio:cons_prio:stream
  cfg   = bdmm_cfg tile index; threads,vec,minb = rot_persist template variant
  ctas  = rotation grid size; *_prio = CUDA stream priorities; stream=1 -> evict-first (__ldcs/__stcs)

Build the extensions in-place first (see README.md), then run on an idle GPU:
    CUDA_VISIBLE_DEVICES=<idle> python profile_parallel_stage.py --check
"""
import argparse, statistics, sys
from pathlib import Path
import torch

_here = Path(__file__).parent
sys.path.insert(0, str(_here))                      # givens_perm.py, colmajor_setup.py
sys.path.insert(0, str(_here / "givens_colmajor"))
sys.path.insert(0, str(_here / "bdmm_only"))
sys.path.insert(0, str(_here / "parallel_stage"))
import bdmm_ext as B                                # CUTLASS block-diagonal GEMM (cfg-swept tiles)
import givens_colmajor_ext as RotCM                 # stock rot_fixed (baseline + correctness ref)
import parallel_stage_ext as P                      # persistent-CTA rotation (rot_persist)
from givens_perm import perm1 as monarch_perm
from colmajor_setup import build_stage

BDMM_BYTES = 1.149e9   # ncu, cfg5 bs32 (main+tail)
ROT_BYTES = 1.071e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cfg", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--check", action="store_true", help="verify rot_persist vs rot_fixed")
    ap.add_argument("--combos", nargs="+", default=[
        # best non-streaming reference
        "5:128,1,8:216:0:-1:0",
        # streaming (evict-first) variants
        "5:128,1,8:108:0:-1:1", "5:128,1,8:162:0:-1:1", "5:128,1,8:216:0:-1:1",
        "5:128,1,8:324:0:-1:1", "5:128,1,8:432:0:-1:1", "5:128,1,8:216:0:0:1",
        "5:128,1,8:324:0:0:1",  "5:128,1,8:648:0:-1:1",
        "6:256,2,3:108:0:-1:1", "6:256,2,3:216:0:-1:1", "6:256,1,4:216:0:-1:1",
        "6:128,1,8:432:0:-1:1",
        "2:128,1,8:216:0:-1:1",
    ])
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU"

    dev, dt, n = "cuda", torch.float16, 2048
    nb, K, N = 8, 512, 256
    M = args.batch * 4096
    scale = M / 131072
    bdmm_b, rot_b = BDMM_BYTES * scale, ROT_BYTES * scale

    gen = torch.Generator().manual_seed(0)
    _, c1 = build_stage(monarch_perm(n, nb, N), gen, dev)
    cf, cc, sn = c1["cyc_feat"], c1["fx_cos"], c1["fx_sin"]
    X = torch.randn(M, n, device=dev, dtype=dt)
    W = torch.randn(nb, K, N, device=dev, dtype=dt)
    interA = torch.empty(n, M, device=dev, dtype=dt)
    interB = torch.randn(n, M, device=dev, dtype=dt)
    bcfg = args.base_cfg

    if args.check:
        print("== correctness (rot_persist variants vs stock rot_fixed) ==")
        base = torch.randn(n, M, device=dev, dtype=dt)
        seen = set()
        for combo in args.combos:
            _, var, ctas_s, _, _, st_s = combo.split(":")
            key = (var, ctas_s, st_s)
            if key in seen: continue
            seen.add(key)
            th, ve, mb = (int(x) for x in var.split(","))
            ref, got = base.clone(), base.clone()
            with torch.inference_mode():
                RotCM.rot_fixed(ref, cf, cc, sn)
                P.rot_persist(got, cf, cc, sn, th, ve, mb, int(st_s), int(ctas_s))
                torch.cuda.synchronize()
            err = (ref.float() - got.float()).abs().max().item()
            print(f"  {var} ctas={ctas_s:<4} stream={st_s}  err={err:.3e} {'OK' if err == 0 else 'MISMATCH'}")
        print()

    def seq():
        B.bdmm_cfg_into(bcfg, X, W, interA, N)
        RotCM.rot_fixed(interB, cf, cc, sn)

    def prod_base(): B.bdmm_cfg_into(bcfg, X, W, interA, N)
    def cons_base(): RotCM.rot_fixed(interB, cf, cc, sn)

    streams = {}
    def get_streams(pp, cp):
        if (pp, cp) not in streams:
            streams[(pp, cp)] = (torch.cuda.Stream(priority=pp), torch.cuda.Stream(priority=cp))
        return streams[(pp, cp)]

    names = {"seq": seq, "prod": prod_base, "cons": cons_base}
    with torch.inference_mode():
        for _ in range(args.warmup):
            for f in names.values(): f()
        torch.cuda.synchronize()
        acc = {k: [] for k in names}
        for _ in range(args.iters):
            for k, f in names.items():
                e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                e0.record(); f(); e1.record(); torch.cuda.synchronize()
                acc[k].append(e0.elapsed_time(e1))
    m = {k: statistics.median(v) for k, v in acc.items()}
    ceiling = max(m["prod"], (bdmm_b + rot_b) / 1332e9 * 1e3)
    print(f"batch={args.batch} M={M}  traffic total={(bdmm_b+rot_b)/1e9:.3f} GB")
    print(f"seq={m['seq']:.3f}  prod={m['prod']:.3f}  cons={m['cons']:.3f}  "
          f"ceiling={ceiling:.3f}  max_speedup={m['seq']/ceiling:.3f}x\n")

    print(f"{'cfg':>3} {'variant':>9} {'ctas':>5} {'pp':>3} {'cp':>3} {'st':>2} | "
          f"{'seq':>6} {'race':>6} {'bdmm':>6} {'rot':>6} | {'GB/s':>6} {'infl':>5} {'speedup':>8}")
    best = None
    for combo in args.combos:
        cfg_s, var, ctas_s, pp_s, cp_s, st_s = combo.split(":")
        cfg, ctas, pp, cp, st = int(cfg_s), int(ctas_s), int(pp_s), int(cp_s), int(st_s)
        th, ve, mb = (int(x) for x in var.split(","))
        try:
            B.bdmm_cfg(cfg, X, W, N)
        except Exception:
            print(f"{cfg:>3} unsupported"); continue
        s_prod, s_cons = get_streams(pp, cp)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

        def race():
            cur = torch.cuda.current_stream()
            ev[0].record(cur)
            s_prod.wait_stream(cur); s_cons.wait_stream(cur)
            with torch.cuda.stream(s_prod):
                ev[1].record(s_prod); B.bdmm_cfg_into(cfg, X, W, interA, N); ev[2].record(s_prod)
            with torch.cuda.stream(s_cons):
                ev[3].record(s_cons); P.rot_persist(interB, cf, cc, sn, th, ve, mb, st, ctas); ev[4].record(s_cons)
            cur.wait_stream(s_prod); cur.wait_stream(s_cons)

        arms = {"seq": seq, "race": race, "prod": prod_base}
        with torch.inference_mode():
            for _ in range(args.warmup):
                for f in arms.values(): f()
            torch.cuda.synchronize()
            acc = {k: [] for k in arms}
            sub = []
            for _ in range(args.iters):
                for k, f in arms.items():
                    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
                    e0.record(); f(); e1.record(); torch.cuda.synchronize()
                    acc[k].append(e0.elapsed_time(e1))
                sub.append((ev[0].elapsed_time(ev[2]), ev[0].elapsed_time(ev[4])))
        mm = {k: statistics.median(v) for k, v in acc.items()}
        bd, rt = (statistics.median(x) for x in zip(*sub))
        sp = mm["seq"] / mm["race"]
        bw = (bdmm_b + rot_b) / (mm["race"] * 1e-3) / 1e9
        infl = bd / mm["prod"]
        flag = ""
        if best is None or sp > best[0]:
            best = (sp, combo, mm["race"]); flag = " *"
        print(f"{cfg:>3} {var:>9} {ctas:>5} {pp:>3} {cp:>3} {st:>2} | {mm['seq']:>6.3f} {mm['race']:>6.3f} "
              f"{bd:>6.3f} {rt:>6.3f} | {bw:>6.0f} {infl:>5.2f} {sp:>8.3f}{flag}")
    print(f"\nbest: {best[1]}  race={best[2]:.3f} ms  speedup={best[0]:.3f}x  (ceiling {m['seq']/ceiling:.3f}x)")


if __name__ == "__main__":
    main()
