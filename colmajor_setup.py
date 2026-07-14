"""Build the Givens-layer tensors for one stage (one permutation, its two layers tau,rho), in both
forms, sharing the SAME random angles so the two paths produce identical output:
  - row-major  : plane-index/angle tensors for givens_fused_ext.fused_two_givens
  - col per-cycle: (cyc_feat, swap_lo, swap_hi, swap_cos, swap_sin) for givens_colmajor_ext.rot_cycle_vec
Assumes the 11-cycle case (n = 2^11 layout): all non-trivial cycles have length 11.
"""
import torch
from givens_perm import cycles_of

# within an 11-cycle: tau = psi_0 pairs, rho = psi_1 pairs (positions)
_TAU_POS = [(1, 10), (2, 9), (3, 8), (4, 7), (5, 6)]
_RHO_POS = [(0, 1), (2, 10), (3, 9), (4, 8), (5, 7)]


def build_planes(perm):
    """Global tau, rho involutions (as [(i,j)] with i<j) for an odd-cycle permutation."""
    N = len(perm); tau = list(range(N)); rho = list(range(N))
    for c in cycles_of(perm):
        L = len(c)
        if L % 2 and L > 1:
            for i in range(L):
                tau[c[i]] = c[(-i) % L]; rho[c[i]] = c[(1 - i) % L]
    return ([(i, tau[i]) for i in range(N) if tau[i] > i],
            [(i, rho[i]) for i in range(N) if rho[i] > i])


def build_stage(perm, gen, dev):
    tp, rp = build_planes(perm)
    Ct = torch.rand(len(tp), generator=gen); St = torch.rand(len(tp), generator=gen)
    Cr = torch.rand(len(rp), generator=gen); Sr = torch.rand(len(rp), generator=gen)
    taud = {tp[k]: (Ct[k].item(), St[k].item()) for k in range(len(tp))}
    rhod = {rp[k]: (Cr[k].item(), Sr[k].item()) for k in range(len(rp))}

    def itensor(pl, which):
        return torch.tensor([p[which] for p in pl], dtype=torch.int32, device=dev)
    row = dict(
        I1=itensor(tp, 0), J1=itensor(tp, 1), C1=Ct.to(dev), S1=St.to(dev),
        I2=itensor(rp, 0), J2=itensor(rp, 1), C2=Cr.to(dev), S2=Sr.to(dev),
    )

    cyc = [c for c in cycles_of(perm) if len(c) == 11]
    CF = []; LO = []; HI = []; CO = []; SI = []
    # fixed-position form (rot_fixed kernel): pairs are the compile-time canonical (a,b); the lo/hi
    # direction is folded into sin's sign (+ if fa<fb i.e. lo==a, - otherwise). cos unchanged.
    FXC = []; FXS = []
    for c in cyc:
        CF.append(list(c)); los = []; his = []; cos = []; sin = []; fxc = []; fxs = []
        for src, pairs in [(taud, _TAU_POS), (rhod, _RHO_POS)]:
            for (a, b) in pairs:
                fa, fb = c[a], c[b]
                cs, sn = src[(min(fa, fb), max(fa, fb))]
                lo, hi = (a, b) if fa < fb else (b, a)
                los.append(lo); his.append(hi); cos.append(cs); sin.append(sn)
                fxc.append(cs); fxs.append(sn if fa < fb else -sn)
        LO.append(los); HI.append(his); CO.append(cos); SI.append(sin)
        FXC.append(fxc); FXS.append(fxs)
    col = dict(
        cyc_feat=torch.tensor(CF, dtype=torch.int32, device=dev),
        swap_lo=torch.tensor(LO, dtype=torch.int32, device=dev),
        swap_hi=torch.tensor(HI, dtype=torch.int32, device=dev),
        swap_cos=torch.tensor(CO, dtype=torch.float32, device=dev),
        swap_sin=torch.tensor(SI, dtype=torch.float32, device=dev),
        fx_cos=torch.tensor(FXC, dtype=torch.float32, device=dev),
        fx_sin=torch.tensor(FXS, dtype=torch.float32, device=dev),
    )
    return row, col
