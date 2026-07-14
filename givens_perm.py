#!/usr/bin/env python3
"""Realize the Monarch/strided-butterfly permutation (perm1) as TWO parallel layers of
disjoint Givens rotations.

perm1(f) = (f % nblocks)*per_block + (f // nblocks)   -- the transpose of a (per_block x nblocks)
grid, equivalently multiplication by `per_block` mod (n-1) with the two endpoints fixed. Its
cycle structure for n=2048 is 186 cycles of length 11 + 2 fixed points.

A 90-degree Givens rotation on plane (i, j) is the transposition (i j) up to sign. One parallel
layer of disjoint Givens = an involution (disjoint-transposition matching). Any permutation is a
product of two involutions, so perm1 = sig . tau with each of sig, tau a single parallel layer.
We build them from the dihedral fact "rotation of an L-cycle = product of two reflections":

  for a cycle [p_0, ..., p_{L-1}] with perm1(p_i) = p_{(i+1) mod L}:
    tau : p_i <-> p_{(-i)  mod L}     (reflection f0(i) = -i)
    sig : p_i <-> p_{(1-i) mod L}     (reflection f1(i) = 1-i)
  then  sig(tau(p_i)) = p_{i+1} = perm1(p_i).

Each layer is realized as 90-degree Givens rotations on the swapped planes; the composed orthogonal
matrix equals perm1 up to a diagonal of +-1 (sign), which is all we ask for.
"""
import argparse
import numpy as np


def perm1(n, nblocks, per_block):
    """The forward permutation as a list: index f -> perm[f]."""
    assert n == nblocks * per_block
    return [(f % nblocks) * per_block + (f // nblocks) for f in range(n)]


def cycles_of(perm):
    """Disjoint-cycle decomposition of a permutation given as f -> perm[f]."""
    n = len(perm)
    seen = [False] * n
    cycles = []
    for s in range(n):
        if seen[s]:
            continue
        c, x = [], s
        while not seen[x]:
            seen[x] = True
            c.append(x)
            x = perm[x]
        cycles.append(c)
    return cycles


def two_matchings(perm):
    """Return (tau, sig): two involutions (as f->p[f] lists) with sig . tau == perm.

    Each is a set of disjoint transpositions (a parallel Givens layer). Built per cycle via the
    two dihedral reflections f0(i) = -i and f1(i) = 1-i (indices taken mod cycle length).
    """
    n = len(perm)
    tau = list(range(n))
    sig = list(range(n))
    for c in cycles_of(perm):
        L = len(c)
        for i in range(L):
            tau[c[i]] = c[(-i) % L]
            sig[c[i]] = c[(1 - i) % L]
    return tau, sig


def transpositions_of(involution):
    """The unordered swapped pairs (i, j) of an involution; fixed points omitted."""
    pairs = []
    for i in range(len(involution)):
        j = involution[i]
        if j > i:
            pairs.append((i, j))
    return pairs


def givens_matrix(n, pairs, theta=np.pi / 2):
    """Orthogonal matrix applying a disjoint set of plane rotations by `theta`.

    Plane (i, j) rotation R: e_i -> cos*e_i + sin*e_j,  e_j -> -sin*e_i + cos*e_j.
    At theta=90deg this is the swap of i,j up to sign (e_i->e_j, e_j->-e_i).
    Disjoint planes => one well-defined orthogonal layer.
    """
    G = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    for (i, j) in pairs:
        G[i, i] = c
        G[j, j] = c
        G[i, j] = -s   # column j: e_j -> -s e_i + c e_j
        G[j, i] = s    # column i: e_i ->  c e_i + s e_j
    return G


def permutation_matrix(perm):
    """Matrix P with P @ x routing input f to output perm[f]:  (P x)[perm[f]] = x[f]."""
    n = len(perm)
    P = np.zeros((n, n))
    for f in range(n):
        P[perm[f], f] = 1.0
    return P


def solve_orientations(perm, tau, sig):
    """Pick a +90 / -90 angle per plane so the two Givens layers equal perm1 EXACTLY (no sign).

    Net sign on input f is  sigma(f,p)*sigma(p,q)*(-1)^(u+w)  with p=tau[f], q=perm[f], and u,w the
    orientation bits of f's tau-plane and p's sig-plane (sigma(a,b)=+1 if a<b else -1). Demanding +1
    everywhere is one XOR equation per coordinate in >=1 plane bit -- a 2-variable parity system,
    solved by union-find with parity (a GND node anchors the unary constraints from coords that pass
    through only one rotation). Returns (t_angles, s_angles), each a list of +/-pi/2 per plane.

    Bit 0 -> +pi/2, bit 1 -> -pi/2. Raises if the system is inconsistent.
    """
    n = len(perm)
    tpairs, spairs = transpositions_of(tau), transpositions_of(sig)
    t_id = {x: k for k, pr in enumerate(tpairs) for x in pr}
    s_id = {x: k for k, pr in enumerate(spairs) for x in pr}
    nt, ns = len(tpairs), len(spairs)
    GND = nt + ns                         # ground node, parity pinned to 0
    par, rel = list(range(GND + 1)), [0] * (GND + 1)

    def find(x):
        if par[x] == x:
            return x, 0
        r, p = find(par[x])
        par[x] = r
        rel[x] ^= p
        return r, rel[x]

    def union(a, b, bit):
        ra, pa = find(a)
        rb, pb = find(b)
        need = pa ^ pb ^ bit
        if ra == rb:
            return need == 0
        par[ra] = rb
        rel[ra] = need
        return True

    sgn = lambda a, b: 1 if a < b else -1
    for f in range(n):
        p, q = tau[f], perm[f]
        rot_t, rot_s = (p != f), (q != p)
        if rot_t and rot_s:
            ok = union(t_id[f], nt + s_id[p], 0 if sgn(f, p) * sgn(p, q) == 1 else 1)
        elif rot_t:                       # only tau acts on f
            ok = union(t_id[f], GND, 0 if sgn(f, p) == 1 else 1)
        elif rot_s:                       # only sig acts (tau fixes f)
            ok = union(nt + s_id[p], GND, 0 if sgn(p, q) == 1 else 1)
        else:
            continue                      # global fixed point, sign already +1
        if not ok:
            raise RuntimeError(f"orientation parity system inconsistent at coord {f}")

    half = np.pi / 2
    t_ang = [half if find(k)[1] == 0 else -half for k in range(nt)]
    s_ang = [half if find(nt + k)[1] == 0 else -half for k in range(ns)]
    return t_ang, s_ang


def oriented_givens_matrix(n, pairs, angles):
    """Givens layer with a per-plane angle (lets us pick +90 vs -90 to cancel signs)."""
    G = np.eye(n)
    for (i, j), th in zip(pairs, angles):
        c, s = np.cos(th), np.sin(th)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = -s
        G[j, i] = s
    return G


def signfree_layers(perm):
    """Return (tpairs, t_angles, spairs, s_angles): two oriented Givens layers whose product is
    EXACTLY perm1 (apply tau layer first, then sig layer). No residual signs."""
    tau, sig = two_matchings(perm)
    t_ang, s_ang = solve_orientations(perm, tau, sig)
    return transpositions_of(tau), t_ang, transpositions_of(sig), s_ang


def verify(n=2048, nblocks=32, per_block=64):
    perm = perm1(n, nblocks, per_block)
    assert sorted(perm) == list(range(n)), "perm1 is not a valid permutation"

    cyc = cycles_of(perm)
    lengths = sorted(set(len(c) for c in cyc))
    print(f"n={n}  perm1 = transpose of {per_block}x{nblocks} grid (x{per_block} mod {n-1})")
    print(f"  cycles: {len(cyc)} total, lengths present = {lengths}, "
          f"fixed points = {sum(1 for c in cyc if len(c)==1)}")

    tau, sig = two_matchings(perm)

    # (1) each layer is a genuine involution / disjoint matching
    inv_ok = all(tau[tau[i]] == i for i in range(n)) and all(sig[sig[i]] == i for i in range(n))
    # (2) the two layers compose to perm1 as permutations
    comp = [sig[tau[i]] for i in range(n)]
    perm_ok = (comp == perm)
    t_pairs = transpositions_of(tau)
    s_pairs = transpositions_of(sig)
    print(f"  tau: {len(t_pairs)} disjoint swaps,  sig: {len(s_pairs)} disjoint swaps "
          f"(max per layer = {n//2})")
    print(f"  [perm] both layers are involutions : {inv_ok}")
    print(f"  [perm] sig . tau == perm1          : {perm_ok}")

    # (3) the ACTUAL Givens rotation matrices reproduce perm1 up to sign.
    #     M = sig_layer @ tau_layer (tau applied first). Compare |M| to the permutation matrix,
    #     and confirm M equals P * diag(+-1) exactly (a signed permutation == perm1 up to sign).
    Gt = givens_matrix(n, t_pairs)
    Gs = givens_matrix(n, s_pairs)
    M = Gs @ Gt
    P = permutation_matrix(perm)

    orth_err = np.abs(M.T @ M - np.eye(n)).max()           # M is orthogonal
    abs_err = np.abs(np.abs(M) - P).max()                  # |M| == permutation matrix
    # signed-permutation check: every nonzero of M is +-1 at exactly perm1's positions
    signs = M[perm, range(n)]                               # the (perm[f], f) entries
    is_signed_perm = np.allclose(np.abs(signs), 1.0) and np.allclose(np.abs(M).sum(0), 1.0) \
        and np.allclose(np.abs(M).sum(1), 1.0)
    print(f"  [matrix] orthogonal (max|MᵀM-I|)   : {orth_err:.2e}")
    print(f"  [matrix] |M| == perm-matrix (maxerr): {abs_err:.2e}")
    print(f"  [matrix] M is perm1 up to sign     : {is_signed_perm}")

    # (4) end-to-end on a random vector: routing matches up to per-coordinate sign.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    y = M @ x                       # apply the two Givens layers
    y_perm = x[np.array(perm).argsort()]   # plain perm1 routing of x (no sign): y_perm[perm[f]] = x[f]
    route_ok = np.allclose(np.abs(y), np.abs(y_perm))      # same magnitudes, possibly flipped sign
    print(f"  [vector] |Givens(x)| == |perm1(x)| : {route_ok}")

    all_ok = inv_ok and perm_ok and is_signed_perm and route_ok and orth_err < 1e-10
    print(f"\n  ALL CHECKS PASS: {all_ok}")
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--nblocks", type=int, default=32)
    ap.add_argument("--per-block", type=int, default=64)
    args = ap.parse_args()
    ok = verify(args.n, args.nblocks, args.per_block)
    raise SystemExit(0 if ok else 1)
