"""Approach (b): hard-square Monte Carlo compression (gravity toward a point / wall / corner),
then search for the waste point p maximizing the K_w count.
Usage: python jam.py OUTFILE SEED TIME_LIMIT_SEC MODES
"""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from geom import signed_gap, point_sq, rect_margin, wall_gap, wall_foot, analyse, verts
from fastsa import axes_of, verts1

H = 0.5


def sat_one(ci, Vi, Ai, C, V, A):
    pj = V @ Ai.T
    cip = Ai @ ci
    s1 = np.maximum(pj.min(1) - (cip + H), (cip - H) - pj.max(1)).max(1)
    pi = np.einsum('kd,mad->mka', Vi, A)
    cjp = np.einsum('md,mad->ma', C, A)
    s2 = np.maximum(pi.min(1) - (cjp + H), (cjp - H) - pi.max(1)).max(1)
    return np.maximum(s1, s2)


def jam(rng, n, mode, steps):
    # walls: region n.x >= o
    if mode == 'free':
        walls = []
        grav = lambda c: c @ c
    elif mode == 'wall':
        walls = [(np.array([0.0, 1.0]), 0.0)]
        grav = lambda c: c[1] * 3 + 0.3 * c[0] ** 2
    else:
        walls = [(np.array([0.0, 1.0]), 0.0), (np.array([1.0, 0.0]), 0.0)]
        grav = lambda c: (c[0] + c[1]) * 3
    # random sequential insertion
    C = []; T = []
    R = 0.9 * np.sqrt(n) + 1
    tries = 0
    while len(C) < n and tries < 100000:
        tries += 1
        if mode == 'free':
            c = rng.uniform(-R, R, 2)
            if c @ c > R * R: continue
        elif mode == 'wall':
            c = np.array([rng.uniform(-R, R), rng.uniform(0.71, 2 * R)])
        else:
            c = rng.uniform(0.71, 1.6 * R, 2)
        t = rng.uniform(0, np.pi / 2)
        Vi = verts1(c, t)
        if any((Vi @ w[0]).min() - w[1] <= 0 for w in walls):
            continue
        if C:
            Ca = np.array(C); Ta = np.array(T)
            V = np.array([verts1(Ca[k], Ta[k]) for k in range(len(Ca))])
            A = np.array([axes_of(x) for x in Ta])
            if sat_one(c, Vi, axes_of(t), Ca, V, A).min() <= 0: continue
        C.append(c); T.append(t)
    C = np.array(C); T = np.array(T); n = len(C)
    V = np.array([verts1(C[k], T[k]) for k in range(n)])
    A = np.array([axes_of(x) for x in T])
    sig = 0.2; acc = 0; tot = 0
    for k in range(steps):
        fr = k / steps
        Temp = 1.0 * (1e-7 / 1.0) ** fr
        i = rng.integers(n)
        nc = C[i] + rng.normal(0, sig, 2); nt = T[i] + rng.normal(0, sig)
        dE = grav(nc) - grav(C[i])
        tot += 1
        if dE > 0 and rng.random() >= np.exp(-dE / Temp):
            pass
        else:
            Vi = verts1(nc, nt)
            ok = all((Vi @ w[0]).min() - w[1] > 0 for w in walls)
            if ok:
                m_ = np.arange(n) != i
                ok = sat_one(nc, Vi, axes_of(nt), C[m_], V[m_], A[m_]).min() > 0
            if ok:
                C[i] = nc; T[i] = nt; V[i] = Vi; A[i] = axes_of(nt); acc += 1
        if tot == 500:
            r = acc / tot
            sig *= 1.3 if r > 0.3 else 0.75
            sig = min(max(sig, 1e-9), 0.3)
            acc = tot = 0
    return C, T, [(float(w[0][0]), float(w[0][1]), float(w[1])) for w in walls]


def pair_list(C, T, walls, d):
    n = len(C)
    I, J = np.triu_indices(n, 1)
    near = np.hypot(*(C[I] - C[J]).T) < 1.5
    I, J = I[near], J[near]
    g = signed_gap(C[I], T[I], C[J], T[J])
    sel = g < d
    segs = [(C[i], C[j]) for i, j in zip(I[sel], J[sel])]
    lab = [(int(i), int(j)) for i, j in zip(I[sel], J[sel])]
    for wi, w in enumerate(walls):
        wg = wall_gap(C, T, w)
        F = wall_foot(C, w)
        for i in np.nonzero(wg < d)[0]:
            segs.append((C[i], F[i])); lab.append((int(i), 'W%d' % wi))
    return segs, lab


def counts_at(P, segs, C, T, walls):
    """P (m,2) candidate points; returns count array (0 where not waste)."""
    if not segs:
        return np.zeros(len(P), int)
    A = np.array([s[0] for s in segs]); B = np.array([s[1] for s in segs])
    D = B - A; L = np.hypot(*D.T); e = D / L[:, None]
    Q = P[:, None, :] - A[None]
    t = (Q * e[None]).sum(-1)
    s = np.abs(Q[..., 0] * e[None, :, 1] - Q[..., 1] * e[None, :, 0])
    inside = (t >= 0) & (t <= L[None]) & (s <= H)
    cnt = inside.sum(1)
    # waste
    q = P[:, None, :] - C[None]
    c, sn = np.cos(T), np.sin(T)
    lx = q[..., 0] * c + q[..., 1] * sn; ly = -q[..., 0] * sn + q[..., 1] * c
    insq = (np.abs(lx) <= H + 1e-12) & (np.abs(ly) <= H + 1e-12)
    waste = ~insq.any(1)
    for w in walls:
        waste &= (P @ np.array(w[:2]) - w[2]) > 1e-12
    return np.where(waste, cnt, 0)


def best_p(C, T, walls, d, rng):
    segs, lab = pair_list(C, T, walls, d)
    if not segs:
        return None, 0
    # only points near some segment rectangle matter: use segment midpoints region
    mids = np.array([(a + b) / 2 for a, b in segs])
    cand = []
    lo = mids.min(0) - 1; hi = mids.max(0) + 1
    xs = np.arange(lo[0], hi[0], 0.02); ys = np.arange(lo[1], hi[1], 0.02)
    G = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    G = G + rng.uniform(-0.01, 0.01, G.shape)
    cand.append(G)
    V = verts(C, T)  # (n,4,2)
    tt = np.linspace(0, 1, 41)
    for off in (1e-9, 1e-6, 1e-4, 3e-3):
        for k in range(4):
            a = V[:, k]; b = V[:, (k + 1) % 4]
            nrm = (a + b) / 2 - C; nrm = nrm / np.hypot(*nrm.T)[:, None]
            P = a[:, None] + tt[None, :, None] * (b - a)[:, None] + off * nrm[:, None]
            cand.append(P.reshape(-1, 2))
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
        cand.append((V.reshape(-1, 2)[:, None] + off * 3 * dirs[None]).reshape(-1, 2))
    P = np.concatenate(cand)
    best = (None, 0)
    for k0 in range(0, len(P), 20000):
        cc = counts_at(P[k0:k0 + 20000], segs, C, T, walls)
        j = int(cc.argmax())
        if cc[j] > best[1]:
            best = (P[k0 + j].copy(), int(cc[j]))
    # local refinement
    p, c0 = best
    if p is None:
        return None, 0
    for sc in (3e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        Q = p + rng.normal(0, sc, (3000, 2))
        cc = counts_at(Q, segs, C, T, walls)
        j = int(cc.argmax())
        if cc[j] > c0:
            p, c0 = Q[j].copy(), int(cc[j])
    return p, c0


def main():
    outf, seed, tlim = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    modes = sys.argv[4].split(',')
    rng = np.random.default_rng(seed)
    t0 = time.time(); run = 0
    while time.time() - t0 < tlim:
        mode = modes[run % len(modes)]
        n = int(rng.integers(12, 26))
        steps = int(rng.integers(80000, 200000))
        C, T, walls = jam(rng, n, mode, steps)
        # make strictly disjoint: tiny expansion about the gravity point
        C = C * (1 + 2e-9) + (np.array([0, 1e-9]) if mode != 'free' else 0)
        rec = dict(seed=seed, run=run, mode=mode, n=len(C), steps=steps, walls=walls)
        for d in (1e-4, 1e-3, 1e-2):
            p, c = best_p(C, T, walls, d, rng)
            if p is None:
                rec['d%g' % d] = None; continue
            A = analyse(C, T, p, d, walls)
            rec['d%g' % d] = dict(count=A['count'], valid=A['valid'], strict=A['strict_count'], deg=A['deg'],
                                 p=[float(x) for x in p], min_gap=A['min_gap'], min_psq=A['min_psq'])
        rec['C'] = [[float(a), float(b)] for a, b in C]; rec['T'] = [float(x) for x in T]
        rec['t'] = time.time() - t0
        with open(outf, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(run, mode, len(C), [(rec[k]['count'], rec[k]['valid']) if rec[k] else None for k in ('d0.0001', 'd0.001', 'd0.01')], flush=True)
        run += 1


if __name__ == '__main__':
    main()
