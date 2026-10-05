"""(i) targeted angle test: neighbours placed exactly at gap in (0,d) (uniform sampling never hits gap<d);
(ii) coarse single-centre degree search for r >= 0.55 (non-degenerate regime). Time-limited (~8 min)."""
import time, json
import numpy as np
from geom import signed_gap, point_sq, rect_margin

rng = np.random.default_rng(11)
d = 1e-4
Lam = d + np.sqrt(2)
gamma0 = np.degrees(np.arccos(1 - 1 / (2 * Lam ** 2)))
T0 = time.time()


def place(alpha, theta, gap):
    u = np.stack([np.cos(alpha), np.sin(alpha)], -1)
    lo = np.full(len(alpha), 0.9); hi = np.full(len(alpha), 1.6)
    Z = np.zeros((len(alpha), 2)); Tz = np.zeros(len(alpha))
    for _ in range(55):
        mid = (lo + hi) / 2
        g = signed_gap(Z, Tz, u * mid[:, None], theta)
        big = g > gap
        hi = np.where(big, mid, hi); lo = np.where(big, lo, mid)
    return u * hi[:, None]

# ---- (i)
minang = 180.0; nchk = 0; viol = 0
for rep in range(4):
    M = 20000
    a1 = rng.uniform(0, 2 * np.pi, M); da = rng.uniform(0, np.radians(60), M) * rng.choice([-1, 1], M)
    # bias toward small angular separation
    a2 = a1 + da
    t1 = rng.uniform(0, np.pi / 2, M); t2 = rng.uniform(0, np.pi / 2, M)
    g1 = rng.uniform(1e-9, d, M); g2 = rng.uniform(1e-9, d, M)
    c1 = place(a1, t1, g1); c2 = place(a2, t2, g2)
    ok = signed_gap(c1, t1, c2, t2) > 0
    if ok.any():
        ang = np.degrees(np.abs(np.angle(np.exp(1j * (a2[ok] - a1[ok])))))
        nchk += int(ok.sum()); minang = min(minang, float(ang.min())); viol += int((ang < gamma0 - 1e-9).sum())
print('(i) gamma0 = %.6f deg; %d disjoint neighbour pairs at gap<d tested; min angle = %.6f; violations = %d' % (gamma0, nchk, minang, viol), flush=True)


def bound(r):
    return 1 + int(np.floor(2 * np.degrees(np.arcsin(min(1, 1 / (2 * r)))) / gamma0))


def max_degree(r, phi, da=1.5, dt=5.0):
    p = r * np.array([np.cos(phi), np.sin(phi)])
    if point_sq(p, np.zeros(2), 0.0) <= 1e-12:
        return None
    al = np.radians(np.arange(0, 360, da)); th = np.radians(np.arange(0, 90, dt))
    A, Tq = np.meshgrid(al, th, indexing='ij'); A = A.ravel(); Tq = Tq.ravel()
    dirp = np.arctan2(p[1], p[0])
    keep = np.abs(np.angle(np.exp(1j * (A - dirp)))) <= np.arcsin(min(1, 1 / (2 * r))) + np.radians(2)
    A, Tq = A[keep], Tq[keep]
    Cq = place(A, Tq, np.full(len(A), d / 2))
    ok = (rect_margin(p, np.zeros_like(Cq), Cq) >= 0) & (point_sq(p, Cq, Tq) > 1e-12)
    A, Tq, Cq = A[ok], Tq[ok], Cq[ok]
    if len(A) == 0:
        return 0
    o = np.argsort(np.angle(np.exp(1j * (A - dirp)))); A, Tq, Cq = A[o], Tq[o], Cq[o]
    n = len(A); best = np.ones(n, int)
    for j in range(1, n):
        I = np.arange(j)
        g = signed_gap(Cq[I], Tq[I], np.repeat(Cq[j:j+1], j, 0), np.repeat(Tq[j:j+1], j))
        cand = np.where(g > 1e-12, best[I], 0)
        best[j] = max(best[j], int(cand.max()) + 1)
    return int(best.max())   # longest compatible chain: upper bound for the grid instance (consecutive compat.)

res = []
for r in [0.55, 0.6, 0.7071, 0.75, 0.8, 0.866]:
    if time.time() - T0 > 300: break
    bk = 0
    for phd in np.linspace(0, 45, 3):
        k = max_degree(r, np.radians(phd))
        if k is not None: bk = max(bk, k)
    print('(ii) r=%.4f bound=%d grid-chain=%d  (%.0fs)' % (r, bound(r), bk, time.time() - T0), flush=True)
    res.append(dict(r=r, bound=bound(r), grid_chain=bk))
json.dump(dict(gamma0=gamma0, min_angle=minang, n_checked=nchk, viol=viol, degree=res), open('angle_degree2.json', 'w'), indent=1)
