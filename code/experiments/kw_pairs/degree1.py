"""Task 4: (i) angle fact check, (ii) max single-centre degree.
(ii): square P axis-aligned at 0; waste point p at distance r, polar angle phi. Candidate neighbours Q
(direction alpha, rotation theta, gap = d/2 to P), kept if p outside Q (strictly) and p in R(P,Q).
Max set of pairwise disjoint candidates = longest compatible chain in alpha order (DP upper bound), then verified.
"""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from geom import signed_gap, point_sq, rect_margin

rng = np.random.default_rng(7)
d = 1e-4
Lam = d + np.sqrt(2)
gamma0 = np.degrees(np.arccos(1 - 1 / (2 * Lam ** 2)))
# ---------- (i)
N = 2_000_000
a = rng.uniform(1, Lam, N); b = rng.uniform(1, Lam, N)
# include the extreme corner explicitly
a[:1000] = Lam - rng.uniform(0, 1e-9, 1000); b[:1000] = Lam - rng.uniform(0, 1e-9, 1000)
cosmax = (a ** 2 + b ** 2 - 1) / (2 * a * b)
phi_min = np.degrees(np.arccos(np.clip(cosmax, -1, 1)))
print('(i) gamma0 =', gamma0, ' sampled min angle =', phi_min.min(), ' (should be >= gamma0 up to 1e-6)')
# random geometric check with actual disjoint squares:
viol = 0; minang = 180
for _ in range(20):
    M = 200000
    c1 = rng.uniform(-1.5, 1.5, (M, 2)); c2 = rng.uniform(-1.5, 1.5, (M, 2))
    t0 = rng.uniform(0, np.pi / 2, M); t1 = rng.uniform(0, np.pi / 2, M); t2 = rng.uniform(0, np.pi / 2, M)
    Z = np.zeros((M, 2))
    r1 = np.hypot(*c1.T); r2 = np.hypot(*c2.T)
    s = (r1 < Lam) & (r2 < Lam)
    c1, c2, t0, t1, t2, Z = c1[s], c2[s], t0[s], t1[s], t2[s], Z[s]
    g01 = signed_gap(Z, t0, c1, t1); g02 = signed_gap(Z, t0, c2, t2); g12 = signed_gap(c1, t1, c2, t2)
    ok = (g01 > 0) & (g02 > 0) & (g12 > 0) & (g01 < d) & (g02 < d)
    if ok.any():
        cc1, cc2 = c1[ok], c2[ok]
        ang = np.degrees(np.arccos(np.clip((cc1 * cc2).sum(1) / np.hypot(*cc1.T) / np.hypot(*cc2.T), -1, 1)))
        minang = min(minang, ang.min()); viol += int((ang < gamma0 - 1e-9).sum())
print('(i) random disjoint-square triples: min angle seen =', minang, ' violations =', viol)


def bound(r):
    return 1 + int(np.floor(2 * np.degrees(np.arcsin(min(1, 1 / (2 * r)))) / gamma0))


def place(alpha, theta):
    """distance s along direction alpha with gap(P,Q)=d/2 (bisection)."""
    u = np.stack([np.cos(alpha), np.sin(alpha)], -1)
    lo = np.full(len(alpha), 0.9); hi = np.full(len(alpha), 1.6)
    Z = np.zeros((len(alpha), 2)); Tz = np.zeros(len(alpha))
    for _ in range(60):
        mid = (lo + hi) / 2
        g = signed_gap(Z, Tz, u * mid[:, None], theta)
        big = g > d / 2
        hi = np.where(big, mid, hi); lo = np.where(big, lo, mid)
    return u * hi[:, None]


def max_degree(r, phi, da=1.0, dt=3.0):
    p = r * np.array([np.cos(phi), np.sin(phi)])
    if point_sq(p, np.zeros(2), 0.0) <= 1e-12:
        return None
    al = np.radians(np.arange(0, 360, da)); th = np.radians(np.arange(0, 90, dt))
    A, Tq = np.meshgrid(al, th, indexing='ij'); A = A.ravel(); Tq = Tq.ravel()
    # quick prefilter by cone
    dirp = np.arctan2(p[1], p[0])
    dang = np.angle(np.exp(1j * (A - dirp)))
    keep = np.abs(dang) <= np.arcsin(min(1, 1 / (2 * r))) + np.radians(2)
    A, Tq = A[keep], Tq[keep]
    Cq = place(A, Tq)
    m = rect_margin(p, np.zeros_like(Cq), Cq)
    ps = point_sq(p, Cq, Tq)
    ok = (m >= 0) & (ps > 1e-12)
    A, Tq, Cq = A[ok], Tq[ok], Cq[ok]
    if len(A) == 0:
        return 0, None
    order = np.argsort(np.angle(np.exp(1j * (A - dirp))))
    A, Tq, Cq = A[order], Tq[order], Cq[order]
    n = len(A)
    # compatibility (only needed for pairs that are close)
    best = np.ones(n, int); prev = -np.ones(n, int)
    for j in range(n):
        if j == 0: continue
        I = np.arange(j)
        g = signed_gap(Cq[I], Tq[I], np.repeat(Cq[j:j+1], j, 0), np.repeat(Tq[j:j+1], j))
        comp = g > 1e-12
        if comp.any():
            cand = np.where(comp, best[I], 0)
            k = int(cand.argmax())
            if cand[k] + 1 > best[j]:
                best[j] = cand[k] + 1; prev[j] = k
    j = int(best.argmax()); chain = []
    while j >= 0:
        chain.append(j); j = prev[j]
    chain = chain[::-1]
    # verify full pairwise
    Cc, Tc = Cq[chain], Tq[chain]
    allok = True
    for x in range(len(chain)):
        for y in range(x + 1, len(chain)):
            if signed_gap(Cc[x:x+1], Tc[x:x+1], Cc[y:y+1], Tc[y:y+1])[0] <= 1e-12:
                allok = False
    return int(best.max()), dict(verified=allok, C=Cc.tolist(), T=Tc.tolist(), p=p.tolist())


res = []
t0 = time.time()
for r in [0.5000001, 0.501, 0.51, 0.53, 0.55, 0.58, 0.6, 0.65, 0.7, 0.7071, 0.75, 0.8, 0.85, 0.866]:
    bestk = -1; bestinfo = None
    phmax = np.degrees(np.arccos(min(1, 0.5 / r))) if r > 0.5 else 0
    for phd in np.linspace(0, 45, 10):
        out = max_degree(r, np.radians(phd))
        if out is None: continue
        k, info = out
        if k > bestk: bestk, bestinfo = k, (phd, info)
    print('r=%.7f  bound=%d  found=%d at phi=%.1f deg  verified=%s  (%.0fs)' % (r, bound(r), bestk, bestinfo[0], bestinfo[1]['verified'] if bestinfo[1] else None, time.time() - t0), flush=True)
    res.append(dict(r=r, bound=bound(r), found=bestk, phi=bestinfo[0], cfg=bestinfo[1]))
json.dump(dict(gamma0=gamma0, results=res), open('degree1_results.json', 'w'), indent=1)
