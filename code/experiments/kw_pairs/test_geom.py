"""Fuzz geom.py against independent brute force (boundary sampling, polygon clipping area)."""
import numpy as np
from geom import *

rng = np.random.default_rng(1)


def brute_dist(VA, VB, k=400):
    t = np.linspace(0, 1, k, endpoint=False)
    def samp(V):
        pts = []
        for i in range(4):
            a, b = V[i], V[(i + 1) % 4]
            pts.append(a[None] + t[:, None] * (b - a)[None])
        return np.concatenate(pts)
    SA, SB = samp(VA), samp(VB)
    D = np.sqrt(((SA[:, None] - SB[None]) ** 2).sum(-1))
    return D.min()


def inside(V, q):
    s = []
    for i in range(4):
        a, b = V[i], V[(i + 1) % 4]
        s.append((b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]))
    s = np.array(s)
    return np.all(s >= 0) or np.all(s <= 0)


def seg_int(a, b, c, d):
    def cr(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1, d2, d3, d4 = cr(c, d, a), cr(c, d, b), cr(a, b, c), cr(a, b, d)
    return (d1 * d2 <= 0) and (d3 * d4 <= 0)


def intersects(VA, VB):
    if any(inside(VB, q) for q in VA) or any(inside(VA, q) for q in VB):
        return True
    for i in range(4):
        for j in range(4):
            if seg_int(VA[i], VA[(i + 1) % 4], VB[j], VB[(j + 1) % 4]):
                return True
    return False


bad = 0
N = 3000
CA = rng.uniform(-1, 1, (N, 2)); CB = rng.uniform(-1, 1, (N, 2)) * 1.3
TA = rng.uniform(0, np.pi, N); TB = rng.uniform(0, np.pi, N)
g = signed_gap(CA, TA, CB, TB)
VA = verts(CA, TA); VB = verts(CB, TB)
maxerr = 0
for k in range(N):
    it = intersects(VA[k], VB[k])
    if it != (g[k] <= 0):
        bad += 1
        print('disjoint mismatch', k, g[k])
    if g[k] > 0 and k < 600:
        bd = brute_dist(VA[k], VB[k])
        maxerr = max(maxerr, bd - g[k])
        if g[k] > bd + 1e-9 or bd - g[k] > 5e-3:
            bad += 1; print('dist mismatch', g[k], bd)
print('intersection/dist fuzz bad =', bad, ' max(brute-exact) =', maxerr)
# point-square
P = rng.uniform(-1, 1, (N, 2))
ps = point_sq(P, CA, TA)
for k in range(N):
    if inside(VA[k], P[k]) != (ps[k] <= 0):
        bad += 1; print('ps mismatch')
print('point-square fuzz bad =', bad)
# rectangle margin vs direct definition
A = rng.uniform(-1, 1, (N, 2)); B = rng.uniform(-1, 1, (N, 2))
m = rect_margin(P, A, B)
for k in range(N):
    D = B[k] - A[k]; L = np.linalg.norm(D)
    t = np.dot(P[k] - A[k], D) / L
    perp = np.linalg.norm(P[k] - A[k] - t * D / L)
    inR = (0 <= t <= L) and perp <= 0.5
    if inR != (m[k] >= 0):
        bad += 1; print('rect mismatch')
print('total bad =', bad)
