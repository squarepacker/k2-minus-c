# Independent geometry helpers (written from scratch for the adversarial review of cstar_v7.tex).
# Squares are closed unit squares given by 4 vertices (CCW numpy array 4x2).
import math
import numpy as np

HALF_PI = math.pi / 2


def square(cx, cy, phi):
    c, s = math.cos(phi), math.sin(phi)
    hx = np.array([c, s]) * 0.5
    hy = np.array([-s, c]) * 0.5
    C = np.array([cx, cy], dtype=float)
    return np.array([C - hx - hy, C + hx - hy, C + hx + hy, C - hx + hy])


def incl_phi(phi):
    m = phi % HALF_PI
    return min(m, HALF_PI - m)


def incl_poly(P):
    e = P[1] - P[0]
    return incl_phi(math.atan2(e[1], e[0]))


def sat_gap(P, Q):
    """max over edge normals of both polygons of the separation; >0 iff closed convex polygons disjoint."""
    best = -np.inf
    for poly in (P, Q):
        n = len(poly)
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            nrm = np.array([-e[1], e[0]]) / math.hypot(e[0], e[1])
            pP = P @ nrm
            pQ = Q @ nrm
            g = max(pQ.min() - pP.max(), pP.min() - pQ.max())
            if g > best:
                best = g
    return best


def vchord(P, xs):
    """vertical chord [ylo, yhi] of convex polygon P at abscissae xs (array). NaN where missed."""
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    lo = np.full(xs.shape, np.inf)
    hi = np.full(xs.shape, -np.inf)
    n = len(P)
    for i in range(n):
        p0 = P[i]
        p1 = P[(i + 1) % n]
        x0, x1 = p0[0], p1[0]
        if x0 == x1:
            m = xs == x0
            if m.any():
                lo[m] = np.minimum(lo[m], min(p0[1], p1[1]))
                hi[m] = np.maximum(hi[m], max(p0[1], p1[1]))
            continue
        a, b = (x0, x1) if x0 < x1 else (x1, x0)
        m = (xs >= a) & (xs <= b)
        if m.any():
            y = p0[1] + (xs[m] - x0) * (p1[1] - p0[1]) / (x1 - x0)
            lo[m] = np.minimum(lo[m], y)
            hi[m] = np.maximum(hi[m], y)
    miss = lo > hi
    lo[miss] = np.nan
    hi[miss] = np.nan
    return lo, hi


def hchord(P, ys):
    Q = P[:, ::-1]
    return vchord(Q, ys)


def minkowski_pts(A, B):
    """points A_i - B_j (their convex hull is A - B)."""
    return (A[:, None, :] - B[None, :, :]).reshape(-1, 2)


def hull(pts):
    pts = sorted(set(map(tuple, np.round(pts, 17))))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def push_right_contact(A, B, dy=0.0):
    """B shifted by (s, dy): returns s_hi = largest s such that A and B+(s,dy) intersect, or None.
    So for s > s_hi, B+(s,dy) lies strictly to the right of A (if they share a horizontal line)."""
    H = hull(minkowski_pts(A, B))
    # (s, dy) in A - B  <=>  B + (s,dy) meets A ; horizontal chord of hull at height dy
    lo, hi = hchord(H, np.array([dy]))
    if np.isnan(hi[0]):
        return None
    return hi[0]


def push_down_contact(A, B):
    """smallest... returns t_hi such that B+(0,-t) meets A iff t in [t_lo,t_hi]; i.e. dropping B by t
    first touches A at t = t_lo. Returns t_lo or None (no vertical overlap)."""
    H = hull(minkowski_pts(A, B))
    # B + (0,-t) meets A  <=>  (0,-t) in A - B
    lo, hi = vchord(H, np.array([0.0]))
    if np.isnan(lo[0]):
        return None
    # -t in [lo,hi]  <=> t in [-hi, -lo]
    return -hi[0]
