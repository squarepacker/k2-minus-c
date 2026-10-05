"""Own geometry for the Lemma K_w attack (written from scratch).
Unit squares: centre c=(x,y), angle t. half side H=0.5.
Walls: list of (nx, ny, o): admissible region n.x >= o (unit normal n).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

H = 0.5


def verts(C, T):
    """C (...,2), T (...) -> (...,4,2) in cyclic order."""
    c, s = np.cos(T), np.sin(T)
    u = np.stack([c, s], -1) * H
    v = np.stack([-s, c], -1) * H
    V = np.stack([C + u + v, C - u + v, C - u - v, C + u - v], -2)
    return V


def _pt_seg(P, A, B):
    """distance from points P to segments AB, broadcasting over leading dims."""
    AB = B - A
    AP = P - A
    den = np.sum(AB * AB, -1)
    t = np.clip(np.sum(AP * AB, -1) / den, 0.0, 1.0)
    D = AP - t[..., None] * AB
    return np.sqrt(np.sum(D * D, -1))


def sat_sep(VA, TA, VB, TB):
    """max over the 4 edge normals of the separation (positive = separated by that gap along axis).
    VA,VB (N,4,2), TA,TB (N,)."""
    ax = np.stack([np.stack([np.cos(TA), np.sin(TA)], -1),
                   np.stack([-np.sin(TA), np.cos(TA)], -1),
                   np.stack([np.cos(TB), np.sin(TB)], -1),
                   np.stack([-np.sin(TB), np.cos(TB)], -1)], -2)  # (N,4,2)
    pa = np.einsum('nkd,nad->nak', VA, ax)  # (N,axes,verts)
    pb = np.einsum('nkd,nad->nak', VB, ax)
    s1 = pb.min(-1) - pa.max(-1)
    s2 = pa.min(-1) - pb.max(-1)
    return np.maximum(s1, s2).max(-1)


def edist(VA, VB):
    """Euclidean distance between disjoint convex quads (vertex-edge minimum)."""
    A0 = VA; A1 = np.roll(VA, -1, axis=-2)
    B0 = VB; B1 = np.roll(VB, -1, axis=-2)
    d1 = _pt_seg(VA[:, :, None, :], B0[:, None, :, :], B1[:, None, :, :])
    d2 = _pt_seg(VB[:, :, None, :], A0[:, None, :, :], A1[:, None, :, :])
    return np.minimum(d1.min((-1, -2)), d2.min((-1, -2)))


def signed_gap(CA, TA, CB, TB):
    """Euclidean distance if disjoint (SAT>0), else SAT separation (<=0, minus penetration depth)."""
    VA = verts(CA, TA); VB = verts(CB, TB)
    s = sat_sep(VA, TA, VB, TB)
    e = edist(VA, VB)
    return np.where(s > 0, e, s)


def point_sq(p, C, T):
    """signed distance of point p to squares (positive outside)."""
    q = p - C
    c, s = np.cos(T), np.sin(T)
    lx = q[..., 0] * c + q[..., 1] * s
    ly = -q[..., 0] * s + q[..., 1] * c
    dx = np.abs(lx) - H; dy = np.abs(ly) - H
    out = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
    inn = np.maximum(dx, dy)
    return np.where(inn > 0, out, inn)


def rect_margin(p, A, B):
    """margin of p in the width-1 rectangle around segment AB (positive = interior)."""
    D = B - A
    L = np.sqrt(np.sum(D * D, -1))
    e = D / L[..., None]
    q = p - A
    t = np.sum(q * e, -1)
    s = np.abs(q[..., 0] * e[..., 1] - q[..., 1] * e[..., 0])
    return np.minimum(np.minimum(t, L - t), H - s)


def wall_gap(C, T, wall):
    n = np.array(wall[:2]); o = wall[2]
    V = verts(C, T)
    return (V @ n).min(-1) - o


def wall_foot(C, wall):
    n = np.array(wall[:2]); o = wall[2]
    h = C @ n - o
    return C - h[..., None] * n


def analyse(C, T, p, d, walls=(), tol=1e-12):
    """Full exact-ish (float64) analysis. Returns dict."""
    C = np.asarray(C, float); T = np.asarray(T, float); p = np.asarray(p, float)
    n = len(C)
    I, J = np.triu_indices(n, 1)
    g = signed_gap(C[I], T[I], C[J], T[J])
    ps = point_sq(p, C, T)
    m = rect_margin(p, C[I], C[J])
    ok_disjoint = bool(np.all(g >= tol))
    ok_waste = bool(np.all(ps > tol))
    pairs = []
    for k in range(len(I)):
        if g[k] < d and m[k] >= -tol:
            pairs.append((int(I[k]), int(J[k]), float(g[k]), float(m[k])))
    wpairs = []
    ok_walls = True
    for w_i, w in enumerate(walls):
        wg = wall_gap(C, T, w)
        if np.any(wg < tol):
            ok_walls = False
        # p must be in admissible region
        if np.dot(w[:2], p) - w[2] <= tol:
            ok_walls = False
        F = wall_foot(C, w)
        wm = rect_margin(p, C, F)
        for i in range(n):
            if wg[i] < d and wm[i] >= -tol:
                wpairs.append((i, 'W%d' % w_i, float(wg[i]), float(wm[i])))
    deg = [0] * n
    for a, b, *_ in pairs:
        deg[a] += 1; deg[b] += 1
    for a, *_ in wpairs:
        deg[a] += 1
    return dict(valid=ok_disjoint and ok_waste and ok_walls, disjoint=ok_disjoint, waste=ok_waste,
                walls_ok=ok_walls, count=len(pairs) + len(wpairs), pairs=pairs, wpairs=wpairs,
                deg=deg, min_gap=float(g.min()) if len(g) else None, min_psq=float(ps.min()),
                strict_count=sum(1 for x in pairs if x[3] > tol) + sum(1 for x in wpairs if x[3] > tol),
                dist_p=[float(x) for x in np.hypot(*(C - p).T)])
