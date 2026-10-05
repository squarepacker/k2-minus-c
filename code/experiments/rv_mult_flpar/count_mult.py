# Target 3(b): multiplicity of the rectangles R_e over edges e of the horizontal visibility graph G_h
# whose members are at distance < d (Lemma lem:count(b) claims <= K = 27).
import sys, json, math, time
import numpy as np
from geom import square, vchord, hchord, sat_gap, hull, minkowski_pts
from crack_full import drop_dist, push_left_dist

D_THR = 1e-4

def poly_dist(A, B):
    # exact distance between disjoint convex polygons: min over vertex-edge distances
    def pe(p, a, b):
        ab = b - a; t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
        return np.hypot(*(a + t * ab - p))
    best = np.inf
    for P, Q in ((A, B), (B, A)):
        for p in P:
            for i in range(4):
                best = min(best, pe(p, Q[i], Q[(i + 1) % 4]))
    return best

def edges_Gh(sqs, k, nlines=6000):
    ys = np.linspace(1e-7, k - 1e-7, nlines)
    L = np.full((len(sqs), nlines), np.nan); Rr = np.full((len(sqs), nlines), np.nan)
    for j, S in enumerate(sqs):
        lo, hi = hchord(S, ys)
        L[j] = lo; Rr[j] = hi
    E = {}
    for i in range(nlines):
        idx = np.where(~np.isnan(L[:, i]))[0]
        order = idx[np.argsort(L[idx, i])]
        seq = ["W-"] + list(order) + ["W+"]
        for a, b in zip(seq[:-1], seq[1:]):
            key = (a, b)
            E[key] = E.get(key, 0) + 1
    return E

def rect(c1, c2):
    u = c2 - c1; Lh = np.hypot(*u); u = u / Lh; n = np.array([-u[1], u[0]])
    return c1, u, n, Lh

def in_rect(P, R):
    c1, u, n, Lh = R
    D = P - c1
    s = D @ u; t = D @ n
    return (s >= -1e-12) & (s <= Lh + 1e-12) & (np.abs(t) <= 0.5 + 1e-12)

def multiplicity(sqs, k, d=D_THR, grid=0.004, region=None):
    E = edges_Gh(sqs, k)
    rects = []
    for (a, b), cnt in E.items():
        if a == "W-" and b == "W+":
            continue
        if a == "W-" or b == "W+":
            S = sqs[b] if a == "W-" else sqs[a]
            dist = S[:, 0].min() if a == "W-" else k - S[:, 0].max()
            if dist >= d:
                continue
            c = S.mean(0)
            # nearest point of the boundary of [0,k]^2
            cand = [np.array([0, c[1]]), np.array([k, c[1]]), np.array([c[0], 0]), np.array([c[0], k])]
            dd = [c[0], k - c[0], c[1], k - c[1]]
            c2 = cand[int(np.argmin(dd))]
            rects.append(rect(c, c2))
        else:
            if poly_dist(sqs[a], sqs[b]) >= d:
                continue
            rects.append(rect(sqs[a].mean(0), sqs[b].mean(0)))
    if not rects:
        return 0, None, 0
    if region is None:
        region = (0, k, 0, k)
    xs = np.arange(region[0], region[1] + 1e-12, grid); ys = np.arange(region[2], region[3] + 1e-12, grid)
    X, Y = np.meshgrid(xs, ys); P = np.stack([X.ravel(), Y.ravel()], 1)
    cnt = np.zeros(len(P), int)
    for R in rects:
        cnt += in_rect(P, R)
    i = int(np.argmax(cnt))
    # refine around the best point on a finer grid
    p0 = P[i]
    xs = np.linspace(p0[0] - grid, p0[0] + grid, 81); ys = np.linspace(p0[1] - grid, p0[1] + grid, 81)
    X, Y = np.meshgrid(xs, ys); P2 = np.stack([X.ravel(), Y.ravel()], 1)
    cnt2 = np.zeros(len(P2), int)
    for R in rects:
        cnt2 += in_rect(P2, R)
    j = int(np.argmax(cnt2))
    return int(max(cnt.max(), cnt2.max())), P2[j].tolist(), len(rects)

def columns_config(rng, k, ncols, g):
    """columns of stacked axis-parallel squares, random vertical offsets, gaps g < d."""
    sqs = []
    x = g * rng.random() if rng.random() < 0.5 else 0.3 + rng.random()
    for c in range(ncols):
        off = rng.random()
        y = off - 1 + g
        tilt = 0.0
        while y + 1 <= k - g:
            if y >= g:
                sqs.append(square(x + 0.5, y + 0.5, tilt))
            y += 1 + g * rng.random()
        x += 1 + g * rng.random()
    return sqs

def jammed_config(rng, k, n, g, anyangle):
    sqs = []
    for i in range(n):
        phi = rng.random() * math.pi / 2 if anyangle else 0.05 * (2 * rng.random() - 1)
        B = square(0.8 + (k - 1.6) * rng.random(), k + 1.0, phi)
        B = B - np.array([0, 2.0])
        for it in range(4):
            t = drop_dist(sqs, B)
            if t < 0:
                B = None; break
            B = B - np.array([0, max(t - g * rng.random(), 0)])
            t2 = push_left_dist(sqs, B)
            if t2 < 0:
                B = None; break
            B = B - np.array([max(t2 - g * rng.random(), 0), 0])
        if B is None or B[:, 1].max() > k or B[:, 0].max() > k:
            continue
        if all(sat_gap(A, B) > 0 for A in sqs) and B.min() >= 0:
            sqs.append(B)
    return sqs

if __name__ == "__main__":
    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    t0 = time.time(); best = {"mult": 0}; hist = {}
    n = 0
    while time.time() - t0 < tl:
        n += 1
        k = 5
        r = rng.random()
        g = D_THR * 10 ** (-2 * rng.random()) * 0.3
        if r < 0.4:
            sqs = columns_config(rng, k, 4, g); fam = "columns"
        elif r < 0.7:
            sqs = jammed_config(rng, k, 25, g, False); fam = "jammed-small-tilt"
        else:
            sqs = jammed_config(rng, k, 25, g, True); fam = "jammed-any-angle"
        if len(sqs) < 3:
            continue
        m, p, ne = multiplicity(sqs, k, grid=0.01)
        hist[m] = hist.get(m, 0) + 1
        if m > best["mult"]:
            best = {"mult": m, "p": p, "family": fam, "nedges": ne, "squares": [S.tolist() for S in sqs]}
            print(fam, m, ne, flush=True)
            with open("count_mult_results.json", "w") as f:
                json.dump({"best": best, "hist": hist, "configs": n}, f)
    with open("count_mult_results.json", "w") as f:
        json.dump({"best": best, "hist": {str(a): b for a, b in hist.items()}, "configs": n}, f)
    print("configs", n, "hist", hist, "best", best["mult"])
