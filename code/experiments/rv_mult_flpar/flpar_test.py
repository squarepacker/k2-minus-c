# Target 4: Proposition prop:FLpar.  S1 = [-1/2,1/2]^2, S2 a square at distance < d with inclination
# theta <= 1e-5 (d = 1e-5), extra squares Z pushed into the free triangles by exact ray casting
# (always feasible: closed squares pairwise disjoint).  Minimise sum lambda_r^2 (paper: >= 0.32).
import sys, json, math, time
import numpy as np
from geom import square, hull, minkowski_pts, sat_gap

DMAX = 1e-5; TH = 1e-5

def ray_entry(H, w):
    """smallest tau >= 0 with tau*w in convex polygon H (CCW); None if never."""
    t0, t1 = 0.0, np.inf
    n = len(H)
    for i in range(n):
        a = H[i]; b = H[(i + 1) % n]
        e = b - a
        nrm = np.array([e[1], -e[0]])  # outward for CCW
        # constraint: (tau*w - a).nrm <= 0  ->  tau*(w.nrm) <= a.nrm
        den = w @ nrm; num = a @ nrm
        if abs(den) < 1e-300:
            if num < 0:
                return None
            continue
        tt = num / den
        if den > 0:
            t1 = min(t1, tt)
        else:
            t0 = max(t0, tt)
    if t0 > t1:
        return None
    return t0

def hull_ccw(pts):
    H = hull(pts)
    # monotone chain returns CCW
    return H

def place(obst, B, w, gap):
    """move B along unit w until first contact with obstacles; stop 'gap' before. returns moved B or None."""
    tmin = 10.0
    for A in obst:
        H = hull_ccw(minkowski_pts(A, B))
        te = ray_entry(H, w)
        if te is None:
            continue
        if te <= 0:
            return None  # overlapping at start
        tmin = min(tmin, te)
    if tmin >= 10.0:
        return None
    return B + (tmin - gap) * w

def setup(b, th, gap2, flip):
    S1 = square(0, 0, 0)
    S2 = square(0, 0, th)
    w = 0.5 * (math.cos(th) + math.sin(abs(th)))
    S2 = S2 + np.array([3.0, b])
    # push S2 left until contact with S1, leave gap2
    H = hull_ccw(minkowski_pts(S1, S2))
    te = ray_entry(H, np.array([-1.0, 0.0]))
    if te is None:
        return None
    S2 = S2 - np.array([te - gap2, 0])
    return S1, S2

def sum_lambda2(S1, S2, Zs):
    c1 = S1.mean(0); c2 = S2.mean(0)
    u = c2 - c1; Lh = np.hypot(*u); u /= Lh; n = np.array([-u[1], u[0]])
    T = []
    for Z in Zs:
        D = Z - c1
        s = D @ u; t = D @ n
        for idx in (int(np.argmax(t)), int(np.argmin(t))):
            # all vertices attaining the extreme
            ext = t[idx]
            for j in range(4):
                if abs(t[j] - ext) <= 1e-15 and -1e-15 <= s[j] <= Lh + 1e-15 and abs(t[j]) < 0.5:
                    T.append(t[j]); break
    pts = np.sort(np.array([-0.5] + T + [0.5]))
    lam = np.diff(pts)
    return float((lam ** 2).sum()), sorted(T)

def random_Z(rng, S1, S2, Zs, gap):
    c2 = S2.mean(0)
    # target near one of the two free triangles: top (above S1, left of S2) or bottom (below S2, right of S1)
    if rng.random() < 0.5:
        tgt = np.array([0.5 + 0.02 * rng.random(), 0.5 + 0.3 * rng.random()])
    else:
        tgt = np.array([c2[0] - 0.5 * math.cos(0) - 0.02 * rng.random() + 0.0, c2[1] - 0.5 - 0.3 * rng.random()])
        tgt[0] = 0.5 + 0.3 * rng.random() + 0.0
        tgt[1] = c2[1] - 0.5 - 0.02 * rng.random()
    psi = rng.random() * math.pi / 2
    om = rng.random() * 2 * math.pi
    return (tgt, psi, om)

def build_Z(params, S1, S2, gap):
    Zs = []
    for tgt, psi, om in params:
        w = np.array([math.cos(om), math.sin(om)])
        B = square(tgt[0] - 3 * w[0], tgt[1] - 3 * w[1], psi)
        obst = [S1, S2] + Zs
        if any(sat_gap(A, B) <= 0 for A in obst):
            continue
        B2 = place(obst, B, w, gap)
        if B2 is None:
            continue
        if any(sat_gap(A, B2) <= 0 for A in obst):
            continue
        Zs.append(B2)
    return Zs

if __name__ == "__main__":
    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    t0 = time.time(); best = (9, None); nrest = 0; neval = 0
    hist = []
    while time.time() - t0 < tl:
        nrest += 1
        b = rng.random() * 1.0
        th = TH * (2 * rng.random() - 1)
        gap2 = DMAX * rng.random() * 0.9
        st = setup(b, th, gap2, False)
        if st is None:
            continue
        S1, S2 = st
        # check distance < d: gap2 along horizontal; actual distance <= gap2
        nz = int(rng.integers(1, 6))
        params = [random_Z(rng, S1, S2, [], 0) for _ in range(nz)]
        gap = 1e-9
        Zs = build_Z(params, S1, S2, gap)
        val, T = sum_lambda2(S1, S2, Zs); neval += 1
        # hill climb
        for it in range(150):
            p2 = []
            for tgt, psi, om in params:
                if rng.random() < 0.5:
                    tgt = tgt + rng.normal(0, 0.03, 2); psi = (psi + rng.normal(0, 0.1)) % (math.pi / 2)
                    om = om + rng.normal(0, 0.3)
                p2.append((tgt, psi, om))
            if rng.random() < 0.1 and len(p2) < 6:
                p2.append(random_Z(rng, S1, S2, [], 0))
            Z2 = build_Z(p2, S1, S2, gap)
            v2, T2 = sum_lambda2(S1, S2, Z2); neval += 1
            if v2 <= val:
                val, T, params, Zs = v2, T2, p2, Z2
        hist.append(val)
        if val < best[0]:
            best = (val, {"b": b, "theta": th, "gap": gap2, "T": T, "S1": S1.tolist(), "S2": S2.tolist(),
                          "Z": [Z.tolist() for Z in Zs]})
            print(nrest, val, T, flush=True)
            with open("flpar_results_%s.json" % (sys.argv[2] if len(sys.argv) > 2 else "1"), "w") as f:
                json.dump({"best": best[0], "config": best[1], "restarts": nrest, "evals": neval}, f)
    h = np.array(hist)
    with open("flpar_results_%s.json" % (sys.argv[2] if len(sys.argv) > 2 else "1"), "w") as f:
        json.dump({"best": best[0], "config": best[1], "restarts": nrest, "evals": neval,
                   "quantiles": [float(np.quantile(h, q)) for q in (0, 0.01, 0.1, 0.5)]}, f)
    print("restarts", nrest, "evals", neval, "best", best[0])
