# Target 3(b), hill climbing: columns of axis-parallel squares (horizontal gaps < d between columns,
# arbitrary vertical gaps inside a column, optional column at the left wall); maximise the number of
# rectangles R_e (edges of G_h with members at distance < d) containing a common point.
import sys, json, time
import numpy as np
from geom import square
from count_mult import multiplicity

K = 6

def build(params):
    x0, cols = params
    sqs = []
    x = x0
    for (off, gaps) in cols:
        y = off
        for g in gaps:
            if y >= 0 and y + 1 <= K:
                sqs.append(square(x + 0.5, y + 0.5, 0.0))
            y += 1 + g
        x += 1 + 3e-5
    return sqs

def rand_params(rng, ncol):
    x0 = 0.0 if rng.random() < 0.5 else 1e-5 + 1.5 * rng.random()
    cols = []
    for c in range(ncol):
        gaps = list(np.where(rng.random(8) < 0.5, 1e-6, rng.random(8) * 0.8))
        cols.append((rng.random() * 1.2, gaps))
    return (x0, cols)

def mutate(rng, p):
    x0, cols = p
    cols = [(o, list(g)) for o, g in cols]
    c = int(rng.integers(len(cols)))
    o, g = cols[c]
    if rng.random() < 0.5:
        o = float(np.clip(o + rng.normal(0, 0.1), 0, 1.5))
    else:
        i = int(rng.integers(len(g)))
        g[i] = float(max(1e-6, g[i] + rng.normal(0, 0.15))) if rng.random() < 0.8 else 1e-6
    cols[c] = (o, g)
    if rng.random() < 0.1:
        x0 = 0.0 if x0 > 0 else 1e-5 + 1.5 * rng.random()
    return (x0, cols)

if __name__ == "__main__":
    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    t0 = time.time(); best = (0, None)
    while time.time() - t0 < tl:
        ncol = int(rng.integers(2, 5))
        p = rand_params(rng, ncol)
        m, pt, ne = multiplicity(build(p), K, grid=0.02)
        for it in range(60):
            if time.time() - t0 > tl:
                break
            p2 = mutate(rng, p)
            m2, pt2, ne2 = multiplicity(build(p2), K, grid=0.02)
            if m2 >= m:
                p, m, pt = p2, m2, pt2
        if m > best[0]:
            best = (m, {"p": pt, "squares": [S.tolist() for S in build(p)]})
            print("mult", m, flush=True)
            with open("mult_climb_results.json", "w") as f:
                json.dump({"best": best[0], "config": best[1]}, f)
    print("best", best[0])
