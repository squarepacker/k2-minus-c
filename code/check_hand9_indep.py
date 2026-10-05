"""Independent re-check (own code, mpmath 80 digits) of the multiplicity-9 configuration in attack_kw/hand9.json:
closed squares pairwise disjoint, p = (0,0) outside all squares, and the number of square pairs at distance < d whose
rectangle R(P,Q) contains p."""
import json, os, itertools
from mpmath import mp, mpf, cos, sin, sqrt
mp.dps = 80
HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, 'attack_kw', 'hand9.json')))


def verts(sq):
    c = (mpf(sq['cx']), mpf(sq['cy'])); a = mpf(sq['angle'])
    e1 = (cos(a)/2, sin(a)/2); e2 = (-sin(a)/2, cos(a)/2)
    return [(c[0]+s*e1[0]+t*e2[0], c[1]+s*e1[1]+t*e2[1]) for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))], c, a


def sep_gap(V, W):
    """max over the 8 edge normals of the gap between projections (>0 iff disjoint closed convex polygons)."""
    best = None
    for P in (V, W):
        for i in range(4):
            dx, dy = P[(i+1) % 4][0]-P[i][0], P[(i+1) % 4][1]-P[i][1]
            L = sqrt(dx*dx+dy*dy); n = (-dy/L, dx/L)
            a = [v[0]*n[0]+v[1]*n[1] for v in V]; b = [w[0]*n[0]+w[1]*n[1] for w in W]
            g = max(min(b)-max(a), min(a)-max(b))
            best = g if best is None else max(best, g)
    return best


def seg_dist(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy); t = min(max(t, mpf(0)), mpf(1))
    qx, qy = a[0]+t*dx-p[0], a[1]+t*dy-p[1]
    return sqrt(qx*qx+qy*qy)


def dist(V, W):
    return min(min(seg_dist(v, W[i], W[(i+1) % 4]) for v in V for i in range(4)),
               min(seg_dist(w, V[i], V[(i+1) % 4]) for w in W for i in range(4)))


def inside(p, c, a):
    x, y = p[0]-c[0], p[1]-c[1]
    u = x*cos(a)+y*sin(a); v = -x*sin(a)+y*cos(a)
    return abs(u) <= mpf(1)/2 and abs(v) <= mpf(1)/2


def in_rect(p, c1, c2):
    dx, dy = c2[0]-c1[0], c2[1]-c1[1]; L = sqrt(dx*dx+dy*dy)
    ux, uy = dx/L, dy/L
    s = (p[0]-c1[0])*ux+(p[1]-c1[1])*uy; t = -(p[0]-c1[0])*uy+(p[1]-c1[1])*ux
    return mpf(0) <= s <= L and abs(t) <= mpf(1)/2


p = (mpf(0), mpf(0))
for key, rec in data.items():
    if not isinstance(rec, dict) or 'squares' not in rec:
        continue
    d = mpf(key)
    sq = [verts(s) for s in rec['squares']]
    gaps = [sep_gap(sq[i][0], sq[j][0]) for i, j in itertools.combinations(range(len(sq)), 2)]
    disjoint = all(g > 0 for g in gaps)
    p_waste = not any(inside(p, c, a) for _, c, a in sq)
    cnt = 0
    for i, j in itertools.combinations(range(len(sq)), 2):
        if dist(sq[i][0], sq[j][0]) < d and in_rect(p, sq[i][1], sq[j][1]):
            cnt += 1
    print(f"d={key}: {len(sq)} squares, pairwise disjoint={disjoint} (min sep gap {mp.nstr(min(gaps), 5)}), p waste={p_waste}, count={cnt}")
