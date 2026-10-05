"""Hand construction with 9 counted pairs (slit + doubly-degenerate tilts), verified in mpmath (80 digits)
with an implementation independent of geom.py (exact SAT disjointness, vertex-edge distances, band test).

Squares (p = (0,0)):
  0: A0  centre (0, -1/2 - eA), angle 0          (p just above its top edge midpoint)
  1: B0  centre (0, +1/2 + eB), angle 0          (p just below its bottom edge midpoint)
  2: A1  rotated -th, leftmost vertex at x = 1/2 + g, centre height -1/2 - eA + sA   (raised)
  3: B1  rotated -th, leftmost vertex at x = 1/2 + g + D, centre height 1/2 + eB - sB (lowered)
  4,5: mirror images of 2,3 in the y-axis (rotation +th).
Counted: A0-B0, A0-A1, A0-A(-1), B0-B1, B0-B(-1), A0-B1, A0-B(-1), B0-A1, B0-A(-1)  -> 9.
Key trick: A1 raised and B1 lowered by ~2*sqrt(e) (needed for p to be in the horizontal bands) are made
compatible by a common tilt th with horizontal offset D:  D*th >= sA + sB + th^2/2.
"""
import sys, json
import mpmath as mp

mp.mp.dps = 80
H = mp.mpf(1) / 2


def square(cx, cy, t):
    c, s = mp.cos(t), mp.sin(t)
    u = (c * H, s * H); v = (-s * H, c * H)
    V = [(cx + u[0] + v[0], cy + u[1] + v[1]), (cx - u[0] + v[0], cy - u[1] + v[1]),
         (cx - u[0] - v[0], cy - u[1] - v[1]), (cx + u[0] - v[0], cy + u[1] - v[1])]
    return dict(c=(cx, cy), t=t, V=V, ax=[(c, s), (-s, c)])


def sat(P, Q):
    best = None
    for a in P['ax'] + Q['ax']:
        pa = [x * a[0] + y * a[1] for x, y in P['V']]; pb = [x * a[0] + y * a[1] for x, y in Q['V']]
        s = max(min(pb) - max(pa), min(pa) - max(pb))
        best = s if best is None else max(best, s)
    return best


def pt_seg(p, a, b):
    abx, aby = b[0] - a[0], b[1] - a[1]
    t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / (abx ** 2 + aby ** 2)
    t = min(max(t, 0), 1)
    return mp.sqrt((p[0] - a[0] - t * abx) ** 2 + (p[1] - a[1] - t * aby) ** 2)


def dist(P, Q):
    m = None
    for X, Y in ((P, Q), (Q, P)):
        for v in X['V']:
            for k in range(4):
                dd = pt_seg(v, Y['V'][k], Y['V'][(k + 1) % 4])
                m = dd if m is None else min(m, dd)
    return m


def point_out(p, P):
    """signed: positive = outside (max-norm in local frame minus 1/2)."""
    qx, qy = p[0] - P['c'][0], p[1] - P['c'][1]
    (c, s), _ = P['ax']
    lx = qx * c + qy * s; ly = -qx * s + qy * c
    return max(abs(lx), abs(ly)) - H


def band(p, A, B):
    ex, ey = B[0] - A[0], B[1] - A[1]; L = mp.sqrt(ex ** 2 + ey ** 2)
    qx, qy = p[0] - A[0], p[1] - A[1]
    t = (qx * ex + qy * ey) / L; perp = abs(qx * ey - qy * ex) / L
    return min(t, L - t, H - perp)


def build(d, eA, eB, sA, sB, th, g, D, mirror=True):
    sq = [square(mp.mpf(0), -H - eA, mp.mpf(0)), square(mp.mpf(0), H + eB, mp.mpf(0))]
    c, s = mp.cos(th), mp.sin(th)
    xA = H + g + H * (c + s)          # leftmost vertex (bottom-left after CW rotation) at x = 1/2 + g
    xB = H + g + D + H * (c + s)
    sq.append(square(xA, -H - eA + sA, -th))
    sq.append(square(xB, H + eB - sB, -th))
    if mirror:
        sq.append(square(-xA, -H - eA + sA, th))
        sq.append(square(-xB, H + eB - sB, th))
    return sq


def check(sq, d, verbose=True):
    p = (mp.mpf(0), mp.mpf(0))
    n = len(sq)
    ok = True
    waste = min(point_out(p, P) for P in sq)
    if not waste > 0: ok = False
    cnt = 0; pairs = []; mingap = None
    for i in range(n):
        for j in range(i + 1, n):
            s = sat(sq[i], sq[j])
            if not s > 0:
                ok = False; gap = s
            else:
                gap = dist(sq[i], sq[j])
            mingap = gap if mingap is None else min(mingap, gap)
            m = band(p, sq[i]['c'], sq[j]['c'])
            if s > 0 and gap < d and m >= 0:
                cnt += 1; pairs.append((i, j, mp.nstr(gap, 5), mp.nstr(m, 5)))
    if verbose:
        print('valid', ok, 'count', cnt, 'p-distance to nearest square', mp.nstr(waste, 5), 'min gap', mp.nstr(mingap, 5))
        for x in pairs: print('   pair', x)
    return ok, cnt, pairs


if __name__ == '__main__':
    out = {}
    for d in ('1e-4', '1e-5', '1e-6', '1e-3', '1e-2'):
        dd = mp.mpf(d)
        # parameters scaled with d
        g = dd / 100; D = dd * 4 / 10; th = dd / 5
        budget = D * th - th ** 2 / 2          # must exceed sA + sB + gapAB
        sA = sB = budget / 4                    # leaves budget/2 for the A1-B1 gap
        eA = eB = (sA / (2 * (1 + 2 * dd))) ** 2 / 2   # sA >= 2 X sqrt(e + e^2) with margin
        sq = build(dd, eA, eB, sA, sB, th, g, D)
        print('d =', d, ' eA=eB=', mp.nstr(eA, 4), ' sA=sB=', mp.nstr(sA, 4), ' th=', mp.nstr(th, 4), ' D=', mp.nstr(D, 4), ' g=', mp.nstr(g, 4))
        ok, cnt, pairs = check(sq, dd)
        out[d] = dict(valid=ok, count=cnt, eA=str(eA), sA=str(sA), th=str(th), D=str(D), g=str(g),
                      squares=[dict(cx=mp.nstr(P['c'][0], 40), cy=mp.nstr(P['c'][1], 40), angle=mp.nstr(P['t'], 40)) for P in sq])
    json.dump(out, open('hand9.json', 'w'), indent=1)
