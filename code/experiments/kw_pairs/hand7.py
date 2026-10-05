"""Hand construction: slit between rows A (below) and B (above), aligned columns -1,0,1.
A(+-1) raised by s (V-shape of row A) -> horizontal pairs A0-A1, A0-A(-1) count, plus A0-B0 and 4 diagonals = 7.
Verified in float64 (geom.analyse) and in mpmath (50 digits, independent formulas)."""
import json, sys
import numpy as np
import mpmath as mp
from geom import analyse

def build(d, eps=1e-10, s=None, g=None, gam=None, dB=1e-10):
    s = s if s is not None else 0.5 * d
    g = g if g is not None else 0.1 * d      # horizontal gaps
    gam = gam if gam is not None else 0.1 * d  # gap between A(+-1) tops and B(+-1) bottoms
    yA0 = -0.5 - eps; yA1 = yA0 + s
    yB0 = 0.5 + dB
    yB1 = (yA1 + 0.5) + gam + 0.5
    C = [[-(1 + g), yA1], [0, yA0], [1 + g, yA1], [-(1 + g), yB1], [0, yB0], [1 + g, yB1]]
    return np.array(C), np.zeros(6)

res = {}
for d in (1e-4, 1e-3, 1e-2):
    C, T = build(d)
    A = analyse(C, T, np.zeros(2), d)
    print('d=%g valid=%s count=%d strict=%d deg=%s min_gap=%.3g min_psq=%.3g' % (d, A['valid'], A['count'], A['strict_count'], A['deg'], A['min_gap'], A['min_psq']))
    print('   pairs', [(a, b, '%.3g' % g, '%.3g' % m) for a, b, g, m in A['pairs']])
    res[str(d)] = dict(C=C.tolist(), T=T.tolist(), count=A['count'], pairs=A['pairs'])

# exact-ish recheck with mpmath, axis-aligned squares: distances and bands computed directly
mp.mp.dps = 50
def mp_check(C, d):
    C = [[mp.mpf(repr(float(x))) for x in c] for c in C]
    h = mp.mpf('0.5'); p = [mp.mpf(0), mp.mpf(0)]
    n = len(C)
    for c in C:  # p waste
        assert max(abs(p[0] - c[0]), abs(p[1] - c[1])) > h
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = max(abs(C[i][0] - C[j][0]) - 1, 0); dy = max(abs(C[i][1] - C[j][1]) - 1, 0)
            assert abs(C[i][0] - C[j][0]) > 1 or abs(C[i][1] - C[j][1]) > 1  # disjoint (axis-aligned)
            dist = mp.sqrt(dx ** 2 + dy ** 2)
            if dist < d:
                ex, ey = C[j][0] - C[i][0], C[j][1] - C[i][1]; L = mp.sqrt(ex ** 2 + ey ** 2)
                qx, qy = p[0] - C[i][0], p[1] - C[i][1]
                t = (qx * ex + qy * ey) / L; perp = abs(qx * ey - qy * ex) / L
                if 0 <= t <= L and perp <= h:
                    cnt += 1
    return cnt
for d in (1e-4, 1e-3, 1e-2):
    print('mpmath count d=%g:' % d, mp_check(res[str(d)]['C'], mp.mpf(d)))
json.dump(res, open('hand7.json', 'w'), indent=1)
