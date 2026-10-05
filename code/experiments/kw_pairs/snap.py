"""Approach (c'): 'snap' Monte Carlo. State = valid closed packing + waste point p.
Move: perturb one square, then re-snap it to k (1..3) nearby squares/walls (gap = d/2) by a 3-variable
least squares solve; or move p. Accept by Metropolis on E = -count (count recomputed exactly by geom.analyse).
Usage: python snap.py SRC_JSONL(or 'none') OUTFILE SEED TIME_LIMIT_SEC D MODES
"""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import least_squares
from geom import signed_gap, point_sq, rect_margin, wall_gap, wall_foot, analyse


def gaps_of(i, C, T, walls):
    n = len(C)
    o = np.array([j for j in range(n) if j != i])
    g = signed_gap(np.repeat(C[i:i+1], n - 1, 0), np.repeat(T[i:i+1], n - 1), C[o], T[o])
    wg = [float(wall_gap(C[i:i+1], T[i:i+1], w)[0]) for w in walls]
    return o, g, wg


def snap(i, C, T, walls, d, rng):
    o, g, wg = gaps_of(i, C, T, walls)
    items = [('s', int(j), float(x)) for j, x in zip(o, g)] + [('w', k, x) for k, x in enumerate(wg)]
    items.sort(key=lambda t: t[2])
    k = int(rng.integers(1, 4))
    tgt = items[:k]
    others = items[k:]
    def res(x):
        c = x[:2][None]; t = np.array([x[2]])
        r = []
        for typ, j, _ in tgt:
            if typ == 's':
                gg = signed_gap(c, t, C[j:j+1], T[j:j+1])[0]
            else:
                gg = wall_gap(c, t, walls[j])[0]
            r.append((gg - 0.5 * d) * 100)
        for typ, j, g0 in others[:6]:
            if g0 > 0.3: continue
            if typ == 's':
                gg = signed_gap(c, t, C[j:j+1], T[j:j+1])[0]
            else:
                gg = wall_gap(c, t, walls[j])[0]
            r.append(max(0.0, 0.2 * d - gg) * 1000)
        return np.array(r)
    x0 = np.array([C[i, 0], C[i, 1], T[i]])
    try:
        sol = least_squares(res, x0, method='lm' if len(res(x0)) >= 3 else 'trf', xtol=1e-14, ftol=1e-14, max_nfev=200)
    except Exception:
        sol = least_squares(res, x0, method='trf', xtol=1e-14, ftol=1e-14, max_nfev=200)
    C2 = C.copy(); T2 = T.copy(); C2[i] = sol.x[:2]; T2[i] = sol.x[2]
    return C2, T2


def evaluate(C, T, p, walls, d):
    A = analyse(C - p, T, np.zeros(2), d, [(w[0], w[1], w[2] - (w[0] * p[0] + w[1] * p[1])) for w in walls])
    return A


def random_start(rng, mode):
    n = int(rng.integers(6, 12))
    walls = []
    if mode == 'wall':
        walls = [(0.0, 1.0, -float(rng.uniform(0.0, 1.0)))]
    elif mode == 'corner':
        walls = [(0.0, 1.0, -float(rng.uniform(0.0, 1.0))), (1.0, 0.0, -float(rng.uniform(0.0, 1.0)))]
    C = []; T = []
    tries = 0
    while len(C) < n and tries < 20000:
        tries += 1
        r = rng.uniform(0.55, 1.9); a = rng.uniform(0, 2 * np.pi)
        c = np.array([r * np.cos(a), r * np.sin(a)]); t = rng.uniform(0, np.pi / 2)
        if point_sq(np.zeros(2), c, t) <= 1e-3: continue
        if any(wall_gap(c[None], np.array([t]), w)[0] <= 1e-3 for w in walls): continue
        if C and signed_gap(np.array(C), np.array(T), np.repeat(c[None], len(C), 0), np.repeat(t, len(C))).min() <= 1e-3: continue
        C.append(c); T.append(t)
    return np.array(C), np.array(T), walls


def load_starts(src, d):
    key = 'd%g' % d
    out = []
    for line in open(src):
        r = json.loads(line)
        x = r.get(key)
        if not x or not x.get('valid', True): continue
        C = np.array(r.get('C') or x.get('C')); T = np.array(r.get('T') or x.get('T'))
        p = np.array(x.get('p', [0, 0]))
        walls = [tuple(w) for w in r['walls']]
        C = C - p
        walls = [(w[0], w[1], w[2] - (w[0] * p[0] + w[1] * p[1])) for w in walls]
        keep = np.hypot(*C.T) < 2.2
        out.append((x['count'], C[keep], T[keep], walls))
    out.sort(key=lambda t: -t[0])
    return out


def main():
    src, outf, seed, tlim, d = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    modes = sys.argv[6].split(',') if len(sys.argv) > 6 else ['free']
    rng = np.random.default_rng(seed)
    starts = load_starts(src, d) if src != 'none' else []
    t0 = time.time(); run = 0
    gbest = 0
    while time.time() - t0 < tlim:
        mode = modes[run % len(modes)]
        if starts and run % 2 == 0:
            c0, C, T, walls = starts[(run // 2) % min(len(starts), 30)]
            C = C.copy(); T = T.copy()
        else:
            C, T, walls = random_start(rng, mode)
        p = np.zeros(2)
        A = evaluate(C, T, p, walls, d)
        if not A['valid']:
            run += 1; continue
        cur = A['count']; best = (cur, C.copy(), T.copy(), p.copy(), A)
        steps = int(rng.integers(1500, 4000))
        ts = time.time()
        for it in range(steps):
            Temp = 0.6 * (0.05 / 0.6) ** (it / steps)
            mv = rng.random()
            n = len(C)
            if mv < 0.1:
                p2 = p + rng.normal(0, 10 ** rng.uniform(-4, -1), 2); C2, T2 = C, T
            elif mv < 0.15 and n < 14:
                r = rng.uniform(0.55, 1.9); a = rng.uniform(0, 2 * np.pi)
                C2 = np.vstack([C, p + [r * np.cos(a), r * np.sin(a)]]); T2 = np.append(T, rng.uniform(0, np.pi / 2)); p2 = p
                C2, T2 = snap(n, C2, T2, walls, d, rng)
            elif mv < 0.18 and n > 3:
                j = int(rng.integers(n)); C2 = np.delete(C, j, 0); T2 = np.delete(T, j); p2 = p
            else:
                dist = np.hypot(*(C - p).T)
                w = np.exp(-dist); w /= w.sum()
                i = int(rng.choice(n, p=w))
                sc = 10 ** rng.uniform(-3, -0.8)
                C2 = C.copy(); T2 = T.copy()
                C2[i] += rng.normal(0, sc, 2); T2[i] += rng.normal(0, sc)
                C2, T2 = snap(i, C2, T2, walls, d, rng)
                p2 = p
            A2 = evaluate(C2, T2, p2, walls, d)
            if not A2['valid']:
                continue
            dE = cur - A2['count']
            if dE <= 0 or rng.random() < np.exp(-dE / Temp):
                C, T, p, cur = C2, T2, p2, A2['count']
                if cur > best[0]:
                    best = (cur, C.copy(), T.copy(), p.copy(), A2)
        cnt, C, T, p, A = best
        rec = dict(run=run, mode=mode, d=d, count=cnt, strict=A['strict_count'], deg=A['deg'], min_gap=A['min_gap'],
                   min_psq=A['min_psq'], p=[float(x) for x in p], walls=[list(map(float, w)) for w in walls],
                   C=[[float(a), float(b)] for a, b in C], T=[float(x) for x in T],
                   pairs=[[a, b, g, m] for a, b, g, m in A['pairs']], wpairs=[[a, b, g, m] for a, b, g, m in A['wpairs']],
                   t=time.time() - t0, steps=steps)
        with open(outf, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        gbest = max(gbest, cnt)
        print(run, mode, 'count', cnt, 'deg', [x for x in A['deg'] if x], 'gbest', gbest, '%.0fs' % (time.time() - ts), flush=True)
        run += 1


if __name__ == '__main__':
    main()
