"""Approach (c): basin hopping with targeted feasibility polish at a fixed waste point p = 0.
Usage: python improve.py SRC_JSONL OUTFILE SEED TIME_LIMIT_SEC D [TOPK]
Reads jam/SA results, takes the TOPK best configurations (for the given d), recentres at p, and tries to raise the count.
"""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from geom import signed_gap, point_sq, rect_margin, wall_gap, wall_foot, analyse
from sa import polish

P0 = np.zeros(2)


def targets(C, T, walls, thr):
    n = len(C)
    I, J = np.triu_indices(n, 1)
    near = np.hypot(*(C[I] - C[J]).T) < 1.5
    I, J = I[near], J[near]
    g = signed_gap(C[I], T[I], C[J], T[J]); m = rect_margin(P0, C[I], C[J])
    tg = [(int(a), int(b)) for a, b, gg, mm in zip(I, J, g, m) if gg < thr and mm > -thr]
    tw = []
    for wi, w in enumerate(walls):
        wg = wall_gap(C, T, w); wm = rect_margin(P0, C, wall_foot(C, w))
        tw += [(i, wi) for i in range(n) if wg[i] < thr and wm[i] > -thr]
    return tg, tw


def polish_count(C, T, walls, d, thr):
    tg, tw = targets(C, T, walls, thr)
    best = None
    for it in range(5):
        C, T = polish(C, T, walls, d, tg, tw)
        A = analyse(C, T, P0, d, walls)
        if A['valid'] and (best is None or A['count'] > best[0]['count']):
            best = (A, C.copy(), T.copy())
        ok = {(a, b) for a, b, *_ in A['pairs']}
        okw = {(a, int(b[1:])) for a, b, *_ in A['wpairs']}
        bad = [x for x in tg if x not in ok]; badw = [x for x in tw if x not in okw]
        if not bad and not badw:
            break
        if bad:
            gg = {x: float(signed_gap(C[x[0]:x[0]+1], T[x[0]:x[0]+1], C[x[1]:x[1]+1], T[x[1]:x[1]+1])[0]) for x in bad}
            tg.remove(max(gg, key=gg.get))
        else:
            tw.remove(badw[0])
    return best


def load(src, d, topk):
    key = 'd%g' % d
    recs = []
    for line in open(src):
        r = json.loads(line)
        x = r.get(key)
        if x and x.get('valid', True):
            recs.append(r)
    recs.sort(key=lambda r: -r[key]['count'])
    out = []
    for r in recs[:topk]:
        x = r[key]
        C = np.array(r.get('C', x.get('C')), float); T = np.array(r.get('T', x.get('T')), float)
        p = np.array(x.get('p', [0, 0]), float)
        walls = [tuple(w) for w in r['walls']]
        # recentre: p -> 0; walls (n,o) -> o - n.p
        C = C - p
        walls = [(w[0], w[1], w[2] - (w[0] * p[0] + w[1] * p[1])) for w in walls]
        keep = np.zeros(len(C), bool)
        for pr in x.get('pairs_used', []): keep[pr] = True
        if not keep.any():
            A0 = analyse(C, T, P0, d, walls)
            for a, b_, *_ in A0['pairs']: keep[a] = keep[b_] = True
            for a, *_ in A0['wpairs']: keep[a] = True
            keep |= np.hypot(*C.T) < 1.2
        out.append((C[keep], T[keep], walls, x['count']))
    return out


def main():
    src, outf, seed, tlim, d = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    topk = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    rng = np.random.default_rng(seed)
    starts = load(src, d, topk)
    t0 = time.time()
    per = tlim / max(1, len(starts))
    for si, (C, T, walls, c0) in enumerate(starts):
        b = polish_count(C, T, walls, d, 1e-6)
        if b is None:
            continue
        cur = b; best = b
        ts = time.time(); it = 0
        while time.time() - ts < per and time.time() - t0 < tlim:
            it += 1
            A, C, T = cur
            C2, T2 = C.copy(), T.copy()
            mv = rng.random()
            n = len(C2)
            near = np.argsort(np.hypot(*C2.T))
            if mv < 0.6:
                k = int(rng.integers(1, 4))
                idx = rng.choice(near[:min(n, 8)], size=min(k, n), replace=False)
                sc = 10 ** rng.uniform(-2.5, -0.7)
                C2[idx] += rng.normal(0, sc, (len(idx), 2)); T2[idx] += rng.normal(0, sc * 2, len(idx))
            elif mv < 0.8:
                r = rng.uniform(0.55, 1.6); a = rng.uniform(0, 2 * np.pi)
                C2 = np.vstack([C2, [r * np.cos(a), r * np.sin(a)]]); T2 = np.append(T2, rng.uniform(0, np.pi / 2))
            elif mv < 0.9 and n > 3:
                j = int(rng.integers(n)); C2 = np.delete(C2, j, 0); T2 = np.delete(T2, j)
            else:
                # rotate a near square about p-ish (slide around the hole)
                j = near[int(rng.integers(min(n, 6)))]
                a = rng.normal(0, 0.15); R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                C2[j] = R @ C2[j]; T2[j] += a
            thr = 10 ** rng.uniform(-2.5, -1)
            b = polish_count(C2, T2, walls, d, thr)
            if b is None:
                continue
            if b[0]['count'] >= cur[0]['count']:
                cur = b
            if b[0]['count'] > best[0]['count']:
                best = b
                print('start', si, 'it', it, 'new best', best[0]['count'], best[0]['deg'], flush=True)
        A, C, T = best
        rec = dict(start=si, start_count=c0, count=A['count'], strict=A['strict_count'], deg=A['deg'], d=d,
                   min_gap=A['min_gap'], min_psq=A['min_psq'], walls=[list(map(float, w)) for w in walls],
                   C=[[float(a), float(b_)] for a, b_ in C], T=[float(x) for x in T], iters=it,
                   pairs=[[a, b_, g, m] for a, b_, g, m in A['pairs']], wpairs=[[a, b_, g, m] for a, b_, g, m in A['wpairs']])
        with open(outf, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print('start', si, 'from', c0, '->', A['count'], flush=True)


if __name__ == '__main__':
    main()
