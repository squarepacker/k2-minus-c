"""Simulated annealing + polish search for many near-touching pairs whose rectangle contains the waste point p=0.
Usage: python sa.py OUTFILE SEED TIME_LIMIT_SEC [modes]
Writes one JSON line per run (checkpointing), only valid (re-verified) configurations are scored.
"""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import least_squares
from geom import signed_gap, point_sq, rect_margin, wall_gap, wall_foot, analyse

P0 = np.zeros(2)


def sig(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


class State:
    def __init__(self, C, T, walls, d):
        self.C = C; self.T = T; self.walls = walls; self.d = d
        self.n = len(C)

    def row(self, i, C, T, par):
        """energy contributions of square i (pairs with all others, waste, walls)."""
        tg, tm, K = par
        n = self.n
        idx = np.array([j for j in range(n) if j != i])
        Ci = np.repeat(C[i][None], n - 1, 0); Ti = np.repeat(T[i], n - 1)
        g = signed_gap(Ci, Ti, C[idx], T[idx])
        m = rect_margin(P0, Ci, C[idx])
        F = np.exp(-np.maximum(g - 0.5 * self.d, 0) / tg)
        rew = F * sig(m / tm)
        pen = K * np.maximum(-g, 0) ** 2
        e = np.zeros(n); e[idx] = pen - rew
        self_e = K * max(0.0, 1e-3 - float(point_sq(P0, C[i], T[i]))) ** 2
        for w in self.walls:
            wg = float(wall_gap(C[i:i + 1], T[i:i + 1], w)[0])
            F = wall_foot(C[i:i + 1], w)
            wm = float(rect_margin(P0, C[i:i + 1], F)[0])
            self_e += K * max(-wg, 0) ** 2 - np.exp(-max(wg - 0.5 * self.d, 0) / tg) * sig(wm / tm)
        return e, self_e


def total(st, par):
    E = np.zeros((st.n, st.n)); S = np.zeros(st.n)
    for i in range(st.n):
        e, s = st.row(i, st.C, st.T, par)
        E[i] = e; S[i] = s
    return E, S


def anneal(rng, n, walls, d, steps, init=None):
    if init is None:
        r = np.sqrt(rng.uniform(0.6 ** 2, 2.3 ** 2, n)); a = rng.uniform(0, 2 * np.pi, n)
        C = np.stack([r * np.cos(a), r * np.sin(a)], 1); T = rng.uniform(0, np.pi / 2, n)
        for w in walls:  # push inside walls
            nn = np.array(w[:2]); h = C @ nn - w[2]
            C = C + np.maximum(0.75 - h, 0)[:, None] * nn
    else:
        C, T = init
    st = State(C.copy(), T.copy(), walls, d)
    for k in range(steps):
        fr = k / steps
        tg = 0.15 * (0.01 / 0.15) ** fr
        tm = 0.1 * (0.003 / 0.1) ** fr
        K = 30 * (3e5 / 30) ** fr
        Temp = 0.6 * (0.003 / 0.6) ** fr
        par = (tg, tm, K)
        if k % 2000 == 0:
            E, S = total(st, par)
        i = rng.integers(n)
        sc = 0.25 * (1 - fr) + 0.004
        if rng.random() < 0.02 * (1 - fr):
            r = np.sqrt(rng.uniform(0.6 ** 2, 2.3 ** 2)); a = rng.uniform(0, 2 * np.pi)
            nc = np.array([r * np.cos(a), r * np.sin(a)]); nt = rng.uniform(0, np.pi / 2)
        else:
            nc = st.C[i] + rng.normal(0, sc, 2); nt = st.T[i] + rng.normal(0, sc * 0.8)
        C2 = st.C.copy(); T2 = st.T.copy(); C2[i] = nc; T2[i] = nt
        e_old = E[i].sum() + S[i]
        e_new_row, s_new = st.row(i, C2, T2, par)
        dE = e_new_row.sum() + s_new - e_old
        if dE < 0 or rng.random() < np.exp(-dE / Temp):
            st.C = C2; st.T = T2
            E[i] = e_new_row; E[:, i] = e_new_row; S[i] = s_new
    return st


def polish(C, T, walls, d, target_pairs, target_walls, delta=1e-9):
    n = len(C)
    I, J = np.triu_indices(n, 1)
    tp = np.array(target_pairs, int).reshape(-1, 2)
    # only pairs that are anywhere near matter for disjointness
    def res(x):
        Cx = x[:2 * n].reshape(n, 2); Tx = x[2 * n:]
        g = signed_gap(Cx[I], Tx[I], Cx[J], Tx[J])
        r = [100 * np.maximum(4 * delta - g, 0)]
        r.append(100 * np.maximum(4 * delta - point_sq(P0, Cx, Tx), 0))
        if len(tp):
            gt = signed_gap(Cx[tp[:, 0]], Tx[tp[:, 0]], Cx[tp[:, 1]], Tx[tp[:, 1]])
            r.append(10 * np.maximum(gt - 0.3 * d, 0) / d * 1e-3)
            mt = rect_margin(P0, Cx[tp[:, 0]], Cx[tp[:, 1]])
            r.append(100 * np.maximum(4 * delta - mt, 0))
        for wi, w in enumerate(walls):
            wg = wall_gap(Cx, Tx, w)
            r.append(100 * np.maximum(4 * delta - wg, 0))
            sel = [i for (i, ww) in target_walls if ww == wi]
            if sel:
                sel = np.array(sel)
                r.append(10 * np.maximum(wg[sel] - 0.3 * d, 0) / d * 1e-3)
                F = wall_foot(Cx[sel], w)
                r.append(100 * np.maximum(4 * delta - rect_margin(P0, Cx[sel], F), 0))
        return np.concatenate(r)
    x0 = np.concatenate([C.ravel(), T])
    sol = least_squares(res, x0, method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=120)
    x = sol.x
    return x[:2 * n].reshape(n, 2), x[2 * n:]


def finish(st, d_list):
    """from annealed state choose target set, polish for each d, return analyses."""
    out = {}
    n = st.n
    I, J = np.triu_indices(n, 1)
    g = signed_gap(st.C[I], st.T[I], st.C[J], st.T[J])
    m = rect_margin(P0, st.C[I], st.C[J])
    for d in d_list:
        tgt = [(int(I[k]), int(J[k])) for k in range(len(I)) if g[k] < 0.03 and m[k] > -0.03]
        tw = []
        for wi, w in enumerate(st.walls):
            wg = wall_gap(st.C, st.T, w); wm = rect_margin(P0, st.C, wall_foot(st.C, w))
            tw += [(i, wi) for i in range(n) if wg[i] < 0.03 and wm[i] > -0.03]
        C, T = st.C.copy(), st.T.copy()
        best = None
        for it in range(4):
            C, T = polish(C, T, st.walls, d, tgt, tw)
            A = analyse(C, T, P0, d, st.walls)
            if A['valid'] and (best is None or A['count'] > best[0]['count']):
                best = (A, C.copy(), T.copy())
            # drop targets that failed
            ok = {(a, b) for a, b, *_ in A['pairs']}
            okw = {(a, int(b[1:])) for a, b, *_ in A['wpairs']}
            if set(tgt) <= ok and set(tw) <= okw:
                break
            # remove worst violator
            gg = {}
            for (a, b) in tgt:
                if (a, b) not in ok:
                    gg[(a, b)] = float(signed_gap(C[a:a+1], T[a:a+1], C[b:b+1], T[b:b+1])[0])
            if gg:
                worst = max(gg, key=lambda k: gg[k]); tgt.remove(worst)
            tw = [x for x in tw if x in okw] if not gg else tw
        out[d] = best
    return out


def main():
    outf, seed, tlim = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    modes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['free']
    rng = np.random.default_rng(seed)
    t0 = time.time(); run = 0
    while time.time() - t0 < tlim:
        mode = modes[run % len(modes)]
        n = int(rng.integers(10, 19))
        if mode == 'free':
            walls = []
        elif mode == 'wall':
            walls = [(0.0, 1.0, -float(rng.uniform(0.05, 1.0)))]
        else:
            walls = [(0.0, 1.0, -float(rng.uniform(0.05, 1.0))), (1.0, 0.0, -float(rng.uniform(0.05, 1.0)))]
        d_target = 1e-4
        steps = int(rng.integers(60000, 160000))
        st = anneal(rng, n, walls, d_target, steps)
        res = finish(st, [1e-4, 1e-3, 1e-2])
        rec = dict(seed=seed, run=run, mode=mode, n=n, steps=steps, walls=walls, t=time.time() - t0)
        for d, b in res.items():
            if b is None:
                rec['d%g' % d] = None
            else:
                A, C, T = b
                rec['d%g' % d] = dict(count=A['count'], strict=A['strict_count'], deg=A['deg'],
                                      min_gap=A['min_gap'], min_psq=A['min_psq'],
                                      C=[[float(a), float(b_)] for a, b_ in C], T=[float(x) for x in T])
        with open(outf, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(run, mode, n, {k: (v['count'] if v else None) for k, v in rec.items() if k.startswith('d0')}, flush=True)
        run += 1


if __name__ == '__main__':
    main()
