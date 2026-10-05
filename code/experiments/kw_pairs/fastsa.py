"""Faster annealer: SAT separation (lower bound of distance, = -penetration when overlapping)."""
import numpy as np

H = 0.5


def axes_of(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s], [-s, c]])


def verts1(c, t):
    A = axes_of(t) * H
    u, v = A[0], A[1]
    return np.array([c + u + v, c - u + v, c - u - v, c + u - v])


class Fast:
    def __init__(self, C, T, walls, d, p=np.zeros(2)):
        self.C = C.copy(); self.T = T.copy(); self.n = len(C)
        self.walls = [(np.array(w[:2]), w[2]) for w in walls]
        self.d = d; self.p = p
        self.V = np.array([verts1(C[i], T[i]) for i in range(self.n)])
        self.A = np.array([axes_of(T[i]) for i in range(self.n)])

    def pair_terms(self, i, ci, ti, Vi, Ai, par):
        tg, tm, K = par
        m_ = np.arange(self.n) != i
        Cj = self.C[m_]; Vj = self.V[m_]; Aj = self.A[m_]
        # axes of i: (2,2); project Vj (m,4,2) -> (m,4,2)
        pj = Vj @ Ai.T                      # (m,4,2)
        ci_p = Ai @ ci                      # (2,)
        s1 = np.maximum(pj.min(1) - (ci_p + H), (ci_p - H) - pj.max(1)).max(1)
        # axes of j: Aj (m,2,2); project Vi (4,2) -> (m,4,2)
        pi = np.einsum('kd,mad->mka', Vi, Aj)
        cj_p = np.einsum('md,mad->ma', Cj, Aj)
        s2 = np.maximum(pi.min(1) - (cj_p + H), (cj_p - H) - pi.max(1)).max(1)
        g = np.maximum(s1, s2)
        # rect margin of p
        D = Cj - ci
        L = np.sqrt((D * D).sum(1)); e = D / L[:, None]
        q = self.p - ci
        t = e @ q
        s = np.abs(q[0] * e[:, 1] - q[1] * e[:, 0])
        mg = np.minimum(np.minimum(t, L - t), H - s)
        rew = np.exp(-np.maximum(g - 0.5 * self.d, 0) / tg) * 0.5 * (1 + np.tanh(0.5 * mg / tm))
        pen = K * np.maximum(-g, 0) ** 2
        row = np.zeros(self.n); row[m_] = pen - rew
        # self terms: waste and walls
        lq = Ai @ (self.p - ci)
        dx, dy = abs(lq[0]) - H, abs(lq[1]) - H
        ins = max(dx, dy)
        ps = ins if ins <= 0 else np.hypot(max(dx, 0), max(dy, 0))
        se = K * max(0.0, 2e-3 - ps) ** 2
        for nn, o in self.walls:
            wg = (Vi @ nn).min() - o
            h = ci @ nn - o
            f = ci - h * nn
            Dw = f - ci; Lw = abs(h)
            if Lw < 1e-12:
                wm = -1.0
            else:
                ew = Dw / Lw
                tw = ew @ q; sw = abs(q[0] * ew[1] - q[1] * ew[0])
                wm = min(tw, Lw - tw, H - sw)
            se += K * max(-wg, 0) ** 2 - np.exp(-max(wg - 0.5 * self.d, 0) / tg) * 0.5 * (1 + np.tanh(0.5 * wm / tm))
        return row, se

    def anneal(self, rng, steps, sched):
        n = self.n
        E = np.zeros((n, n)); S = np.zeros(n)
        for k in range(steps):
            fr = k / steps
            tg, tm, K, Temp, sc = sched(fr)
            par = (tg, tm, K)
            if k % 1000 == 0:
                for i in range(n):
                    r, s = self.pair_terms(i, self.C[i], self.T[i], self.V[i], self.A[i], par)
                    E[i] = r; S[i] = s
            i = rng.integers(n)
            if rng.random() < 0.01 * (1 - fr):
                rr = np.sqrt(rng.uniform(0.6 ** 2, 2.3 ** 2)); a = rng.uniform(0, 2 * np.pi)
                nc = self.p + np.array([rr * np.cos(a), rr * np.sin(a)]); nt = rng.uniform(0, np.pi / 2)
            else:
                nc = self.C[i] + rng.normal(0, sc, 2); nt = self.T[i] + rng.normal(0, sc)
            Vi = verts1(nc, nt); Ai = axes_of(nt)
            r, s = self.pair_terms(i, nc, nt, Vi, Ai, par)
            dE = r.sum() + s - E[i].sum() - S[i]
            if dE < 0 or rng.random() < np.exp(-dE / Temp):
                self.C[i] = nc; self.T[i] = nt; self.V[i] = Vi; self.A[i] = Ai
                E[i] = r; E[:, i] = r; S[i] = s
        return self
