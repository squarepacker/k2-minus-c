"""Targeted feasibility: 2x3 grid around p (rows A: y<0, B: y>0), try to realise subsets of the 9 'grid pairs'
(vertical A0B0, 4 diagonals, 4 horizontals) by least-squares polish from randomized tiny perturbations."""
import sys, json, itertools
import numpy as np
from sa import polish
from geom import analyse

P0 = np.zeros(2)
# indices: 0:A-1 1:A0 2:A1 3:B-1 4:B0 5:B1 ; plus optional extra squares
C0 = np.array([[-1, -.5], [0, -.5], [1, -.5], [-1, .5], [0, .5], [1, .5]], float)
T0 = np.zeros(6)
V = [(1, 4)]
D = [(1, 3), (1, 5), (4, 0), (4, 2)]
Hh = [(0, 1), (1, 2), (3, 4), (4, 5)]
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
best = {}
for d in (1e-4, 1e-3, 1e-2):
    bc = 0
    for trial in range(60):
        k = rng.integers(0, 5)
        hs = [Hh[i] for i in rng.choice(4, size=k, replace=False)]
        tgt = V + D + hs
        C = C0 + rng.normal(0, 1e-3, C0.shape) + np.array([rng.uniform(-.45, .45), 0])
        C[:3, 1] -= 1e-3; C[3:, 1] += 1e-3
        T = T0 + rng.normal(0, 1e-3, 6)
        for it in range(3):
            C, T = polish(C, T, [], d, tgt, [], delta=1e-11)
        A = analyse(C, T, P0, d)
        if A['valid'] and A['count'] > bc:
            bc = A['count']; best[d] = dict(count=bc, deg=A['deg'], C=C.tolist(), T=T.tolist(), pairs=A['pairs'], target=tgt)
            print('d', d, 'trial', trial, 'count', bc, 'target size', len(tgt), [(a, b) for a, b, *_ in A['pairs']], flush=True)
json.dump({str(k): v for k, v in best.items()}, open('grid_try_best.json', 'w'), indent=1)
