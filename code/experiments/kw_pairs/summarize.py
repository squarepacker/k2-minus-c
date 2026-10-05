"""Summarize jsonl results: distribution of counts per d, and show best configs re-verified."""
import sys, json, collections
import numpy as np
from geom import analyse

for fn in sys.argv[1:]:
    recs = [json.loads(l) for l in open(fn)]
    print('==', fn, len(recs), 'records')
    for key in ('d0.0001', 'd0.001', 'd0.01', None):
        if key is None:
            vals = [(r['count'], r) for r in recs if 'count' in r]
            lab = 'count'
        else:
            vals = [(r[key]['count'], r) for r in recs if r.get(key) and r[key].get('valid', True)]
            lab = key
        if not vals: continue
        dist = collections.Counter(v for v, _ in vals)
        print(lab, 'distribution', dict(sorted(dist.items())))
        by_mode = collections.defaultdict(list)
        for v, r in vals: by_mode[r.get('mode')].append(v)
        print('   max by mode', {m: max(v) for m, v in by_mode.items()})
        best = max(vals, key=lambda t: t[0])[1]
        x = best if key is None else best[key]
        C = np.array(best.get('C') or x.get('C')); T = np.array(best.get('T') or x.get('T'))
        p = np.array(x.get('p', [0, 0])); d = best.get('d', None) or float(key[1:])
        walls = [tuple(w) for w in best['walls']]
        A = analyse(C, T, p, d, walls)
        print('   best re-verified: valid', A['valid'], 'count', A['count'], 'strict', A['strict_count'], 'min_gap %.3g' % A['min_gap'], 'min_psq %.3g' % A['min_psq'])
        part = sorted({a for a, *_ in A['pairs']} | {b for _, b, *_ in A['pairs']} | {a for a, *_ in A['wpairs']})
        for i in part:
            print('     sq %d c=(%.6f,%.6f) t=%.6f |c-p|=%.4f deg=%d' % (i, C[i, 0] - p[0], C[i, 1] - p[1], T[i], np.hypot(*(C[i] - p)), A['deg'][i]))
        print('     pairs', [(a, b, '%.2g' % g, '%.2g' % m) for a, b, g, m in A['pairs']], A['wpairs'], 'walls', walls)
