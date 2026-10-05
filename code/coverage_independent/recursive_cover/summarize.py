"""Summarise results.jsonl of coverage_lowmem.py. Usage: python summarize.py [results.jsonl]"""
import json, collections, sys
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")]
by = collections.defaultdict(list)
for r in rows:
    by[(r['wall'], r['n'])].append(r)
tot = collections.Counter()
allok = True
leaves = {False: 0, True: 0}
for key in sorted(by):
    rs = by[key]
    units = rs[0]['units']
    us = sorted(r['unit'] for r in rs)
    complete = us == list(range(units))
    g = sum(r['gaps'] for r in rs)
    tot['gaps'] += g
    for k in ('nodes', 'contained', 'unordered'):
        tot[k] += sum(r[k] for r in rs)
    leaves[key[0]] += rs[0]['leaves_total']
    kinds = rs[0]['kinds']
    ok = complete and g == 0 and all(r['leaves_total'] == rs[0]['leaves_total'] for r in rs)
    allok &= ok
    print(f"wall={key[0]!s:5} n={key[1]}: units {len(rs)}/{units} complete={complete} leaves={rs[0]['leaves_total']} kinds={kinds} "
          f"gaps={g} nodes={sum(r['nodes'] for r in rs)} contained={sum(r['contained'] for r in rs)} "
          f"unordered={sum(r['unordered'] for r in rs)} maxdepth={max(r['maxdepth'] for r in rs)} max_sec={max(r['sec'] for r in rs)}")
print("rows", len(rows), "dup units:", len(rows) - len({(r['wall'], r['n'], r['unit']) for r in rows}))
print("leaves no-side", leaves[False], "side", leaves[True], "total", leaves[False] + leaves[True])
print("TOTAL", dict(tot))
print("peak_private_MB max", max(r['mem']['peak_private'] for r in rows), "peak_ws_MB max", max(r['mem']['peak_ws'] for r in rows))
print("min available during run (MB):", round(min(r['min_avail'] for r in rows)))
print("ALL GAP-FREE" if allok and tot['gaps'] == 0 else "GAPS OR INCOMPLETE")
