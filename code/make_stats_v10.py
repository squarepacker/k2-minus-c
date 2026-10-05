"""Collect the statistics of the final T=9 runs into stats_v10.json (read by make_v10.py) and report whether everything is complete.
Complete means:
  - for every n = 1..5, with and without a side, all task indices 0..ntask-1 have a leaf file (and the B&B checkpoint says done);
  - every leaf file is fully re-verified by verify_leaves2.py (all 4 parts finished, checked = number of leaves, no failure,
    max rigorous bound <= 9);
  - every leaf file passed coverage_check.py.
(verify_leaves2.py contains the exact-2pi treatment, so the separate verify_edge.py results are no longer needed.)
Usage: python make_stats_v10.py"""
import json, glob, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
leaf_files = sorted(glob.glob('bnb_n*_T9_*_final_leaves.jsonl') + glob.glob('bnbw_n*_T9_*_final_leaves.jsonl'))
rows = []; complete = True
tot_leaves = {False: 0, True: 0}; tot_pieces = 0; tot_rig = 0; maxU = 0; fails = 0
groups = {}
for lf in leaf_files:
    m = re.match(r'(bnbw?)_n(\d)_T9_(\d+)of(\d+)_final_leaves\.jsonl$', lf)
    wall = m.group(1) == 'bnbw'; n = int(m.group(2)); task = int(m.group(3)); ntask = int(m.group(4))
    groups.setdefault((wall, n, ntask), set()).add(task)
    n_l = sum(1 for _ in open(lf))
    parts = sorted(glob.glob(lf.replace('.jsonl', '_verify2_*of4.json')))
    c = 0; fin = len(parts) == 4
    for p in parts:
        D = json.load(open(p)); c += D['checked']; fails += len(D['fail']); fin &= bool(D.get('finished'))
        maxU = max(maxU, D['maxU']); tot_pieces += D.get('pieces', 0); tot_rig += D.get('rig_pieces', 0)
    cov = lf.replace('.jsonl', '_coverage.json')
    cov_ok = os.path.exists(cov) and json.load(open(cov))['ok']
    ok = fin and c == n_l and cov_ok
    complete &= ok
    tot_leaves[wall] += n_l
    rows.append(dict(file=lf, leaves=n_l, verified=c, verify_finished=fin, coverage_ok=cov_ok, ok=ok))
# every (wall, n) present with a complete set of tasks
for wall in (False, True):
    for n in range(1, 6):
        g = [(k, v) for k, v in groups.items() if k[0] == wall and k[1] == n]
        good = len(g) == 1 and g[0][1] == set(range(g[0][0][2]))
        if not good: print('MISSING TASKS', 'wall' if wall else 'nowall', n, g)
        complete &= good
complete &= (fails == 0 and maxU <= 9)


def num(x):
    return f"${x:,}$".replace(',', '{,}')


nl = tot_leaves[False] + tot_leaves[True]
ST = dict(
    boxes_text=f"{num(tot_leaves[False])} accepted boxes without a side and {num(tot_leaves[True])} with a side",
    verify_text=f"all {num(nl)} boxes confirmed, with {num(tot_pieces)} further bisections and largest rigorous bound ${maxU}$",
    complete=complete, fails=fails, maxU=maxU, n_leaves=nl, pieces=tot_pieces, rig_pieces=tot_rig, rows=rows)
json.dump(ST, open('stats_v10.json', 'w'), indent=1)
for r in rows: print(('OK  ' if r['ok'] else 'TODO') + f"  {r['file']}: leaves {r['leaves']}, verified(v2) {r['verified']}, finished {r['verify_finished']}, coverage {r['coverage_ok']}")
print('COMPLETE' if complete else 'INCOMPLETE', '| fails', fails, '| maxU', maxU, '| rig pieces', tot_rig, '|', ST['boxes_text'], '|', ST['verify_text'])
