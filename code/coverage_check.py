"""Coverage check for the final T=9 branch-and-bound runs (independent of the leaf verification).
For each leaf file (one B&B task) it checks that the recorded terminal boxes ('closed', 'inf', 'nowall') together with the
boxes pruned by the ordering condition tile the task's initial parameter boxes:
  (1) tree reconstruction: starting from the initial boxes, every node is either a recorded leaf (matched exactly, used once),
      or violates the ordering condition theta_i <= theta_{i+1} (no configuration with sorted angles lies in it), or is
      bisected at its midpoint (the two halves cover it exactly); at the end every recorded leaf must have been used;
  (2) volume: with exact rational arithmetic, sum of volumes(leaves) + volumes(pruned) = volume(initial boxes);
  (3) the initial boxes are r in [1/2, RHO1], theta slices of [0, 2 pi_float] with matching endpoints, h in [0, WALL_R]; that
      RHO1 >= exact rho1 and WALL_R >= exact d + sqrt2/2 is checked with mpmath in verify_v10.py (not in this file);
  (4) boxes whose theta-interval ends at 2 pi_float (2 pi_float < 2 pi by 2.4e-16) are counted ('edge_boxes'); the missing arc
      (2 pi_float, 2 pi] is covered by verify_leaves2.py, which extends every such interval to the exact 2 pi in Arb
      (for the wall runs the list is also written to *_edge_leaves.jsonl, used by the superseded verify_edge.py).
The split rule used for the reconstruction is that of bnb.py / bnb_wall.py (a mismatch would make the reconstruction fail,
never pass wrongly, because a node is accepted only if it equals a recorded leaf exactly).
Usage: python coverage_check.py LEAVES.jsonl"""
import sys, json, math, os
from fractions import Fraction as Fr
import bnb as B
import bnb_wall as BW

path = sys.argv[1]
name = os.path.basename(path)
wall = name.startswith('bnbw_')
# bnb_n{n}_T{T}_{task}of{ntask}_final_leaves.jsonl
core = name.split('_final_leaves')[0]
parts = core.split('_')
n = int(parts[1][1:]); T = int(parts[2][1:]); task, ntask = map(int, parts[3].split('of'))

leaves = {}
for line in open(path):
    L = json.loads(line)
    key = (json.dumps([list(b) for b in L['box']]), json.dumps(list(L['h'])) if wall else '')
    leaves[key] = leaves.get(key, 0) + 1
dup = sum(1 for v in leaves.values() if v > 1)


def key_of(box, h):
    return (json.dumps([list(map(float, b)) for b in box]), json.dumps(list(map(float, h))) if wall else '')


def vol(box, h):
    v = Fr(1)
    for i, (rl, rh, tl, th) in enumerate(box):
        v *= (Fr(rh) - Fr(rl))
        if wall or i > 0: v *= (Fr(th) - Fr(tl))
    if wall: v *= (Fr(h[1]) - Fr(h[0]))
    return v


if wall:
    m = ntask*4; init = []
    for s in range(m):
        if s % ntask != task: continue
        b = [(0.5, B.RHO1, 2*math.pi*s/m, 2*math.pi*(s+1)/m)] + [(0.5, B.RHO1, 0.0, 2*math.pi)]*(n - 1)
        init.append((b, (0.0, BW.WALL_R)))
else:
    init = [(b, None) for b in B.initial_boxes(n, task, ntask)]

used = {}; v_leaf = Fr(0); v_pruned = Fr(0); edge = []; nodes = 0
stack = list(init)
v_init = sum(vol(b, h) for b, h in init)
TWO_PI = 2*math.pi
while stack:
    box, h = stack.pop(); nodes += 1
    k = key_of(box, h)
    if k in leaves:
        used[k] = used.get(k, 0) + 1; v_leaf += vol(box, h)
        if any(b[3] == TWO_PI for b in (box if wall else box[1:])): edge.append(dict(box=box, h=h))
        continue
    ordered = BW.ordered_ok(box) if wall else B.ordered_ok(box)
    if not ordered:
        v_pruned += vol(box, h); continue
    if wall:
        w, A, Bb = BW.split_wall(box, tuple(h))
        if w < B.MINW: print('FAIL: reached a tiny unmatched box'); sys.exit(1)
        stack.append((A[0], list(A[1]))); stack.append((Bb[0], list(Bb[1])))
    else:
        w, b1, b2 = B.split(box)
        if w < B.MINW: print('FAIL: reached a tiny unmatched box'); sys.exit(1)
        stack.append((b1, None)); stack.append((b2, None))
unused = [k for k in leaves if k not in used]
overused = [k for k, c in used.items() if c > 1]
ok = (not unused) and (not overused) and dup == 0 and (v_leaf + v_pruned == v_init)
res = dict(file=name, n=n, wall=wall, task=task, ntask=ntask, leaves=len(leaves), nodes=nodes, unused=len(unused), overused=len(overused),
           duplicates=dup, volume_equal=(v_leaf + v_pruned == v_init), pruned_fraction=float(v_pruned/v_init) if v_init else None,
           edge_boxes=len(edge), ok=ok)
print(json.dumps(res))
json.dump(res, open(path.replace('.jsonl', '_coverage.json'), 'w'), indent=1)
if wall and edge:
    with open(path.replace('_final_leaves.jsonl', '_edge_leaves.jsonl'), 'w') as f:
        for e in edge: f.write(json.dumps(dict(box=[list(map(float, b)) for b in e['box']], h=list(map(float, e['h'])))) + '\n')
