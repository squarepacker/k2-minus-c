"""Rigorous re-verification, version 2 (supersedes verify_leaves.py and verify_edge.py for the final check).
Same method as verify_leaves.py (Arb ball arithmetic, sub-branch-and-bound of each accepted box), with two gaps closed
(found in review14; the floating-point cell end points do not tile the circle exactly):
  (a) direction cells: the float cells [PHI_LO[k], PHI_HI[k]] leave 264 gaps of one ulp (PHI_HI[k] < PHI_LO[k+1]);
      here cell k is taken as [PHI_LO[k], max(PHI_HI[k], PHI_LO[k+1])], so the cells cover [0, 2pi_float] without gaps;
  (b) every interval whose upper end is the float 2pi (the last direction cell, and angle intervals theta_i of the boxes)
      is extended to contain the exact 2pi (Arb), so the arc (2pi_float, 2pi) is covered.
max_points (float, cells treated as closed arcs [PHI_LO, PHI_LO + DPHI], separation gamma0 - 1e-12) is unchanged: the certified cells are at most 1e-15
longer than the float arcs it uses, far below the 1e-12 margin.
Output: *_final_leaves_verify2_{START}of{STEP}.json (checkpoint every 60 s, resumable); 'rig_pieces' counts pieces decided by
the direction count (rig_bound).  Usage: python verify_leaves2.py LEAVES.jsonl T [START STEP]"""
import sys, json, math, os, time
import numpy as np
from flint import arb
import verify_leaves as V
import bnb as B

TWO_PI_F = 2*math.pi
_iv0 = V.iv


def iv2(lo, hi):
    x = _iv0(lo, hi)
    if hi == TWO_PI_F:
        x = x.union(2*arb.pi())
    return x


V.iv = iv2                              # used by certified_violation, check_inf, check_nowall
PHI_HI_EFF = [max(float(B.PHI_HI[k]), float(B.PHI_LO[k + 1])) for k in range(B.NC - 1)] + [float(B.PHI_HI[B.NC - 1])]
assert all(PHI_HI_EFF[k] >= B.PHI_LO[k + 1] for k in range(B.NC - 1)) and PHI_HI_EFF[-1] == TWO_PI_F and B.PHI_LO[0] == 0.0
assert all(B.RHI[q] == B.RLO[q + 1] for q in range(B.NR - 1))


def rig_bound2(box, T, h=None):
    n = len(box)
    R = [iv2(b[0], b[1]) for b in box]; TH = [iv2(b[2], b[3]) for b in box]
    wh = iv2(h[0], h[1]) if h is not None else None
    E = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = R[i]*TH[i].cos() - R[j]*TH[j].cos(); dy = R[i]*TH[i].sin() - R[j]*TH[j].sin()
            if not (dx*dx + dy*dy >= V.LAM*V.LAM): E += 1
    W = 0
    if h is not None:
        for i in range(n):
            cx, cy = R[i]*TH[i].cos(), R[i]*TH[i].sin()
            impossible = (cy + wh >= V.WALL_R) or (abs(cx) > V.HALF) or (cy < 0)
            if not impossible: W += 1
    total = E + W
    for i in range(n):
        others = [(R[j], TH[j]) for j in range(n) if j != i]
        if h is None:
            fmask = B.allowed_cells(i, box)
        else:
            import bnb_wall
            fmask = bnb_wall.allowed_cells_wall(i, box, h[0], h[1])
        mask = np.ones(B.NC, bool)
        for k in np.nonzero(~fmask)[0]:
            if all(V.certified_violation(R[i], TH[i], others, float(B.PHI_LO[k]), PHI_HI_EFF[k], float(B.RLO[q]), float(B.RHI[q]), wh)
                   for q in range(B.NR)):
                mask[k] = False
        G0_saved = B.G0; B.G0 = V.G0_SAFE
        try:
            total += B.max_points(mask)
        finally:
            B.G0 = G0_saved
        if total > T: return total
    return total


if __name__ == '__main__':
    path, T = sys.argv[1], int(sys.argv[2])
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0; step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    out = path.replace('.jsonl', f'_verify2_{start}of{step}.json')
    done = json.load(open(out)) if os.path.exists(out) else dict(checked=0, ok=0, fail=[], maxU=0, next=start, pieces=0, rig_pieces=0)
    if done.get('finished'):
        print('RESULT(already)', path, done['checked']); sys.exit(0)
    lines = open(path).read().splitlines()
    t0 = time.time(); last = t0
    for idx in range(done['next'], len(lines), step):
        L = json.loads(lines[idx]); box0 = [tuple(b) for b in L['box']]; h0 = L.get('h')
        work = [(box0, h0)]; ok = True; u = None; pieces = 0
        while work:
            box, h = work.pop()
            if V.check_inf(box, h): continue
            if h is not None and V.check_nowall(box, h): continue
            uu = rig_bound2(box, T, h)
            done['rig_pieces'] += 1
            if uu <= T:
                done['maxU'] = max(done['maxU'], uu); continue
            pieces += 1
            if pieces > V.MAXSPLIT: ok = False; u = uu; break
            if h is None:
                w, b1, b2 = B.split(box); work += [(b1, None), (b2, None)]
            else:
                import bnb_wall as BW
                w, A_, B_ = BW.split_wall(box, tuple(h)); work += [(A_[0], list(A_[1])), (B_[0], list(B_[1]))]
        done['pieces'] += pieces
        done['checked'] += 1; done['ok'] += int(ok)
        if not ok: done['fail'].append(dict(idx=idx, kind=L['kind'], U=u))
        done['next'] = idx + step
        if time.time() - last > 60:
            json.dump(done, open(out, 'w')); last = time.time()
            print(f'{time.time()-t0:6.0f}s checked={done["checked"]} ok={done["ok"]} fail={len(done["fail"])} maxU={done["maxU"]} rig={done["rig_pieces"]}', flush=True)
    done['finished'] = True; done['n_lines'] = len(lines)
    json.dump(done, open(out, 'w'), indent=1)
    print('RESULT', path, 'checked', done['checked'], 'ok', done['ok'], 'fail', len(done['fail']), 'maxU', done['maxU'], 'rig', done['rig_pieces'], flush=True)
