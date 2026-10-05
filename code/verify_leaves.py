"""Rigorous re-verification (Arb ball arithmetic via python-flint) of every terminal box dumped by bnb.py / bnb_wall.py.
For each leaf:
  kind 'inf'    : certify that the box is infeasible: some inner pair is certainly at distance < 1, or (wall) some centre is
                  certainly closer than 1/2 to the side.
  kind 'nowall' : certify that no inner centre can have a square-side pair containing p.
  kind 'closed' : recompute an upper bound U_rig = E + W + sum_i P_i where
                  - E counts inner pairs not certified to be at distance >= Lam,
                  - P_i is the max number of points with separation >= gamma0 (minus 1e-12) in the cells NOT certified
                    to be violated (a cell is certified violated if for each rho-piece one of the constraints is certainly
                    violated; pieces that fail are subdivided up to depth 2 in phi and rho),
                  and check U_rig <= T.
All constants (Lam, rho1, gamma0, rho1^2, ...) are computed in Arb. Usage: python verify_leaves.py LEAVES.jsonl T [START STEP]"""
import sys, json, math, os
import numpy as np
from flint import arb, ctx
ctx.prec = 96
import bnb as B

D = arb('1e-4')
LAM = arb(2).sqrt() + D
RHO1SQ = (D + arb(2).sqrt()/2)**2 + arb('0.25')
WALL_R = D + arb(2).sqrt()/2
HALF = arb('0.5')
G0_SAFE = B.G0 - 1e-12            # float, slightly smaller than gamma0 (gamma0 is >= this): conservative for packing
PI2 = 2*arb.pi()
MAXSPLIT = 4096


def iv(lo, hi):
    return arb(lo).union(arb(hi))


def certified_violation(r, th, others, phi_lo, phi_hi, rho_lo, rho_hi, wall_h=None, depth=0):
    """True if for every phi in [phi_lo,phi_hi], rho in [rho_lo,rho_hi] some constraint is certainly violated."""
    ph = iv(phi_lo, phi_hi); rho = iv(rho_lo, rho_hi)
    dl = ph - th
    cd = dl.cos(); sd = dl.sin()
    ok = False
    # (1) |x|^2 <= rho1^2 certainly
    x2 = r*r + 2*r*rho*cd + rho*rho
    if x2 < RHO1SQ: return True
    # (2) p not in R(c, x): t = -r cos(delta) < 0 certainly, or t > rho certainly, or perp = r|sin(delta)| > 1/2 certainly
    t = -r*cd
    if t < 0: return True
    if t > rho: return True
    if abs(r*sd) > HALF: return True
    # (3) |x - c_j| < 1 certainly
    ex, ey = ph.cos(), ph.sin()
    cx, cy = r*th.cos(), r*th.sin()
    xx, xy = cx + rho*ex, cy + rho*ey
    for (oj_r, oj_t) in others:
        dx = xx - oj_r*oj_t.cos(); dy = xy - oj_r*oj_t.sin()
        if dx*dx + dy*dy < 1: return True
    if wall_h is not None and xy < -wall_h + HALF: return True
    if depth >= 2: return False
    pm = 0.5*(phi_lo + phi_hi); rm = 0.5*(rho_lo + rho_hi)
    return all(certified_violation(r, th, others, a, b, c, e, wall_h, depth + 1)
               for (a, b) in ((phi_lo, pm), (pm, phi_hi)) for (c, e) in ((rho_lo, rm), (rm, rho_hi)))


def rig_bound(box, T, h=None):
    n = len(box)
    R = [iv(b[0], b[1]) for b in box]; TH = [iv(b[2], b[3]) for b in box]
    wh = iv(h[0], h[1]) if h is not None else None
    E = 0
    for i in range(n):
        for j in range(i+1, n):
            dx = R[i]*TH[i].cos() - R[j]*TH[j].cos(); dy = R[i]*TH[i].sin() - R[j]*TH[j].sin()
            if not (dx*dx + dy*dy >= LAM*LAM): E += 1
    W = 0
    if h is not None:
        for i in range(n):
            cx, cy = R[i]*TH[i].cos(), R[i]*TH[i].sin()
            impossible = (cy + wh >= WALL_R) or (abs(cx) > HALF) or (cy < 0)
            if not impossible: W += 1
    total = E + W
    for i in range(n):
        others = [(R[j], TH[j]) for j in range(n) if j != i]
        fmask = B.allowed_cells(i, box) if h is None else __import__('bnb_wall').allowed_cells_wall(i, box, h[0], h[1])
        mask = np.ones(B.NC, bool)
        for k in np.nonzero(~fmask)[0]:
            if all(certified_violation(R[i], TH[i], others, B.PHI_LO[k], B.PHI_HI[k], B.RLO[q], B.RHI[q], wh)
                   for q in range(B.NR)):
                mask[k] = False
        G0_saved = B.G0; B.G0 = G0_SAFE
        try:
            total += B.max_points(mask)
        finally:
            B.G0 = G0_saved
        if total > T: return total
    return total


def check_inf(box, h=None):
    n = len(box)
    R = [iv(b[0], b[1]) for b in box]; TH = [iv(b[2], b[3]) for b in box]
    if h is not None:
        wh = iv(h[0], h[1])
        for i in range(n):
            if R[i]*TH[i].sin() < -wh + HALF: return True
    for i in range(n):
        for j in range(i+1, n):
            dx = R[i]*TH[i].cos() - R[j]*TH[j].cos(); dy = R[i]*TH[i].sin() - R[j]*TH[j].sin()
            if dx*dx + dy*dy < 1: return True
    return False


def check_nowall(box, h):
    wh = iv(h[0], h[1])
    for b in box:
        r = iv(b[0], b[1]); t = iv(b[2], b[3]); cx, cy = r*t.cos(), r*t.sin()
        if not ((cy + wh >= WALL_R) or (abs(cx) > HALF) or (cy < 0)): return False
    return True


if __name__ == '__main__':
    path, T = sys.argv[1], int(sys.argv[2])
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0; step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    out = path.replace('.jsonl', f'_verify_{start}of{step}.json')
    done = json.load(open(out)) if os.path.exists(out) else dict(checked=0, ok=0, fail=[], maxU=0, next=start)
    lines = open(path).read().splitlines()
    import time; t0 = time.time(); last = t0
    for idx in range(done['next'], len(lines), step):
        L = json.loads(lines[idx]); box0 = [tuple(b) for b in L['box']]; h0 = L.get('h')
        # rigorous sub-branch-and-bound from this leaf (MAXSPLIT pieces at most): a piece is OK if it is certified
        # infeasible, certified to have no side pair (wall runs), or has rigorous bound <= T; otherwise it is split
        work = [(box0, h0)]; ok = True; u = None; pieces = 0
        while work:
            box, h = work.pop()
            if check_inf(box, h): continue
            if h is not None and check_nowall(box, h): continue
            uu = rig_bound(box, T, h)
            if uu <= T:
                done['maxU'] = max(done['maxU'], uu); continue
            pieces += 1
            if pieces > MAXSPLIT: ok = False; u = uu; break
            if h is None:
                w, b1, b2 = B.split(box); work += [(b1, None), (b2, None)]
            else:
                import bnb_wall as BW
                w, A_, B_ = BW.split_wall(box, tuple(h)); work += [(A_[0], list(A_[1])), (B_[0], list(B_[1]))]
        done['pieces'] = done.get('pieces', 0) + pieces
        done['checked'] += 1; done['ok'] += int(ok)
        if not ok: done['fail'].append(dict(idx=idx, kind=L['kind'], U=u, box=L['box'], h=h))
        done['next'] = idx + step
        if time.time() - last > 60:
            json.dump(done, open(out, 'w')); last = time.time()
            print(f'{time.time()-t0:6.0f}s checked={done["checked"]} ok={done["ok"]} fail={len(done["fail"])} maxU={done["maxU"]}', flush=True)
    done['finished'] = True
    json.dump(done, open(out, 'w'), indent=1)
    print('RESULT', path, 'checked', done['checked'], 'ok', done['ok'], 'fail', len(done['fail']), 'maxU', done['maxU'], flush=True)
