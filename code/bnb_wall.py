"""Stage 2 (wall case): branch-and-bound for  U_w = E_II + W + sum_i P_i  <= T  with one container side present.
Frame: the side is the line y = -h (container above it), h in (0, d + sqrt2/2]: if h > d + sqrt2/2 no square-side pair can
contain p = 0 in its rectangle (the rectangle of such a pair lies within d + sqrt2/2 of the side), and then the side only
adds constraints, so the no-wall bound applies. The rectangles of all relevant side edges stand on one side (proof of Lemma 4.10
in the paper: "two of them on opposite sides would force k < 2d + sqrt2, which is impossible since k >= 2", and on adjacent sides
the two centres would satisfy |c - c'|^2 <= 1/2), and the other sides are ignored (conservative).
Constraints added to bnb.py: every centre (inner c_i and outer neighbour x) has y >= -h + 1/2.
W <= number of inner centres that may have a square-side pair containing p: dist(c_i, side) < d + sqrt2/2, |c_ix| <= 1/2, c_iy >= 0.
(The side's foot direction is >= 78 deg from every centre neighbour direction; we do not use this, which is conservative.)
No rotation normalisation (the side fixes the frame); reflection x -> -x is used: theta_1 in [-pi/2, pi/2] (c_1 = the inner
centre with the smallest index is taken in the right half-plane or on the axis) -- for safety we do NOT use it by default.
Usage: python bnb_wall.py N T TASK NTASK SECONDS"""
import math, sys, os, time, pickle, json
import numpy as np
import bnb as B

WALL_R = B.d + math.sqrt(2)/2


def allowed_cells_wall(i, box, h_lo, h_hi):
    ok = B.allowed_cells(i, box)
    # outer neighbour x = c_i + rho e(phi) must have x_y >= -h + 1/2: violated if max x_y < -h_hi + 1/2 - MARG
    rl, rh, tl, th = box[i]
    cxl, cxh, cyl, cyh = B.centre_xy(rl, rh, tl, th)
    esl, esh = B.isin(B.PHI_LO, B.PHI_HI)
    ymax = cyh + np.maximum(B.RLO[0]*esh, B.RHI[-1]*esh)        # max over rho in [1, Lam] of rho*sin (sin may be negative)
    viol = ymax < -h_hi + 0.5 - B.MARG
    return ok & ~viol


def box_bound_wall(box, h_lo, h_hi):
    n = len(box)
    xy = [B.centre_xy(*b) for b in box]
    for (xl, xh, yl, yh) in xy:
        if yh < -h_hi + 0.5 - B.MARG: return -1, None            # centre too close to (or beyond) the side
    E = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = xy[i], xy[j]
            dxl = a[0] - b[1]; dxh = a[1] - b[0]; dyl = a[2] - b[3]; dyh = a[3] - b[2]
            mx2 = max(dxl*dxl, dxh*dxh) + max(dyl*dyl, dyh*dyh)
            mn2 = float(B.iabs_lo(dxl, dxh))**2 + float(B.iabs_lo(dyl, dyh))**2
            if mx2 < 1 - B.MARG: return -1, None
            if mn2 < B.LAM*B.LAM + B.MARG: E += 1
    W = 0
    for (xl, xh, yl, yh) in xy:
        possible = (yl + h_lo < WALL_R + B.MARG) and (float(B.iabs_lo(xl, xh)) <= 0.5 + B.MARG) and (yh >= -B.MARG)
        W += int(possible)
    if W == 0: return -2, None                                     # no side pair possible: covered by the no-wall run
    P = [B.max_points(allowed_cells_wall(i, box, h_lo, h_hi)) for i in range(n)]
    return E + W + sum(P), (E, W, P)


def split_wall(box, h):
    best = None
    for i, (rl, rh, tl, th) in enumerate(box):
        for kind, w in (('r', (rh - rl)/(B.RHO1 - 0.5)), ('t', (th - tl)/(math.pi/2))):
            if best is None or w > best[0]: best = (w, i, kind)
    wh = (h[1] - h[0])/0.7
    if wh > best[0]:
        m = 0.5*(h[0] + h[1]); return wh, (box, (h[0], m)), (box, (m, h[1]))
    w, i, kind = best
    rl, rh, tl, th = box[i]
    if kind == 'r':
        m = 0.5*(rl + rh); A = (rl, m, tl, th); Bx = (m, rh, tl, th)
    else:
        m = 0.5*(tl + th); A = (rl, rh, tl, m); Bx = (rl, rh, m, th)
    b1 = list(box); b1[i] = A; b2 = list(box); b2[i] = Bx
    return w, (b1, h), (b2, h)


def ordered_ok(box):
    for i in range(len(box) - 1):
        if box[i][2] > box[i+1][3]: return False
    return True


if __name__ == '__main__':
    n, T, task, ntask, secs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    here = os.path.dirname(os.path.abspath(__file__))
    ck = os.path.join(here, f'bnbw_n{n}_T{T}_{task}of{ntask}' + os.environ.get('BNB_TAG', '') + '.pkl')
    if os.path.exists(ck):
        st = pickle.load(open(ck, 'rb')); stack, stats, witnesses = st['stack'], st['stats'], st['witnesses']
    else:
        m = ntask*4; stack = []
        for s in range(m):
            if s % ntask != task: continue
            b = [(0.5, B.RHO1, 2*math.pi*s/m, 2*math.pi*(s+1)/m)] + [(0.5, B.RHO1, 0.0, 2*math.pi)]*(n - 1)
            stack.append((b, (0.0, WALL_R)))
        stats = dict(closed=0, infeasible=0, nowall=0, split=0, maxU_closed=0); witnesses = []
    t0 = time.time(); last = t0
    dump = open(ck.replace('.pkl', '_leaves.jsonl'), 'a') if os.environ.get('BNB_DUMP') else None
    while stack and time.time() - t0 < secs:
        box, h = stack.pop()
        if not ordered_ok(box): stats['infeasible'] += 1; continue
        u, info = box_bound_wall(box, *h)
        if u == -1:
            stats['infeasible'] += 1
            if dump: dump.write(json.dumps(dict(kind='inf', box=box, h=h)) + '\n')
            continue
        if u == -2:
            stats['nowall'] += 1
            if dump: dump.write(json.dumps(dict(kind='nowall', box=box, h=h)) + '\n')
            continue
        if u <= T:
            stats['closed'] += 1; stats['maxU_closed'] = max(stats['maxU_closed'], u)
            if dump: dump.write(json.dumps(dict(kind='closed', U=int(u), box=box, h=h)) + '\n')
            continue
        w, A, Bb = split_wall(box, h)
        if w < B.MINW:
            witnesses.append(dict(U=int(u), info=str(info), h=list(h), box=[list(map(float, b)) for b in box])); stats['witness'] = stats.get('witness', 0) + 1
            if len(witnesses) >= 20: break
            continue
        stats['split'] += 1; stack.append(A); stack.append(Bb)
        if time.time() - last > 120:
            pickle.dump(dict(stack=stack, stats=stats, witnesses=witnesses), open(ck, 'wb'))
            print(f'{time.time()-t0:7.0f}s stack={len(stack)} {stats}', flush=True); last = time.time()
    pickle.dump(dict(stack=stack, stats=stats, witnesses=witnesses), open(ck, 'wb'))
    res = dict(n=n, T=T, task=task, ntask=ntask, done=(len(stack) == 0 and not witnesses), stack=len(stack), stats=stats, witnesses=witnesses[:5])
    json.dump(res, open(ck.replace('.pkl', '.json'), 'w'), indent=1, default=str)
    print('RESULT', json.dumps(res, default=str)[:2000], flush=True)
