"""Stage 2: branch-and-bound for the disc-model bound  U(I) = E_II + sum_i P_i  <= T  (no wall).
Model (p = 0, worst case d = 1e-4, Lam = sqrt2 + 1e-4, rho1 = sqrt((d+sqrt2/2)^2 + 1/4), gamma0 = arccos(1 - 1/(2 Lam^2))):
  inner centres c_i = r_i (cos th_i, sin th_i), 1/2 < r_i <= rho1, pairwise >= 1, n = |I| in 1..5 (ALL centres within rho1);
  E_II <= number of inner pairs that may be at distance < Lam;
  P_i  <= max number of points, pairwise >= gamma0 apart, in the union of the direction cells phi in which some
          x = c_i + rho e(phi), rho in [1, Lam], may satisfy |x| > rho1, p in R(c_i, x), |x - c_j| >= 1 (j != i).
All bounds are computed with interval arithmetic in float64 with an explicit safety margin MARG (a constraint is declared
violated only if violated by more than MARG); leaf boxes are re-checked later in Arb ball arithmetic via python-flint (verify_leaves2.py).
A box is closed when its bound <= T; otherwise it is split. A box smaller than MINW whose CENTRE point already has
U > T (computed pointwise with a fine grid) is reported as a witness that the method cannot prove T.
Checkpointing: the work stack and statistics are pickled every CKPT seconds; rerun the same command to resume.
Usage: python bnb.py N T TASK_ID N_TASKS SECONDS"""
import math, sys, os, time, pickle, json
import numpy as np

d = 1e-4
LAM = math.sqrt(2) + d
RHO1 = math.sqrt((d + math.sqrt(2)/2)**2 + 0.25)
G0 = math.acos(1 - 1/(2*LAM**2))
MARG = 1e-9
NC = 1440                    # direction cells of 0.25 deg
DPHI = 2*math.pi/NC
PHI_LO = np.arange(NC)*DPHI
PHI_HI = PHI_LO + DPHI
NR = 6
RHO_E = np.linspace(1.0, LAM, NR + 1)
RLO, RHI = RHO_E[:-1], RHO_E[1:]
MINW = 1e-7


# ---------------- interval helpers (vectorised over numpy arrays) ----------------
def icos(lo, hi):
    """interval of cos over [lo, hi] (arrays)."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    c1, c2 = np.cos(lo), np.cos(hi)
    mn = np.minimum(c1, c2); mx = np.maximum(c1, c2)
    full = (hi - lo) >= 2*math.pi
    # contains a multiple of 2pi -> max 1; contains pi + 2k pi -> min -1
    k0 = np.ceil(lo/(2*math.pi)); has0 = k0*2*math.pi <= hi
    kp = np.ceil((lo - math.pi)/(2*math.pi)); hasp = kp*2*math.pi + math.pi <= hi
    mx = np.where(has0 | full, 1.0, mx); mn = np.where(hasp | full, -1.0, mn)
    return mn - 1e-15, mx + 1e-15


def isin(lo, hi):
    return icos(np.asarray(lo) - math.pi/2, np.asarray(hi) - math.pi/2)


def imul(a_lo, a_hi, b_lo, b_hi):
    p = np.stack([a_lo*b_lo, a_lo*b_hi, a_hi*b_lo, a_hi*b_hi])
    return p.min(0), p.max(0)


def iabs_lo(lo, hi):
    return np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(np.abs(lo), np.abs(hi)))


# ---------------- bounds for a box ----------------
def centre_xy(rl, rh, tl, th):
    cl, ch = icos(tl, th); sl, sh = isin(tl, th)
    xl, xh = imul(rl, rh, cl, ch); yl, yh = imul(rl, rh, sl, sh)
    return xl, xh, yl, yh


def allowed_cells(i, box):
    rl, rh, tl, th = box[i]
    # delta = phi - theta_i
    dl = PHI_LO - th; dh = PHI_HI - tl
    cdl, cdh = icos(dl, dh); sdl, sdh = isin(dl, dh)
    ok = np.ones((NC, NR), bool)
    # (1) |x| > rho1 : violated if max |x|^2 <= rho1^2 - MARG; |x|^2 = r^2 + 2 r rho cos(delta) + rho^2 (convex in r, rho)
    mx = None
    for r in (rl, rh):
        for rho in (RLO, RHI):
            v = r*r + 2*r*rho[None, :]*cdh[:, None] + (rho*rho)[None, :]
            mx = v if mx is None else np.maximum(mx, v)
    ok &= ~(mx < RHO1**2 - MARG)
    # (2) p in R(c_i, x): t = -r cos(delta) in [0, rho], perp = r |sin(delta)| <= 1/2
    t_lo, t_hi = imul(np.full(NC, -rh), np.full(NC, -rl), cdl, cdh)     # -r * cos  (r>0): product of [-rh,-rl] and [cdl,cdh]
    ok &= ~((t_hi < -MARG)[:, None])
    ok &= ~(t_lo[:, None] > RHI[None, :] + MARG)
    perp_lo = rl*iabs_lo(sdl, sdh)
    ok &= ~((perp_lo > 0.5 + MARG)[:, None])
    # (3) |x - c_j| >= 1 : violated if max |x - c_j|^2 < 1 - MARG
    cxl, cxh, cyl, cyh = centre_xy(rl, rh, tl, th)
    ecl, ech = icos(PHI_LO, PHI_HI); esl, esh = isin(PHI_LO, PHI_HI)
    for j in range(len(box)):
        if j == i: continue
        jxl, jxh, jyl, jyh = centre_xy(*box[j])
        for k in range(NR):
            pxl, pxh = imul(RLO[k], RHI[k], ecl, ech); pyl, pyh = imul(RLO[k], RHI[k], esl, esh)
            vxl = cxl - jxh + pxl; vxh = cxh - jxl + pxh
            vyl = cyl - jyh + pyl; vyh = cyh - jyl + pyh
            m2 = np.maximum(vxl*vxl, vxh*vxh) + np.maximum(vyl*vyl, vyh*vyh)
            ok[:, k] &= ~(m2 < 1 - MARG)
    return ok.any(axis=1)


def max_points(mask):
    """max number of points in the union of allowed cells with circular separation >= G0 (cells closed, conservative)."""
    idx = np.nonzero(mask)[0]
    if len(idx) == 0: return 0
    if len(idx) == NC: return int(math.floor(2*math.pi/G0 + 1e-12))
    best = 0
    # start at the beginning (lower end) of each maximal run of allowed cells
    starts = [k for k in idx if not mask[(k - 1) % NC]]
    for s in starts:
        pos = PHI_LO[s]; cnt = 1; first = pos
        while True:
            target = pos + G0
            # earliest allowed point >= target: find first allowed cell whose hi >= target
            found = None
            for off in range(NC):
                k = (s + off) % NC
                base = PHI_LO[s] + off*DPHI
                if base + DPHI < target - 1e-15: continue
                if base - PHI_LO[s] > 2*math.pi: break
                if mask[k]:
                    found = max(target, base); break
            if found is None or found > first + 2*math.pi - G0 + 1e-12: break
            pos = found; cnt += 1
        best = max(best, cnt)
    return best


def box_bound(box):
    n = len(box)
    # feasibility: pairwise >= 1 possible?
    xy = [centre_xy(*b) for b in box]
    E = 0
    for i in range(n):
        for j in range(i+1, n):
            a, b = xy[i], xy[j]
            dxl = a[0] - b[1]; dxh = a[1] - b[0]; dyl = a[2] - b[3]; dyh = a[3] - b[2]
            mx2 = np.maximum(dxl*dxl, dxh*dxh) + np.maximum(dyl*dyl, dyh*dyh)
            mn2 = iabs_lo(dxl, dxh)**2 + iabs_lo(dyl, dyh)**2
            if mx2 < 1 - MARG: return -1, None          # infeasible box
            if mn2 < LAM*LAM + MARG: E += 1
    P = [max_points(allowed_cells(i, box)) for i in range(n)]
    return E + sum(P), (E, P)


def split(box):
    best = None
    for i, (rl, rh, tl, th) in enumerate(box):
        wr = (rh - rl)/(RHO1 - 0.5)
        wt = (th - tl)/(math.pi/2) if i > 0 else 0.0
        for kind, w in (('r', wr), ('t', wt)):
            if best is None or w > best[0]: best = (w, i, kind)
    w, i, kind = best
    rl, rh, tl, th = box[i]
    if kind == 'r':
        m = 0.5*(rl + rh); A = (rl, m, tl, th); B = (m, rh, tl, th)
    else:
        m = 0.5*(tl + th); A = (rl, rh, tl, m); B = (rl, rh, m, th)
    b1 = list(box); b1[i] = A; b2 = list(box); b2[i] = B
    return w, b1, b2


def ordered_ok(box):
    # theta_1 = 0 fixed; require theta_2 <= ... <= theta_n possible
    for i in range(1, len(box) - 1):
        if box[i][2] > box[i+1][3]: return False
    return True


def initial_boxes(n, task, ntask):
    # split theta_2 range into ntask*8 slices, distribute round-robin
    base = [(0.5, RHO1, 0.0, 0.0)] + [(0.5, RHO1, 0.0, 2*math.pi)]*(n - 1)
    if n == 1: return [base] if task == 0 else []
    m = ntask*8; out = []
    for s in range(m):
        if s % ntask != task: continue
        b = list(base); b[1] = (0.5, RHO1, 2*math.pi*s/m, 2*math.pi*(s+1)/m); out.append(b)
    return out


if __name__ == '__main__':
    n, T, task, ntask, secs = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
    here = os.path.dirname(os.path.abspath(__file__))
    ck = os.path.join(here, f'bnb_n{n}_T{T}_{task}of{ntask}' + os.environ.get('BNB_TAG', '') + '.pkl')
    if os.path.exists(ck):
        st = pickle.load(open(ck, 'rb')); stack, stats, witnesses = st['stack'], st['stats'], st['witnesses']
        print('resumed', len(stack), stats, flush=True)
    else:
        stack = initial_boxes(n, task, ntask); stats = dict(closed=0, infeasible=0, split=0, maxU_closed=0); witnesses = []
    t0 = time.time(); last = t0
    dump = open(ck.replace('.pkl', '_leaves.jsonl'), 'a') if os.environ.get('BNB_DUMP') else None
    while stack and time.time() - t0 < secs:
        box = stack.pop()
        if not ordered_ok(box): stats['infeasible'] += 1; continue
        u, info = box_bound(box)
        if u < 0:
            stats['infeasible'] += 1
            if dump: dump.write(json.dumps(dict(kind='inf', box=box)) + '\n')
            continue
        if u <= T:
            stats['closed'] += 1; stats['maxU_closed'] = max(stats['maxU_closed'], u)
            if dump: dump.write(json.dumps(dict(kind='closed', U=int(u), box=box)) + '\n')
            continue
        w, b1, b2 = split(box)
        if w < MINW:
            witnesses.append(dict(U=int(u), info=str(info), box=[list(map(float, b)) for b in box]))
            stats['witness'] = stats.get('witness', 0) + 1
            if len(witnesses) >= 20: break
            continue
        stats['split'] += 1; stack.append(b1); stack.append(b2)
        if time.time() - last > 120:
            pickle.dump(dict(stack=stack, stats=stats, witnesses=witnesses), open(ck, 'wb'))
            print(f'{time.time()-t0:7.0f}s stack={len(stack)} {stats}', flush=True); last = time.time()
    pickle.dump(dict(stack=stack, stats=stats, witnesses=witnesses), open(ck, 'wb'))
    res = dict(n=n, T=T, task=task, ntask=ntask, done=(len(stack) == 0 and not witnesses), stack=len(stack), stats=stats, witnesses=witnesses[:5])
    json.dump(res, open(ck.replace('.pkl', '.json'), 'w'), indent=1, default=str)
    print('RESULT', json.dumps(res, default=str)[:2000], flush=True)
