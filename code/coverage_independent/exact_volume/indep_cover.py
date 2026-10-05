"""Independent coverage checker for the T9 branch-and-bound leaf files.
Written from scratch; does not import or read the author's coverage scripts.
Domain (from cstar_v10.tex Lemma 4.10 proof, constants re-derived as in bnb.py / bnb_wall.py):
  no side : r_i in [1/2, RHO1] (i=1..n), theta_1 = 0, 0 <= theta_2 <= ... <= theta_n <= 2pi_f,
            theta_2 in slices [2pi s/m, 2pi (s+1)/m], m = 8*ntask, s = task mod ntask
  side    : r_i in [1/2, RHO1], 0 <= theta_1 <= ... <= theta_n <= 2pi_f, theta_1 in slices of m = 4*ntask, h in [0, WALL_R]
Method A (exact measure): (1) every box inside the bounding domain; (2) boxes pairwise interior-disjoint (float compare, exact);
  (3) sum over boxes of vol(box cap S) == vol(D) in exact rationals (Fraction of floats), S = sorted region.
  With (1)-(3), the finite union of closed boxes covers D up to a null set; as D is a finite union of convex bodies with
  non-empty interior and the union is closed, it covers D entirely.
Method B (point location): many random / boundary / tie / grid / ulp-nudged points, each checked against all boxes.
Usage: python indep_cover.py FILE [NPOINTS] [SEED]"""
import sys, json, math, re, os, time
from fractions import Fraction as F
import numpy as np

D_ = 1e-4
RHO1 = math.sqrt((D_ + math.sqrt(2) / 2) ** 2 + 0.25)
WALL_R = D_ + math.sqrt(2) / 2
TWO_PI = 2 * math.pi


# ---------------- exact volume of {x_1 <= ... <= x_m, x_k in [a_k, b_k]} ----------------
def padd(p, q):
    n = max(len(p), len(q)); return [(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)]


def pint(p):  # antiderivative with zero constant
    return [F(0)] + [c / (i + 1) for i, c in enumerate(p)]


def peval(p, x):
    s = F(0)
    for c in reversed(p): s = s * x + c
    return s


def chain_volume(iv):
    """iv: list of (a,b) Fractions. exact volume of sorted chain inside the box."""
    m = len(iv)
    if m == 0: return F(1)
    for a, b in iv:
        if b < a: return F(0)
    P = sorted(set([a for a, b in iv] + [b for a, b in iv]))
    K = len(P) - 1
    if K == 0: return F(0)
    # G_{k-1} represented on each interval j as polynomial g[j]; G_0 = 1
    g = [[F(1)] for _ in range(K)]
    for k in range(m):
        a, b = iv[k]
        newg = []; acc = F(0)
        for j in range(K):
            lo, hi = P[j], P[j + 1]
            if a <= lo and hi <= b:
                Q = pint(g[j]); q0 = peval(Q, lo)
                poly = padd(Q, [acc - q0])
                newg.append(poly); acc = peval(poly, hi)
            else:
                newg.append([acc])
        g = newg
    return peval(g[-1], P[-1])


# ---------------- load ----------------
def load(path):
    name = os.path.basename(path)
    mt = re.match(r'(bnbw?)_n(\d)_T9_(\d)of(\d)_final_leaves\.jsonl', name)
    wall = mt.group(1) == 'bnbw'; n = int(mt.group(2)); task = int(mt.group(3)); ntask = int(mt.group(4))
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return wall, n, task, ntask, recs


def setup(wall, n, task, ntask, recs):
    """returns lo, hi arrays (N x dim), kinds, coordinate names, slice coordinate index, angle indices, slices."""
    names = []; ang = []
    for i in range(n):
        names.append(f'r{i+1}')
    if wall:
        for i in range(n): ang.append(len(names)); names.append(f't{i+1}')
        names.append('h')
        m = 4 * ntask
    else:
        for i in range(1, n): ang.append(len(names)); names.append(f't{i+1}')
        m = 8 * ntask
    slices = [(TWO_PI * s / m, TWO_PI * (s + 1) / m) for s in range(m) if s % ntask == task]
    if not wall and n == 1: slices = None
    N = len(recs); dim = len(names)
    lo = np.zeros((N, dim)); hi = np.zeros((N, dim)); kinds = []
    bad = []
    for k, rec in enumerate(recs):
        box = rec['box']; kinds.append(rec['kind'])
        if len(box) != n: bad.append((k, 'wrong number of centres')); continue
        for i in range(n):
            lo[k, i], hi[k, i] = box[i][0], box[i][1]
        if wall:
            for i in range(n):
                lo[k, n + i], hi[k, n + i] = box[i][2], box[i][3]
            lo[k, 2 * n], hi[k, 2 * n] = rec['h'][0], rec['h'][1]
        else:
            if not (box[0][2] == 0.0 and box[0][3] == 0.0): bad.append((k, f'theta_1 not fixed 0: {box[0][2:]}'))
            for i in range(1, n):
                lo[k, n + i - 1], hi[k, n + i - 1] = box[i][2], box[i][3]
    return names, ang, slices, lo, hi, kinds, bad


def bounds(names, wall, n):
    blo = []; bhi = []
    for nm in names:
        if nm[0] == 'r': blo.append(0.5); bhi.append(RHO1)
        elif nm[0] == 't': blo.append(0.0); bhi.append(TWO_PI)
        else: blo.append(0.0); bhi.append(WALL_R)
    return np.array(blo), np.array(bhi)


def domain_volume(names, ang, slices, wall, n):
    v = F(1)
    for nm in names:
        if nm[0] == 'r': v *= F(RHO1) - F(0.5)
        elif nm == 'h': v *= F(WALL_R)
    if not ang: return v
    tot = F(0)
    for (a, b) in slices:
        iv = [(F(a), F(b))] + [(F(0), F(TWO_PI))] * (len(ang) - 1)
        tot += chain_volume(iv)
    return v * tot


def box_volume_sorted(k, lo, hi, names, ang, cache):
    v = F(1)
    for d, nm in enumerate(names):
        if nm[0] != 't': v *= F(hi[k, d]) - F(lo[k, d])
    key = tuple((lo[k, d], hi[k, d]) for d in ang)
    if key not in cache:
        cache[key] = chain_volume([(F(a), F(b)) for a, b in key])
    return v * cache[key]


def interior_overlaps(lo, hi, limit=50):
    N = len(lo); out = []
    order = np.argsort(lo[:, 0])
    for i in range(N):
        # strict interior overlap in every dim
        m = np.ones(N, bool); m[:i + 1] = False
        idx = np.nonzero(m)[0]
        for d in range(lo.shape[1]):
            if len(idx) == 0: break
            sel = (lo[idx, d] < hi[i, d]) & (lo[i, d] < hi[idx, d])
            idx = idx[sel]
        for j in idx:
            out.append((i, int(j)))
            if len(out) >= limit: return out
    return out


# ---------------- method B: point location ----------------
class Locator:
    def __init__(self, lo, hi, sd, nb=256):
        self.lo, self.hi, self.sd = lo, hi, sd
        a = lo[:, sd].min(); b = hi[:, sd].max(); self.a, self.b, self.nb = a, b, nb
        w = (b - a) / nb if b > a else 1.0; self.w = w
        self.bins = []
        for t in range(nb):
            blo, bhi = a + t * w, a + (t + 1) * w
            self.bins.append(np.nonzero((lo[:, sd] <= bhi + 1e-12) & (hi[:, sd] >= blo - 1e-12))[0])

    def covered(self, pts):
        res = np.zeros(len(pts), bool); cnt = np.zeros(len(pts), int)
        t = np.clip(((pts[:, self.sd] - self.a) / self.w).astype(int), 0, self.nb - 1)
        for b in np.unique(t):
            pi = np.nonzero(t == b)[0]
            # also include neighbours for points on a bin edge (safety)
            cand = np.unique(np.concatenate([self.bins[x] for x in (b - 1, b, b + 1) if 0 <= x < self.nb]))
            L = self.lo[cand]; H = self.hi[cand]
            for s in range(0, len(pi), 256):
                P = pts[pi[s:s + 256]]
                inside = np.all((L[None, :, :] <= P[:, None, :]) & (P[:, None, :] <= H[None, :, :]), axis=2)
                res[pi[s:s + 256]] = inside.any(1); cnt[pi[s:s + 256]] = inside.sum(1)
        return res, cnt


def in_domain(pts, names, ang, slices, sd, blo, bhi):
    ok = np.all((pts >= blo) & (pts <= bhi), axis=1)
    for u, v in zip(ang[:-1], ang[1:]): ok &= pts[:, u] <= pts[:, v]
    if slices is not None:
        s_ok = np.zeros(len(pts), bool)
        for a, b in slices: s_ok |= (pts[:, sd] >= a) & (pts[:, sd] <= b)
        ok &= s_ok
    return ok


def gen_points(rng, M, names, ang, slices, sd, blo, bhi, lo, hi):
    dim = len(names); out = []
    # 1. uniform: sample, sort angle coordinates
    P = rng.uniform(blo, bhi, size=(M, dim))
    if ang: P[:, ang] = np.sort(P[:, ang], axis=1)
    out.append(('uniform', P))
    # 2. slice-directed: put slice coord uniformly in a task slice, rest above it sorted
    if slices is not None:
        P = rng.uniform(blo, bhi, size=(M, dim)); s = rng.integers(len(slices), size=M)
        a = np.array([slices[x][0] for x in s]); b = np.array([slices[x][1] for x in s])
        P[:, sd] = rng.uniform(a, b)
        rest = ang[1:]
        if rest:
            P[:, rest] = np.sort(rng.uniform(P[:, sd][:, None], TWO_PI, size=(M, len(rest))), axis=1)
        out.append(('in-slice', P))
        # 2b. slice-directed with ties
        Q = P.copy()
        if len(ang) > 1:
            for r in range(M):
                k = rng.integers(1, len(ang)); i = rng.integers(0, len(ang) - k + 1) if len(ang) - k + 1 > 0 else 0
                i = min(i, len(ang) - 1); j = min(i + k, len(ang) - 1)
                Q[r, ang[i:j + 1]] = Q[r, ang[i]]
        out.append(('ties', Q))
        # 2c. slice endpoints exactly
        R = P.copy(); e = rng.integers(2, size=M); R[:, sd] = np.where(e == 0, a, b)
        if rest: R[:, rest] = np.sort(rng.uniform(R[:, sd][:, None], TWO_PI, size=(M, len(rest))), axis=1)
        out.append(('slice-endpoints', R))
    # 3. domain boundary values substituted
    P = out[-1][1].copy() if slices is not None else out[0][1].copy()
    for r in range(M):
        for d in range(dim):
            if rng.random() < 0.3:
                P[r, d] = blo[d] if rng.random() < 0.5 else bhi[d]
        if ang: P[r, ang] = np.sort(P[r, ang])
    out.append(('domain-boundary', P))
    # 4. grid: coordinates drawn from the set of box end points of that coordinate
    P = np.empty((M, dim))
    for d in range(dim):
        vals = np.unique(np.concatenate([lo[:, d], hi[:, d]]))
        P[:, d] = vals[rng.integers(len(vals), size=M)]
    if ang: P[:, ang] = np.sort(P[:, ang], axis=1)
    out.append(('box-endpoint-grid', P))
    # 5. ulp-nudged box corners: random corner of a random box, every coordinate nudged one ulp outward (or kept)
    k = rng.integers(len(lo), size=M); side = rng.integers(2, size=(M, dim))
    C = np.where(side == 0, lo[k], hi[k])
    nud = rng.integers(3, size=(M, dim))  # 0 keep, 1 outward, 2 inward
    out_dir = np.where(side == 0, -np.inf, np.inf); in_dir = -out_dir
    C = np.where(nud == 1, np.nextafter(C, out_dir), np.where(nud == 2, np.nextafter(C, in_dir), C))
    if ang: C[:, ang] = np.sort(C[:, ang], axis=1)
    out.append(('ulp-nudged-corners', C))
    # 6. face midpoints nudged one ulp outward across one face
    k = rng.integers(len(lo), size=M); d0 = rng.integers(dim, size=M); side = rng.integers(2, size=M)
    C = rng.uniform(lo[k], hi[k])
    v = np.where(side == 0, lo[k, d0], hi[k, d0])
    C[np.arange(M), d0] = np.nextafter(v, np.where(side == 0, -np.inf, np.inf))
    if ang: C[:, ang] = np.sort(C[:, ang], axis=1)
    out.append(('ulp-across-face', C))
    return out


def main(path, M, seed):
    t0 = time.time()
    wall, n, task, ntask, recs = load(path)
    mut = os.environ.get('MUTATE')  # self-test of the checker: deliberately damage the box list
    if mut:
        import random, copy
        R = random.Random(seed); k = R.randrange(len(recs)); recs = copy.deepcopy(recs)
        if mut == 'drop': recs.pop(k)
        elif mut == 'dup': recs.append(copy.deepcopy(recs[k]))
        elif mut == 'shrink':   # move one non-domain-boundary endpoint inward by one ulp
            box = recs[k]['box']
            for _ in range(1000):
                i = R.randrange(n); e = R.randrange(4)
                if e >= 2 and i == 0 and not wall: continue
                v = box[i][e]
                if v not in (0.5, RHO1, 0.0, TWO_PI):
                    box[i][e] = float(np.nextafter(v, np.inf if e % 2 == 0 else -np.inf)); rep_e = (i, e); break
        rep_mut = f'{mut} box #{k}'
    names, ang, slices, lo, hi, kinds, bad = setup(wall, n, task, ntask, recs)
    rep = dict(file=os.path.basename(path), wall=wall, n=n, task=task, ntask=ntask, boxes=len(recs),
               kinds={k: kinds.count(k) for k in sorted(set(kinds))}, dims=names, format_problems=bad[:10])
    if mut: rep['MUTATION'] = rep_mut
    blo, bhi = bounds(names, wall, n)
    sd = ang[0] if ang else 0
    # (1) inside domain bounds, positive widths, slice containment
    outside = np.nonzero(np.any((lo < blo) | (hi > bhi), axis=1))[0]
    zerow = np.nonzero(np.any(hi <= lo, axis=1))[0]
    rep['boxes_outside_bounds'] = [(int(k), lo[k].tolist(), hi[k].tolist()) for k in outside[:5]]
    rep['n_outside_bounds'] = int(len(outside)); rep['n_zero_or_negative_width'] = int(len(zerow))
    if slices is not None:
        ins = np.zeros(len(lo), bool)
        for a, b in slices: ins |= (lo[:, sd] >= a) & (hi[:, sd] <= b)
        rep['n_not_in_a_task_slice'] = int((~ins).sum())
        rep['slices'] = slices
    # boxes with no sorted point (should have been pruned): lo_t[i] > hi_t[i+1]
    if len(ang) > 1:
        unsorted = np.zeros(len(lo), bool)
        for u, v in zip(ang[:-1], ang[1:]): unsorted |= lo[:, u] > hi[:, v]
        rep['n_boxes_without_sorted_point'] = int(unsorted.sum())
    rep['max_theta_hi'] = float(hi[:, ang].max()) if ang else None
    rep['n_boxes_touching_2pi_f'] = int((hi[:, ang] == TWO_PI).any(1).sum()) if ang else 0
    # (2) interior disjointness
    ov = interior_overlaps(lo, hi)
    rep['n_interior_overlapping_pairs(first50)'] = len(ov)
    rep['overlap_examples'] = [(i, j, kinds[i], kinds[j]) for i, j in ov[:5]]
    # (3) exact volume
    cache = {}
    tot = F(0); vol_by_kind = {}
    for k in range(len(lo)):
        v = box_volume_sorted(k, lo, hi, names, ang, cache); tot += v
        if v == 0: rep['n_boxes_zero_sorted_volume'] = rep.get('n_boxes_zero_sorted_volume', 0) + 1
        vol_by_kind[kinds[k]] = vol_by_kind.get(kinds[k], F(0)) + v
    if ov:  # subtract nothing, but report the sorted-measure of overlaps
        ovv = F(0)
        for i, j in ov:
            L = np.maximum(lo[i], lo[j]); H = np.minimum(hi[i], hi[j])
            tmp_lo = L[None, :]; tmp_hi = H[None, :]
            ovv += box_volume_sorted(0, tmp_lo, tmp_hi, names, ang, {})
        rep['overlap_sorted_volume(first50 pairs)'] = float(ovv)
    Dv = domain_volume(names, ang, slices, wall, n)
    rep['domain_volume'] = float(Dv); rep['sum_box_sorted_volume'] = float(tot)
    rep['volume_exact_equal'] = (tot == Dv); rep['volume_difference_exact'] = str(Dv - tot) if tot != Dv else '0'
    rep['volume_rel_diff'] = float((Dv - tot) / Dv)
    rep['volume_fraction_by_kind'] = {k: float(v / Dv) for k, v in vol_by_kind.items()}
    rep['methodA_pass'] = bool(tot == Dv and not ov and len(outside) == 0 and len(zerow) == 0 and not bad
                               and (slices is None or rep['n_not_in_a_task_slice'] == 0))
    # Method B
    rng = np.random.default_rng(seed)
    loc = Locator(lo, hi, sd)
    tested = {}; uncovered = []; total = 0
    for rnd in range(max(1, M // 2000)):
        for label, P in gen_points(rng, 2000 // 1, names, ang, slices, sd, blo, bhi, lo, hi):
            okd = in_domain(P, names, ang, slices, sd, blo, bhi); P = P[okd]
            if len(P) == 0: continue
            cov, cnt = loc.covered(P)
            tested[label] = tested.get(label, 0) + len(P); total += len(P)
            for x in np.nonzero(~cov)[0][:5]:
                if len(uncovered) < 20: uncovered.append((label, P[x].tolist()))
    rep['methodB_points_tested'] = total; rep['methodB_by_family'] = tested
    rep['methodB_uncovered'] = uncovered
    rep['seconds'] = round(time.time() - t0, 1)
    return rep


if __name__ == '__main__':
    path = sys.argv[1]; M = int(sys.argv[2]) if len(sys.argv) > 2 else 20000; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rep = main(path, M, seed)
    if len(sys.argv) > 4:
        with open(sys.argv[4], 'w') as fo: fo.write(json.dumps(rep, default=str, indent=1))
    keys = ['file', 'MUTATION', 'boxes', 'kinds', 'methodA_pass', 'volume_exact_equal', 'volume_rel_diff', 'n_outside_bounds',
            'n_interior_overlapping_pairs(first50)', 'methodB_points_tested', 'methodB_uncovered', 'seconds']
    print(json.dumps({k: rep[k] for k in keys if k in rep}, default=str)[:1500])
