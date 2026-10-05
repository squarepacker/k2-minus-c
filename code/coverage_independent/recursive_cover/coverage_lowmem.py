"""Low-memory independent gap check of the K_w* = 9 branch-and-bound leaves (pure Python, no numpy).

Same algorithm as the reviewer's earlier numpy versions, not included here (recursive covering of the closed parameter domain:
a query box is accepted iff it contains no point with ordered angles or is contained in one recorded leaf; otherwise
it is split at a leaf face strictly inside it; a box with ordered points meeting no leaf in positive measure is a GAP),
re-implemented without numpy to minimise memory:
  * leaves are kept in array('d') (8 bytes per number);
  * the domain of each case is cut into S closed slabs along its first angle coordinate, and for every slab the
    JSONL file is READ AGAIN, keeping only the leaves that meet the slab in positive measure ("reading in pieces");
  * every finished slab is appended to the results file (checkpoint); finished slabs are skipped on restart;
  * memory guard: before every slab and every GUARD_EVERY nodes the system's available physical memory is read;
    below FLOOR_MB the program stops (exit code 3) so that its memory is released; the supervisor restarts it later.
usage: python coverage_lowmem.py DATADIR RESULTS.jsonl [--slabs S] [--floor MB] [--only wall,n]
       python coverage_lowmem.py DATADIR --selftest"""
import os, sys, json, glob, math, time, ctypes, random
from array import array
# Public copy. The original run imported a local resource-limit module ("limits": process priority, CPU affinity and a
# memory cap); it does not affect the algorithm and is optional here. The memory guard below works on Windows only;
# on other systems avail_mb() returns infinity and the guard never stops the run.
try:
    import limits  # noqa
except ImportError:
    pass

d = 1e-4
RHO1 = math.sqrt((d + math.sqrt(2) / 2) ** 2 + 0.25)
WALL_R = d + math.sqrt(2) / 2
TWO_PI = 2 * math.pi
sys.setrecursionlimit(100000)
GUARD_EVERY = 2000
FLOOR_MB = 1024.0


if os.name != 'nt':
    def avail_mb():
        return float('inf')

    def proc_mem_mb():
        return dict(peak_ws=-1.0, peak_private=-1.0, private=-1.0)
else:
  from ctypes import wintypes

  class MEMORYSTATUSEX(ctypes.Structure):
      _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                  ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                  ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                  ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                  ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


  class PMC_EX(ctypes.Structure):
      _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                  ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                  ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                  ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                  ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                  ("PrivateUsage", ctypes.c_size_t)]


  def avail_mb():
      m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(m)
      ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
      return m.ullAvailPhys / 2 ** 20


  def proc_mem_mb():
      c = PMC_EX(); c.cb = ctypes.sizeof(c)
      k32 = ctypes.WinDLL("kernel32"); k32.GetCurrentProcess.restype = wintypes.HANDLE
      ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(k32.GetCurrentProcess()), ctypes.byref(c), c.cb)
      return dict(peak_ws=round(c.PeakWorkingSetSize / 2 ** 20, 1), peak_private=round(c.PeakPagefileUsage / 2 ** 20, 1),
                  private=round(c.PrivateUsage / 2 ** 20, 1))


class MemoryFloor(Exception):
    pass


def guard():
    a = avail_mb()
    if a < FLOOR_MB:
        raise MemoryFloor(a)
    return a


def dims(wall, n):
    out = []
    for i in range(n):
        out.append(('r', i))
        if wall or i > 0:
            out.append(('t', i))
    if wall:
        out.append(('h', 0))
    return out


def domain(wall, n):
    dl = dims(wall, n)
    Qlo = [0.5 if k == 'r' else 0.0 for (k, i) in dl]
    Qhi = [RHO1 if k == 'r' else (TWO_PI if k == 't' else WALL_R) for (k, i) in dl]
    tdims = [j for j, (k, i) in enumerate(dl) if k == 't']
    return Qlo, Qhi, tdims


def files_of(datadir, wall, n):
    return sorted(glob.glob(os.path.join(datadir, f"{'bnbw' if wall else 'bnb'}_n{n}_T9_*_final_leaves.jsonl")))


def leaf_vec(o, wall):
    l, h_ = [], []
    for i, (rl, rh, tl, th) in enumerate(o['box']):
        l.append(rl); h_.append(rh)
        if wall or i > 0:
            l.append(tl); h_.append(th)
        elif not (tl == 0.0 and th == 0.0):
            raise ValueError("theta1 not fixed")
    if wall:
        l.append(o['h'][0]); h_.append(o['h'][1])
    return l, h_


def load_slab(files, wall, D, slo, shi):
    """stream the JSONL files; keep only leaves meeting the slab box [slo, shi] in positive measure"""
    LO, HI = array('d'), array('d')
    total = 0; kinds = {}; outside = 0
    for f in files:
        with open(f) as fh:
            for line in fh:
                o = json.loads(line)
                total += 1
                kinds[o['kind']] = kinds.get(o['kind'], 0) + 1
                l, h_ = leaf_vec(o, wall)
                if len(l) != D:
                    raise ValueError("dimension mismatch")
                if all(l[j] < shi[j] and h_[j] > slo[j] for j in range(D)):
                    LO.extend(l); HI.extend(h_)
    return LO, HI, total, kinds


def has_ordered(qlo, qhi, tdims):
    t = 0.0
    for k in tdims:
        if qlo[k] > t:
            t = qlo[k]
        if t > qhi[k]:
            return False
    return True


def check_box(Qlo, Qhi, tdims, LO, HI, D, idx0, stats, gaps):
    def rec(qlo, qhi, idx, depth):
        stats['nodes'] += 1
        if stats['nodes'] % GUARD_EVERY == 0:
            stats['min_avail'] = min(stats['min_avail'], guard())
        if depth > stats['maxdepth']:
            stats['maxdepth'] = depth
        if not has_ordered(qlo, qhi, tdims):
            stats['unordered'] += 1
            return
        sub = []
        contained = False
        for i in idx:
            b = i * D
            meets = True
            cont = True
            for j in range(D):
                lj = LO[b + j]; hj = HI[b + j]
                if not (lj < qhi[j] and hj > qlo[j]):
                    meets = False
                    break
                if not (lj <= qlo[j] and hj >= qhi[j]):
                    cont = False
            if meets:
                if cont:
                    contained = True
                    break
                sub.append(i)
        if contained:
            stats['contained'] += 1
            return
        if not sub:
            stats['gaps'] += 1
            if len(gaps) < 20:
                gaps.append((list(qlo), list(qhi)))
            return
        best = None
        for j in range(D):
            w = qhi[j] - qlo[j]
            if w <= 0:
                continue
            mid = 0.5 * (qlo[j] + qhi[j])
            bc = None; bdist = None
            for arr in (LO, HI):          # same candidate order as the numpy version: all lower faces, then upper
                for i in sub:
                    c = arr[i * D + j]
                    if qlo[j] < c < qhi[j]:
                        dd_ = abs(c - mid)
                        if bdist is None or dd_ < bdist:
                            bdist = dd_; bc = c
            if bc is None:
                continue
            score = bdist / w
            if best is None or score < best[0]:
                best = (score, j, bc)
        if best is None:
            stats['gaps'] += 1
            gaps.append(('nosplit', list(qlo), list(qhi)))
            return
        _, j, c = best
        a_hi = list(qhi); a_hi[j] = c
        b_lo = list(qlo); b_lo[j] = c
        rec(qlo, a_hi, sub, depth + 1)
        rec(b_lo, qhi, sub, depth + 1)

    rec(Qlo, Qhi, idx0, 0)


def slabs_of(Qlo, Qhi, tdims, S):
    if not tdims or S <= 1:
        return [(list(Qlo), list(Qhi))]
    j = tdims[0]
    cuts = [Qlo[j] + (Qhi[j] - Qlo[j]) * s / S for s in range(S + 1)]
    cuts[0], cuts[-1] = Qlo[j], Qhi[j]
    out = []
    for s in range(S):
        a = list(Qlo); b = list(Qhi); a[j] = cuts[s]; b[j] = cuts[s + 1]
        out.append((a, b))
    return out


def selftest(datadir):
    print("SELFTEST(lowmem) start, avail", round(avail_mb()), "MB", flush=True)
    rng = random.Random(7)
    allok = True
    for (wall, n) in ((False, 2), (False, 3), (True, 2), (True, 3)):
        guard()
        Qlo, Qhi, tdims = domain(wall, n)
        D = len(Qlo)
        LO, HI, total, kinds = load_slab(files_of(datadir, wall, n), wall, D, Qlo, Qhi)
        N = len(LO) // D

        def gaps_of(LO_, HI_, N_):
            st = dict(nodes=0, unordered=0, contained=0, maxdepth=0, gaps=0, min_avail=1e9); g = []
            check_box(Qlo, Qhi, tdims, LO_, HI_, D, list(range(N_)), st, g)
            return st['gaps'], st['nodes']
        base, base_nodes = gaps_of(LO, HI, N)

        def interior_ordered(i, jcut=None):
            lo = [LO[i * D + j] for j in range(D)]; hi = [HI[i * D + j] for j in range(D)]
            if jcut is not None:
                lo[jcut] = lo[jcut] + 0.999 * (hi[jcut] - lo[jcut])
            eps = [1e-9 * (hi[j] - lo[j]) for j in range(D)]
            return has_ordered([lo[j] + eps[j] for j in range(D)], [hi[j] - eps[j] for j in range(D)], tdims)
        cand = [i for i in range(N) if interior_ordered(i)]
        picks = rng.sample(cand, min(5, len(cand)))
        rem, shr = [], []
        for i in picks:
            LO2 = array('d', LO[:i * D] + LO[(i + 1) * D:]); HI2 = array('d', HI[:i * D] + HI[(i + 1) * D:])
            rem.append(gaps_of(LO2, HI2, N - 1)[0])
            js = list(range(D)); rng.shuffle(js)
            jj = next((j for j in js if HI[i * D + j] > LO[i * D + j] and interior_ordered(i, j)), None)
            if jj is None:
                shr.append(None); continue
            HI3 = array('d', HI); HI3[i * D + jj] = LO[i * D + jj] + 0.999 * (HI[i * D + jj] - LO[i * D + jj])
            shr.append(gaps_of(LO, HI3, N)[0])
        ok = base == 0 and all(x > 0 for x in rem) and all(x is None or x > 0 for x in shr)
        allok = allok and ok
        print(json.dumps(dict(wall=wall, n=n, leaves=N, base_gaps=base, base_nodes=base_nodes,
                              removed_leaf_gaps=rem, shrunk_leaf_gaps=shr, ok=ok)), flush=True)
    print("SELFTEST", "PASS" if allok else "FAIL", "mem", proc_mem_mb(), flush=True)


def main():
    global FLOOR_MB
    datadir = sys.argv[1]
    if '--floor' in sys.argv:
        FLOOR_MB = float(sys.argv[sys.argv.index('--floor') + 1])
    if '--selftest' in sys.argv:
        try:
            return selftest(datadir)
        except MemoryFloor as e:
            print(f"STOPPED(selftest): available {e.args[0]:.0f} MB < floor {FLOOR_MB} MB", flush=True)
            sys.exit(3)
    results = sys.argv[2]
    S = int(sys.argv[sys.argv.index('--slabs') + 1]) if '--slabs' in sys.argv else 8
    only = None
    if '--only' in sys.argv:
        w_, n_ = sys.argv[sys.argv.index('--only') + 1].split(',')
        only = (w_ == '1', int(n_))
    done = set()
    if os.path.exists(results):
        for line in open(results):
            o = json.loads(line)
            done.add((o['wall'], o['n'], o['unit']))
    t0 = time.time()
    try:
        print(f"start: avail {guard():.0f} MB, floor {FLOOR_MB} MB, slabs {S}", flush=True)
        for wall in (False, True):
            for n in range(1, 6):
                if only and (wall, n) != only:
                    continue
                Qlo, Qhi, tdims = domain(wall, n)
                D = len(Qlo)
                files = files_of(datadir, wall, n)
                units = slabs_of(Qlo, Qhi, tdims, S if n >= 2 else 1)
                for u, (a_, b_) in enumerate(units):
                    if (wall, n, u) in done:
                        continue
                    guard()
                    LO, HI, total, kinds = load_slab(files, wall, D, a_, b_)
                    N = len(LO) // D
                    st = dict(nodes=0, unordered=0, contained=0, maxdepth=0, gaps=0, min_avail=avail_mb())
                    g = []
                    t1 = time.time()
                    check_box(a_, b_, tdims, LO, HI, D, list(range(N)), st, g)
                    rec = dict(wall=wall, n=n, unit=u, units=len(units), leaves_total=total, leaves_in_slab=N,
                               kinds=kinds, files=[os.path.basename(f) for f in files],
                               slab=[a_[tdims[0]], b_[tdims[0]]] if tdims else None, first_gaps=g[:3],
                               sec=round(time.time() - t1, 2), mem=proc_mem_mb(), **st)
                    with open(results, 'a') as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(f"[{time.time()-t0:7.1f}s] wall={wall} n={n} slab {u+1}/{len(units)}: leaves {N}/{total} "
                          f"gaps={st['gaps']} nodes={st['nodes']} contained={st['contained']} unordered={st['unordered']} "
                          f"min_avail={st['min_avail']:.0f}MB peak_private={rec['mem']['peak_private']}MB", flush=True)
                    del LO, HI
        print("FINISHED", flush=True)
    except MemoryFloor as e:
        print(f"STOPPED: available memory {e.args[0]:.0f} MB < floor {FLOOR_MB} MB (finished slabs saved)", flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
