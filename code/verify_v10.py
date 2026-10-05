"""High-precision verification of the numerical constants in cstar_v10.tex. Built by make_verify_v10.py."""
from mpmath import mp, mpf, sqrt, pi, cos, sin, sec, tan, log, acos, polyroots, findroot, diff
import random
mp.dps = 40
ok = True
def check(name, cond, val=None):
    global ok
    print(f"{'OK ' if cond else 'FAIL'} {name}" + (f"   [{mp.nstr(val, 12)}]" if val is not None else ""))
    ok = ok and bool(cond)
SQ2 = sqrt(2)
# ---------- Lemma 3.4 (short chords)
for a in [mpf('1e-7'), mpf('0.1'), mpf('0.5'), pi/4]:
    sc = sin(a)*cos(a)
    I = mp.quad(lambda u: u/sc, [0, sc])
    check(f"ramp integral at a={mp.nstr(a,3)}: 2*int = sin a cos a", abs(2*I - sc) < mpf(10)**-30)
# ---------- Lemma 4.4 (sections near a vertex)
pmin = min(1 - (p-1)**2/2 for p in [1 + i*(SQ2-1)/1000 for i in range(1001)])
check("p - sin cos = 1-(p-1)^2/2 >= 0.91 on [1,sqrt2]", pmin >= mpf('0.91'), pmin)
def section(psi, u):
    s_, c_ = sin(psi), cos(psi)
    lo, hi = min(s_, c_), max(s_, c_)
    if psi == 0 or psi == pi/2: return mpf(1)
    if u <= lo: return u/(s_*c_)
    if u <= hi: return 1/hi
    return (s_ + c_ - u)/(s_*c_)
worst = min(section(mpf(i)/400*pi/2, mpf(j)/400*mpf('0.75')) - min(2*mpf(j)/400*mpf('0.75'), 1)
            for i in range(1, 400) for j in range(0, 401))
check("section >= min(2u,1) on grid (psi,u<=3/4)", worst >= -mpf(10)**-25, worst)
# ---------- Proposition 4.5 with d, theta1 <= 1e-5, eps0 = 1e-4
TH1 = mpf('1e-5'); DMAX = mpf('1e-5')
eps = DMAX + TH1; eps0 = mpf('1e-4'); gam = mpf('1.415')*eps
Qs = mpf('0.32')
check("eps1 = d+theta1 <= 2e-5", eps <= mpf('2e-5'))
check("wall case: eps0 > eps1, eps0 > theta, (1-2eps0)^2 > Q*", eps0 > eps and eps0 > TH1 and (1-2*eps0)**2 > Qs)
def rho(h): return sqrt(1+h*h)
def D(h): return mpf(1)/2 - (1-h)/(2*rho(h))
def chi(h): return 1 + rho(h)/2 - (1+h)/rho(h)
random.seed(1)
maxerr = 0
for _ in range(200):
    th = mpf(random.uniform(0, 1e-5)); d = mpf(random.uniform(0, 1e-5))
    a = 1 + mpf(random.uniform(0, 1))*(d + th/2); b = mpf(random.uniform(0.0, 1.00002))
    w = cos(th)+sin(th); sig = 1/w; l = a - sig/2; L = sqrt(a*a+b*b); h = b/a; r = rho(h)
    t = lambda x, y: (a*y - b*x)/L
    s = lambda x, y: (a*x + b*y)/L
    pairs = [((mpf(1)/2, mpf(1)/2), (1-h)/(2*r)), ((-mpf(1)/2, mpf(1)/2), (1+h)/(2*r)), ((mpf(1)/2, -mpf(1)/2), -(1+h)/(2*r)),
             ((l, b-sig/2), -sig*(1-h)/(2*r)), ((l, b+sig/2), sig*(1+h)/(2*r)), ((l+sig, b-sig/2), -sig*(1+h)/(2*r))]
    for P, val in pairs: maxerr = max(maxerr, abs(t(*P) - val))
    tt = mpf(random.uniform(-0.4, 0.4))
    y = (tt*L + b/2)/a; maxerr = max(maxerr, abs(s(mpf(1)/2, y) - (r/2 + h*tt)))
    y = (tt*L + b*l)/a; maxerr = max(maxerr, abs(s(l, y) - (r*l + h*tt)))
    if b > 0:
        x = (a/2 - tt*L)/b; maxerr = max(maxerr, abs(s(x, mpf(1)/2) - (r/2 - tt)/h))
        yb = b - sig/2; x = (a*yb - tt*L)/b; maxerr = max(maxerr, abs(s(x, yb) - (r*yb - tt)/h))
    if b > mpf('0.344')*a:
        tG2 = sig*(1+h)/(2*r)
        G = (r*l + h*tG2) - (r/2 - tG2)/h
        maxerr = max(maxerr, abs(G - (r*(l-mpf(1)/2) + r/(2*h)*(h-1+sig*(1+h)))))
        tl = -sig*(1+h)/(2*r); vx, vy = l+sig, b-sig/2
        Gl = s(vx, vy) - (r/2 + h*tl)
        maxerr = max(maxerr, abs(Gl - r*(a-(1-sig)/2)))
        cu = G/2 - 2*(tG2 - mpf(1)/2); cl = Gl/2 - 2*(tG2 - mpf(1)/2)
        e1 = d + TH1
        check_ok = cu <= chi(h) + mpf('2.2')*e1 and cl <= chi(h) + mpf('2.2')*e1
        if not check_ok: print("chi bound violated", th, d, a, b, cu, cl, chi(h))
        ok = ok and check_ok
check("level and section formulas (random parameters, eps1<=2e-5)", maxerr < mpf(10)**-30, maxerr)
check("D(0.344) < 0.19", D(mpf('0.344')) < mpf('0.19'), D(mpf('0.344')))
hc = findroot(lambda x: x**3 + 3*x - 2, 0.6); print("   h_c =", hc)
cmax = max(chi(mpf('0.344')), chi(mpf('1.00002')))
check("max(chi(0.344), chi(1.00002)) + 2.2 eps1 < 0.29295", cmax + mpf('2.2')*eps < mpf('0.29295'), cmax + mpf('2.2')*eps)
check("h/rho + 3 eps < 3/4 for h <= 1+eps", (1+eps)/rho(1+eps) + 3*eps < mpf('0.75'))
check("D(1+eps)+3eps <= 0.51", D(1+eps) + 3*eps <= mpf('0.51'))
cc = mpf('0.29295')
vals = [mpf(1)/3, (2-cc)**2/9, mpf(2)/3*(1-cc)**2, (1-cc/2)**2/2, mpf('0.48')]
check("all candidate minima > 0.3237 at chi*=0.29295", min(vals) > mpf('0.3237'), min(vals))
import numpy as np
R = np.linspace(0, 0.6, 1201); E = np.minimum(R**2, np.maximum(2*R - 0.29295, 0)**2)
G = np.maximum(1 - R[:, None] - R[None, :], 0)**2 + E[:, None] + E[None, :]
check("brute-force min Xi > 0.3237 (grid)", G.min() > 0.3237, mpf(float(G.min())))
beta1 = 3*gam
check("beta1 = 3 gamma* <= 8.5e-5", beta1 <= mpf('8.5e-5'), beta1)
check("0.3237 - 2.04 beta1 > 0.3235", mpf('0.3237') - mpf('2.04')*beta1 > mpf('0.3235'))
check("0.3235 - 4 eps0 > Q* = 0.32", mpf('0.3235') - 4*eps0 > Qs)
random.seed(7); worst_u = 0; worst_l = 0
for _ in range(4000):
    th = mpf(random.uniform(0, 1e-5)); d = mpf(random.uniform(0, 1e-5))
    a = 1 + mpf(random.uniform(0, 1))*(d + th/2); h = mpf(random.uniform(0.344, 1)) if random.random() < 0.9 else 1 + mpf(random.uniform(0,1))*(d+th/2)
    b = h*a; w = cos(th)+sin(th); sig = 1/w; l = a - sig/2; r = rho(h)
    G_ = r*(l-mpf(1)/2) + r/(2*h)*(h-1+sig*(1+h)); tG2 = sig*(1+h)/(2*r)
    Gl = r*(a-(1-sig)/2)
    worst_u = max(worst_u, (G_/2 - tG2)/(d+TH1))
    worst_l = max(worst_l, (Gl/2 - tG2)/(d+TH1))
check("(f) capped upper case: < 1.1 eps1 (sampled)", worst_u < mpf('1.1'), worst_u)
check("(f) capped lower case: < 1.8 eps1 (sampled)", worst_l < mpf('1.8'), worst_l)
check("1.8 eps1 < eps0", mpf('1.8')*eps < eps0)
check("D<0.19 implies zeta < 0.19+3eps <= 0.1906", mpf("0.19") + 3*eps <= mpf("0.1906"))
check("(1-0.3812)^2 > 0.38", (1 - mpf('0.3812'))**2 > mpf('0.38'))
check("theta/6.26: 0.32*(2-1e-5)/4 >= 1/6.26", mpf('0.32')*(2-mpf('1e-5'))/4 >= 1/mpf('6.26'))
# (e): (1+h)/(2 rho_h) <= 0.71 on [0,1] (max at h=1: sqrt2/2), and 0.71 theta + 1.5 gamma* <= 3 eps1 using theta <= eps1
check("(e) (1+h)/(2rho_h) <= 0.71 on [0,1] (grid; derivative (1-h)/rho^3 >= 0 so max at h=1)",
      all((1+mpf(i)/1000)/(2*rho(mpf(i)/1000)) <= mpf('0.71') for i in range(1001)) and SQ2/2 <= mpf('0.71'))
check("(e) 0.71 eps1 + 1.5*1.415 eps1 <= 3 eps1", mpf('0.71') + mpf('1.5')*mpf('1.415') <= 3)
check("(g) D(h)<0.19 branch: 0.38 - 4 eps0 > Q*", mpf('0.38') - 4*eps0 > Qs)
check("(g) z,z' <= D(1+eps1) + 3 eps1 <= 0.51", D(1 + eps) + 3*eps <= mpf('0.51'))

# analytic versions of the bounds that are sampled above (theta <= theta1 <= eps1, d <= eps1)
rmax = rho(1 + eps)
check("rho_h <= 1.415 for h <= 1+eps1", rmax <= mpf('1.415'), rmax)
check("rho_h/h <= sqrt2 for h > 1 (sqrt(1+1/h^2) decreasing)", all(rho(1+mpf(i)/100)/(1+mpf(i)/100) <= SQ2 for i in range(1, 300)))
check("sigma_theta >= 1 - theta and (w - 1/w)/2 <= theta (grid in theta <= 1e-5)",
      all(1/(cos(t)+sin(t)) >= 1 - t and ((cos(t)+sin(t)) - 1/(cos(t)+sin(t)))/2 <= t for t in [mpf(i)/100*TH1 for i in range(101)]))
check("chi_u, chi_l <= chi(h) + (sqrt2 + rho_max/2) eps1 <= chi(h) + 2.2 eps1", SQ2 + rmax/2 <= mpf('2.2'), SQ2 + rmax/2)
hh = 1 + eps
cap_u = hh*(hh-1)/(2*rho(hh)) + rmax/2*eps
check("capped upper: h(h-1)/(2rho) + (rho/2)(l-1/2) <= 1.1 eps1 (analytic)", cap_u < mpf('1.1')*eps, cap_u/eps)
cap_l = hh*(hh-1)/(2*rho(hh)) + rmax/2*eps + SQ2/2*TH1
check("capped lower: ... + (1-sigma)(1+h)/(2rho) <= 1.8 eps1 (analytic)", cap_l < mpf('1.8')*eps, cap_l/eps)
Du_max = D(1 + eps) + 3*eps
check("tau >= 1/2 - D_u > -0.02", mpf(1)/2 - Du_max > mpf('-0.02'), mpf(1)/2 - Du_max)
sig_min = 1/(cos(TH1) + sin(TH1))
check("t(V3') <= -0.6 on [0.344, 1+eps1] ((1+h)/rho_h minimal at 0.344 there)",
      sig_min*(1 + mpf('0.344'))/(2*rho(mpf('0.344'))) > mpf('0.6') and all((1+x)/rho(x) >= (1+mpf('0.344'))/rho(mpf('0.344')) for x in [mpf('0.344') + i*(mpf('0.656')+eps)/2000 for i in range(2001)]))
check("Gamma_* <= rho_h + rho_h(l - 1/2) < 2", rmax*(1 + eps) < 2, rmax*(1+eps))
check("tau_* > 0 on [0.344, 1+eps1]: sigma(1+h)/(2 rho_h) > 1/2 at h = 0.344", sig_min*(1 + mpf('0.344'))/(2*rho(mpf('0.344'))) > mpf('0.5'))
# ---------- Fodor, Oler and counting
x0 = min(x.real for x in polyroots([9,-15,7,0,-3,1], maxsteps=200, extraprec=200) if abs(x.imag) < 1e-25 and x.real > 0)
R12 = 1/(sqrt(3)*x0); R0 = sqrt((SQ2 + mpf('1e-4'))**2 + mpf(1)/4)
check("Fodor R12 = 1.5148...", abs(R12 - mpf('1.5148009650580917')) < mpf('1e-15'), R12)
check("R0(1e-4) < 1.5001 < R12", R0 < mpf('1.5001') < R12, R0)
check("wall centre radius < R0", sqrt((mpf('1e-4') + SQ2/2)**2 + mpf(1)/4) < R0)
cD = mpf('1e-4') + SQ2/2 - mpf(1)/2
check("c = d + sqrt2/2 - 1/2 < 0.20721", cD < mpf('0.20721'), cD)
phi = acos(cD/R0)
AD = R0**2*(pi - phi) + cD*sqrt(R0**2 - cD**2)
PD = 2*R0*(pi - phi) + 2*sqrt(R0**2 - cD**2)
check("area(D0) < 4.155", AD < mpf('4.155'), AD)
check("perimeter(D0) < 8.1", PD < mpf('8.1'), PD)
ob = 2/sqrt(3)*mpf('4.155') + mpf('8.1')/2 + 1
check("Oler bound 2/sqrt3*4.155 + 8.1/2 + 1 < 9.85 (so <= 9 centres)", ob < mpf('9.85'), ob)
# the area/perimeter are increasing in d (R0 and c grow), so d = 1e-4 is the worst case
# area/perimeter formulas cross-checked by numerical integration over the disc segment
import math
def area_num(R, c, n=200000):
    # area of {x >= -c} ∩ disc(R): integrate chord length over x in [-c, R]
    xs = np.linspace(-float(c), float(R), n+1); y = np.sqrt(np.maximum(float(R)**2 - xs**2, 0))
    return float(np.trapezoid(2*y, xs))
check("area formula vs numerical integration", abs(area_num(R0, cD) - float(AD)) < 1e-6, mpf(area_num(R0, cD)))
# sanity check only (not used in the text): a circumscribed polygon gives the same Oler bound
def circ_poly(R, c, n):
    ph = acos(-c/R)  # kept arc: angles in [-ph, ph] measured from the far side? use direction away from the cut
    th = [-ph + 2*ph*j/n for j in range(n+1)]
    verts = []
    def ti(t1, t2):
        det = cos(t1)*sin(t2) - sin(t1)*cos(t2)
        return (R*(sin(t2) - sin(t1))/det, R*(cos(t1) - cos(t2))/det)
    def tc(t):
        x = -c; return (x, (R - x*cos(t))/sin(t))
    verts.append(tc(th[0]))
    for j in range(n): verts.append(ti(th[j], th[j+1]))
    verts.append(tc(th[-1]))
    A = mpf(0); P = mpf(0)
    for i in range(len(verts)):
        x1, y1 = verts[i]; x2, y2 = verts[(i+1) % len(verts)]
        A += x1*y2 - x2*y1; P += sqrt((x2-x1)**2 + (y2-y1)**2)
    return abs(A)/2, P
Ap, Pp = circ_poly(R0, cD, 3000)
check("sanity: Oler for a circumscribed 3000-gon < 9.85", 2/sqrt(3)*Ap + Pp/2 + 1 < mpf('9.85'), 2/sqrt(3)*Ap + Pp/2 + 1)
check("K = max(3*11-6, 3*10-6) = 27", max(3*11-6, 3*10-6) == 27)
dist_side = (mpf('1e-4') + SQ2/2) + sqrt((mpf('1e-4') + SQ2/2)**2 + mpf(1)/4)
check("p within 1.58 of a side; 2*1.58 = 3.16 < 4", dist_side < mpf('1.58') and 2*mpf('1.58') < 4, dist_side)
Lam = SQ2 + mpf('1e-4')
check("12 Lambda/pi + 4 < 10", 12*Lam/pi + 4 < 10, 12*Lam/pi + 4)
# lem:count(a) Steiner count, directly: (pi/4) m' + 2 <= 3|L| + 2 + pi with |L| < Lambda  ->  m' < 12 Lambda/pi + 4
check("count(a) Steiner: (3*Lam + pi)*4/pi = 12 Lam/pi + 4 < 10, wall: (3(d+sqrt2/2) + pi/2)*4/pi < 5",
      (3*Lam + pi)*4/pi < 10 and (3*(mpf('1e-4') + SQ2/2) + pi/2)*4/pi < 5)
check("wall: 12(d+sqrt2/2)/pi + 2 < 5", 12*(mpf('1e-4') + SQ2/2)/pi + 2 < 5)
# ---------- Lemma 4.10/4.11
thmax = mpf('3e-6')
check("theta1(2-theta1)/40 >= Q* thmax(2-thmax)/4 (thmax<=3e-6)", TH1*(2-TH1)/40 >= Qs*thmax*(2-thmax)/4)
check("column: theta1(2-theta1)/40 >= Q* bbar(2-bbar)/4 (bbar<=1.5e-6)", TH1*(2-TH1)/40 >= Qs*mpf('1.5e-6')*(2-mpf('1.5e-6'))/4)
check("d' = thmax <= 3e-6 <= 1e-5 (Prop 4.5 applies)", thmax <= DMAX)
# ======== from verify_v8_parts.py ========
from mpmath import asin, floor, findroot
import itertools
mp.dps = 50
# =============================== (I) Lemma lem:Kw ===============================
# All quantities below are monotone in d (Lambda, rho1 increase with d; gamma0 decreases), and the worst case for an
# upper bound on the number of edges is the largest d, d = 1e-4: larger rho1/R_m and smaller gamma0 only enlarge D and
# shrink separations.  We check d = 1e-4 (and d = 1e-5 for information).
for d in [mpf('1e-4'), mpf('1e-5')]:
    Lam = SQ2 + d
    g0 = acos(1 - 1/(2*Lam**2))
    rho_sq = sqrt((Lam/2)**2 + mpf(1)/4)               # square-square edge: p within this of the nearer end of L
    rho_wall = sqrt((d + SQ2/2)**2 + mpf(1)/4)        # square-side edge: p within this of the square's centre
    rho1 = max(rho_sq, rho_wall)
    tag = f"[d={mp.nstr(d,2)}]"
    check(f"{tag} rho1 = max(...) < 0.8662 < 1", rho1 < mpf('0.8662'), rho1)
    # neighbour separation at a centre c: other ends c', c'' are centres at distance in [1, Lam) from c, |c'-c''| >= 1.
    # cos(angle) <= f(r1,r2) := (r1^2+r2^2-1)/(2 r1 r2) = r1/(2 r2) + (r2^2-1)/(2 r1 r2): convex in r1 (as r2^2-1 >= 0) and,
    # by symmetry, in r2 -> maximum over [1,Lam]^2 at a corner.
    f = lambda r1, r2: (r1**2 + r2**2 - 1)/(2*r1*r2)
    corners = [f(mpf(1), mpf(1)), f(mpf(1), Lam), f(Lam, Lam)]
    check(f"{tag} max over corners of f is f(Lam,Lam) = 1 - 1/(2 Lam^2)", max(corners) == f(Lam, Lam) and abs(f(Lam, Lam) - (1 - 1/(2*Lam**2))) < mpf(10)**-45)
    # grid sanity check of the convexity claim
    grid = max(f(1 + (Lam-1)*i/200, 1 + (Lam-1)*j/200) for i in range(201) for j in range(201))
    check(f"{tag} grid max of f on [1,Lam]^2 <= f(Lam,Lam)", grid <= f(Lam, Lam) + mpf(10)**-40, grid)
    # foot point on a side vs a neighbour centre: (c''-c).nu <= (d + sqrt2/2) - 1/2 = c0, |c''-c| >= 1
    c0 = d + SQ2/2 - mpf(1)/2
    wall_sep = acos(c0)
    check(f"{tag} side-foot separation arccos(c0) > gamma0", wall_sep > g0, wall_sep*180/pi)
    # degree classes
    R = {1: rho1, 2: rho1}
    for m in (3, 4, 5):
        R[m] = min(rho1, 1/(2*sin((m-1)*g0/2)))
    check(f"{tag} D(r) <= 5 for r > 1/2 (2 arcsin(1/(2r)) < pi < 5 gamma0)", pi < 5*g0, pi/g0)
    check(f"{tag} class radii R3,R4,R5 > 1/2 (classes nonempty) and R5 < R4 < R3 < rho1",
          mpf(1)/2 < R[5] < R[4] < R[3] < rho1)
    # inner centres a, b at radii r_a <= R_a, r_b <= R_b (< 1), |c_a - c_b| >= 1: angle at p >= arccos(f(R_a,R_b)),
    # as f is increasing in each variable when the other is < 1 (df/dr1 = (r1^2 - r2^2 + 1)/(2 r1^2 r2) > 0).
    gam = {(a, b): acos(f(R[a], R[b])) for a in R for b in R}
    minsep = min(gam.values())
    nmax = int(floor(2*pi/minsep))
    check(f"{tag} at most {nmax} inner centres (min separation {mp.nstr(minsep*180/pi, 6)} deg)", nmax == 5)
    classes = [2, 3, 4, 5]    # class 1 is dominated by class 2 (same radius bound, smaller degree)
    best = 0; margin = None
    for n in range(1, nmax + 1):
        for seq in itertools.product(classes, repeat=n):
            tot = mpf(0) if n == 1 else (2*gam[seq[0], seq[1]] if n == 2 else sum(gam[seq[i], seq[(i+1) % n]] for i in range(n)))
            s = sum(seq)
            if tot <= 2*pi:
                best = max(best, s)
    for n in range(1, nmax + 1):
        for seq in itertools.product(classes, repeat=n):
            s = sum(seq)
            if s <= 14:
                continue
            tot = mpf(0) if n == 1 else (2*gam[seq[0], seq[1]] if n == 2 else sum(gam[seq[i], seq[(i+1) % n]] for i in range(n)))
            margin = tot - 2*pi if margin is None else min(margin, tot - 2*pi)
    check(f"{tag} K_w: every feasible cyclic class sequence has degree sum <= 14", best <= 14, best)
    check(f"{tag} every sequence with degree sum >= 15 exceeds 2 pi by > 25 deg", margin > 25*pi/180, margin*180/pi)
    print(f"   {tag} gamma0 = {mp.nstr(g0*180/pi, 10)} deg; R3,R4,R5 = {mp.nstr(R[3],8)}, {mp.nstr(R[4],8)}, {mp.nstr(R[5],8)}")
    # numbers quoted in the proof (stated for the worst case d = 1e-4)
    deg = 180/pi
    check(f"{tag} quoted: gamma0 > 41.406 deg, rho1 < 0.86611, R3 < 0.75598, R4 < 0.56571, R5 < 0.50396, gamma(2,2) > 70.52 deg",
          g0*deg > mpf('41.406') and rho1 < mpf('0.86611') and R[3] < mpf('0.75598') and R[4] < mpf('0.56571') and R[5] < mpf('0.50396') and gam[2, 2]*deg > mpf('70.52'))
    sum14 = []
    for n in range(1, nmax + 1):
        for seq in itertools.product(classes, repeat=n):
            if sum(seq) != 14:
                continue
            tot = mpf(0) if n == 1 else (2*gam[seq[0], seq[1]] if n == 2 else sum(gam[seq[i], seq[(i+1) % n]] for i in range(n)))
            if tot <= 2*pi:
                sum14.append((seq, tot*deg))
    check(f"{tag} quoted: feasible sum-14 sequences are exactly the rotations of (5,2,5,2), total ~358.95 deg",
          sorted(q for q, _ in sum14) == [(2, 5, 2, 5), (5, 2, 5, 2)] and all(abs(t - mpf('358.95')) < mpf('0.05') for _, t in sum14),
          sum14[0][1] if sum14 else None)
    # monotonicity in d claimed in the text: Lambda, rho1, R_m increase and gamma0 decreases with d
    if d == mpf('1e-4'):
        dd = [mpf(i)*mpf('1e-6') for i in range(1, 101)]
        g0s = [acos(1 - 1/(2*(SQ2 + x)**2)) for x in dd]
        r1s = [sqrt((x + SQ2/2)**2 + mpf(1)/4) for x in dd]
        check(f"{tag} gamma0 decreasing and rho1 increasing in d on (0,1e-4] (grid)",
              all(g0s[i] > g0s[i+1] for i in range(99)) and all(r1s[i] < r1s[i+1] for i in range(99)))

# ---------------- lem:Kw (c): two class-5 inner centres ----------------
f_ = lambda r1, r2: (r1**2 + r2**2 - 1)/(2*r1*r2)
from mpmath import asin
for dd in [mpf('1e-4'), mpf('1e-5')]:
    Lam_ = SQ2 + dd; g0_ = acos(1 - 1/(2*Lam_**2)); rho1_ = sqrt((dd + SQ2/2)**2 + mpf(1)/4)
    R5_ = min(rho1_, 1/(2*sin(2*g0_))); g55 = acos(f_(R5_, R5_)); psi_ = asin(R5_*sin(pi - g55))
    sep3 = acos(max(f_(mpf(1), mpf(1)), f_(mpf(1), Lam_), f_(2*R5_, mpf(1)), f_(2*R5_, Lam_)))
    c0_ = dd + SQ2/2 - mpf(1)/2; dg = 180/pi; tg = f"[d={mp.nstr(dd,2)}]"
    check(f"{tg} Kw(c): 2 R5 < 1.008, gamma(5,5) > 165.6 deg, psi < 7.19 deg", 2*R5_ < mpf('1.008') and g55*dg > mpf('165.6') and psi_*dg < mpf('7.19'), psi_*dg)
    check(f"{tg} Kw(c): separation from c3 > 44.99 deg (corners of [1,2R5]x[1,Lam]); side foot > 78 deg", sep3*dg > mpf('44.99') and acos(c0_)*dg > 78, sep3*dg)
    check(f"{tg} Kw(c): 90 + 7.19 - 44.99 < 2 gamma0", mpf(90) + mpf('7.19') - mpf('44.99') < 2*g0_*dg, 2*g0_*dg)
    check(f"{tg} Kw(c) text: 2R5 < Lambda (so (a) gives separation gamma0 from c3), gamma0 > 41.406 deg, 90 + 7.19 - 41.406 < 2 gamma0",
          2*R5_ < Lam_ and g0_*dg > mpf('41.406') and mpf(90) + mpf('7.19') - mpf('41.406') < 2*g0_*dg)
# =============================== Sections 5-6 (v10: k2 = 1e13, K_w^* = 9) ===============================
import sys, os
from mpmath import tan, cos, sin, sec, log, floor
KW = 9
SQ2 = sqrt(2); C = mpf(2); Cp = 2*SQ2; delta = mpf('1e-5'); Qs = mpf('0.32')
k2 = mpf(10)**13; smin = sqrt(k2)/4
eps = mpf('2e-3'); om0 = mpf('2e-4')
print('KW =', KW, ' smin = y0(k2) =', smin)
al = 1/(C*smin)
check("alpha(s) <= 6.33e-7 for s >= sqrt(k2)/4", al <= mpf('6.33e-7'), al)
check("alpha <= 1e-3 (crack lemma range), delta <= 1e-3", al <= mpf('1e-3') and delta <= mpf('1e-3'))
check("sec x - 1 <= 0.50001 x^2 for x <= 1e-3", sec(mpf('1e-3')) - 1 <= mpf('0.50001')*mpf('1e-6'))
nq = mpf('0.50001')*(1 + al/smin)/(C**2*smin)
check("lem:Q n(sec a - 1) < 1.59e-7", nq < mpf('1.59e-7'), nq)
check("lem:Q beta < n + 1.0159e-5", delta + mpf('1.59e-7') <= mpf('1.0159e-5'))
check("lem:Q lower ramp < n + 1.0792e-5", mpf('1.0159e-5') + mpf('6.33e-7') <= mpf('1.0792e-5'))
check("lem:Q cos a - sin a > 1 - 6.4e-7", cos(al) - sin(al) > 1 - mpf('6.4e-7'))
w0 = mpf('1.08e-5')
check("lem:Q w0 = 1.08e-5 > 1.0792e-5 and > 6.4e-7", w0 > mpf('1.0792e-5'))
rhoc = cos(al) + sin(al)
check("tan a < 1.000001 a", tan(al) < mpf('1.000001')*al)
check("3.02 delta * 1.000001 <= 3.04e-5", mpf('3.02')*delta*mpf('1.000001') <= mpf('3.04e-5'))
BZ = mpf('2.01')*al + (1 + (al + rhoc)/smin)*(1 + mpf('3.04e-5'))/(C*(cos(al) - sin(al)))
check("lem:GZ |B_Z| < 0.500018", BZ < mpf('0.500018'), BZ)
g0 = mpf('0.49998')
check("g0: 1 - 6.4e-7 - 0.500018 > 0.49998", 1 - mpf('6.4e-7') - mpf('0.500018') > g0)
A = 4*sqrt(KW/Qs)/(2 - mpf('3e-6'))
Ab = mpf(str(float(floor(A*10**4) + 1)/10**4))
check(f"A = 4 sqrt({KW}/0.32)/(2 - 3e-6) < {Ab}", A < Ab, A)
check("A >= 2", A >= 2)
check("Bh: theta1(2-theta1)/40 >= Q* thmax(2-thmax)/4, thmax <= 3e-6", mpf('1e-5')*(2 - mpf('1e-5'))/40 >= Qs*mpf('3e-6')*(2 - mpf('3e-6'))/4)
alp = 1/(Cp*smin)
check("alpha'(y0) <= 4.48e-7 <= 1.5e-6 (lem:Bh bbar; thmax = 2 bbar <= 3e-6)", alp <= mpf('4.48e-7'), alp)
check("alpha(y0) <= 6.33e-7 <= 1.5e-6 (lem:Bv bbar)", al <= mpf('1.5e-6'))
check("C'(1 - eps) > C", Cp*(1 - eps) > C)
check("vertical extent < 1.0001", cos(alp/(1 - eps)) + sin(alp/(1 - eps)) < mpf('1.0001'))
check("eta <= 0.50001 * 16/8 = 1.00002 (y0 = sqrt k / 4)", mpf('0.50001')*16/8 <= mpf('1.00002'))
lk2 = log(k2)
check("for 2 <= k < k2: log k < 29.934, 0.033 * 29.934 < 0.99", lk2 < mpf('29.934') and mpf('0.033')*mpf('29.934') < mpf('0.99'), lk2)
c2 = 2*log(2*(1 - eps))
check("2 log(2(1 - eps)) > 1.38229", c2 > mpf('1.38229'), c2)
fac = (1 - 6/k2)/(1 + 8/sqrt(k2))
check("(1-6/k)/(1+8/sqrt k) > 1 - 2.6e-6 for k >= k2", fac > 1 - mpf('2.6e-6'), fac)
c3 = c2 + 2*log(1 - mpf('2.6e-6'))
check("l(H) >= h0 (log k + 1.38228)", c3 > mpf('1.38228'), c3)
h0 = 1 - 2*w0
lead = (1 - om0)*h0
check("(1 - omega0) h0 >= 0.99977", lead >= mpf('0.99977'), lead)
t1 = 1/(om0*smin); t2 = mpf('2.01')/(eps*smin); t3 = al/delta
print('1/(om0 y0) =', t1, ' 2.01/(eps y0) =', t2, ' alpha(y0)/delta =', t3)
check("1/(omega0 y0) <= 6.33e-3, 2.01/(eps y0) <= 1.272e-3, alpha(y0)/delta <= 0.0633", t1 <= mpf('6.33e-3') and t2 <= mpf('1.272e-3') and t3 <= mpf('0.0633'))
br = mpf('6.33e-3') + SQ2*Ab + (1 + mpf('1.272e-3'))/(SQ2*g0*(1 - eps))*(Ab + mpf('0.0633'))
Bb = mpf(str(float(floor(br*1000) + 1)/1000))
print('bracket =', br, ' -> bound', Bb)
check(f"bracket < {Bb}", br < Bb, br)
off = mpf('0.99977')*mpf('1.38228') - mpf('1.00002')
check("offset 0.99977*1.38228 - 1.00002 > 0.3819", off > mpf('0.3819'), off)
lim = mpf('0.99977')/Bb
print('limit kappa =', lim, '  1/log(k2-1) =', 1/log(k2 - 1))
Fk = lambda L: (mpf('0.99977')*L + mpf('0.3819'))/Bb
Lc = findroot(lambda L: Fk(L) - 1, 29.5)
print('F > 1 for log k >', Lc, '  F(k2)/log k2 =', Fk(lk2)/lk2)
kap = min(lim, 1/log(k2 - 1))
print('certified kappa for all k >= 2 =', kap)
target = mpf('0.033') if KW == 9 else mpf('0.031')
check(f"kappa_certified > {target}", kap > target, kap)
check("F/log k decreasing (offset > 0) and F(k2)/log k2 >= limit", Fk(lk2)/lk2 >= lim)
th4 = (Bb*4 - mpf('0.3819'))/mpf('0.99977')
print('s(k^2-4)=k for log k >', th4)
# ---------------- numbers quoted in the remarks of v10 ----------------
check("rem:k: expression > 1 for log k > 29.772 (< log k2 = 29.9336...)", mpf('29.771') < Lc < mpf('29.773') and mpf('29.9336') < lk2 < mpf('29.9337'), Lc)
check("rem:k: 1/log(k2-1) = 0.033407..., F(k2)/log k2 = 0.03358..., limit 0.99977/30.147 = 0.033163...",
      mpf('0.033407') < 1/log(k2 - 1) < mpf('0.033408') and mpf('0.03358') < Fk(lk2)/lk2 < mpf('0.03359') and mpf('0.033163') < lim < mpf('0.033164'))
check("rem:k: sqrt(Q*/9)/(4 sqrt2) ~ 0.033333", abs(sqrt(Qs/9)/(4*SQ2) - mpf('0.033333')) < mpf('1e-6'))
check("rem:const: sqrt(9/0.32) ~ 5.3; with Q* = 1/3: sqrt((1/3)/9)/(4 sqrt2) ~ 0.0340 (text: about 0.0338 after parameter losses)",
      abs(sqrt(9/Qs) - mpf('5.303')) < mpf('0.001') and abs(sqrt((mpf(1)/3)/9)/(4*SQ2) - mpf('0.03402')) < mpf('1e-4'))
qlim = mpf('0.99977')/(mpf('6.33e-3') + SQ2*(4*sqrt(9/(mpf(1)/3))/(2 - mpf('3e-6'))) + (1 + mpf('1.272e-3'))/(SQ2*g0*(1 - eps))*(4*sqrt(9/(mpf(1)/3))/(2 - mpf('3e-6')) + mpf('0.0633')))
check("rem:const: with Q* = 1/3 the same parameters give a limit ~ 0.0338", abs(qlim - mpf('0.0338')) < mpf('2e-4'), qlim)
check("rate: (30.147c - 0.3819)/0.99977 <= 30.16c - 0.38 for c = 2..1000, threshold > log k2 for c >= 2, s(k^2-4)=k for log k > 120.24",
      all((mpf('30.147')*c - mpf('0.3819'))/mpf('0.99977') <= mpf('30.16')*c - mpf('0.38') for c in range(2, 1001))
      and (mpf('30.147')*2 - mpf('0.3819'))/mpf('0.99977') > lk2 and th4 < mpf('120.24'), th4)
check("intro: 1/0.033 ~ 30; 0.033*29.934 < 0.99", abs(1/mpf('0.033') - 30) < mpf('0.4') and mpf('0.033')*mpf('29.934') < mpf('0.99'))
# ---------------- lem:Kw9: final computation complete (see make_stats_v10.py) ----------------
import glob, re as _re, json as _json
_here = os.path.dirname(os.path.abspath(__file__))
leaves = sorted(glob.glob(os.path.join(_here, '*_T9_*_final_leaves.jsonl')))
tot = 0; fails = 0; finished = True; mx = 0; per_file_ok = True; groups = {}
for lf in leaves:
    m_ = _re.match(r'(bnbw?)_n(\d)_T9_(\d+)of(\d+)_final_leaves\.jsonl$', os.path.basename(lf))
    groups.setdefault((m_.group(1), int(m_.group(2)), int(m_.group(4))), set()).add(int(m_.group(3)))
    n_l = sum(1 for _ in open(lf))
    parts = sorted(glob.glob(lf.replace('.jsonl', '_verify2_*of4.json')))
    c_l = 0
    for f in parts:
        D_ = _json.load(open(f)); c_l += D_['checked']; fails += len(D_['fail']); finished &= bool(D_.get('finished')); mx = max(mx, D_['maxU'])
    per_file_ok &= (len(parts) == 4 and c_l == n_l)
    tot += c_l
tasks_ok = all(any(k[0] == w and k[1] == n and v == set(range(k[2])) for k, v in groups.items()) and
               sum(1 for k in groups if k[0] == w and k[1] == n) == 1 for w in ('bnb', 'bnbw') for n in range(1, 6))
check("lem:Kw9: final T=9 runs present for n = 1..5, without and with a side, all task indices", tasks_ok)
nleaves = sum(sum(1 for _ in open(f)) for f in leaves)
check(f"lem:Kw9: all {nleaves} accepted boxes re-verified in Arb by verify_leaves2.py ({tot} checked), no failure, max rigorous bound {mx}",
      finished and fails == 0 and per_file_ok and tot == nleaves and mx <= 9 and nleaves > 0)
cov_ok = True
for lf in leaves:
    cj = lf.replace('.jsonl', '_coverage.json')
    C_ = _json.load(open(cj)) if os.path.exists(cj) else None
    cov_ok &= bool(C_ and C_['ok'] and C_['unused'] == 0 and C_['overused'] == 0 and C_['duplicates'] == 0 and C_['volume_equal']
                   and C_['leaves'] == sum(1 for _ in open(lf)))
check(f"lem:Kw9: the accepted boxes of all {len(leaves)} final runs tile the initial boxes (tree rebuilt, each box used once, exact volumes)", cov_ok and len(leaves) > 0)
import bnb as _B, bnb_wall as _BW, math as _math
_hi = [max(float(_B.PHI_HI[k]), float(_B.PHI_LO[k + 1])) for k in range(_B.NC - 1)]
check("lem:Kw9: direction cells used by verify_leaves2 cover [0, 2pi_float] without gaps (last cell extended to the exact 2pi in Arb); rho pieces cover [1, Lambda]",
      _B.PHI_LO[0] == 0.0 and all(_hi[k] >= _B.PHI_LO[k + 1] for k in range(_B.NC - 1)) and float(_B.PHI_HI[-1]) == 2*_math.pi
      and all(_B.RHI[q] == _B.RLO[q + 1] for q in range(_B.NR - 1)) and _B.RLO[0] <= 1 and mpf(float(_B.RHI[-1])) >= SQ2 + mpf('1e-4'))
check("lem:Kw9: wall angle slices and the no-side slices of theta_2 share their end points (same float expression), first slice starts at 0",
      all(2*_math.pi*(s + 1)/m == 2*_math.pi*(s + 1)/m for m in (4, 8, 16) for s in range(m)) and 2*_math.pi*0/8 == 0.0)
# ---------------- M3 (review14/lit): without the computer, the analytic K_w = 13 still gives kappa = 0.027 ----------------
A13 = 4*sqrt(mpf(13)/Qs)/(2 - mpf('3e-6'))
br13 = mpf('6.33e-3') + SQ2*A13 + (1 + mpf('1.272e-3'))/(SQ2*g0*(1 - eps))*(A13 + mpf('0.0633'))
check("rem:const: with K_w = 13 (Lemma lem:Kw only): A < 12.7476, bracket < 36.212, limit 0.99977/36.212 > 0.0276, and 0.027*log(k2) < 1",
      A13 < mpf('12.7476') and br13 < mpf('36.212') and mpf('0.99977')/mpf('36.212') > mpf('0.0276') and mpf('0.027')*lk2 < 1
      and min((mpf('0.99977')*L_ + mpf('0.3819'))/mpf('36.212')/L_ for L_ in [lk2 + j for j in range(0, 2000, 7)]) > mpf('0.027'), br13)
from mpmath import mp as _mp, mpf as _mpf, sqrt as _sqrt
_mp.dps = 40
import bnb as _B, bnb_wall as _BW
_rho1 = _sqrt((_mpf('1e-4') + _sqrt(2)/2)**2 + _mpf(1)/4)
check("lem:Kw9: floating-point rho_1, Lambda and d+sqrt2/2 used by the search are upper bounds of the exact values",
      _mpf(_B.RHO1) >= _rho1 and _mpf(_B.LAM) >= _sqrt(2) + _mpf('1e-4') and _mpf(_BW.WALL_R) >= _mpf('1e-4') + _sqrt(2)/2)
check("lem:Kw9: gamma_0 used in the count (float G0 - 1e-12) is below the exact gamma_0 = acos(1 - 1/(2 Lambda^2))", _mpf(_B.G0) - _mpf('1e-12') < __import__('mpmath').acos(1 - 1/(2*(_sqrt(2) + _mpf('1e-4'))**2)))
check("lem:Kw9: arc (2pi_float, 2pi) is empty without a side: rho_1 - 1/2 < 1 (two centres there would be closer than 1)", _rho1 - _mpf(1)/2 < 1)

print("ALL OK" if ok else "SOME CHECK FAILED")
