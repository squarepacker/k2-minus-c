"""v9: checks for the refinement K_w <= 13 (mpmath 50 digits) and the resulting constants.
Refinement (part (c) of lem:Kw): if two inner centres c1, c3 both have class 5 (radius <= R5), then the number of relevant
edges is at most sum_i m_i - 1.  Proof ingredients checked here:
 (i)   |c1 - c3| in [1, 2 R5];
 (ii)  the angle at c1 between p - c1 and c3 - c1 is at most psi_max = arcsin(R5 sin(pi - gamma(5,5)))  (law of sines);
 (iii) a neighbour centre c' of c1 (|c' - c1| in [1, Lambda), |c' - c3| >= 1) is seen from c1 at angle >= sep3 from c3, where
       cos(sep3) = max of f over [1, 2R5] x [1, Lambda] (convex in each variable -> corners);
       a side foot is >= arccos(c0) > sep3 from c3;
 (iv)  if c3 is not an other end of a relevant edge at c1, the other ends lie in [-beta, beta] minus (psi - sep3, psi + sep3),
       beta < 90 deg, so in two arcs of length < 90 + psi_max - sep3 < 2 gamma0  ->  at most 2 + 2 = 4 of them: deg(c1) <= 4;
 (v)   if c3 is such an other end, the edge {c1, c3} is counted in deg(c1) and in deg(c3).
Either way the count is <= sum m_i - 1, and (5,2,5,2) is the only class sequence with sum 14, so K_w <= 13."""
from mpmath import mp, mpf, sqrt, acos, asin, sin, pi, log, findroot
mp.dps = 50
ok = True


def check(name, cond, val=None):
    global ok
    print(f"{'OK ' if cond else 'FAIL'} {name}" + (f"   [{mp.nstr(val, 12)}]" if val is not None else ""))
    ok = ok and bool(cond)


SQ2 = sqrt(2); deg = 180/pi
f = lambda r1, r2: (r1**2 + r2**2 - 1)/(2*r1*r2)
for d in [mpf('1e-4'), mpf('1e-5')]:
    Lam = SQ2 + d; g0 = acos(1 - 1/(2*Lam**2))
    rho1 = sqrt((d + SQ2/2)**2 + mpf(1)/4)
    R5 = min(rho1, 1/(2*sin(2*g0)))
    g55 = acos(f(R5, R5))
    psi = asin(R5*sin(pi - g55))
    cand = [f(mpf(1), mpf(1)), f(mpf(1), Lam), f(2*R5, mpf(1)), f(2*R5, Lam)]
    sep3 = acos(max(cand))
    c0 = d + SQ2/2 - mpf(1)/2
    tag = f"[d={mp.nstr(d,2)}]"
    check(f"{tag} 2 R5 < 1.008 (|c1-c3| in [1, 2R5])", 2*R5 < mpf('1.008'), 2*R5)
    check(f"{tag} psi_max = arcsin(R5 sin(pi - gamma(5,5))) < 7.2 deg", psi*deg < mpf('7.2'), psi*deg)
    check(f"{tag} f convex in r1 for r2 >= 1 (coefficient (r2^2-1)/(2 r2) >= 0) -> corners; sep3 > 44.99 deg", sep3*deg > mpf('44.99'), sep3*deg)
    check(f"{tag} side-foot separation arccos(c0) > sep3", acos(c0) > sep3)
    # two remaining arcs, each of length < 90 + psi_max - sep3, must each hold < 3 points (needs < 2 gamma0)
    Lmax = pi/2 + psi - sep3
    check(f"{tag} each remaining arc has length < 90 + psi - sep3 < 2 gamma0 (so <= 2 points per arc)", Lmax < 2*g0, Lmax*deg)
    # the inner centres c1, c3 subtend at p an angle >= gamma(5,5) > 165 deg
    check(f"{tag} gamma(5,5) > 165 deg", g55*deg > 165, g55*deg)
print("ALL OK (lem:Kw refinement)" if ok else "SOME CHECK FAILED")

# ---------------- constants with K_w = 13 (otherwise v8 parameters) ----------------
Kw = 13; Qs = mpf('0.32')
A = 4*sqrt(Kw/Qs)/(2 - mpf('3e-6'))
print("A =", A)
Ab = mpf('12.7476')
check("A < 12.7476", A < Ab, A)
g0b = mpf('0.49998'); eps = mpf('1e-4')
br = mpf('4e-3') + SQ2*Ab + (1 + mpf('8.04e-4'))/(SQ2*g0b*(1 - eps))*(Ab + mpf('2e-3'))
print("bracket =", br)
Bb = mpf('36.081')
check("bracket < 36.081", br < Bb, br)
check("0.99996*1.386089 - 1.00002 > 0.386", mpf('0.99996')*mpf('1.386089') - mpf('1.00002') > mpf('0.386'))
lim = mpf('0.99996')/Bb
check("limit 0.99996/36.081 > 0.0277 > 0.027", lim > mpf('0.0277'), lim)
lk2 = log(mpf(10)**16)
check("small k: log k < 36.842 and 0.027*36.842 < 0.995 < 1", lk2 < mpf('36.842') and mpf('0.027')*mpf('36.842') < mpf('0.995'), mpf('0.027')*mpf('36.842'))
Fk = lambda L: (mpf('0.99996')*L + mpf('0.386'))/Bb
Lc = findroot(lambda L: Fk(L) - 1, 35.7)
print("F > 1 for log k >", Lc, " ; 1/Lc =", 1/Lc)
check("F(k2) > 1 (Lc < log k2): kappa(2) is limited by 1/log k2 = 0.027143... > 0.027", Lc < lk2 and 1/lk2 > mpf('0.027143') and Fk(lk2)/lk2 > 1/lk2, 1/lk2)
print("rate c=4: log k >", (Bb*4 - mpf('0.386'))/mpf('0.99996'))
print("limit sqrt(Q/Kw)/(4 sqrt2) =", sqrt(Qs/Kw)/(4*SQ2))
print("ALL OK" if ok else "SOME CHECK FAILED")
