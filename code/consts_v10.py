"""v10 constants (mpmath, 50 digits): Sections 5-6 with threshold k2 = 1e13 and K_w given on the command line (9 or 10).
Parameters: C = 2, C' = 2 sqrt2, delta = 1e-5, s >= smin = sqrt(k2)/4, eps = 2e-3, omega0 = 2e-4, Q* = 0.32.
Prints every number to be quoted in the text and checks it.  Usage: python consts_v10.py KW"""
import sys
from mpmath import mp, mpf, sqrt, cos, sin, tan, sec, log, findroot, floor
mp.dps = 50
ok = True


def check(name, cond, val=None):
    global ok
    print(f"{'OK ' if cond else 'FAIL'} {name}" + (f"   [{mp.nstr(val, 12)}]" if val is not None else ""))
    ok = ok and bool(cond)


KW = int(sys.argv[1]) if len(sys.argv) > 1 else 9
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
print("ALL OK" if ok else "SOME CHECK FAILED")
