# Superseded verification outputs (not rigorous; do not cite)

The first rigorous-verification program `verify_leaves.py` and the edge re-check `verify_edge.py` used the floating-point direction
cells [PHI_LO[k], PHI_HI[k]] of bnb.py. These do not tile the circle: in 264 places PHI_HI[k] < PHI_LO[k+1] (gaps of one ulp),
and the last cell ends at the floating-point 2*pi, 2.45e-16 below the exact value. A direction in such a gap could escape the
Arb certification when both neighbouring cells were discarded. Found 2026-10-05 by review14 (literature reviewer, point m1;
independently by the mathematics reviewer) and confirmed by the main session.

Therefore the following outputs are **superseded and must not be cited as the certificate**:
- `*_final_leaves_verify_*.json` (verify_leaves.py, version 1)
- `*_edge_leaves_verify*.json` (verify_edge.py)
- logs `verify_1..7.log`, `verify_5b.log`, `verify_w5_*.log`, `edge_w5_*.log`

The certificate is produced by `verify_leaves2.py` (cells extended to max(PHI_HI[k], PHI_LO[k+1]), every interval ending at the
floating-point 2*pi extended to the exact 2*pi in Arb): outputs `*_final_leaves_verify2_{w}of4.json`, logs `v2_worker{w}.log`,
checked by `make_stats_v10.py` and `verify_v10.py`.
