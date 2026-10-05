# For every fixed c, s(k²−c) = k for all large k — with an explicit constant

Preprint and verification programs by Sungjoon Ryu (2026). Version 1.1.

**Main result.** For every fixed c ≥ 0 one has s(k²−c) = k for all sufficiently large integers k, where s(n) is the side of the smallest square containing n non-overlapping unit squares. Equivalently, if M(k) is the largest number of unit squares that fit in [0,k]² pairwise disjoint as closed sets, then k² − M(k) → ∞. The constant is explicit:

- k² − M(k) ≥ 0.033 · log k for every integer k ≥ 2;
- k² − M(k) ≥ (0.99977 · log k + 0.3819) / 30.147 for k ≥ 10^13.

The value 0.033 uses a computer-assisted lemma (Lemma 4.10, see below). Without it, the analytic Lemma 4.9 alone gives the constant 0.027, so the qualitative result does not depend on the computer.

So c*(k) := max{c : s(k²−c) = k} → ∞. Whether c*(k) → ∞ was recorded as open in E. Daniel's notes (evand/square-packing, `s12/search/FRIEDMAN.md`).

**What it does not do.** The result is qualitative: it says nothing about any k within computational reach. For example, it yields s(k²−4) = k only for log k > 120.24, and it says nothing about s(12). (For concrete values, see the computer-assisted, unrefereed announcement of s(k²−4) = k for all k ≥ 5 in evand/square-packing, which addresses a different question.)

## Constants with and without the computer

- Lemma 4.9 (analytic): at uncovered points at most 13 of the Roth–Vaughan rectangles of nearly touching pairs overlap. With this alone the proof gives k² − M(k) ≥ 0.027 · log k for every k ≥ 2.
- Lemma 4.10 (computer-assisted): at most 9 overlap. This gives the constant 0.033. The bound 9 is sharp for the number of all pairs at distance < d; whether it is sharp for the edges of a single graph is not known.
## The computation behind Lemma 4.10

- **Disc model.** The lemma reduces to bounding U(I) = E_II + W_I + Σ D_i*(I) over all configurations I of at most 5 centres near the uncovered point, without a side and with one side.
- **Branch-and-bound.** A floating-point search (`bnb.py`, `bnb_wall.py`) found a cover of the parameter domain by 78,673 accepted boxes. Every box is either infeasible, has no possible side pair, or has bound ≤ 9.
- **Rigorous re-verification.** Every accepted box was re-verified in Arb ball arithmetic through python-flint (`verify_leaves2.py`), with 59,238 further bisections. There were 0 failures, and the largest rigorous bound was 9.
- **Coverage check.** A separate program (`coverage_check.py`) rebuilds the bisection tree and checks, with exact rational volumes, that the accepted boxes leave no gap. It reuses the bisection rule of the search, so it is not independent of it.
- **Independent coverage checks.** Two further programs, written independently of the search by different methods, confirm that the accepted boxes leave no gap: an exact-volume check and a recursive covering check (`code/coverage_independent/`, with results and a README).
- **Superseded verifier.** An earlier verifier (`verify_leaves.py` used alone, and `verify_edge.py`) did not cover one-ulp gaps between floating-point direction cells. It is superseded (see `code/SUPERSEDED.md`). The certificate is the output of `verify_leaves2.py`.
- **What is independently reproduced.** The coverage (no gap) is confirmed by the two independent checks above. The verification of the individual boxes (the Arb re-verification) has not yet been reproduced by an independent implementation.

## Contents

| Path | Content |
|---|---|
| `paper/paper.pdf`, `paper/paper.tex` | The preprint (25 pages) |
| `paper/LICENSE` | CC BY 4.0 for the paper |
| `code/verify_v10.py` | Recomputes every numerical constant in high-precision arithmetic (mpmath). It also checks that the computation of Lemma 4.10 is complete (needs `data/kw9_data.zip` unpacked into `code/`). Prints `ALL OK` (132 checks) |
| `code/consts_v10.py` | Constants of Sections 5–6 |
| `code/kw13_check.py` | Numerical facts used in Lemma 4.9(c) |
| `code/bnb.py`, `code/bnb_wall.py` | Branch-and-bound search (without / with a side) |
| `code/verify_leaves.py`, `code/verify_leaves2.py` | Rigorous Arb re-verification (`verify_leaves2.py` is the certificate; it imports routines from `verify_leaves.py`) |
| `code/coverage_check.py`, `code/make_stats_v10.py`, `code/run_seq.py`, `code/run_v2.py` | Coverage check, statistics, job runners |
| `code/check_hand9_indep.py`, `code/attack_kw/hand9.json` | The configuration with 9 pairs (sharpness) and its check in 80-digit arithmetic |
| `code/SUPERSEDED.md` | Note on the superseded first verifier |
| `code/coverage_independent/` | Two independent checks that the accepted boxes of Lemma 4.10 cover the parameter domain (exact volumes; recursive covering), with results; see its README |
| `code/experiments/` | Search programs and stored results behind the numerical experiments of Remark 7.5 (not part of the proof); see `code/experiments/README.md` |
| `data/kw9_data.zip` | Accepted boxes of the final runs, verification and coverage results, logs |
| `reviews/REVIEWS.md` | Summary of the independent (AI) reviews |
| `LICENSE` | MIT License for `code/` and `data/` |

Reproduce the checks (Python 3.12; packages `mpmath`, `numpy`, `python-flint`):

```
pip install mpmath numpy python-flint
python code/kw13_check.py
python code/check_hand9_indep.py
```

Unpack `data/kw9_data.zip` into `code/`, then run:

```
cd code
python coverage_check.py bnb_n3_T9_0of1_final_leaves.jsonl
python verify_leaves2.py bnb_n3_T9_0of1_final_leaves.jsonl 9 0 1
python make_stats_v10.py
python verify_v10.py
```

The full re-verification of all boxes took about 1.7 hours on 4 cores.

## Changes in version 1.1

- Added `code/coverage_independent/`: two independent coverage checks for Lemma 4.10, with their results.
- One sentence in the proof of Lemma 4.10 (and the corresponding line above) now states precisely what has been independently reproduced: the coverage, by two independent checks; not yet the verification of the individual boxes.
- `SHA256SUMS` now covers every file except `README.md` and `.zenodo.json` (metadata that may be edited on the GitHub web page). The mathematics is unchanged.

## Status and use of AI

Developed with extensive assistance from Claude (Anthropic), including the proofs, the text, the programs and the computation for Lemma 4.10; the author takes full responsibility. Not peer reviewed. Comments and corrections are welcome (please open an issue).

## License

- Paper (`paper/paper.tex`, `paper/paper.pdf`): Creative Commons Attribution 4.0 International (CC BY 4.0), see `paper/LICENSE`.
- Programs and data (`code/`, including `code/experiments/`, and `data/`): MIT License, see `LICENSE`.

`LICENSE` contains the MIT License text; it applies to `code/` and `data/`. The paper is licensed separately under CC BY 4.0 (`paper/LICENSE`).

## How to cite

Ryu, Sungjoon. *Packing k²−c unit squares: s(k²−c) = k for all large k.* Preprint, version 1.1, 2026. DOI (v1.1): added after the release. DOI (v1.0): https://doi.org/10.5281/zenodo.23164302

DOI (all versions, always the latest): https://doi.org/10.5281/zenodo.23164301
