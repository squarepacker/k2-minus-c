# Search programs behind the numerical experiments (Remark 7.5)

These programs are **not part of the proof**. They produced the numerical observations reported in Remark 7.5 and in the remark after Proposition 4.5. Each folder has its own geometry module (`geom.py`) and is run from inside that folder. They need Python 3 with `numpy` and `mpmath`. The stored results are the ones quoted in the paper; random searches give different runs with other seeds.

## `kw_pairs/`: pairs whose rectangles share an uncovered point

This folder counts all pairs of squares at distance < d, whether or not they are edges of a graph, whose rectangles R(P,Q) contain a common uncovered point p.

| Program | What it does | Quoted result |
|---|---|---|
| `hand9.py`, `hand9.json`, `hand9.log` | Builds the degenerate configuration with 9 pairs and checks it in 80-digit arithmetic. Two axis-parallel squares sit on both sides of a slit of width about 1e-20 through p, each nearly touching two slightly tilted squares on either side. | 9 pairs (see also `code/check_hand9_indep.py`, an independent check for d = 1e-6 … 1e-2) |
| `jam.py`, `jam_run1.jsonl`, `jam_run1.log` | Hard-square Monte Carlo compression towards a point, a wall or a corner, then a search for the best uncovered point | at most 6 |
| `sa.py`, `fastsa.py`, `improve.py`, `snap.py`, `grid_try.py` | Annealing, basin hopping, re-snapping and grid-based local searches | 3–6 |
| `hand7.py`, `hand7.json` | A simpler construction with 7 pairs | 7 |
| `degree1.py`, `angle_degree2.py` | Angle facts and the degree of a single centre | γ₀ = 41.40656°, degree ≤ 5 |
| `geom.py`, `test_geom.py` | Geometry; fuzz test against brute force | 0 mismatches |
| `summarize.py` | Summarises `.jsonl` result files | |

Reproduce:

```
cd code/experiments/kw_pairs
python hand9.py
python jam.py my_run.jsonl 1 600 free,wall,corner
python summarize.py my_run.jsonl
```

## `rv_mult_flpar/`: multiplicity of R_e, and Σλ² in Proposition 4.5

| Program | What it does | Quoted result |
|---|---|---|
| `count_mult.py`, `count_mult_results.json`, `count_mult.log` | Randomised dense packings: largest multiplicity of the rectangles R_e over edges of the horizontal visibility graph at distance < d (Lemma 4.8(b) bound: 27) | 5 (991 configurations) |
| `mult_climb.py`, `mult_climb_results.json` | Hill climbing on columns of squares for the same multiplicity | 5 |
| `flpar_test.py`, `flpar_results.json`, `flpar_results_3.json` | Proposition 4.5: squares pushed into the free triangles between two nearly parallel, nearly touching squares; minimises Σλ_r² (proved bound 0.32) | smallest value 0.33334; nothing below 1/3 |

Reproduce. The arguments are the time limit in seconds and the random seed:

```
cd code/experiments/rv_mult_flpar
python count_mult.py 300 3
python mult_climb.py 300 1
python flpar_test.py 300 1
```

An earlier search in a previous working environment reported a multiplicity of 8 for R_e. Its program was not preserved, so the paper now quotes the reproducible value 5 and mentions the earlier value separately.
