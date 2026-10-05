# Independent coverage checks for Lemma 4.10

Lemma 4.10 (K_w* = 9) rests on a branch-and-bound computation whose 78,673 accepted boxes are listed in `data/kw9_data.zip` (files `*_T9_*_final_leaves.jsonl`). Two things must hold:

1. **Labels**: every box is correctly labelled. It is infeasible, has no possible side pair, or has rigorous bound ≤ 9. This is checked by the Arb re-verification `code/verify_leaves2.py` and is **not** the subject of this folder.
2. **Coverage**: the boxes leave no gap in the parameter domain. The author's program `code/coverage_check.py` checks this, but it rebuilds the bisection tree with the same splitting rule as the search, so it is not independent of it.

This folder contains **two further coverage checks**. They were written independently of the search and of `coverage_check.py`, and they use different methods.
- Check 1 was written from scratch by a reviewer who was instructed not to open `coverage_check.py`.
- The algorithm of check 2 was designed by a blank-slate reviewer before that reviewer read `coverage_check.py`.

Both reviewers were AI-based (Claude, Anthropic), working in sessions separate from the one that wrote the search.

## The domain that is checked

As in the proof of Lemma 4.10, with d = 10⁻⁴, ρ₁ = sqrt((d + √2/2)² + 1/4) and the floating-point value 2π_f of 2π:

- **No side** (`bnb_*` files): n = 1, …, 5 centres.
  - r_i ∈ [1/2, ρ₁], θ₁ = 0 ≤ θ₂ ≤ … ≤ θ_n ≤ 2π_f.
  - θ₂ is cut into 8·ntask slices; task t takes the slices s with s ≡ t (mod ntask).
- **One side** (`bnbw_*` files): r_i ∈ [1/2, ρ₁], 0 ≤ θ₁ ≤ … ≤ θ_n ≤ 2π_f, h ∈ [0, d + √2/2].
  - θ₁ is cut into 4·ntask slices.
- Boxes containing no point with ordered angles were pruned by the search and are not recorded.

## Check 1: `exact_volume/`, exact volumes and disjointness

- **Method A (a proof, up to the correctness of the program).** For every file:
  - every box lies in the domain and in one slice of its task;
  - the boxes are pairwise interior-disjoint (exact floating-point comparisons, all pairs);
  - in exact rational arithmetic (`fractions.Fraction` of the floating-point end points), the volumes of the boxes intersected with the region of ordered angles add up exactly to the volume of the domain.

  Disjoint closed boxes whose volumes add up to that of the domain cover it up to a null set. The domain is a finite union of convex bodies with non-empty interior and the union of the boxes is closed, so they cover it entirely.
- **Method B (supporting evidence).** Random, boundary, angle-tie, grid and one-ulp-nudged points are located in the boxes.
- **Run**:
  - Unpack `data/kw9_data.zip` into `code/`, then:

    ```
    cd code/coverage_independent/exact_volume
    python run_all.py 40000
    ```

  - `run_all.py` runs `indep_cover.py` on the 13 files, 4 processes at a time.
  - One file: `python indep_cover.py ../../bnb_n3_T9_0of1_final_leaves.jsonl 40000 12345 out.json`.
  - Needs Python 3 and numpy.
- **Result** (`run_all_log.txt`, `results/*.json`):
  - all 13 files pass Method A, with exact volume equality and 0 interior-overlapping pairs;
  - Method B: 3,333,005 points, 0 uncovered;
  - total running time about 590 s.
- **Self-test of the checker.** Set the environment variable `MUTATE` to `drop`, `shrink` or `dup` to damage the box list before checking. These options remove a random box, move one non-boundary end point inward by one ulp, or duplicate a box. The reviewer reports that such damage was detected:
  - a drop or a shrink is detected whenever the damaged box has positive volume in the region of ordered angles;
  - a duplicate is detected by the overlap test.

  The outputs of those mutation runs are not included here.

## Check 2: `recursive_cover/`, recursive covering

- **Method.** A query box is accepted if it contains no point with ordered angles, or if it lies inside one recorded box. Otherwise it is split at a box face strictly inside it.
  - A query box that contains ordered points but meets no recorded box in positive measure is reported as a **gap**.
  - The test can over-report on null sets but cannot miss a gap.
  - The domain of each case is cut into 32 slabs along its first angle coordinate (1 slab for n = 1), 258 slabs in all. The leaf files are read again for every slab, so the program runs in little memory (pure Python, no numpy).
- **Run** (data unpacked into `code/` as above):

  ```
  cd code/coverage_independent/recursive_cover
  python coverage_lowmem.py ../.. results_new.jsonl --slabs 32
  python summarize.py results_new.jsonl
  python coverage_lowmem.py ../.. --selftest
  ```

  - On Windows the run stops, keeping finished slabs, when the free memory of the machine falls below `--floor` MB (default 1024).
  - Run the same command again to resume. On a machine with little free memory, pass a smaller value, for example `--floor 300`. The program itself uses only about 25 MB.

- **Result** (`results.jsonl`, `summary.log`, `run.log`):
  - all 258 slabs are complete, with 78,673 boxes (30,365 without a side, 48,308 with a side);
  - **0 gaps**, 4,568,856 nodes;
  - the slabs took about 160 s in total;
  - `run.log` shows several restarts, because the original run stopped whenever the free memory of the machine fell below 1 GB and resumed from its checkpoint.
- **Mutation test** (`selftest.log`), for no side n = 2, 3 and one side n = 2, 3:
  - with the original boxes, 0 gaps;
  - removing any of 5 boxes that contain ordered interior points, or shrinking one of their faces by 0.1 %, always produced gaps (1–34).
- **Public copy.**
  - The original script imported a local resource-limit module (`limits`: process priority, CPU affinity, memory cap). It does not affect the algorithm and is optional here.
  - The memory guard uses Windows calls. On other systems it is disabled and never stops the run.
  - The comments no longer refer to internal folders.
  - The algorithm is unchanged. The first line of `selftest.log` and the start of `run.log` still show the resource-limit module as it was used.

## Limits of these checks

- They check **coverage only**: that the recorded boxes, together with the pruned regions without ordered angles, leave no gap. Whether each box is correctly labelled is the job of the Arb re-verification (`code/verify_leaves2.py`), which has **not** yet been reproduced by an independent implementation.
- Both checks work with the domain as the search stored it, that is, with the floating-point end points ρ₁, d + √2/2 and 2π_f.
  - The floating-point values of ρ₁, Λ and d + √2/2 are upper bounds of the exact values. `code/verify_v10.py` checks this.
  - **The arc (2π_f, 2π] is not covered by any box.** It is handled in the verification instead: `verify_leaves2.py` extends every interval that ends at 2π_f to the exact 2π in Arb. Without a side the arc contains no admissible configuration in any case, as a centre there would be at distance less than ρ₁ − 1/2 < 1 from the first centre. See the proof of Lemma 4.10.
- Both checks were written by AI-based reviewers working independently of the search code. They are independent implementations, but not independent of the AI system as such.
