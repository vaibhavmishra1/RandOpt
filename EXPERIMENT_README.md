# Neural Fractal vs Flat RandOpt — MATH-500 / Qwen2.5-0.5B-Instruct

This experiment compares the paper's flat RandOpt against our **Fractal RandOpt**
(iterated greedy with σ-annealing, plus density-at-depth measurement).

## Files added

- `fractal_randopt.py` — main script for the iterated/fractal method
- `scripts/fetch_math500.sh` — downloads `data/math-500/test.jsonl`
- `scripts/math500_qwen05b_baseline.sh` — flat RandOpt baseline run
- `scripts/math500_qwen05b_fractal.sh` — fractal run, matched FLOPs
- `scripts/compare_results.py` — side-by-side summary

## Budget matching

Both runs evaluate the same total number of perturbed forward passes on the
training set (depth × samples_per_level = 256). The fractal run partitions
that budget into 4 levels of 64 samples each, with σ schedule
`[0.002, 0.001, 0.0005, 0.00025]`.

## Run

```bash
# 1) data
bash scripts/fetch_math500.sh

# 2) baseline
bash scripts/math500_qwen05b_baseline.sh

# 3) fractal
bash scripts/math500_qwen05b_fractal.sh

# 4) compare (fill in the two run dirs with their actual timestamps)
python scripts/compare_results.py \
  --baseline_dir experiments/math500_qwen05b_baseline/math500_<TS> \
  --fractal_dir  experiments/math500_qwen05b_fractal/math500_fractal_<TS>
```

## Logged information (per run)

**Baseline (`randopt.py`, unmodified)** writes:
- `args.json`, `results.json` with base train/test, sigma stats, top-K ensemble accuracy
- `model_saves/top_k_seeds.json` — chosen top-K (seed, σ, train_reward)

**Fractal (`fractal_randopt.py`)** writes:
- `args.json` — full CLI snapshot
- `levels.json` — for **every depth** d:
  - all (seed, σ, train_reward) tuples evaluated at that level
  - density `δ_d(m)` at thresholds m ∈ {0, 0.01, 0.02, 0.05, 0.1} relative to the *current* center
  - the winner chosen and Δ-vs-center
  - the top-K of that level (used for the final ensemble)
- `chain.json` — the chain of winners θ₀ → θ₁ → ... → θ_D, including each
  intermediate center's train reward AND test accuracy
- `results.json` — base, final-chain, final-ensemble accuracy at multiple K, full density-per-depth

## What the density logs prove

The central scientific question: does `rel_density_m=0` (fraction of perturbations
at depth d that improve over θ_d) **stay high** as depth grows?

- **Fractal** — δ_d(m=0) approximately constant or growing across d
- **Shell** — δ_d(m=0) collapses immediately at d ≥ 1
- **Basin** — δ_d(m=0) decreases monotonically

These are extracted directly from `levels.json` per run.

## Notes on initial settings

For Qwen2.5-0.5B-Instruct on MATH-500 the paper reports (Table 4):
- Base: 43.2%, RandOpt (N=5000, K=50): 35.3% — the 0.5B model is BELOW thicket threshold
  per Figure 8 of the paper.
- This run uses N=256 (much smaller) — we expect modest absolute improvements at best,
  but the **density curve** is the actual measurement of interest, not the final accuracy.
- If you want stronger absolute numbers, scale up `--population_size` / `--samples_per_level`
  and re-run; the script structure is the same.
