#!/usr/bin/env python3
"""
Compare flat-RandOpt baseline vs fractal-RandOpt run on the same dataset/model.

Usage:
  python scripts/compare_results.py \
      --baseline_dir experiments/math500_qwen05b_baseline/math500_<TS> \
      --fractal_dir  experiments/math500_qwen05b_fractal/math500_fractal_<TS>

Prints a side-by-side table and writes a JSON summary next to each run.
"""
import argparse
import json
import os
from pathlib import Path


def load_results(run_dir: str) -> dict:
    results_path = Path(run_dir) / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results.json in {run_dir}")
    with open(results_path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", required=True, help="Flat RandOpt run dir.")
    ap.add_argument("--fractal_dir", required=True, help="Fractal RandOpt run dir.")
    args = ap.parse_args()

    base = load_results(args.baseline_dir)
    frac = load_results(args.fractal_dir)

    print("=" * 80)
    print(f"  COMPARISON")
    print("=" * 80)
    print(f"{'':<30}{'Baseline (flat)':>22}{'Fractal':>22}")
    print("-" * 80)

    # Budget
    base_budget = base.get("train_samples")  # not exact, see below
    # randopt.py saves: dataset, model, base_train_reward, base_test_accuracy,
    # ensemble_results, top_k_perturbs, etc. It does NOT save population_size in results.
    # We compute it from top_k_perturbs length being the max_top_k, not population. So
    # just print what we have.
    print(f"{'Dataset':<30}{base.get('dataset',''):>22}{frac.get('dataset',''):>22}")
    print(f"{'Model':<30}{base.get('model',''):>22}{frac.get('model',''):>22}")

    base_budget = "?"
    frac_budget = frac.get("total_budget", "?")
    print(f"{'Total budget (samples)':<30}{str(base_budget):>22}{str(frac_budget):>22}")

    print(f"{'Base train reward (%)':<30}"
          f"{base.get('base_train_reward', 0)*100:>22.2f}"
          f"{frac.get('base_train_reward', 0)*100:>22.2f}")
    print(f"{'Base test accuracy (%)':<30}"
          f"{base.get('base_test_accuracy', 0)*100:>22.2f}"
          f"{frac.get('base_test_accuracy', 0)*100:>22.2f}")

    # Ensemble accuracy at matched K
    print("-" * 80)
    print("Ensemble test accuracy by K (final-level ensemble for fractal):")
    base_ens = base.get("ensemble_results", {})
    frac_ens = frac.get("ensemble_results", {})
    all_ks = sorted(set(int(k) for k in list(base_ens.keys()) + list(frac_ens.keys())))
    for k in all_ks:
        b = base_ens.get(str(k), {}).get("accuracy", float("nan"))
        f = frac_ens.get(str(k), {}).get("accuracy", float("nan"))
        print(f"  K={k:<4}                       {b:>22.2f}{f:>22.2f}")

    # Fractal-only: density at depth
    print("-" * 80)
    print("Density at depth (fractal only):")
    for d in frac.get("density_per_depth", []):
        print(f"  d={d['depth']}  sigma={d['sigma']:.5g}  "
              f"mean={d['mean']:.3f}  rel_density(m=0)={d.get('rel_density_m=0', 0):.3f}  "
              f"rel_density(m=0.05)={d.get('rel_density_m=0.05', 0):.3f}")

    print("-" * 80)
    print("Final chain (fractal): test acc =",
          f"{frac.get('final_chain_test_accuracy', 0)*100:.2f}%")
    print("=" * 80)

    # Persist
    out = {
        "baseline_dir": args.baseline_dir,
        "fractal_dir": args.fractal_dir,
        "baseline": {
            "base_train_reward": base.get("base_train_reward"),
            "base_test_accuracy": base.get("base_test_accuracy"),
            "ensemble_results": base_ens,
        },
        "fractal": {
            "base_train_reward": frac.get("base_train_reward"),
            "base_test_accuracy": frac.get("base_test_accuracy"),
            "final_chain_test_accuracy": frac.get("final_chain_test_accuracy"),
            "ensemble_results": frac_ens,
            "density_per_depth": frac.get("density_per_depth"),
        },
    }
    out_path = os.path.join(os.path.dirname(args.fractal_dir.rstrip("/")), "comparison.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote summary: {out_path}")


if __name__ == "__main__":
    main()
