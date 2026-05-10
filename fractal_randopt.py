#!/usr/bin/env python3
"""
Fractal RandOpt — iterated greedy random search with sigma annealing.

Hypothesis: the neural thicket is self-similar. Once we land on a good
perturbation theta_1 = theta_0 + sigma_0 * eps, theta_1 is itself surrounded
by a (possibly denser, possibly sparser) thicket. We probe this by:

  for d = 0 ... D-1:
      sample N_d perturbations around the current center theta_d
      measure solution density delta_d(m) at this depth
      pick the top-1 and update center: theta_{d+1} = theta_d + sigma_d * eps_winner
      (sigma is annealed: sigma_d = sigma_0 * decay^d)

At the final depth, we evaluate either the top-1 chain or an ensemble of the
top-K perturbations from the last level.

This script logs (and saves) at every depth:
  - all (seed, sigma, train_reward) tuples
  - density delta_d(m) at multiple thresholds m
  - the winner that becomes the next center
  - cumulative chain of winners from theta_0 to theta_D
"""

import argparse
from collections import Counter
from datetime import datetime
import gc
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from data_handlers import get_dataset_handler, list_datasets
from core import launch_engines, cleanup_engines


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fractal RandOpt — iterated greedy with sigma annealing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / model
    parser.add_argument("--dataset", type=str, default="math500", choices=list_datasets())
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--test_samples", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--precision", type=str, choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max_tokens", type=int, default=None)

    # Fractal-specific
    parser.add_argument("--depth", type=int, default=4,
                        help="Number of hierarchical search levels (D).")
    parser.add_argument("--samples_per_level", type=int, default=64,
                        help="Population size N_d at each level.")
    parser.add_argument("--sigma_start", type=float, default=0.002,
                        help="sigma_0 used at depth 0.")
    parser.add_argument("--sigma_decay", type=float, default=0.5,
                        help="Multiplicative decay applied each depth: sigma_d = sigma_start * decay^d.")
    parser.add_argument("--sigma_schedule", type=str, default=None,
                        help="Optional explicit comma-separated schedule, overrides start/decay. "
                             "Example: '0.002,0.001,0.0005,0.00025'.")
    parser.add_argument("--ensemble_top_k", type=str, default="1,5,10",
                        help="Comma-separated K values for final-level ensembling.")
    parser.add_argument("--density_thresholds", type=str, default="0.0,0.01,0.02,0.05,0.1",
                        help="Comma-separated reward-margin thresholds m used to compute delta_d(m).")

    # Infra
    parser.add_argument("--num_engines", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cuda_devices", type=str, default="0")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--experiment_dir", type=str, default="fractal-experiment")

    # vLLM throughput knobs (cranked defaults for max GPU utilization)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92,
                        help="Fraction of GPU memory vLLM may use (KV cache + activations). 0.90-0.95 is aggressive.")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Disable CUDA graphs (slower). Default: graphs enabled.")
    parser.add_argument("--max_num_seqs", type=int, default=512,
                        help="Max concurrent sequences in vLLM scheduler. Higher = better batch packing.")
    parser.add_argument("--max_num_batched_tokens", type=int, default=16384,
                        help="Max total tokens per forward step.")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Cap context length to bound KV cache memory. None = use model's default.")
    parser.add_argument("--kv_cache_dtype", type=str, default="auto",
                        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
                        help="fp8 halves KV-cache memory; usually no quality hit on inference.")

    args = parser.parse_args()

    # Build sigma schedule
    if args.sigma_schedule:
        args.sigma_schedule_list = [float(s.strip()) for s in args.sigma_schedule.split(",")]
        if len(args.sigma_schedule_list) != args.depth:
            raise ValueError(
                f"--sigma_schedule has {len(args.sigma_schedule_list)} values "
                f"but --depth is {args.depth}."
            )
    else:
        args.sigma_schedule_list = [
            args.sigma_start * (args.sigma_decay ** d) for d in range(args.depth)
        ]

    args.ensemble_top_k_list = sorted(
        set(int(k.strip()) for k in args.ensemble_top_k.split(",")), reverse=True
    )
    args.density_threshold_list = [float(m.strip()) for m in args.density_thresholds.split(",")]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    torch.manual_seed(args.global_seed)
    torch.cuda.manual_seed_all(args.global_seed)
    return args


# ============================================================================
# Data
# ============================================================================

def load_data(handler, args):
    """Load train/test. Mirrors logic in randopt.py."""
    train_path = args.train_data_path or handler.default_train_path
    test_path = args.test_data_path or handler.default_test_path

    print(f"Loading {handler.name} data...")
    if train_path == test_path:
        all_data = handler.load_data(train_path, split="train", max_samples=None)
        train_datas = all_data[: args.train_samples]
        test_datas = (
            all_data[args.train_samples :]
            if args.test_samples is None
            else all_data[args.train_samples : args.train_samples + args.test_samples]
        )
        if len(test_datas) < 50:
            print(f"  Warning: only {len(test_datas)} test samples; using all data for test.")
            test_datas = all_data
    else:
        train_datas = handler.load_data(train_path, split="train", max_samples=args.train_samples)
        test_datas = handler.load_data(test_path, split="test", max_samples=args.test_samples)

    print(f"  Train: {len(train_datas)} | Test: {len(test_datas)}")
    return train_datas, test_datas


# ============================================================================
# Evaluation helpers
# ============================================================================

def evaluate_center(engines, handler, train_prompts, test_prompts, train_datas, test_datas,
                    sampling_params, label="CENTER"):
    """
    Evaluate the *current* model on train and test sets.

    The current model is whatever is loaded into the engines right now —
    base weights at depth 0, the chain-of-winners at later depths.
    Engine 0 is treated as the canonical model (engines may be out of sync
    after a sampling batch; caller is responsible for syncing if needed).
    """
    print(f"\n--- Evaluating {label} ---")
    train_outputs = ray.get(
        engines[0].generate.remote(train_prompts, sampling_params, use_tqdm=False)
    )
    train_reward = handler.postprocess_outputs(train_outputs, train_datas)
    print(f"  Train reward: {train_reward * 100:.2f}%")

    test_outputs = ray.get(
        engines[0].generate.remote(test_prompts, sampling_params, use_tqdm=False)
    )
    correct = 0
    for i, output in enumerate(test_outputs):
        response_text = output.outputs[0].text
        if handler.name == "countdown":
            numbers = test_datas[i].get("numbers")
            answer, is_valid, _ = handler.extract_answer_for_voting(response_text, numbers=numbers)
            answer = answer if is_valid else ""
        elif hasattr(handler, "extract_answer_for_voting"):
            answer = handler.extract_answer_for_voting(response_text) or ""
        else:
            answer = handler.extract_answer(response_text) or ""

        if not answer:
            continue
        if hasattr(handler, "is_voted_answer_correct"):
            ok = handler.is_voted_answer_correct(answer, test_datas[i]["ground_truth"])
        else:
            ok = handler.is_answer_correct(handler.format_answer_for_check(answer),
                                           test_datas[i]["ground_truth"])
        if ok:
            correct += 1
    test_acc = correct / len(test_datas) if test_datas else 0.0
    print(f"  Test acc:     {test_acc * 100:.2f}% ({correct}/{len(test_datas)})")
    return train_reward, test_acc


# ============================================================================
# Sampling at one level (around current _base_weights)
# ============================================================================

def sample_one_level(engines, handler, train_prompts, train_datas, sampling_params,
                     n_samples: int, sigma: float, seed_rng: np.random.Generator,
                     num_engines: int) -> List[Tuple[int, float, float]]:
    """
    Sample n_samples random perturbations around the current center
    (engines' _base_weights), evaluate on train set, return list of
    (seed, sigma, train_reward).

    Uses `apply_perturbation` (which always re-centers to _base_weights
    before adding noise) so the level's center stays fixed across samples.
    """
    seeds = seed_rng.choice(2**31, size=n_samples, replace=False).tolist()
    results: List[Tuple[int, float, float]] = []

    samples_done, batch_idx = 0, 0
    while samples_done < n_samples:
        bs = min(num_engines, n_samples - samples_done)
        batch_seeds = [int(seeds[samples_done + i]) for i in range(bs)]

        # Apply each engine its own perturbation from base
        ray.get([
            engines[i].collective_rpc.remote("apply_perturbation", args=(s, float(sigma)))
            for i, s in enumerate(batch_seeds)
        ])

        # Generate
        outputs = ray.get([
            engines[i].generate.remote(train_prompts, sampling_params, use_tqdm=False)
            for i in range(bs)
        ])

        rewards = []
        for i, s in enumerate(batch_seeds):
            r = handler.postprocess_outputs(outputs[i], train_datas)
            results.append((s, float(sigma), float(r)))
            rewards.append(r)

        samples_done += bs
        batch_idx += 1
        print(f"    batch {batch_idx} | {samples_done}/{n_samples} | "
              f"rewards={['%.3f' % r for r in rewards]}")

    # Reset all engines to base weights so caller starts from a clean state
    ray.get([e.collective_rpc.remote("reset_to_base_weights", args=()) for e in engines])
    return results


def lock_in_winner(engines, winner_seed: int, winner_sigma: float):
    """
    Move the engines' model weights to the winner perturbation, then
    update _base_weights to this new point so it becomes the new center
    for subsequent levels.
    """
    # Apply perturbation on every engine (each engine becomes the winner)
    ray.get([
        e.collective_rpc.remote("apply_perturbation", args=(int(winner_seed), float(winner_sigma)))
        for e in engines
    ])
    # Update each engine's _base_weights to the new center
    ray.get([e.collective_rpc.remote("store_base_weights", args=()) for e in engines])


# ============================================================================
# Density measurement
# ============================================================================

def compute_density(rewards: List[float], center_reward: float,
                    thresholds: List[float]) -> Dict[str, float]:
    """
    Solution density delta(m) = fraction of perturbations whose train reward
    is at least center_reward + m. Computes both:
      - relative density vs the *current* center (the fractal claim)
      - absolute density vs a fixed reference (passed via thresholds offset)

    Returns a dict keyed by 'rel_m=...' so we can stash multiple thresholds.
    """
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    out = {
        "n": len(rewards_arr),
        "mean": float(rewards_arr.mean()) if len(rewards_arr) else 0.0,
        "max": float(rewards_arr.max()) if len(rewards_arr) else 0.0,
        "min": float(rewards_arr.min()) if len(rewards_arr) else 0.0,
        "std": float(rewards_arr.std()) if len(rewards_arr) else 0.0,
        "center_reward": float(center_reward),
    }
    for m in thresholds:
        density = float((rewards_arr >= center_reward + m).mean()) if len(rewards_arr) else 0.0
        out[f"rel_density_m={m:g}"] = density
    return out


# ============================================================================
# Final ensemble (majority vote over top-K perturbations of the LAST level)
# ============================================================================

def run_ensemble_at_final_level(engines, handler, test_prompts, test_datas, sampling_params,
                                top_k_perturbs: List[Tuple[int, float]], top_k_list: List[int],
                                num_engines: int) -> Dict[int, Dict]:
    """
    Engines' _base_weights are the final-level center. We ensemble the top-K
    perturbations *of the final level* via majority vote on the test set.
    """
    max_k = min(max(top_k_list), len(top_k_perturbs))
    eval_k_values = sorted([k for k in top_k_list if k <= max_k], reverse=True)
    num_samples = len(test_datas)
    print(f"\n=== Final-level ensemble (K in {eval_k_values}) ===")

    all_answers: List[Optional[List[str]]] = [None] * max_k
    total_batches = (max_k + num_engines - 1) // num_engines

    for batch_idx in range(total_batches):
        start = batch_idx * num_engines
        end = min(start + num_engines, max_k)
        batch = top_k_perturbs[start:end]

        ray.get([
            engines[i].collective_rpc.remote("apply_perturbation", args=(int(s), float(sig)))
            for i, (s, sig) in enumerate(batch)
        ])

        batch_outputs = ray.get([
            engines[i].generate.remote(test_prompts, sampling_params, use_tqdm=False)
            for i in range(len(batch))
        ])

        # Reset before next batch (and for cleanliness)
        ray.get([engines[i].collective_rpc.remote("reset_to_base_weights", args=())
                 for i in range(len(batch))])

        for local_idx, global_idx in enumerate(range(start, end)):
            outs = batch_outputs[local_idx]
            answers_for_model = []
            for i in range(num_samples):
                response_text = outs[i].outputs[0].text
                if handler.name == "countdown":
                    numbers = test_datas[i].get("numbers")
                    a, is_valid, _ = handler.extract_answer_for_voting(response_text, numbers=numbers)
                    answers_for_model.append(a if is_valid else "")
                elif hasattr(handler, "extract_answer_for_voting"):
                    answers_for_model.append(handler.extract_answer_for_voting(response_text) or "")
                else:
                    answers_for_model.append(handler.extract_answer(response_text) or "")
            all_answers[global_idx] = answers_for_model

        del batch_outputs
        gc.collect()

    # Majority vote at each K
    results = {}
    for k in eval_k_values:
        correct = 0
        for idx, data in enumerate(test_datas):
            answers = [all_answers[m][idx] for m in range(k) if all_answers[m][idx]]
            if not answers:
                continue
            final = Counter(answers).most_common(1)[0][0]
            if hasattr(handler, "is_voted_answer_correct"):
                ok = handler.is_voted_answer_correct(final, data["ground_truth"])
            else:
                ok = handler.is_answer_correct(handler.format_answer_for_check(final),
                                               data["ground_truth"])
            if ok:
                correct += 1
        acc = correct / num_samples * 100 if num_samples else 0.0
        results[k] = {"accuracy": acc, "correct": correct, "total": num_samples}
        print(f"  K={k}: {acc:.2f}% ({correct}/{num_samples})")

    del all_answers
    gc.collect()
    return results


# ============================================================================
# Main fractal loop
# ============================================================================

def main(args):
    handler = get_dataset_handler(args.dataset)
    max_tokens = args.max_tokens or handler.default_max_tokens

    # Logging setup
    run_name = f"{args.dataset}_fractal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging_dir = os.path.join(args.experiment_dir, run_name)
    os.makedirs(logging_dir, exist_ok=True)
    with open(os.path.join(logging_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, default=str)

    print("=" * 70)
    print(f"FRACTAL RandOpt — {handler.name.upper()}")
    print("=" * 70)
    print(f"Model:           {args.model_name}")
    print(f"Depth (D):       {args.depth}")
    print(f"Samples / level: {args.samples_per_level}  "
          f"(total budget: {args.depth * args.samples_per_level})")
    print(f"Sigma schedule:  {['%.5g' % s for s in args.sigma_schedule_list]}")
    print(f"Ensemble K:      {args.ensemble_top_k_list}")
    print(f"Density m:       {args.density_threshold_list}")
    print(f"Logs:            {logging_dir}")

    # Ray
    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True)
    else:
        ray.init(address="local", ignore_reinit_error=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    is_instruct = any(x in args.model_name.lower() for x in ["instruct", "chat", "-it"])

    def format_prompt(messages):
        if is_instruct and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in messages) + "\n"

    # Data
    train_datas, test_datas = load_data(handler, args)
    train_prompts = [format_prompt(d["messages"]) for d in train_datas]
    test_prompts = [format_prompt(d["messages"]) for d in test_datas]
    sampling_params = SamplingParams(temperature=0.0, seed=args.global_seed, max_tokens=max_tokens)

    # Engines (cranked vLLM throughput settings)
    engines, pgs = launch_engines(
        args.num_engines, args.model_name,
        precision=args.precision,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
    )

    seed_rng = np.random.default_rng(seed=args.global_seed)
    all_levels_log: List[Dict] = []
    chain: List[Dict] = []  # winners selected at each depth

    try:
        # ---- depth 0: evaluate base model ----
        print("\n" + "#" * 70)
        print("# DEPTH 0 — base pretrained weights")
        print("#" * 70)
        base_train, base_test = evaluate_center(
            engines, handler, train_prompts, test_prompts, train_datas, test_datas,
            sampling_params, label="BASE"
        )
        center_train_reward = base_train

        # All engines already have base weights stored at launch time. Good.
        for d in range(args.depth):
            sigma_d = args.sigma_schedule_list[d]
            print("\n" + "#" * 70)
            print(f"# LEVEL d={d} | sigma={sigma_d:.5g} | center_train_reward={center_train_reward:.4f}")
            print("#" * 70)

            # Sample around current center
            level_results = sample_one_level(
                engines, handler, train_prompts, train_datas, sampling_params,
                n_samples=args.samples_per_level, sigma=sigma_d,
                seed_rng=seed_rng, num_engines=args.num_engines,
            )

            rewards_only = [r for _, _, r in level_results]
            density_stats = compute_density(rewards_only, center_train_reward,
                                            args.density_threshold_list)

            # Sort and pick winner
            level_results_sorted = sorted(level_results, key=lambda x: x[2], reverse=True)
            winner_seed, winner_sigma, winner_reward = level_results_sorted[0]

            print(f"  density stats: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in density_stats.items()})}")
            print(f"  WINNER: seed={winner_seed} sigma={winner_sigma:.5g} train_reward={winner_reward:.4f}")
            print(f"           Δ vs center = {winner_reward - center_train_reward:+.4f}")

            # Log level
            level_log = {
                "depth": d,
                "sigma": sigma_d,
                "n_samples": len(level_results),
                "all_results": [
                    {"seed": int(s), "sigma": float(sg), "train_reward": float(r)}
                    for (s, sg, r) in level_results
                ],
                "density": density_stats,
                "winner": {
                    "seed": int(winner_seed),
                    "sigma": float(winner_sigma),
                    "train_reward": float(winner_reward),
                    "delta_vs_center": float(winner_reward - center_train_reward),
                },
                "top_k_at_this_level": [
                    {"seed": int(s), "sigma": float(sg), "train_reward": float(r)}
                    for (s, sg, r) in level_results_sorted[: max(args.ensemble_top_k_list)]
                ],
                "center_train_reward_before_step": center_train_reward,
            }
            all_levels_log.append(level_log)
            chain.append({
                "depth": d,
                "winner_seed": int(winner_seed),
                "winner_sigma": float(winner_sigma),
                "winner_train_reward": float(winner_reward),
            })

            # Save partial log every level so we don't lose progress on crash
            with open(os.path.join(logging_dir, "levels.json"), "w") as f:
                json.dump(all_levels_log, f, indent=2)
            with open(os.path.join(logging_dir, "chain.json"), "w") as f:
                json.dump(chain, f, indent=2)

            # Lock in winner: this updates each engine's _base_weights
            print(f"  Locking in winner as new center …")
            lock_in_winner(engines, winner_seed, winner_sigma)
            center_train_reward = winner_reward

            # OPTIONAL: also evaluate the new center on test for tracking
            new_train, new_test = evaluate_center(
                engines, handler, train_prompts, test_prompts, train_datas, test_datas,
                sampling_params, label=f"CENTER@d={d+1}"
            )
            chain[-1]["center_after_step_train_reward"] = float(new_train)
            chain[-1]["center_after_step_test_acc"] = float(new_test)
            with open(os.path.join(logging_dir, "chain.json"), "w") as f:
                json.dump(chain, f, indent=2)

        # ---- final ensemble at last depth (top-K of FINAL level around final center) ----
        # The final level's top-K is the last entry in all_levels_log
        final_top_k_records = all_levels_log[-1]["top_k_at_this_level"]
        final_top_k_perturbs = [(r["seed"], r["sigma"]) for r in final_top_k_records]
        ensemble_results = run_ensemble_at_final_level(
            engines, handler, test_prompts, test_datas, sampling_params,
            top_k_perturbs=final_top_k_perturbs,
            top_k_list=args.ensemble_top_k_list,
            num_engines=args.num_engines,
        )

        # ---- Save final results ----
        results = {
            "method": "fractal_randopt",
            "dataset": args.dataset,
            "model": args.model_name,
            "depth": args.depth,
            "samples_per_level": args.samples_per_level,
            "total_budget": args.depth * args.samples_per_level,
            "sigma_schedule": args.sigma_schedule_list,
            "base_train_reward": float(base_train),
            "base_test_accuracy": float(base_test),
            "final_chain_train_reward": float(center_train_reward),
            "final_chain_test_accuracy": float(chain[-1]["center_after_step_test_acc"]),
            "ensemble_results": {str(k): v for k, v in ensemble_results.items()},
            "chain": chain,
            "density_per_depth": [
                {"depth": L["depth"], "sigma": L["sigma"], **L["density"]}
                for L in all_levels_log
            ],
        }
        with open(os.path.join(logging_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Pretty summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Base:                 train={base_train*100:.2f}%  test={base_test*100:.2f}%")
        print(f"Final chain (top-1):  train={center_train_reward*100:.2f}%  "
              f"test={chain[-1]['center_after_step_test_acc']*100:.2f}%")
        for k, v in ensemble_results.items():
            print(f"Final ensemble K={k}:  test={v['accuracy']:.2f}%")
        print("\nDensity at depth (rel to *current* center):")
        for L in all_levels_log:
            print(f"  d={L['depth']:>2}  sigma={L['sigma']:.5g}  "
                  f"mean={L['density']['mean']:.3f}  "
                  f"rel_density_m=0={L['density']['rel_density_m=0']:.3f}  "
                  f"rel_density_m=0.05={L['density'].get('rel_density_m=0.05', 0.0):.3f}")
        print(f"\nLogs written to: {logging_dir}")

    finally:
        cleanup_engines(engines, pgs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
