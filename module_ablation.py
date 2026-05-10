#!/usr/bin/env python3
"""
Module ablation: localize the gain.

Given a baseline RandOpt run (with its top-K winning perturbations), apply each
winning perturbation but ONLY to a chosen subset of parameters, and measure the
resulting train reward. By comparing across subsets, we localize *where in the
network* the perturbation's gain is concentrated.

Usage
-----
python module_ablation.py \
    --baseline_dir experiments/math500_qwen05b_baseline/math500_<TS> \
    --top_k 3 \
    --train_samples 200 \
    --num_engines 1

Outputs (in <baseline_dir>/ablation_<TS>/):
    - results.json: matrix of (winner_idx × filter_name) -> train_reward + counts
    - filters.json: filter definitions used
    - summary.txt:  human-readable ranking of which modules carry the gain
"""

import argparse
import gc
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from data_handlers import get_dataset_handler
from core import launch_engines, cleanup_engines


# ============================================================================
# Filter definitions
# ============================================================================

def build_filter_groups(num_layers: int) -> List[Dict]:
    """
    Define the set of module ablations to test. Each filter is a dict with:
      name: str
      include_substrings: list[str]  (None/empty == all)
      exclude_substrings: list[str]  (None == none)
      layer_min, layer_max: optional ints

    Notes:
      - 'all' is the sanity reference: full perturbation on every param
      - per-component filters (only attn / only mlp / only norms) tell us
        which functional block contributes
      - per-projection filters split attn into qkv vs o, and mlp into gate/up/down
      - depth filters split layers into halves and quarters
    """
    half = num_layers // 2
    q1 = num_layers // 4
    q3 = (3 * num_layers) // 4

    return [
        # Sanity reference
        {"name": "all",                "include_substrings": [], "exclude_substrings": []},

        # Component-level groups (apply only this kind of param, in any layer)
        {"name": "embedding_only",     "include_substrings": ["embed_tokens"],     "exclude_substrings": []},
        {"name": "lm_head_only",       "include_substrings": ["lm_head"],          "exclude_substrings": []},
        {"name": "all_norms_only",     "include_substrings": ["norm"],             "exclude_substrings": []},
        {"name": "attention_only",     "include_substrings": ["self_attn"],        "exclude_substrings": []},
        {"name": "mlp_only",           "include_substrings": ["mlp"],              "exclude_substrings": []},

        # Attention sub-projections
        {"name": "attn_qkv_only",      "include_substrings": ["q_proj", "k_proj", "v_proj"], "exclude_substrings": []},
        {"name": "attn_o_only",        "include_substrings": ["o_proj"],           "exclude_substrings": []},

        # MLP sub-projections
        {"name": "mlp_gate_only",      "include_substrings": ["gate_proj"],        "exclude_substrings": []},
        {"name": "mlp_up_only",        "include_substrings": ["up_proj"],          "exclude_substrings": []},
        {"name": "mlp_down_only",      "include_substrings": ["down_proj"],        "exclude_substrings": []},

        # Necessity tests: perturb everything EXCEPT one component
        {"name": "all_except_attn",    "include_substrings": [],                    "exclude_substrings": ["self_attn"]},
        {"name": "all_except_mlp",     "include_substrings": [],                    "exclude_substrings": ["mlp"]},
        {"name": "all_except_embed",   "include_substrings": [],                    "exclude_substrings": ["embed_tokens", "lm_head"]},

        # Depth-localization (apply to all params *within* a layer band)
        {"name": "first_half_layers",  "include_substrings": [], "exclude_substrings": [],
         "layer_min": 0,    "layer_max": half - 1},
        {"name": "second_half_layers", "include_substrings": [], "exclude_substrings": [],
         "layer_min": half, "layer_max": num_layers - 1},
        {"name": "early_quarter",      "include_substrings": [], "exclude_substrings": [],
         "layer_min": 0,    "layer_max": q1 - 1},
        {"name": "late_quarter",       "include_substrings": [], "exclude_substrings": [],
         "layer_min": q3,   "layer_max": num_layers - 1},
        {"name": "middle_half",        "include_substrings": [], "exclude_substrings": [],
         "layer_min": q1,   "layer_max": q3 - 1},
    ]


# ============================================================================
# Per-layer ablation: optionally drop in for a finer per-layer scan
# ============================================================================

def build_per_layer_filters(num_layers: int) -> List[Dict]:
    return [
        {"name": f"layer_{i:02d}_only",
         "include_substrings": [],
         "exclude_substrings": [],
         "layer_min": i, "layer_max": i}
        for i in range(num_layers)
    ]


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--baseline_dir", required=True,
                   help="A completed RandOpt run dir (must contain model_saves/top_k_seeds.json).")
    p.add_argument("--model_name", type=str, default=None,
                   help="Override base model path. If omitted, reads from baseline_dir/top_k_seeds.json.")
    p.add_argument("--top_k", type=int, default=3,
                   help="How many top winners (from baseline) to ablate.")
    p.add_argument("--train_samples", type=int, default=200)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--per_layer", action="store_true",
                   help="Also do a per-layer scan (one filter per layer index).")
    p.add_argument("--include_test_eval", action="store_true",
                   help="Also evaluate test accuracy for each (winner, filter). Slower.")
    p.add_argument("--test_samples", type=int, default=None)

    # Infra (mirrors fractal_randopt.py)
    p.add_argument("--num_engines", type=int, default=1)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--cuda_devices", type=str, default="0")
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--max_num_seqs", type=int, default=512)
    p.add_argument("--max_num_batched_tokens", type=int, default=16384)
    p.add_argument("--max_model_len", type=int, default=3072)
    p.add_argument("--kv_cache_dtype", type=str, default="auto",
                   choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"])

    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    torch.manual_seed(args.global_seed)
    return args


# ============================================================================
# Main
# ============================================================================

def main(args):
    # Load baseline run metadata
    seeds_path = os.path.join(args.baseline_dir, "model_saves", "top_k_seeds.json")
    results_path = os.path.join(args.baseline_dir, "results.json")
    if not os.path.exists(seeds_path):
        raise FileNotFoundError(f"Missing {seeds_path}")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing {results_path}")

    with open(seeds_path) as f:
        seeds_blob = json.load(f)
    with open(results_path) as f:
        prev_results = json.load(f)

    saved_model = seeds_blob["base_model_path"]
    if args.model_name and args.model_name != saved_model:
        print(f"WARNING: --model_name override ({args.model_name}) differs from "
              f"baseline run's saved model ({saved_model}). Seeds were generated for "
              f"the saved model — perturbations may not be valid on a different "
              f"architecture / param shapes.")
    base_model = args.model_name or saved_model
    dataset_name = prev_results["dataset"]
    top_k_models = seeds_blob["top_k_models"][: args.top_k]
    print(f"Baseline run:  {args.baseline_dir}")
    print(f"Base model:    {base_model}")
    print(f"Dataset:       {dataset_name}")
    print(f"Top-K winners (selected for ablation): {len(top_k_models)}")
    for w in top_k_models:
        print(f"  rank={w['rank']} seed={w['seed']} sigma={w['sigma']} train_reward={w['train_reward']:.4f}")

    # Output dir
    out_dir = os.path.join(args.baseline_dir, f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Handler / data
    handler = get_dataset_handler(dataset_name)
    train_path = prev_results.get("train_data_path") or handler.default_train_path
    test_path = prev_results.get("test_data_path") or handler.default_test_path

    if train_path == test_path:
        all_data = handler.load_data(train_path, split="train", max_samples=None)
        train_datas = all_data[: args.train_samples]
        test_datas = (
            all_data[args.train_samples :]
            if args.test_samples is None
            else all_data[args.train_samples : args.train_samples + args.test_samples]
        )
    else:
        train_datas = handler.load_data(train_path, split="train", max_samples=args.train_samples)
        test_datas = handler.load_data(test_path, split="test", max_samples=args.test_samples)
    print(f"Train: {len(train_datas)} | Test: {len(test_datas)}")

    # Tokenizer / prompts
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    is_instruct = any(x in base_model.lower() for x in ["instruct", "chat", "-it"])

    def fmt(messages):
        if is_instruct and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return "\n".join(m["content"] for m in messages) + "\n"

    train_prompts = [fmt(d["messages"]) for d in train_datas]
    test_prompts = [fmt(d["messages"]) for d in test_datas]
    sampling_params = SamplingParams(temperature=0.0, seed=args.global_seed,
                                     max_tokens=args.max_tokens)

    # Ray
    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True)
    else:
        ray.init(address="local", ignore_reinit_error=True)

    engines, pgs = launch_engines(
        args.num_engines, base_model,
        precision="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
    )

    try:
        # Discover number of transformer layers from any engine
        all_param_names = ray.get(engines[0].collective_rpc.remote("list_param_names", args=()))
        # collective_rpc returns one result per worker; flatten if list-of-lists
        if isinstance(all_param_names, list) and all_param_names and isinstance(all_param_names[0], list):
            all_param_names = all_param_names[0]
        layer_indices = set()
        import re
        for n in all_param_names:
            m = re.search(r"\.layers\.(\d+)\.", n)
            if m:
                layer_indices.add(int(m.group(1)))
        num_layers = (max(layer_indices) + 1) if layer_indices else 0
        print(f"Detected {num_layers} transformer layers, {len(all_param_names)} params total.")

        # Build filter set
        filters = build_filter_groups(num_layers)
        if args.per_layer:
            filters += build_per_layer_filters(num_layers)
        with open(os.path.join(out_dir, "filters.json"), "w") as f:
            json.dump(filters, f, indent=2)
        print(f"Will evaluate {len(filters)} filters × {len(top_k_models)} winners "
              f"= {len(filters) * len(top_k_models)} train evaluations.")

        # Base reward (no perturbation, sanity check)
        ray.get(engines[0].collective_rpc.remote("reset_to_base_weights", args=()))
        outs = ray.get(engines[0].generate.remote(train_prompts, sampling_params, use_tqdm=False))
        base_reward = handler.postprocess_outputs(outs, train_datas)
        print(f"Sanity: base train reward (no perturbation) = {base_reward:.4f}")

        # Run ablation
        # Layout: process (winner × filter) pairs, parallelizing filters across engines
        results_matrix: List[Dict] = []  # one row per (winner, filter) result

        for w_idx, w in enumerate(top_k_models):
            seed = int(w["seed"])
            sigma = float(w["sigma"])
            full_train_reward_recorded = float(w["train_reward"])
            print(f"\n{'='*70}")
            print(f"WINNER {w_idx+1}/{len(top_k_models)}: "
                  f"seed={seed} sigma={sigma} (recorded train_reward={full_train_reward_recorded:.4f})")
            print(f"{'='*70}")

            # Process filters in batches of size num_engines, each engine getting one filter.
            for batch_start in range(0, len(filters), args.num_engines):
                batch = filters[batch_start: batch_start + args.num_engines]

                # Each engine applies its filtered perturbation (same seed, same sigma)
                # but with a different filter spec.
                rpc_calls = []
                for i, flt in enumerate(batch):
                    kwargs = dict(
                        include_substrings=flt.get("include_substrings", []) or [],
                        exclude_substrings=flt.get("exclude_substrings", []) or [],
                        layer_min=flt.get("layer_min"),
                        layer_max=flt.get("layer_max"),
                    )
                    rpc_calls.append(
                        engines[i].collective_rpc.remote(
                            "apply_perturbation_filtered",
                            args=(seed, sigma),
                            kwargs=kwargs,
                        )
                    )
                counts = ray.get(rpc_calls)

                # Generate on train set
                outs = ray.get([
                    engines[i].generate.remote(train_prompts, sampling_params, use_tqdm=False)
                    for i in range(len(batch))
                ])
                rewards = [handler.postprocess_outputs(outs[i], train_datas) for i in range(len(batch))]

                # Optional: test eval
                test_accs = [None] * len(batch)
                if args.include_test_eval:
                    test_outs = ray.get([
                        engines[i].generate.remote(test_prompts, sampling_params, use_tqdm=False)
                        for i in range(len(batch))
                    ])
                    for i in range(len(batch)):
                        correct = 0
                        for ti, output in enumerate(test_outs[i]):
                            response = output.outputs[0].text
                            if hasattr(handler, "extract_answer_for_voting"):
                                ans = handler.extract_answer_for_voting(response) or ""
                            else:
                                ans = handler.extract_answer(response) or ""
                            if not ans:
                                continue
                            if hasattr(handler, "is_voted_answer_correct"):
                                ok = handler.is_voted_answer_correct(ans, test_datas[ti]["ground_truth"])
                            else:
                                ok = handler.is_answer_correct(
                                    handler.format_answer_for_check(ans),
                                    test_datas[ti]["ground_truth"],
                                )
                            if ok:
                                correct += 1
                        test_accs[i] = correct / len(test_datas) if test_datas else 0.0
                    del test_outs

                # Reset all engines to base before next batch
                ray.get([engines[i].collective_rpc.remote("reset_to_base_weights", args=())
                         for i in range(len(batch))])

                # Record
                for i, flt in enumerate(batch):
                    # collective_rpc returns one result per worker — when tp=1 we get a single dict
                    cnt = counts[i]
                    if isinstance(cnt, list):
                        cnt = cnt[0] if cnt else {}
                    row = {
                        "winner_rank": w["rank"],
                        "winner_seed": seed,
                        "winner_sigma": sigma,
                        "winner_recorded_train_reward": full_train_reward_recorded,
                        "filter_name": flt["name"],
                        "filter_spec": {k: flt.get(k) for k in
                                        ["include_substrings", "exclude_substrings", "layer_min", "layer_max"]},
                        "n_perturbed_params": cnt.get("n_perturbed") if isinstance(cnt, dict) else None,
                        "n_skipped_params": cnt.get("n_skipped") if isinstance(cnt, dict) else None,
                        "ablated_train_reward": float(rewards[i]),
                        "delta_vs_base": float(rewards[i]) - float(base_reward),
                        "delta_vs_full": float(rewards[i]) - full_train_reward_recorded,
                        "ablated_test_accuracy": test_accs[i],
                    }
                    results_matrix.append(row)
                    flag = ""
                    if rewards[i] > base_reward + 0.01:
                        flag = " *better than base*"
                    elif rewards[i] < base_reward - 0.01:
                        flag = " (degrades)"
                    print(f"  filter={flt['name']:<22} "
                          f"perturbed={(cnt.get('n_perturbed') if isinstance(cnt, dict) else '?'):<5} "
                          f"reward={rewards[i]:.4f} "
                          f"Δ_base={rewards[i]-base_reward:+.4f}{flag}")

                # Persist incrementally
                with open(os.path.join(out_dir, "results.json"), "w") as f:
                    json.dump({
                        "baseline_dir": args.baseline_dir,
                        "base_model": base_model,
                        "dataset": dataset_name,
                        "base_train_reward": float(base_reward),
                        "num_layers": num_layers,
                        "rows": results_matrix,
                    }, f, indent=2)

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ---- Summary ----
        print("\n" + "=" * 70)
        print("AVERAGE RESULTS ACROSS WINNERS")
        print("=" * 70)
        # Aggregate by filter_name across winners
        from collections import defaultdict
        agg = defaultdict(list)
        for r in results_matrix:
            agg[r["filter_name"]].append(r["ablated_train_reward"])

        avg_rows = []
        for name, vals in agg.items():
            avg_rows.append({
                "filter_name": name,
                "mean_train_reward": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n_winners": len(vals),
                "delta_vs_base": float(np.mean(vals)) - float(base_reward),
            })
        avg_rows.sort(key=lambda r: r["mean_train_reward"], reverse=True)

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump({"base_train_reward": float(base_reward), "rows": avg_rows}, f, indent=2)

        # Pretty text summary
        lines = []
        lines.append(f"Base train reward: {base_reward:.4f}")
        lines.append("")
        lines.append(f"{'Filter':<24}{'mean reward':>14}{'Δ vs base':>14}{'std':>10}")
        lines.append("-" * 62)
        for r in avg_rows:
            lines.append(f"{r['filter_name']:<24}"
                         f"{r['mean_train_reward']:>14.4f}"
                         f"{r['delta_vs_base']:>+14.4f}"
                         f"{r['std']:>10.4f}")
        text = "\n".join(lines)
        print("\n" + text)
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(text + "\n")

        print(f"\nWrote results to {out_dir}/")

    finally:
        cleanup_engines(engines, pgs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
