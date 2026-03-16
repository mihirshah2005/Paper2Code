"""
collect_results.py

After running all experiments + eval.py for each, run this script to collect
all results into clean tables for the paper.

Usage:
    python collect_results.py --base_dir outputs/ --papers transformer,vdc,curbench,gem,fit,sea

It will look for:
    outputs/<paper>/experiments/<mode>_<format>/experiment_summary_*.json   (token counts)
    outputs/<paper>/experiments/<mode>_<format>/eval_results/*_eval_*.json  (rubric scores)
"""

import json
import os
import glob
import argparse
from pathlib import Path


def find_eval_scores(eval_dir, paper_name):
    """Find eval JSON files and extract average score."""
    pattern = os.path.join(eval_dir, f"{paper_name}_eval_*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    # Take the most recent one
    files.sort(key=os.path.getmtime, reverse=True)
    with open(files[0]) as f:
        data = json.load(f)
    return data.get("eval_result", {}).get("score", None)


def find_token_summary(exp_dir, mode, fmt):
    """Find experiment_summary JSON and extract token counts."""
    path = os.path.join(exp_dir, f"experiment_summary_{mode}_{fmt}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        "prompt_tokens": data.get("total_prompt_tokens", 0),
        "completion_tokens": data.get("total_completion_tokens", 0),
        "total_tokens": data.get("total_tokens", 0),
        "per_file": data.get("per_file_iterations", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base output directory containing per-paper folders")
    parser.add_argument("--papers", type=str, required=True,
                        help="Comma-separated paper names")
    args = parser.parse_args()

    papers = [p.strip() for p in args.papers.split(",")]
    experiments = [
        ("llm_only", "json"),
        ("llm_only", "freetext"),
        ("static_only", "json"),
        ("multi_signal", "json"),
    ]

    # ---- Table 1: Rubric scores across configurations ----
    print("\n" + "=" * 70)
    print("TABLE: Rubric Scores (out of 5)")
    print("=" * 70)
    header = f"{'Paper':<25} {'Baseline':>8} {'LLM-JSON':>8} {'LLM-Free':>8} {'Static':>8} {'Multi':>8}"
    print(header)
    print("-" * 70)

    for paper in papers:
        base_dir = os.path.join(args.base_dir, paper)

        # Try to find baseline score from the original eval
        baseline_score = "??"
        baseline_eval_dir = os.path.join(base_dir, "eval_results")
        if os.path.exists(baseline_eval_dir):
            s = find_eval_scores(baseline_eval_dir, paper)
            if s is not None:
                baseline_score = f"{s:.2f}"

        scores = {}
        for mode, fmt in experiments:
            exp_name = f"{mode}_{fmt}"
            eval_dir = os.path.join(base_dir, "experiments", exp_name, "eval_results")
            s = find_eval_scores(eval_dir, paper)
            scores[exp_name] = f"{s:.2f}" if s is not None else "??"

        print(f"{paper:<25} {baseline_score:>8} {scores.get('llm_only_json','??'):>8} "
              f"{scores.get('llm_only_freetext','??'):>8} {scores.get('static_only_json','??'):>8} "
              f"{scores.get('multi_signal_json','??'):>8}")

    # ---- Table 2: Token costs ----
    print("\n" + "=" * 70)
    print("TABLE: Token Consumption (analysis phase)")
    print("=" * 70)
    header = f"{'Paper':<25} {'LLM-JSON':>12} {'LLM-Free':>12} {'Static':>12} {'Multi':>12}"
    print(header)
    print("-" * 70)

    for paper in papers:
        base_dir = os.path.join(args.base_dir, paper)
        tokens = {}
        for mode, fmt in experiments:
            exp_name = f"{mode}_{fmt}"
            exp_dir = os.path.join(base_dir, "experiments", exp_name)
            t = find_token_summary(exp_dir, mode, fmt)
            if t:
                tokens[exp_name] = f"{t['total_tokens'] / 1000:.1f}K"
            else:
                tokens[exp_name] = "??"

        print(f"{paper:<25} {tokens.get('llm_only_json','??'):>12} "
              f"{tokens.get('llm_only_freetext','??'):>12} {tokens.get('static_only_json','??'):>12} "
              f"{tokens.get('multi_signal_json','??'):>12}")

    # ---- Table 3: Per-iteration critique counts ----
    print("\n" + "=" * 70)
    print("TABLE: Per-Iteration High-Severity Critique Counts (multi_signal)")
    print("=" * 70)

    for paper in papers:
        exp_dir = os.path.join(args.base_dir, paper, "experiments", "multi_signal_json")
        t = find_token_summary(exp_dir, "multi_signal", "json")
        if not t or not t.get("per_file"):
            print(f"  {paper}: no data found")
            continue
        print(f"\n  {paper}:")
        for fname, iters in t["per_file"].items():
            iter_str = " -> ".join(
                f"iter{it['iteration']}:H={it['high']},M={it['medium']}"
                for it in iters
            )
            print(f"    {fname}: {iter_str}")

    print("\n" + "=" * 70)
    print("Copy the tables above into your paper (replace ?? with actual values)")
    print("=" * 70)


if __name__ == "__main__":
    main()