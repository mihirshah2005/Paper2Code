"""
2_analyzing_experiments.py

Drop-in replacement for 2_analyzing.py that supports all experiment modes.
Adds: --mode (llm_only | static_only | multi_signal)
      --feedback_format (json | freetext)

Also logs per-iteration rubric scores and token counts.

Usage examples:
  # Experiment 1: LLM-only with JSON (your original setup)
  python 2_analyzing_experiments.py --mode llm_only --feedback_format json ...

  # Experiment 2: LLM-only with free-text
  python 2_analyzing_experiments.py --mode llm_only --feedback_format freetext ...

  # Experiment 3: Static-only (no LLM judge, only ast+pylint+probe)
  python 2_analyzing_experiments.py --mode static_only ...

  # Experiment 4: Multi-signal (all channels combined)
  python 2_analyzing_experiments.py --mode multi_signal --feedback_format json ...
"""

import json
import os
import sys
import copy
from tqdm import tqdm
from utils import extract_planning, content_to_json, print_response
from pathlib import Path
import openai
import argparse
import time

# ---- import the static analysis module ----
from static_analysis import ast_check, pylint_check, import_probe, merge_critiques, filter_high

MAX_FEEDBACK_ITERATIONS = 3

parser = argparse.ArgumentParser()
parser.add_argument('--paper_name', type=str)
parser.add_argument('--gpt_version', type=str, default="o3-mini")
parser.add_argument('--paper_format', type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str)
parser.add_argument('--pdf_latex_path', type=str)
parser.add_argument('--output_dir', type=str, default="")

# ---- NEW FLAGS ----
parser.add_argument('--mode', type=str, default="llm_only",
                    choices=["llm_only", "static_only", "multi_signal"],
                    help="Feedback mode: llm_only, static_only, or multi_signal")
parser.add_argument('--feedback_format', type=str, default="json",
                    choices=["json", "freetext"],
                    help="LLM critique format: json (structured) or freetext")
parser.add_argument('--output_repo_dir', type=str, default="",
                    help="Path to the generated repo (for static analysis on .py files)")

args = parser.parse_args()

paper_name = args.paper_name
paper_format = args.paper_format
gpt_version = args.gpt_version
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir
mode = args.mode
feedback_format = args.feedback_format
output_repo_dir = args.output_repo_dir

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Token tracking ----
total_prompt_tokens = 0
total_completion_tokens = 0


def api_call(msg):
    """Call the API and track tokens."""
    global total_prompt_tokens, total_completion_tokens

    if "o3-mini" in gpt_version:
        completion = client.chat.completions.create(
            model=gpt_version,
            messages=msg,
            reasoning_effort="high"
        )
    else:
        completion = client.chat.completions.create(
            model=gpt_version,
            messages=msg
        )

    # Track tokens
    if hasattr(completion, 'usage') and completion.usage:
        total_prompt_tokens += completion.usage.prompt_tokens
        total_completion_tokens += completion.usage.completion_tokens

    return completion.choices[0].message.content


# ---- Load paper content ----
if paper_format == "JSON":
    with open(pdf_json_path) as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(pdf_latex_path) as f:
        paper_content = f.read()
else:
    print("[ERROR] Invalid paper format.")
    sys.exit(0)

with open(f'{output_dir}/planning_config.yaml') as f:
    config_yaml = f.read()

context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

if os.path.exists(f'{output_dir}/task_list.json'):
    with open(f'{output_dir}/task_list.json') as f:
        task_list = json.load(f)
else:
    task_list = content_to_json(context_lst[2])

# Robust key lookup
for key in ['Task list', 'task_list', 'task list']:
    if key in task_list:
        todo_file_lst = task_list[key]
        break
else:
    print("[ERROR] 'Task list' not found.")
    sys.exit(0)

for key in ['Logic Analysis', 'logic_analysis', 'logic analysis']:
    if key in task_list:
        logic_analysis = task_list[key]
        break
else:
    print("[ERROR] 'Logic Analysis' not found.")
    sys.exit(0)

logic_analysis_dict = {desc[0]: desc[1] for desc in logic_analysis}

# ---- System message ----
analysis_msg = [
    {"role": "system", "content": f"""You are an expert researcher, strategic analyzer and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in {paper_format} format, an overview of the plan, a design in JSON format consisting of "Implementation approach", "File list", "Data structures and interfaces", and "Program call flow", followed by a task in JSON format that includes "Required packages", "Required other language third-party packages", "Logic Analysis", and "Task list", along with a configuration file named "config.yaml".

Your task is to conduct a comprehensive logic analysis to accurately reproduce the experiments and methodologies described in the research paper.
This analysis must align precisely with the paper's methodology, experimental setup, and evaluation criteria.

1. Align with the Paper: Your analysis must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present your analysis in a logical, well-organized, and actionable format that is easy to follow and implement.
3. Prioritize Efficiency: Optimize the analysis for clarity and practical implementation while ensuring fidelity to the original experiments.
4. Follow design: YOU MUST FOLLOW "Data structures and interfaces". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. REFER TO CONFIGURATION: Always reference settings from the config.yaml file. Do not invent or assume any values.
"""}]


def get_write_msg(todo_file_name, todo_file_desc):
    draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
    if len(todo_file_desc.strip()) == 0:
        draft_desc = f"Write the logic analysis in '{todo_file_name}'."
    return [{'role': 'user', 'content': f"""## Paper
{paper_content}

-----

## Overview of the plan
{context_lst[0]}

-----

## Design
{context_lst[1]}

-----

## Task
{context_lst[2]}

-----

## Configuration file
```yaml
{config_yaml}
```
-----

## Instruction
Conduct a Logic Analysis to assist in writing the code, based on the paper, the plan, the design, the task and the previously specified configuration file (config.yaml).
You DON'T need to provide the actual code yet; focus on a thorough, clear analysis.

{draft_desc}

-----

## Logic Analysis: {todo_file_name}"""}]


def run_llm_evaluation_json(todo_file_name, analysis_text):
    """Original structured JSON evaluation (your existing code)."""
    eval_prompt = [
        {"role": "system", "content": "You are a reviewer checking the correctness and completeness of a logic analysis for a scientific implementation."},
        {"role": "user", "content": f"""Here is the logic analysis for `{todo_file_name}`:

{analysis_text}

Please identify issues with the logic, especially if it:
- contradicts the original paper,
- fails to follow the configuration file,
- breaks the planned design,
- uses undefined components.

Output a list of critiques in JSON format like this:
{{
  "critique_list": [
    {{
      "target_func_name": "...",
      "severity_level": "high" | "medium" | "low",
      "critique": "Describe the issue clearly and concisely."
    }}
  ]
}}

Assign "severity_level" as follows:
- **High**: Missing or incorrect implementation of core methods, loss functions, or experimental components that break reproduction.
- **Medium**: Errors in training loops, data preprocessing, or key workflows that significantly affect results but do not fully block reproduction.
- **Low**: Minor deviations or non critical details that do not alter core methodology.

Respond only with the JSON."""}
    ]
    try:
        eval_response = api_call(eval_prompt)
        start = eval_response.find('{')
        end = eval_response.rfind('}') + 1
        result = json.loads(eval_response[start:end])
        # Tag source
        for c in result.get("critique_list", []):
            c["source"] = "llm-judge"
        return result
    except Exception as e:
        print(f"[EVAL ERROR] JSON parse failed for {todo_file_name}: {e}")
        return None


def run_llm_evaluation_freetext(todo_file_name, analysis_text):
    """Free-text evaluation (for ablation comparison)."""
    eval_prompt = [
        {"role": "system", "content": "You are a reviewer checking the correctness and completeness of a logic analysis for a scientific implementation."},
        {"role": "user", "content": f"""Here is the logic analysis for `{todo_file_name}`:

{analysis_text}

Review this analysis against the rubric below. Describe any issues you find in plain English,
noting which functions are affected and how severe each issue is (high, medium, or low).

Severity guide:
- High: Missing or incorrect core methods, loss functions, or components that break reproduction.
- Medium: Errors in training loops or preprocessing that affect results but do not fully block reproduction.
- Low: Minor deviations that do not alter core methodology.

Write your critique as plain text. Do NOT use JSON."""}
    ]
    try:
        eval_response = api_call(eval_prompt)
        # Parse free-text into critique-like dicts heuristically
        critiques = []
        text_lower = eval_response.lower()
        # Check if any high-severity language is present
        if "high" in text_lower and any(w in text_lower for w in
                ["missing", "incorrect", "wrong", "absent", "broken", "critical", "block"]):
            critiques.append({
                "target_func_name": "<parsed-from-freetext>",
                "severity_level": "high",
                "critique": eval_response[:500],
                "source": "llm-judge-freetext",
            })
        elif "medium" in text_lower:
            critiques.append({
                "target_func_name": "<parsed-from-freetext>",
                "severity_level": "medium",
                "critique": eval_response[:500],
                "source": "llm-judge-freetext",
            })
        # If we cannot parse severity, treat as no high-severity issues
        return {"critique_list": critiques, "raw_freetext": eval_response}
    except Exception as e:
        print(f"[EVAL ERROR] Free-text eval failed for {todo_file_name}: {e}")
        return None


def run_static_analysis(todo_file_name, repo_dir):
    """
    Run ast_check + pylint_check + import_probe on the actual .py file.
    If the file does not exist yet (analysis phase generates specs, not code),
    we skip and return empty. For files that DO exist in repo_dir, we check them.
    """
    if not repo_dir:
        return []

    filepath = os.path.join(repo_dir, todo_file_name)
    if not os.path.exists(filepath):
        return []

    c_ast = ast_check(filepath)
    c_lint = pylint_check(filepath)
    c_exec = import_probe(filepath, project_root=repo_dir)

    return merge_critiques(c_ast, c_lint, c_exec)


def get_critiques(todo_file_name, analysis_text, mode, feedback_format, repo_dir):
    """
    Run the appropriate feedback channels based on mode.
    Returns (all_critiques_list, high_or_medium_list).
    """
    llm_critiques = []
    static_critiques = []

    if mode in ("llm_only", "multi_signal"):
        if feedback_format == "json":
            result = run_llm_evaluation_json(todo_file_name, analysis_text)
        else:
            result = run_llm_evaluation_freetext(todo_file_name, analysis_text)

        if result and isinstance(result, dict):
            llm_critiques = result.get("critique_list", [])

    if mode in ("static_only", "multi_signal"):
        static_critiques = run_static_analysis(todo_file_name, repo_dir)

    if mode == "multi_signal":
        all_critiques = merge_critiques(static_critiques, llm_critiques)
    elif mode == "static_only":
        all_critiques = static_critiques
    else:
        all_critiques = llm_critiques

    hi_med = [c for c in all_critiques
              if isinstance(c, dict)
              and c.get("severity_level", "").lower() in ("high", "medium")]

    return all_critiques, hi_med


# ---- Output directories ----
artifact_output_dir = f'{output_dir}/analyzing_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

debug_output_dir = Path(output_dir) / "analyzing_artifacts" / "debug_revisions"
debug_output_dir.mkdir(parents=True, exist_ok=True)

# ---- Per-iteration score log ----
iteration_log = {}  # {file_name: [score_iter0, score_iter1, ...]}

# ---- Main loop ----
for todo_file_name in tqdm(todo_file_lst):
    if todo_file_name == "config.yaml":
        continue

    print(f"\n[ANALYZING] {todo_file_name}  (mode={mode}, format={feedback_format})")

    file_msg = copy.deepcopy(analysis_msg)
    file_msg.extend(get_write_msg(
        todo_file_name,
        logic_analysis_dict.get(todo_file_name, "")
    ))

    safe_name = todo_file_name.replace("/", "_")
    per_iter_scores = []

    for iteration in range(1, MAX_FEEDBACK_ITERATIONS + 1):
        print(f"\n  Iteration {iteration}: generating analysis...")
        analysis_text = api_call(file_msg)

        # Save revision
        (debug_output_dir / f"{safe_name}_rev{iteration}_{mode}_{feedback_format}.txt").write_text(
            analysis_text, encoding="utf-8"
        )

        # Get critiques based on mode
        all_crits, hi_med = get_critiques(
            todo_file_name, analysis_text, mode, feedback_format, output_repo_dir
        )

        # Save critiques
        (debug_output_dir / f"{safe_name}_rev{iteration}_{mode}_{feedback_format}_critiques.json").write_text(
            json.dumps({"all": all_crits, "high_medium": hi_med}, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Log per-iteration score (we use critique count as a proxy;
        # for actual rubric scores, run eval.py after each iteration)
        n_high = len([c for c in all_crits if c.get("severity_level", "").lower() == "high"])
        n_med = len([c for c in all_crits if c.get("severity_level", "").lower() == "medium"])
        n_low = len([c for c in all_crits if c.get("severity_level", "").lower() == "low"])
        per_iter_scores.append({
            "iteration": iteration,
            "high": n_high,
            "medium": n_med,
            "low": n_low,
            "total_critiques": len(all_crits),
        })

        print(f"    {len(all_crits)} total critiques, {len(hi_med)} high/medium")

        if not hi_med:
            print("  No high/medium critiques. Stopping early.")
            break

        if iteration == MAX_FEEDBACK_ITERATIONS:
            print("  Max iterations reached.")
            break

        # Append feedback for next iteration
        file_msg.append({"role": "assistant", "content": analysis_text})
        file_msg.append({
            "role": "user",
            "content": (
                f"The following critiques were raised for `{todo_file_name}`:\n\n"
                f"{json.dumps(hi_med, indent=2)}\n\n"
                "Please revise the analysis to address these critiques."
            )
        })

    iteration_log[todo_file_name] = per_iter_scores

    # Save final analysis (compatible with 3_coding.py)
    artifact_path = Path(output_dir) / "analyzing_artifacts"
    (artifact_path / f"{safe_name}_simple_analysis.txt").write_text(
        analysis_text, encoding="utf-8"
    )
    (Path(output_dir) / f"{safe_name}_simple_analysis_response.json").write_text(
        json.dumps([{"text": analysis_text}], ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"  Final analysis for {todo_file_name} saved.")


# ---- Save experiment summary ----
summary = {
    "paper_name": paper_name,
    "mode": mode,
    "feedback_format": feedback_format,
    "max_iterations": MAX_FEEDBACK_ITERATIONS,
    "total_prompt_tokens": total_prompt_tokens,
    "total_completion_tokens": total_completion_tokens,
    "total_tokens": total_prompt_tokens + total_completion_tokens,
    "per_file_iterations": iteration_log,
}

summary_path = Path(output_dir) / f"experiment_summary_{mode}_{feedback_format}.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 60)
print(f"EXPERIMENT COMPLETE: mode={mode}, format={feedback_format}")
print(f"Total tokens: {summary['total_tokens']:,} "
      f"(prompt: {total_prompt_tokens:,}, completion: {total_completion_tokens:,})")
print(f"Summary saved to: {summary_path}")
print("=" * 60)