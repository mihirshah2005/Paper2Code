import json
import os
import sys
import copy
from tqdm import tqdm
from utils import extract_planning, content_to_json, print_response
from pathlib import Path
import openai
import argparse

MAX_FEEDBACK_ITERATIONS = 3

parser = argparse.ArgumentParser()
parser.add_argument('--paper_name', type=str)
parser.add_argument('--gpt_version',type=str, default="o3-mini")
parser.add_argument('--paper_format', type=str, default="JSON", choices=["JSON", "LaTeX"])
parser.add_argument('--pdf_json_path', type=str)
parser.add_argument('--pdf_latex_path', type=str)
parser.add_argument('--output_dir', type=str, default="")
args = parser.parse_args()

paper_name = args.paper_name
paper_format = args.paper_format
gpt_version = args.gpt_version
pdf_json_path = args.pdf_json_path
pdf_latex_path = args.pdf_latex_path
output_dir = args.output_dir

gpt_version = "o3-mini"  # or o3-mini
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def api_call(msg):
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
    return completion.choices[0].message.content


if paper_format == "JSON":
    with open(f'{pdf_json_path}') as f:
        paper_content = json.load(f)
elif paper_format == "LaTeX":
    with open(f'{pdf_latex_path}') as f:
        paper_content = f.read()
else:
    print(f"[ERROR] Invalid paper format. Please select either 'JSON' or 'LaTeX.")
    sys.exit(0)

with open(f'{output_dir}/planning_config.yaml') as f:
    config_yaml = f.read()

context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

if os.path.exists(f'{output_dir}/task_list.json'):
    with open(f'{output_dir}/task_list.json') as f:
        task_list = json.load(f)
else:
    task_list = content_to_json(context_lst[2])

if 'Task list' in task_list:
    todo_file_lst = task_list['Task list']
elif 'task_list' in task_list:
    todo_file_lst = task_list['task_list']
elif 'task list' in task_list:
    todo_file_lst = task_list['task list']
else:
    print(f"[ERROR] 'Task list' does not exist. Please re-generate the planning.")
    sys.exit(0)

if 'Logic Analysis' in task_list:
    logic_analysis = task_list['Logic Analysis']
elif 'logic_analysis' in task_list:
    logic_analysis = task_list['logic_analysis']
elif 'logic analysis' in task_list:
    logic_analysis = task_list['logic analysis']
else:
    print(f"[ERROR] 'Logic Analysis' does not exist. Please re-generate the planning.")
    sys.exit(0)

done_file_lst = ['config.yaml']
logic_analysis_dict = {desc[0]: desc[1] for desc in logic_analysis}

analysis_msg = [
    {"role": "system", "content": f"""You are an expert researcher, strategic analyzer and software engineer with a deep understanding of experimental design and reproducibility in scientific research.
You will receive a research paper in {paper_format} format, an overview of the plan, a design in JSON format consisting of \"Implementation approach\", \"File list\", \"Data structures and interfaces\", and \"Program call flow\", followed by a task in JSON format that includes \"Required packages\", \"Required other language third-party packages\", \"Logic Analysis\", and \"Task list\", along with a configuration file named \"config.yaml\".

Your task is to conduct a comprehensive logic analysis to accurately reproduce the experiments and methodologies described in the research paper.
This analysis must align precisely with the paper’s methodology, experimental setup, and evaluation criteria.

1. Align with the Paper: Your analysis must strictly follow the methods, datasets, model configurations, hyperparameters, and experimental setups described in the paper.
2. Be Clear and Structured: Present your analysis in a logical, well-organized, and actionable format that is easy to follow and implement.
3. Prioritize Efficiency: Optimize the analysis for clarity and practical implementation while ensuring fidelity to the original experiments.
4. Follow design: YOU MUST FOLLOW \"Data structures and interfaces\". DONT CHANGE ANY DESIGN. Do not use public member functions that do not exist in your design.
5. REFER TO CONFIGURATION: Always reference settings from the config.yaml file. Do not invent or assume any values—only use configurations explicitly provided.
"""}]

def get_write_msg(todo_file_name, todo_file_desc):
    draft_desc = f"Write the logic analysis in '{todo_file_name}', which is intended for '{todo_file_desc}'."
    if len(todo_file_desc.strip()) == 0:
        draft_desc = f"Write the logic analysis in '{todo_file_name}'."
    return [{
        'role': 'user', 'content': f"""## Paper
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
    

def run_evaluation_on_analysis(todo_file_name, analysis_text):
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
  \"critique_list\": [
    {{
      \"target_func_name\": \"...\",
      \"severity_level\": \"high\" | \"medium\" | \"low\",
      \"critique\": \"Describe the issue clearly and concisely.\"
    }}
  ]
}}

Assign “severity_level” as follows:
- **High**: Missing or incorrect implementation of core methods, loss functions, or experimental components that break reproduction (e.g. main algorithm is wrong or absent).
- **Medium**: Errors in training loops, data preprocessing, or key workflows that significantly affect results but do not fully block reproduction (e.g. bad augmentation, wrong loop structure).
- **Low**: Minor deviations or non critical details (e.g. hyperparameter choices, logging, evaluation scripts) that do not alter core methodology.

Respond only with the JSON.
"""}
    ]
    try:
        eval_response = api_call(eval_prompt)
        start = eval_response.find('{')
        end = eval_response.rfind('}') + 1
        return json.loads(eval_response[start:end])
    except Exception as e:
        print(f"[EVAL ERROR] Failed to parse evaluation response for {todo_file_name}: {e}")
        return None

artifact_output_dir = f'{output_dir}/analyzing_artifacts'
os.makedirs(artifact_output_dir, exist_ok=True)

# -------------------------------
# Feedback loop per file
# -------------------------------

for todo_file_name in tqdm(todo_file_lst):
    if todo_file_name == "config.yaml":
        continue

    print(f"[ANALYZING] {todo_file_name}")

    # ── 1. build a fresh message stack for this file ─────────────
    file_msg = copy.deepcopy(analysis_msg)          # use the global template
    file_msg.extend(get_write_msg(
        todo_file_name,
        logic_analysis_dict.get(todo_file_name, "")
    ))

    # ── 2. make sure debug folder exists (first call only) ───────
    debug_output_dir = Path(output_dir) / "analyzing_artifacts" / "debug_revisions"
    debug_output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = todo_file_name.replace("/", "_")
    max_iterations = MAX_FEEDBACK_ITERATIONS
    for iteration in range(1, max_iterations + 1):
        print(f"\n🔄 Iteration {iteration}: generating analysis...")
        analysis_text = api_call(file_msg)

        # save intermediate revision for inspection
        (debug_output_dir / f"{safe_name}_rev{iteration}.txt").write_text(
            analysis_text, encoding="utf-8"
        )

        # run critique pass
                # ---------- run critique pass & log it -----------------
        crit_json = run_evaluation_on_analysis(todo_file_name, analysis_text) or {}

        # save the raw JSON for this revision
        (debug_output_dir / f"{safe_name}_rev{iteration}_critiques.json").write_text(
            json.dumps(crit_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # robust parsing  (handles dict / string / list)
        if isinstance(crit_json, str):
            try:
                crit_json = json.loads(crit_json)
            except json.JSONDecodeError:
                crit_json = {}

        crit_list = []
        if isinstance(crit_json, dict):
            crit_list = crit_json.get("critique_list", [])
        elif isinstance(crit_json, list):
            crit_list = crit_json                   # already a list

        # normalise severity and filter High / Medium
        hi_med = [
            c for c in crit_list
            if isinstance(c, dict)
            and c.get("severity_level", "").lower() in ("high", "medium")
        ]

        # ---------- console log for quick debugging ------------
        print(f"   ↪︎ {len(crit_list)} total critiques, "
              f"{len(hi_med)} High/Medium")

        if not hi_med:
            print("✅ No high or medium critiques. Stopping loop.")
            break

        if iteration == max_iterations:
            print("⏹️ Max iterations reached. Using latest revision.")
            break

        # append feedback and re-prompt
        file_msg.append({"role": "assistant", "content": analysis_text})
        file_msg.append({
            "role": "user",
            "content": (
                f"The following critiques were raised for `{todo_file_name}`:\n\n"
                f"{json.dumps(hi_med, indent=2)}\n\n"
                "Please revise the analysis to address these critiques."
            )
        })

    # ── 3. save FINAL version (coding.py will consume this) ──────
    artifact_path = Path(output_dir) / "analyzing_artifacts"
    (artifact_path / f"{safe_name}_simple_analysis.txt").write_text(
        analysis_text, encoding="utf-8"
    )

    # wrap in a list so coding.py's [0] access works
    (Path(output_dir) / f"{safe_name}_simple_analysis_response.json").write_text(
        json.dumps([{"text": analysis_text}], ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"✅ Final analysis for {todo_file_name} saved.")
