import json
import os
import sys
import copy
from tqdm import tqdm
from utils import extract_planning, content_to_json, print_response
from pathlib import Path
import openai
import argparse

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
import openai
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

for todo_file_name in tqdm(todo_file_lst):
    if todo_file_name == "config.yaml":
        continue

    responses = []
    trajectories = copy.deepcopy(analysis_msg)

    if todo_file_name not in logic_analysis_dict:
        logic_analysis_dict[todo_file_name] = ""

    instruction_msg = get_write_msg(todo_file_name, logic_analysis_dict[todo_file_name])
    trajectories.extend(instruction_msg)

    completion = api_call(trajectories)

    eval_json = run_evaluation_on_analysis(todo_file_name, completion)
    crit_path = Path(output_dir) / f"{todo_file_name}_critiques.json"
    with open(crit_path, "w") as f:
        json.dump(eval_json, f, indent=2)

    actionable = [c for c in eval_json.get("critique_list", []) if c["severity_level"] in ("high", "medium")]
    if actionable:
        feedback_msg = [{"role": "user", "content": f"""The following critiques were raised for your analysis of `{todo_file_name}`:

{json.dumps(actionable, indent=2)}

Please revise the analysis to address the issues."""}]
        trajectories.extend(feedback_msg)
        completion = api_call(trajectories)

    completion_json = {'text': completion}
    print_response(completion_json, is_llm=True)
    responses.append(completion_json)
    trajectories.append({'role': 'assistant', 'content': completion})

    with open(f'{artifact_output_dir}/{todo_file_name}_simple_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(completion)

    done_file_lst.append(todo_file_name)
    safe_name = todo_file_name.replace("/", "_")
    with open(f'{output_dir}/{safe_name}_simple_analysis_response.json', 'w', encoding='utf-8') as f:
        json.dump(responses, f)
    with open(f'{output_dir}/{safe_name}_simple_analysis_trajectories.json', 'w', encoding='utf-8') as f:
        json.dump(trajectories, f)
