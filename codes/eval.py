from openai import OpenAI
import json
import os
import sys
import argparse
from utils import read_python_files, extract_planning, content_to_json, \
        num_tokens_from_messages, read_all_files, extract_json_from_string, get_now_str, print_log_cost

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>  ALL THIS NEW STUFF I ADDED LATER (FOR MYSELF TO NOT GET CONFUSED)  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def infer_stage(file_name: str, func_name: str, critique: str) -> str:
    text = f"{file_name} {func_name} {critique}".lower()
    if any(k in text for k in ('config', 'plan', 'prompt', 'hyper', 'dataset')):
        return 'planning'
    if any(k in text for k in ('forward', 'train', 'loss', 'optimizer',
                               'preprocess', 'layer', 'mask')):
        return 'analyzing'
    if any(k in text for k in ('eval', 'metric', 'validate', 'test',
                               'plot', 'checkpoint')):
        return 'evaluating'
    return 'unknown'
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>  NEW STUFF ENDS HERE  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def api_call(request_json):
    completion = client.chat.completions.create(**request_json)
    return completion

def main(args):

    paper_name       = args.paper_name
    pdf_json_path    = args.pdf_json_path
    output_dir       = args.output_dir
    target_repo_dir  = args.target_repo_dir
    eval_result_dir  = args.eval_result_dir
    gpt_version      = args.gpt_version
    generated_n      = args.generated_n
    data_dir         = args.data_dir
    eval_type        = args.eval_type
    is_papercoder    = True if args.papercoder else False
    gold_repo_dir    = args.gold_repo_dir

    # paper
    with open(pdf_json_path) as f:
        paper_json = json.load(f)

    codes = ""
    if is_papercoder:
        # python files
        target_files_dict = read_python_files(target_repo_dir)

        # configuration
        with open(f'{output_dir}/planning_config.yaml') as f:
            config_yaml = f.read()

        context_lst = extract_planning(f'{output_dir}/planning_trajectories.json')

        if os.path.exists(f'{output_dir}/task_list.json'):
            with open(f'{output_dir}/task_list.json') as f:
                task_list = json.load(f)
        else:
            task_list = content_to_json(context_lst[2])

        todo_file_lst = task_list['Task list']
        for todo_file in todo_file_lst:
            if todo_file.endswith(".yaml"):
                continue
            codes += f"```python\n## File name: {todo_file}\n{target_files_dict[todo_file]}\n```\n\n"

        codes += f"```yaml\n## File name: config.yaml\n{config_yaml}\n```\n\n"
    else:
        target_files_dict = read_all_files(
            target_repo_dir,
            allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".bash"],
            is_print=False
        )
        for file_name, code in target_files_dict.items():
            codes += f"```## File name: {file_name}\n{code}\n```\n\n"

    prompt = open(f"{data_dir}/prompts/{eval_type}.txt").read()
    cur_prompt = prompt.replace('{{Paper}}', f"{paper_json}").replace('{{Code}}', codes)

    # reference-based
    if eval_type == "ref_based" and len(gold_repo_dir) > 0:
        all_files_dict = read_all_files(
            gold_repo_dir,
            allowed_ext=[".py", ".yaml", ".yml", ".md", ".sh", ".bash"],
            is_print=False
        )

        goldcodes, gold_cnt = "", 0
        if len(args.selected_file_path) > 0:
            with open(args.selected_file_path) as f:
                selected_file_lst = [ln.strip() for ln in f.readlines()]

            for all_file, all_file_code in all_files_dict.items():
                if all_file not in selected_file_lst:
                    continue
                goldcodes += f"```## File name: {all_file}\n{all_file_code}\n```\n\n"
                gold_cnt += 1
        else:
            for all_file, all_file_code in all_files_dict.items():
                goldcodes += f"```## File name: {all_file}\n{all_file_code}\n```\n\n"
                gold_cnt += 1

        cur_prompt = cur_prompt.replace('{{GoldCode}}', goldcodes)

    msg = [{"role": "system", "content": cur_prompt}]

    try:
        num_tokens = num_tokens_from_messages(msg)
    except Exception as e:
        print(f"[WARNING] Token-counting failed for {paper_name}.")
        print(e, "-" * 40)
        num_tokens = 0

    if num_tokens > 128_000:
        print(f"[ERROR] {paper_name}: prompt exceeds 128 k tokens")
        sys.exit(0)

    if "o3-mini" in gpt_version and generated_n > 8:
        print("[WARNING] o3-mini does not support n > 8; resetting to 8.")
        generated_n = 8

    request_json = {
        "model": gpt_version,
        "messages": msg,
        **({"reasoning_effort": "high"} if "o3-mini" in gpt_version else
           {"temperature": 1, "frequency_penalty": 0,
            "presence_penalty": 0, "stop": None}),
        "n": generated_n
    }

    completion      = api_call(request_json)
    completion_json = json.loads(completion.model_dump_json())

    score_key     = "score"
    rationale_key = "critique_list"

    all_scores, rationales = [], []
    for n in range(generated_n):
        choice  = completion_json['choices'][n]
        output  = choice['message']['content'].strip()

        try:
            output_json2 = json.loads(output)

            # >>>>>>>>>>>>>>>>>>>>>>>>>  I MODIFIED SOME STUFF HERE TO ADD THE STAGE FIELD  <<<<<<<<<<<<<<<<<<<<<<<<<<<
            score = int(output_json2[score_key])
            critique_entries = output_json2[rationale_key]
            if isinstance(critique_entries, str):
                critique_entries = json.loads(critique_entries)

            # add stage labels
            for c in critique_entries:
                c['stage'] = infer_stage(
                    c.get('file_name', ''),
                    c.get('func_name', ''),
                    c.get('critique', '')
                )

            rationale = json.dumps(critique_entries, ensure_ascii=False)
            # >>>>>>>>>>>>>>>>>>>>>>>>>  END OF MODS  <<<<<<<<<<<<<<<<<<<<<<<<<<<

        except Exception:
            try:
                output_json2 = json.loads(extract_json_from_string(output))

                # >>>>>>>>>>>>>>>>>>>>>>> AGAIN, MODIFIED SOME STUFF (NOT MUCH THOUGH) <<<<<<<<<<<<<<<<<<<<<<<<<
                score = int(output_json2[score_key])
                critique_entries = output_json2[rationale_key]
                if isinstance(critique_entries, str):
                    critique_entries = json.loads(critique_entries)

                for c in critique_entries:
                    c['stage'] = infer_stage(
                        c.get('file_name', ''),
                        c.get('func_name', ''),
                        c.get('critique', '')
                    )

                rationale = json.dumps(critique_entries, ensure_ascii=False)
                # >>>>>>>>>>>>>>>>>>>>>>>  END OF MOD  <<<<<<<<<<<<<<<<<<<<<<<<<

            except Exception as e2:
                print("[WARNING] Invalid response → parsing error")
                print(e2, "-" * 40)
                continue

        if not 1 <= score <= 5:
            print(f"[WARNING] Invalid score {score} (must be 1-5).")
            continue

        all_scores.append(score)
        rationales.append(rationale)

    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    output_json = {
        "paper_name": paper_name,
        "target_repo_dir": target_repo_dir,
        "eval_type": eval_type,
        "gold_repo_dir": gold_repo_dir,
        "generated_n": generated_n,
        "request_json": request_json,
        "completion_json": completion_json,
        "eval_result": {
            "score": avg_score,
            "valid_n": len(all_scores),
            "scroe_lst": all_scores,
            "rationale_lst": rationales,
        },
    }

    now_str = get_now_str()
    os.makedirs(eval_result_dir, exist_ok=True)
    with open(f"{eval_result_dir}/{paper_name}_eval_{eval_type}_{gpt_version}_{now_str}.json",
              'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    # ---------------  Console summary ---------------
    print("\n" + "=" * 40)
    print("🌟 Evaluation Summary 🌟")
    print(f"📄 Paper name:          {paper_name}")
    print(f"🧪 Evaluation type:     {eval_type}")
    print(f"📁 Target repo dir:     {target_repo_dir}")
    print("📊 Evaluation result:")
    print(f"\t📈 Score:  {avg_score:.4f}")
    print(f"\t✅ Valid:  {output_json['eval_result']['valid_n']}/{generated_n}")
    print("=" * 40 + "\n")

    print_log_cost(completion_json, gpt_version,
                   f"[Evaluation] {paper_name} - {eval_type}",
                   output_dir, 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument('--paper_name',        type=str)
    ap.add_argument('--pdf_json_path',     type=str)
    ap.add_argument('--data_dir',          type=str, default="../data")
    ap.add_argument('--output_dir',        type=str)

    ap.add_argument('--target_repo_dir',   type=str)
    ap.add_argument('--gold_repo_dir',     type=str, default="")
    ap.add_argument('--eval_result_dir',   type=str)

    ap.add_argument('--eval_type',         type=str, default="ref_free",
                    choices=["ref_free", "ref_based"])

    ap.add_argument('--generated_n',       type=int, default=8)
    ap.add_argument('--gpt_version',       type=str, default="o3-mini")

    ap.add_argument('--selected_file_path', type=str, default="")
    ap.add_argument('--papercoder',         action="store_true")

    args = ap.parse_args()
    main(args)

# =======================================================================
