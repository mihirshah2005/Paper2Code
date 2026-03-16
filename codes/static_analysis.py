"""
static_analysis.py  

Three zero-token feedback channels + merge/filter helpers.
"""

import ast, json, subprocess, sys, os


def ast_check(filepath: str) -> list:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            ast.parse(f.read(), filename=filepath)
        return []
    except SyntaxError as e:
        return [{"target_func_name": _enclosing_func(filepath, e.lineno),
                 "severity_level": "high",
                 "critique": f"SyntaxError line {e.lineno}: {e.msg}",
                 "source": "ast-parse"}]
    except Exception as e:
        return [{"target_func_name": "<module>", "severity_level": "high",
                 "critique": f"Parse failed: {e}", "source": "ast-parse"}]


def pylint_check(filepath: str, timeout: int = 60) -> list:
    HIGH = {"E0001", "E0401", "E0602"}
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pylint", "--disable=all",
             "--enable=E0001,E0401,E0602,E1101",
             "--output-format=json", filepath],
            capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    try:
        items = json.loads(r.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        mid = it.get("message-id", "")
        out.append({"target_func_name": it.get("obj", "") or "<module>",
                     "severity_level": "high" if mid in HIGH else "medium",
                     "critique": f"[{mid}] {it.get('message','')} (line {it.get('line','?')})",
                     "source": "pylint"})
    return out


def import_probe(filepath: str, project_root: str = None, timeout: int = 15) -> list:
    ap = os.path.abspath(filepath)
    env = os.environ.copy()
    if project_root:
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")
    code = (f"import importlib.util,sys\n"
            f"s=importlib.util.spec_from_file_location('_p',r'{ap}')\n"
            f"if s and s.loader:\n m=importlib.util.module_from_spec(s)\n s.loader.exec_module(m)\n"
            f"else: sys.exit(1)\n")
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return [{"target_func_name": os.path.basename(filepath),
                 "severity_level": "medium",
                 "critique": f"Import timed out ({timeout}s)", "source": "exec-probe"}]
    if r.returncode != 0:
        err = (r.stderr.strip().split("\n") or ["Unknown"])[-1]
        sev = "medium" if any(k in err for k in ("FileNotFoundError", "RuntimeError")) else "high"
        return [{"target_func_name": os.path.basename(filepath),
                 "severity_level": sev, "critique": f"Import failed: {err}",
                 "source": "exec-probe"}]
    return []


def merge_critiques(*lists) -> list:
    PRI = {"ast-parse": 0, "pylint": 1, "exec-probe": 2, "llm-judge": 3}
    seen = {}
    for cl in lists:
        for c in cl:
            key = (c.get("target_func_name", ""), c.get("severity_level", ""))
            if key not in seen or PRI.get(c.get("source","llm-judge"),99) < PRI.get(seen[key].get("source","llm-judge"),99):
                seen[key] = c
    return list(seen.values())


def filter_high(critiques: list) -> list:
    return [c for c in critiques if c.get("severity_level") == "high"]


def _enclosing_func(fp, ln):
    if not ln: return "<module>"
    try:
        with open(fp) as f:
            tree = ast.parse(f.read())
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(n, "end_lineno") and n.lineno <= ln <= n.end_lineno:
                    return n.name
    except Exception: pass
    return "<module>"