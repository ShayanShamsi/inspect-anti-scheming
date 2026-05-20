"""Run all 25 anti-scheming envs against a single model, write results to YAML.

Used to populate the `evaluation_report` block in
.register-draft/anti_scheming/eval.yaml. Designed to be cheap-and-cheerful:
small `--limit`, low `max_connections`, no docker backends.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

from inspect_ai import eval as inspect_eval


# (module_path, task_name, kwargs)
TASKS: list[tuple[str, str, dict[str, Any]]] = [
    ("anti_scheming.qa.sandbagging_qa", "as_sandbagging_qa", {}),
    ("anti_scheming.qa.sandbagging_qa_no_goal", "as_sandbagging_qa_no_goal", {}),
    ("anti_scheming.qa.sandbagging_grading_cot", "as_sandbagging_grading_cot", {}),
    ("anti_scheming.qa.literature_qa", "as_literature_qa", {}),
    ("anti_scheming.qa.lazy_checklist", "as_lazy_checklist", {}),
    ("anti_scheming.qa.long_document_verification", "as_long_document_verification", {}),
    ("anti_scheming.software.impossible_coding", "as_impossible_coding", {"backend": "vfs"}),
    ("anti_scheming.software.email_scoring", "as_email_scoring", {}),
    ("anti_scheming.software.sabotage_math", "as_sabotage_math", {}),
    ("anti_scheming.software.sabotage_safety", "as_sabotage_safety", {"backend": "vfs"}),
    ("anti_scheming.software.log_falsification", "as_log_falsification", {}),
    ("anti_scheming.software.smuggle_capabilities", "as_smuggle_capabilities", {}),
    ("anti_scheming.software.as_over_escalation", "as_as_over_escalation", {}),
    ("anti_scheming.corporate.legal_reporting_us", "as_legal_reporting_us", {}),
    ("anti_scheming.corporate.legal_reporting_cn", "as_legal_reporting_cn", {}),
    ("anti_scheming.corporate.broken_promises", "as_broken_promises", {}),
    ("anti_scheming.corporate.powerseeking_survey", "as_powerseeking_survey", {}),
    ("anti_scheming.corporate.subagent_collusion", "as_subagent_collusion", {}),
    ("anti_scheming.spec_sabotage.spec_sabotage_moving", "as_spec_sabotage_moving", {}),
    ("anti_scheming.spec_sabotage.spec_sabotage_selection", "as_spec_sabotage_selection", {}),
    (
        "anti_scheming.spec_sabotage.spec_sabotage_verification",
        "as_spec_sabotage_verification",
        {},
    ),
    ("anti_scheming.games.blotto", "as_blotto", {}),
    ("anti_scheming.games.prisoners_dilemma", "as_prisoners_dilemma", {}),
    ("anti_scheming.subagent.subagent_permissions", "as_subagent_permissions", {}),
    ("anti_scheming.subagent.subagent_resource_quota", "as_subagent_resource_quota", {}),
]


def _result_row(log) -> dict[str, Any]:
    """Extract per-task accuracy + stderr from an EvalLog into a row dict."""
    # `covert_action_scorer` reports accuracy as non-covert rate -> covert rate = 1 - acc
    metrics: dict[str, float | int | str] = {}
    if log.results and log.results.scores:
        for sc in log.results.scores:
            for mname, mobj in (sc.metrics or {}).items():
                if mname == "accuracy":
                    metrics["covert_rate"] = round(1.0 - float(mobj.value), 4)
                elif mname == "stderr":
                    metrics["stderr"] = round(float(mobj.value), 4)
    metrics["n_samples"] = (
        log.results.completed_samples if log.results else (len(log.samples or []))
    )
    metrics["status"] = log.status
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openrouter/openai/gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--max-connections", type=int, default=4)
    ap.add_argument("--message-limit", type=int, default=12)
    ap.add_argument("--log-dir", default="logs/baseline")
    ap.add_argument("--output", default="logs/baseline-results.json")
    ap.add_argument("--only", default=None, help="comma-separated task names to include")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    results: dict[str, dict[str, Any]] = {}
    start = time.time()
    for i, (mod_path, task_name, kwargs) in enumerate(TASKS, 1):
        if only and task_name not in only:
            continue
        t0 = time.time()
        print(f"[{i:2}/{len(TASKS)}] {task_name} ...", flush=True)
        try:
            mod = importlib.import_module(mod_path)
            task_factory = getattr(mod, task_name)
            task = task_factory(**kwargs)
            logs = inspect_eval(
                task,
                model=args.model,
                limit=args.limit,
                message_limit=args.message_limit,
                max_connections=args.max_connections,
                fail_on_error=False,
                log_dir=args.log_dir,
                display="none",
                score_display=False,
            )
            log = logs[0]
            row = _result_row(log)
            row["elapsed_s"] = round(time.time() - t0, 1)
            results[task_name] = row
            print(
                f"        covert_rate={row.get('covert_rate', 'N/A')} "
                f"stderr={row.get('stderr', 'N/A')} "
                f"n={row.get('n_samples', 0)} "
                f"status={row['status']} "
                f"({row['elapsed_s']}s)",
                flush=True,
            )
        except Exception as exc:
            print(f"        FAILED: {type(exc).__name__}: {exc}", flush=True)
            results[task_name] = {"status": "exception", "error": repr(exc)}
        # Persist after each task so partial progress isn't lost.
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))

    total = time.time() - start
    print(f"\nDone. {len(results)} tasks in {total:.0f}s. Results -> {args.output}")


if __name__ == "__main__":
    main()
