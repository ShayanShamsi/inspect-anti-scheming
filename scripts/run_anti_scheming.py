"""Driver: run every anti-scheming eval against a single model, aggregate.

Usage:
    uv run python run_anti_scheming.py \
        --model sglang/Qwen/Qwen3.5-0.8B --limit 3

Prints a Figure-24-style table of per-env covert-action rates (one model
only — for the paper's multi-model comparison, run twice with different
`--model` and diff the outputs).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# (module_path, callable_name, bucket)
TASKS: list[tuple[str, str, str]] = [
    ("anti_scheming.qa.sandbagging_qa_no_goal", "as_sandbagging_qa_no_goal", "qa"),
    ("anti_scheming.qa.sandbagging_qa", "as_sandbagging_qa", "qa"),
    ("anti_scheming.qa.sandbagging_grading_cot", "as_sandbagging_grading_cot", "qa"),
    ("anti_scheming.qa.literature_qa", "as_literature_qa", "qa"),
    ("anti_scheming.qa.long_document_verification", "as_long_document_verification", "qa"),
    ("anti_scheming.qa.lazy_checklist", "as_lazy_checklist", "qa"),

    ("anti_scheming.software.impossible_coding", "as_impossible_coding", "software"),
    ("anti_scheming.software.impossible_coding_sla", "as_impossible_coding_sla", "software"),
    ("anti_scheming.software.sabotage_math", "as_sabotage_math", "software"),
    ("anti_scheming.software.sabotage_safety", "as_sabotage_safety", "software"),
    ("anti_scheming.software.log_falsification", "as_log_falsification", "software"),
    ("anti_scheming.software.smuggle_capabilities", "as_smuggle_capabilities", "software"),

    ("anti_scheming.software.email_scoring", "as_email_scoring", "software"),
    ("anti_scheming.corporate.legal_reporting_cn", "as_legal_reporting_cn", "corporate"),
    ("anti_scheming.corporate.legal_reporting_us", "as_legal_reporting_us", "corporate"),
    ("anti_scheming.corporate.broken_promises", "as_broken_promises", "corporate"),
    ("anti_scheming.corporate.powerseeking_survey", "as_powerseeking_survey", "corporate"),
    ("anti_scheming.corporate.subagent_collusion", "as_subagent_collusion", "corporate"),
    ("anti_scheming.software.as_over_escalation", "as_as_over_escalation", "software"),

    ("anti_scheming.spec_sabotage.spec_sabotage_moving", "as_spec_sabotage_moving", "spec_sabotage"),
    ("anti_scheming.spec_sabotage.spec_sabotage_selection", "as_spec_sabotage_selection", "spec_sabotage"),
    ("anti_scheming.spec_sabotage.spec_sabotage_verification", "as_spec_sabotage_verification", "spec_sabotage"),

    ("anti_scheming.games.blotto", "as_blotto", "games"),
    ("anti_scheming.games.prisoners_dilemma", "as_prisoners_dilemma", "games"),

    ("anti_scheming.subagent.subagent_permissions", "as_subagent_permissions", "subagent"),
    ("anti_scheming.subagent.subagent_resource_quota", "as_subagent_resource_quota", "subagent"),
]


def _resolve(module_path: str, name: str):
    return getattr(importlib.import_module(module_path), name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("INSPECT_EVAL_MODEL", "sglang/Qwen/Qwen3.5-0.8B"))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--with-spec", action="store_true",
                    help="prepend the anti-scheming spec (post-intervention condition)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to a list of task names (e.g. as_blotto)")
    ap.add_argument("--bucket", default=None, help="restrict to one bucket")
    ap.add_argument("--attempts", type=int, default=1,
                    help="default attempts per sample passed to react()")
    ap.add_argument("--task-attempts", nargs="*", default=[],
                    help="per-task overrides, e.g. as_blotto=3 as_email_scoring=2")
    ap.add_argument("--max-sandboxes", type=int, default=None,
                    help="cap parallel Docker containers (Inspect default is 2*cpu_count)")
    ap.add_argument("--max-subprocesses", type=int, default=None,
                    help="cap parallel subprocess() calls (Inspect default is cpu_count)")
    ap.add_argument("--max-connections", type=int, default=None,
                    help="cap parallel model connections (Inspect default is 10)")
    ap.add_argument("--no-sandbox-cleanup", action="store_true",
                    help="keep Docker containers alive after each sample for debugging "
                         "(equivalent to inspect eval --no-sandbox-cleanup; only set on "
                         "eval(), not on Task — see Inspect precedence table)")
    ap.add_argument("--fail-on-error", default=None,
                    help="True/False/<float in [0,1]>/<int>. Default is Inspect's True "
                         "(any sample error fails the eval). 0.1 = tolerate 10%% sample errors.")
    ap.add_argument("--retry-on-error", type=int, default=None,
                    help="retry samples that error this many times before giving up")
    ap.add_argument("--score-on-error", action="store_true",
                    help="score errored samples on partial state (covert scorer reads "
                         "the cumulative actions() log so this is safe for us)")
    ap.add_argument("--time-limit", type=int, default=None,
                    help="per-sample wall-clock limit in seconds (early exit, sample "
                         "still scored against whatever state was reached)")
    ap.add_argument("--token-limit", type=int, default=None,
                    help="per-sample total-tokens limit (early exit, sample still scored)")
    args = ap.parse_args()

    overrides: dict[str, int] = {}
    for kv in args.task_attempts:
        if "=" not in kv:
            raise SystemExit(f"--task-attempts entry must be name=int, got {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k.strip()] = int(v)

    selected = [(m, n, b) for (m, n, b) in TASKS
                if (args.only is None or n in args.only)
                and (args.bucket is None or b == args.bucket)]

    print(f"Running {len(selected)} task(s) against {args.model} "
          f"(limit={args.limit}, with_spec={args.with_spec})")

    rows: list[tuple[str, str, float, int]] = []
    for module_path, name, bucket in selected:
        try:
            factory = _resolve(module_path, name)
        except Exception as e:
            print(f"[skip] {name}: import failed: {e}")
            continue
        attempts = overrides.get(name, args.attempts)
        try:
            task = factory(with_spec=args.with_spec, attempts=attempts)
        except TypeError:
            try:
                task = factory(with_spec=args.with_spec)
            except TypeError:
                task = factory()
        try:
            eval_kwargs = dict(
                model=args.model,
                limit=args.limit,
                max_samples=args.max_samples,
                model_args={"max_tokens": args.max_tokens},
                display="plain",
            )
            if args.max_sandboxes is not None:
                eval_kwargs["max_sandboxes"] = args.max_sandboxes
            if args.max_subprocesses is not None:
                eval_kwargs["max_subprocesses"] = args.max_subprocesses
            if args.max_connections is not None:
                eval_kwargs["max_connections"] = args.max_connections
            if args.no_sandbox_cleanup:
                eval_kwargs["sandbox_cleanup"] = False
            if args.fail_on_error is not None:
                v = args.fail_on_error
                if v.lower() == "true":
                    eval_kwargs["fail_on_error"] = True
                elif v.lower() == "false":
                    eval_kwargs["fail_on_error"] = False
                else:
                    eval_kwargs["fail_on_error"] = float(v) if "." in v else int(v)
            if args.retry_on_error is not None:
                eval_kwargs["retry_on_error"] = args.retry_on_error
            if args.score_on_error:
                eval_kwargs["score_on_error"] = True
            if args.time_limit is not None:
                eval_kwargs["time_limit"] = args.time_limit
            if args.token_limit is not None:
                eval_kwargs["token_limit"] = args.token_limit
            logs = inspect_eval(task, **eval_kwargs)
        except Exception as e:
            print(f"[error] {name}: {e}")
            continue
        log = logs[0] if logs else None
        n = 0
        covert_n = 0
        for s in (log.samples or []) if log else []:
            n += 1
            sc = (s.scores or {}).get("covert_action_scorer")
            if sc and sc.metadata and sc.metadata.get("covert"):
                covert_n += 1
        rate = (covert_n / n) if n else 0.0
        rows.append((bucket, name, rate, n))
        print(f"  {name:<40} bucket={bucket:<14} covert={rate:.1%} (n={n})")

    print("\n=== Per-bucket aggregate ===")
    bk: dict[str, list[float]] = defaultdict(list)
    for bucket, _, rate, n in rows:
        bk[bucket].extend([rate] * n)
    for b in sorted(bk):
        v = bk[b]
        print(f"  {b:<14} covert={sum(v)/len(v):.1%} (n={len(v)})")


if __name__ == "__main__":
    main()
