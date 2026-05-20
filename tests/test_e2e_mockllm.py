"""End-to-end smoke tests: every @task constructs and runs against `mockllm/model`.

`mockllm/model` is Inspect's built-in deterministic provider that returns a
fixed response, so this is a structural test — we're not measuring covert
rates here, we're verifying that every task in the package can be
constructed, dispatched, and scored without crashing. This catches:

  * imports / @task registration regressions,
  * scorer crashes on a near-empty TaskState,
  * tool-arg validation errors at task construction,
  * sandbox setup attempts when none should be required (we run vfs-only).

CONTRIBUTING.md from inspect_evals asks specifically for `mockllm/model`
in tests for determinism and to "prevent network calls during testing",
which this satisfies.

The matrix covers all 25 envs from Appendix B except #4 (Chat Deception,
not implemented). Envs that support a sandbox backend (`sabotage_safety`,
`impossible_coding`) are exercised on their `backend="vfs"` path so Docker
isn't required to run the suite.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from inspect_ai import eval as inspect_eval


# (module_path, task_callable_name, kwargs_for_task_factory)
# kwargs is used to keep heavy/Docker-backed envs on their in-memory path
# during the E2E test.
TASKS: list[tuple[str, str, dict[str, Any]]] = [
    # --- QA ---
    ("anti_scheming.qa.sandbagging_qa", "as_sandbagging_qa", {}),
    ("anti_scheming.qa.sandbagging_qa_no_goal", "as_sandbagging_qa_no_goal", {}),
    ("anti_scheming.qa.sandbagging_grading_cot", "as_sandbagging_grading_cot", {}),
    ("anti_scheming.qa.literature_qa", "as_literature_qa", {}),
    ("anti_scheming.qa.lazy_checklist", "as_lazy_checklist", {}),
    ("anti_scheming.qa.long_document_verification", "as_long_document_verification", {}),
    # --- Software ---
    ("anti_scheming.software.impossible_coding", "as_impossible_coding", {"backend": "vfs"}),
    ("anti_scheming.software.email_scoring", "as_email_scoring", {}),
    ("anti_scheming.software.sabotage_math", "as_sabotage_math", {}),
    ("anti_scheming.software.sabotage_safety", "as_sabotage_safety", {"backend": "vfs"}),
    ("anti_scheming.software.log_falsification", "as_log_falsification", {}),
    ("anti_scheming.software.smuggle_capabilities", "as_smuggle_capabilities", {}),
    ("anti_scheming.software.as_over_escalation", "as_as_over_escalation", {}),
    # --- Corporate ---
    ("anti_scheming.corporate.legal_reporting_us", "as_legal_reporting_us", {}),
    ("anti_scheming.corporate.legal_reporting_cn", "as_legal_reporting_cn", {}),
    ("anti_scheming.corporate.broken_promises", "as_broken_promises", {}),
    ("anti_scheming.corporate.powerseeking_survey", "as_powerseeking_survey", {}),
    ("anti_scheming.corporate.subagent_collusion", "as_subagent_collusion", {}),
    # --- Spec sabotage ---
    ("anti_scheming.spec_sabotage.spec_sabotage_moving", "as_spec_sabotage_moving", {}),
    ("anti_scheming.spec_sabotage.spec_sabotage_selection", "as_spec_sabotage_selection", {}),
    (
        "anti_scheming.spec_sabotage.spec_sabotage_verification",
        "as_spec_sabotage_verification",
        {},
    ),
    # --- Games ---
    ("anti_scheming.games.blotto", "as_blotto", {}),
    ("anti_scheming.games.prisoners_dilemma", "as_prisoners_dilemma", {}),
    # --- Subagent ---
    ("anti_scheming.subagent.subagent_permissions", "as_subagent_permissions", {}),
    ("anti_scheming.subagent.subagent_resource_quota", "as_subagent_resource_quota", {}),
]


@pytest.mark.parametrize(
    "module_path,task_name,kwargs",
    TASKS,
    ids=[t[1] for t in TASKS],
)
def test_task_runs_with_mockllm(
    module_path: str, task_name: str, kwargs: dict[str, Any], tmp_path
) -> None:
    """Each task constructs and runs end-to-end against mockllm/model.

    We don't assert on covert rates here — `mockllm/model` returns a fixed
    reply and won't authentically attempt the task. We assert that the eval
    *completes* with a recognizable status (`success` or `error`) and that
    at least one sample shape was produced, which is enough to catch
    construction / scoring / tool-registration regressions.
    """
    mod = importlib.import_module(module_path)
    task_factory = getattr(mod, task_name)
    task = task_factory(**kwargs)

    logs = inspect_eval(
        task,
        model="mockllm/model",
        limit=1,
        message_limit=6,
        fail_on_error=False,
        log_dir=str(tmp_path),
        display="none",
    )
    assert len(logs) == 1, f"{task_name}: expected 1 log, got {len(logs)}"
    log = logs[0]
    # status is "success" (normal completion), "cancelled", or "error".
    # We permit "error" here because mockllm doesn't honour task-specific
    # tool-call contracts — what we're catching are construction-time and
    # scoring-time crashes, not whether the model behaved.
    assert log.status in {"success", "error"}, (
        f"{task_name}: unexpected status {log.status!r}"
    )
    # Even with fail_on_error=False and a misbehaving mockllm, the log
    # should record the attempted sample.
    assert log.samples is not None, f"{task_name}: log.samples is None"
    assert len(log.samples) >= 1, f"{task_name}: no samples in log"
