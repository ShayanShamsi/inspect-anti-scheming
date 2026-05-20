"""Agent wrappers shared across anti-scheming environments."""

from __future__ import annotations

from inspect_ai.agent import AgentPrompt, as_solver, react
from inspect_ai.solver import Solver
from inspect_ai.tool import Tool

from anti_scheming.common.prompts import ANTI_SCHEMING_SPEC, DEFAULT_SYSTEM


def scheming_agent(
    *,
    tools: list[Tool],
    with_spec: bool = False,
    extra_system: str | None = None,
    attempts: int = 1,
) -> Solver:
    """ReAct agent. All system text lives in one block at the front so SGLang
    (which rejects system messages after the first turn) stays happy."""

    parts = [DEFAULT_SYSTEM]
    if with_spec:
        parts.append(ANTI_SCHEMING_SPEC)
    if extra_system:
        parts.append(extra_system)
    instructions = "\n\n".join(parts)

    return as_solver(
        react(
            tools=tools,
            attempts=max(1, int(attempts)),
            prompt=AgentPrompt(instructions=instructions),
            truncation="auto",
            retry_refusals=2,
        )
    )
