"""Docker backend for any anti-scheming env that wants real bash + files.

Usage in a `@task`:

    from anti_scheming.common.docker_sandbox import docker_backend
    from inspect_ai.tool import bash

    return Task(
        ...,
        sandbox=docker_backend(),
        solver=[scheming_agent(tools=[bash(timeout=30)], ...)],
    )

The VFS backend remains the default for the 25 evals in this package because
it's faster and doesn't churn containers; Docker is opt-in per CLAUDE.md
constraints (30-CPU ceiling, manual cleanup on hard kill).
"""

from __future__ import annotations

import pathlib

_HERE = pathlib.Path(__file__).parent / "sandbox"


def docker_backend() -> tuple[str, str]:
    """Returns the `sandbox=` argument for a `Task(...)`."""
    return ("docker", str(_HERE / "compose.yaml"))
