"""In-memory virtual filesystem and file tools, backed by Inspect's Store.

This is the default backend for every env in this package. It avoids the
per-sample Docker container churn (which on the A10 box is bounded by
30 CPUs and tears down ungracefully). Tools live in the Store so each
sample has its own isolated file dict.
"""

from __future__ import annotations

from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util import store


_FILES_KEY = "vfs.files"


def _files() -> dict[str, str]:
    # store().get(key, default) auto-initialises the slot with `default` when
    # the key is missing, so in-place mutations (write_file, etc.) persist.
    return store().get(_FILES_KEY, {})


@solver
def seed_vfs(files: dict[str, str] | None = None) -> Solver:
    """Solver step that initialises the per-sample VFS from sample metadata.

    Files come from `sample.metadata['vfs']` if present, else from the
    `files` argument passed at task-construction time.
    """

    async def _solve(state: TaskState, _generate: Generate) -> TaskState:
        meta_files = (state.metadata or {}).get("vfs") if state.metadata else None
        seed = dict(meta_files or files or {})
        store().set(_FILES_KEY, seed)
        return state

    return _solve  # type: ignore[return-value]


@tool
def read_file() -> Tool:
    async def execute(path: str) -> str:
        """Read a file from the workspace.

        Args:
            path: Path to read.
        """
        files = _files()
        if path not in files:
            raise ToolError(f"file not found: {path}")
        return files[path]

    return execute


@tool
def write_file() -> Tool:
    async def execute(path: str, content: str) -> str:
        """Create or overwrite a file in the workspace.

        Args:
            path: Path to write.
            content: New file contents.
        """
        _files()[path] = content
        return f"wrote {len(content)} bytes to {path}"

    return execute


@tool
def list_files() -> Tool:
    async def execute() -> str:
        """List all files in the workspace."""
        files = _files()
        if not files:
            return "(empty)"
        return "\n".join(sorted(files))

    return execute
