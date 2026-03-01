"""Restricted shell command execution with command allowlist.

Only executables explicitly listed in the allowlist may run.
"""

import shlex
import subprocess


def run_shell(
    command: str,
    allowlist: list[str] | None = None,
    timeout: int = 30,
) -> dict:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("Empty command")

    executable = parts[0]
    if allowlist and executable not in allowlist:
        raise PermissionError(
            f"Command '{executable}' not in allowlist. Allowed: {allowlist}"
        )

    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout[:10_000],
            "stderr": result.stderr[:5_000],
        }
    except subprocess.TimeoutExpired:
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
        }
