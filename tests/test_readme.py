"""Tests for README code examples.

This module extracts Python code and expected result blocks from README.md,
executes the code, and verifies the output matches the documented result.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).parent.parent
README = ROOT / "README.md"

# Regex for Python code blocks
CODE_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)

RESULT = re.compile(r"```result\n(.*?)```", re.DOTALL)


def test_readme_runs():
    """Execute README code blocks and compare output to documented results."""
    readme_text = README.read_text(encoding="utf-8")
    code_blocks = CODE_BLOCK.findall(readme_text)
    result_blocks = RESULT.findall(readme_text)

    # Optional: keep docs and expectations in sync.
    assert len(code_blocks) == len(result_blocks), (
        "Mismatch between python and result blocks in README.md"
    )
    code = "".join(code_blocks)  # merged code
    expected = "".join(result_blocks)

    # Trust boundary: we execute Python snippets sourced from README.md in this repo.
    # The README is part of the trusted repository content and reviewed in PRs.
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )  # noqa: S603

    stdout = result.stdout

    assert result.returncode == 0, (
        f"README code exited with {result.returncode}. Stderr:\n{result.stderr}"
    )
    assert stdout.strip() == expected.strip()
