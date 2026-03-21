import os
from pathlib import Path
import re
import sys

README_PATH = Path(__file__).resolve().parent.parent / "README.md"


def _extract_python_blocks(markdown_text: str) -> list[str]:
    pattern = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)
    return [block.strip() for block in pattern.findall(markdown_text)]


def test_readme_python_examples_run():
    readme_text = README_PATH.read_text(encoding="utf-8")
    python_blocks = _extract_python_blocks(readme_text)

    assert python_blocks, "No python code blocks found in README.md"

    repo_root = README_PATH.parent

    # Make package importable
    sys.path.insert(0, str(repo_root))

    # Run from repo root so relative paths work
    old_cwd = os.getcwd()
    os.chdir(repo_root)

    globals_dict = {"__name__": "__main__"}

    try:
        for idx, block in enumerate(python_blocks, start=1):
            exec(block, globals_dict)

    except Exception as e:
        raise AssertionError(f"README python block #{idx} failed:\n{block}") from e

    finally:
        os.chdir(old_cwd)
