from pathlib import Path
import re
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]

ANSI_ESCAPE_RE = re.compile(
    r"\x1B\[[0-?]*[ -/]*[@-~]"  # basic CSI sequences
)


def _strip_ansi(s: str) -> str:
    return ANSI_ESCAPE_RE.sub("", s)


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_make_help_outputs_expected_sections_and_targets():
    # Ensure we run from repo root so that $(MAKEFILE_LIST) resolves correctly
    assert (ROOT / "Makefile").exists(), "Makefile not found at repository root"

    proc = subprocess.run(
        ["make", "help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    # Capture and normalize output
    out = _strip_ansi(proc.stdout)
    err = _strip_ansi(proc.stderr)

    assert proc.returncode == 0, (
        f"`make help` exited with {proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
    )

    # Basic headings from help target
    assert "Usage:" in out
    assert "Targets:" in out

    # Section headers defined in Makefile
    for section in [
        "Bootstrap",
        "Development and Testing",
        "Meta",
    ]:
        assert section in out, (
            f"Section header '{section}' not found in help output.\nOutput was:\n{out}"
        )

    # Targets declared in Makefile should appear in help
    for target in [
        "install-uv",
        "install",
        "test",
        "fmt",
        "help",
    ]:
        assert re.search(rf"\b{re.escape(target)}\b", out) is not None, (
            f"Target '{target}' not found in help output.\nOutput was:\n{out}"
        )
