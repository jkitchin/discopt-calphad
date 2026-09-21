"""Every Python block in the agent skill runs (issue #1).

The blocks of one file run in order in a shared namespace, as a reader would
run them. A block whose first line is ``# slow`` (a certified solve) runs only
in the slow suite.
"""

import re
from pathlib import Path

import pytest

SKILL = Path(__file__).parents[1] / "skills" / "discopt-calphad"
FILES = [SKILL / "SKILL.md", SKILL / "reference" / "phase-protocol.md"]


def blocks(path):
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), flags=re.S)


def run(path, *, slow, tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.chdir(tmp_path)  # blocks may write figures
    ns = {"__name__": "skill"}
    ran = 0
    for i, code in enumerate(blocks(path)):
        if code.startswith("# slow") and not slow:
            continue
        exec(compile(code, f"{path.name}[block {i}]", "exec"), ns)
        ran += 1
    assert ran


def test_skill_has_frontmatter():
    text = FILES[0].read_text(encoding="utf-8")
    front = text.split("---")[1]
    assert re.search(r"^name: discopt-calphad$", front, flags=re.M)
    assert "description:" in front
    for path in FILES:
        assert blocks(path), path


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_skill_blocks_run(path, tmp_path, monkeypatch):
    run(path, slow=False, tmp_path=tmp_path, monkeypatch=monkeypatch)


@pytest.mark.slow
@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_skill_blocks_run_with_certification(path, tmp_path, monkeypatch):
    run(path, slow=True, tmp_path=tmp_path, monkeypatch=monkeypatch)
