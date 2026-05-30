"""Parse QA markdown tables (STT | question | expect_response)."""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QA:
    qid: str
    question: str
    expected: str
    difficulty: str


def _split_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def parse_md(path: Path, difficulty: str) -> list[QA]:
    out: list[QA] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = _split_row(line)
        if len(cells) < 3:
            continue
        first = cells[0]
        if not re.fullmatch(r"\d+", first):
            continue  # skip header / separator
        out.append(
            QA(
                qid=f"{difficulty[0]}{first}",
                question=cells[1],
                expected=cells[2],
                difficulty=difficulty,
            )
        )
    return out


def load_dataset(test_dir: Path) -> list[QA]:
    items: list[QA] = []
    easy = test_dir / "testQA.md"
    hard = test_dir / "testQA_hard.md"
    if easy.exists():
        items += parse_md(easy, "easy")
    if hard.exists():
        items += parse_md(hard, "hard")
    return items
