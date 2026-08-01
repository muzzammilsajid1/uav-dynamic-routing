from __future__ import annotations

from pathlib import Path

from evaluation.experiment_metadata import _source_tree_digest


def test_source_tree_digest_changes_with_source_content(tmp_path: Path) -> None:
    (tmp_path / "baselines").mkdir()
    source = tmp_path / "baselines" / "planner.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy==1\n", encoding="utf-8")
    first_digest, first_count = _source_tree_digest(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second_digest, second_count = _source_tree_digest(tmp_path)
    assert first_count == second_count == 2
    assert first_digest != second_digest
