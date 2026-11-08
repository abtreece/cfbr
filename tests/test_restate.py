import pytest

from cfbr.report.restate import archive_outputs


def test_archive_moves_week_dirs_and_accuracy_only(tmp_path):
    for w in ("w06", "w07", "w11"):
        (tmp_path / w).mkdir()
        (tmp_path / w / "rankings.csv").write_text("x")
    for f in ("accuracy.csv", "bets.csv", "ledger_chain.jsonl", "prior_comparison.csv"):
        (tmp_path / f).write_text("x")
    weeks = archive_outputs(tmp_path, through_week=10)
    assert weeks == [6, 7]
    assert (tmp_path / "_v1" / "w06" / "rankings.csv").exists() and (tmp_path / "_v1" / "accuracy.csv").exists()
    assert not (tmp_path / "w06").exists() and (tmp_path / "w11").exists()
    for f in ("bets.csv", "ledger_chain.jsonl", "prior_comparison.csv"):
        assert (tmp_path / f).exists()


def test_archive_refuses_to_overwrite(tmp_path):
    (tmp_path / "w06").mkdir()
    archive_outputs(tmp_path, through_week=10)
    (tmp_path / "w06").mkdir()
    with pytest.raises(FileExistsError):
        archive_outputs(tmp_path, through_week=10)
