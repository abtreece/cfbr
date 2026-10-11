import json
from datetime import UTC, datetime, timedelta

from cfbr.backup import STAMP, chain_head, last_backup, run_backup, sources
from cfbr.config import Config, Paths

NOW = datetime(2026, 10, 12, 8, 30, tzinfo=UTC)


def _setup(tmp_path):
    root = tmp_path / "repo"
    real_raw = tmp_path / "elsewhere" / "raw"   # a worktree's data/raw is a symlink
    (real_raw / "games" / "2026").mkdir(parents=True)
    (real_raw / "games" / "2026" / "w07.json").write_text("[]")
    (root / "data").mkdir(parents=True)
    (root / "data" / "raw").symlink_to(real_raw)
    (root / "data" / "news").mkdir()
    (root / "data" / "news" / "a.jsonl").write_text("{}\n")
    season = root / "out" / "2026"
    (season / "ledger_history").mkdir(parents=True)
    (season / "bets.csv").write_text("game_id\n1\n")
    (season / "bets_retro.csv").write_text("game_id\n2\n")
    (season / "ledger_history" / "bets-1.csv").write_text("game_id\n1\n")
    (season / "ledger_chain.jsonl").write_text(json.dumps({"ts": "2026-10-11T14:05:00+00:00", "chain": "f" * 64}) + "\n")
    (season / "w07").mkdir()  # rankings are rebuildable: not backed up
    (season / "w07" / "ratings.csv").write_text("x\n")
    cfg = Config(paths=Paths(raw_dir=root / "data" / "raw", db_path=root / "data" / "cfb.duckdb",
                             results_dir=root / "results", out_dir=root / "out"))
    cfg.backup.target = str(tmp_path / "backup")
    return root, cfg


def test_backup_copies_irreplaceable_data_and_keeps_deleted_files(tmp_path):
    root, cfg = _setup(tmp_path)
    assert {str(p) for p in sources(cfg, root)} == {
        "data/raw", "data/news", "out/2026/bets.csv", "out/2026/bets_retro.csv", "out/2026/ledger_chain.jsonl",
        "out/2026/ledger_history"}
    run_backup(cfg, root, NOW)
    dest = tmp_path / "backup"
    assert (dest / "data" / "raw" / "games" / "2026" / "w07.json").is_file()   # followed the symlink
    assert not (dest / "data" / "raw").is_symlink()
    assert (dest / "out" / "2026" / "ledger_history" / "bets-1.csv").exists()
    assert not (dest / "out" / "2026" / "w07").exists()
    assert last_backup(cfg) == NOW

    (root / "data" / "news" / "a.jsonl").unlink()   # e.g. a purge, or an accident
    later = NOW + timedelta(days=1)
    run_backup(cfg, root, later)
    assert not (dest / "data" / "news" / "a.jsonl").exists()
    assert (dest / "_previous" / "20261013" / "data" / "news" / "a.jsonl").exists()
    stamp = run_backup(cfg, root, NOW + timedelta(days=40))
    assert stamp["pruned"] == ["20261013"] and json.loads((dest / STAMP).read_text())["pruned"] == ["20261013"]


def test_chain_head(tmp_path):
    root, cfg = _setup(tmp_path)
    assert chain_head(cfg.paths.out_dir, 2026) == "ledger chain 2026: 1 entries, head ffffffffffffffff (2026-10-11T14:05:00+00:00)"
    assert chain_head(cfg.paths.out_dir, 2025) is None
