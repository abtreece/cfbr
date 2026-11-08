import subprocess
import sys

import duckdb
import polars as pl
import pytest

from cfbr.anomalies.tools import OUTPUT_FILES, ToolSession
from cfbr.config import Investigate


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "snap.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE games AS SELECT range AS game_id, 'T' || range AS home_team FROM range(500)")
    out = tmp_path / "out"
    (out / "2026" / "w09").mkdir(parents=True)
    (out / "2026" / "w08").mkdir(parents=True)
    pl.DataFrame({"team": ["Iowa"], "rating_change": [-5.0]}).write_csv(out / "2026" / "w09" / "rankings.csv")
    pl.DataFrame({"team": ["Iowa"], "rating_change": [1.0]}).write_csv(out / "2026" / "w08" / "rankings.csv")
    s = ToolSession(db, out, 2026, 9, Investigate(query_max_rows=200, query_max_chars=20000))
    yield s
    s.close()


def test_query_returns_table_with_result_id_and_keeps_it(env):
    text, err = env.query("SELECT game_id, home_team FROM games WHERE game_id < 2 ORDER BY 1")
    assert not err
    assert text.splitlines()[:4] == ["result_id 1 (2 rows)", "game_id\thome_team", "0\tT0", "1\tT1"]
    assert env.results[1] == text and env.log[0]["ok"] and env.log[0]["tool"] == "query"


def test_query_caps_rows_and_chars(env):
    text, _ = env.query("SELECT * FROM games")
    assert "result_id 1 (200 rows)" in text and "more than 200 rows" in text
    small = ToolSession(env.db_path, env.out_dir, 2026, 9, Investigate(query_max_chars=50))
    text2, _ = small.query("SELECT * FROM games LIMIT 100")
    assert text2.endswith("… truncated") and len(text2) < 200
    small.close()


@pytest.mark.parametrize("sql", [
    "SELECT * FROM read_csv('/etc/hosts')", "ATTACH '/tmp/x.db'", "COPY (SELECT 1) TO '/tmp/y.csv'",
    "CREATE TABLE t AS SELECT 1", "SET enable_external_access = true", "INSTALL httpfs",
    "SELECT 1; SELECT 2", "CALL enable_logging()", "CHECKPOINT", "CREATE TEMP TABLE x AS SELECT 1",
    "SET VARIABLE v = 1", "SELECT * FROM query('SELECT 1')", "SELECT * FROM QUERY_TABLE('games')",
    "COPY (SELECT 1) TO '{wal}'"])
def test_sandbox_refuses_writes_files_and_config(env, sql):
    text, err = env.query(sql.format(wal=f"{env.db_path}.wal"))
    assert err and text.startswith("result_id 1: error:")
    assert 1 not in env.results and env.log[0]["ok"] is False


def test_enable_logging_select_is_refused_without_executing(tmp_path):
    db = tmp_path / "snap.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE games AS SELECT 1 AS game_id")
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from cfbr.anomalies.tools import ToolSession\n"
        "from cfbr.config import Investigate\n"
        "s = ToolSession(Path(sys.argv[1]), Path(sys.argv[2]), 2026, 9, Investigate())\n"
        "text, err = s.query(\"SELECT * FROM enable_logging(storage='file', storage_path='/nonexistent/x')\")\n"
        "print(text)\n"
        "s.close()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, str(db), str(tmp_path)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "refused" in proc.stdout


def test_tmp_directory_write_refused_even_when_it_exists(env):
    tmp_dir = env.db_path.parent / f"{env.db_path.name}.tmp"
    tmp_dir.mkdir()
    text, err = env.query(f"COPY (SELECT 1) TO '{tmp_dir / 'x.csv'}'")
    assert err and text.startswith("result_id 1: error:")
    assert not (tmp_dir / "x.csv").exists()


def test_allowed_introspection_and_timestamptz(env):
    for sql in ("DESCRIBE games", "SELECT game_id FROM games LIMIT 1",
                "SELECT TIMESTAMPTZ '2026-11-08 12:00:00+00' AS ts"):
        text, err = env.query(sql)
        assert not err, text


def test_query_timeout_interrupts(env):
    slow = ToolSession(env.db_path, env.out_dir, 2026, 9, Investigate(query_timeout_seconds=0.2))
    text, err = slow.query("SELECT count(*) FROM range(10000000000) a")
    assert err and "timed out after 0.2s" in text
    ok_text, ok_err = slow.query("SELECT 1 AS ok")
    assert not ok_err and ok_text.splitlines()[:3] == ["result_id 2 (1 rows)", "ok", "1"]
    slow.close()


def test_read_output_whitelist_and_shared_ids(env):
    env.query("SELECT 1 AS x")
    text, err = env.read_output("rankings_prev.csv")
    assert not err and text.startswith("result_id 2 (rankings_prev.csv)") and "Iowa,1.0" in text
    assert env.results[2] == text
    bad, err2 = env.read_output("../../.keys")
    assert err2 and "allowed:" in bad and 3 not in env.results
    missing, err3 = env.read_output("predictions.csv")
    assert err3 and "not available" in missing
    assert set(OUTPUT_FILES) == {"rankings.csv", "rankings_prev.csv", "predictions.csv", "accuracy.csv"}
