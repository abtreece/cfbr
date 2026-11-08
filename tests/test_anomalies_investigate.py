from cfbr.config import Config, anthropic_api_key


def test_investigate_config_defaults():
    i = Config().investigate
    assert (i.model, i.effort, i.max_turns, i.max_items, i.max_attempts) == ("claude-opus-5-5", "xhigh", 40, 12, 3)
    assert (i.max_tokens, i.task_budget_tokens, i.monthly_budget_usd) == (32000, 60000, 50.0)
    assert (i.wall_clock_minutes, i.query_timeout_seconds, i.query_max_rows, i.query_max_chars) == (15, 30, 200, 20000)


def test_anthropic_key_env_then_keys_file_then_none(monkeypatch, tmp_path):
    monkeypatch.setattr("cfbr.config.REPO_ROOT", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert anthropic_api_key() is None
    (tmp_path / ".keys").write_text("CFBD_API_KEY=abc\nANTHROPIC_API_KEY='sk-file'\n")
    assert anthropic_api_key() == "sk-file"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert anthropic_api_key() == "sk-env"
