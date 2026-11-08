import optuna

from cfbr.backtest.tune import apply_params, patch_frozen, sensitivity_grids, suggest_fcs_two_stage
from cfbr.config import REPO_ROOT, load_config


def test_apply_params_sets_fcs_keys():
    cfg = apply_params(load_config(), {"fcs_two_stage": True, "fcs_weight": 0.2})
    assert cfg.model.fcs_two_stage is True and cfg.model.fcs_weight == 0.2


def test_fcs_space_fixes_the_rule_settings():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(25):
        p = suggest_fcs_two_stage(study.ask())
        assert p["fcs_two_stage"] is True and p["fcs_pooling"] == "conference" and p["gt_enabled"] is False
        assert 0.0 <= p["fcs_weight"] <= 1.0
        assert 20 <= p["alpha_ppa"] <= 3000 and 20 <= p["alpha_sr"] <= 3000
        assert 2 <= p["half_life_weeks"] <= 26


def test_fcs_sensitivity_grid_never_turns_recency_off_or_varies_fixed_settings():
    g = sensitivity_grids({"alpha_ppa": 300.0, "alpha_sr": 90.0, "half_life_weeks": 3.7, "fcs_weight": 0.1},
                          "fcs-two-stage")
    assert set(g) == {"fcs_weight", "alpha_ppa", "alpha_sr", "half_life_weeks"}
    assert min(g["half_life_weeks"]) > 0 and 0.0 in g["fcs_weight"] and 1.0 in g["fcs_weight"]


def test_patch_frozen_changes_only_the_named_keys(tmp_path):
    base = REPO_ROOT / "config" / "tuned_v1.toml"
    out = tmp_path / "t.toml"
    best = {"fcs_two_stage": True, "fcs_weight": 0.12, "alpha_ppa": 250.0, "alpha_sr": 80.0,
            "half_life_weeks": 4.2, "fcs_pooling": "conference", "gt_enabled": False}
    cfg, ref = patch_frozen(base, best, out), load_config(base)
    assert cfg.model.fcs_two_stage is True and cfg.model.fcs_weight == 0.12
    assert (cfg.model.alpha_ppa, cfg.model.alpha_sr, cfg.model.half_life_weeks) == (250.0, 80.0, 4.2)
    assert cfg.baseline == ref.baseline and cfg.ingest == ref.ingest and cfg.seasons == ref.seasons
    assert cfg.prior == ref.prior and cfg.backtest == ref.backtest and cfg.bets == ref.bets
    assert cfg.prep == ref.prep
    assert "[bets]" not in out.read_text()  # unlisted sections stay on pydantic defaults, as in tuned_v1.toml
