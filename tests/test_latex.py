import pandas as pd
import pytest

from pm25.evaluation.latex import MODEL_LABELS, bold_best, metric_table, tabular


def _overall():
    return pd.DataFrame({
        "model": ["persistence", "persistence", "xgboost", "xgboost"],
        "lead": [1, 24, 1, 24], "mae": [2.9, 6.9, 2.7, 5.7], "f1": [0.74, 0.31, 0.73, 0.16],
    })


def test_bold_best_marks_minimum_per_column():
    df = pd.DataFrame({"a": [2.0, 1.0], "b": [0.5, 0.9]})
    out = bold_best(df, lower_is_better={"a": True, "b": False}, digits=2)
    assert out["a"].tolist() == ["2.00", r"\textbf{1.00}"]
    assert out["b"].tolist() == ["0.50", r"\textbf{0.90}"]


def test_bold_best_leaves_nan_blank():
    out = bold_best(pd.DataFrame({"a": [float("nan"), 1.0]}), {"a": True}, digits=1)
    assert out["a"].tolist() == ["--", r"\textbf{1.0}"]


def test_metric_table_pivots_leads_and_labels_models():
    out = metric_table(_overall(), metrics=("mae", "f1"), leads=(1, 24))
    assert list(out.index) == [MODEL_LABELS["persistence"], MODEL_LABELS["xgboost"]]
    assert list(out.columns) == [("mae", 1), ("mae", 24), ("f1", 1), ("f1", 24)]
    assert out.loc[MODEL_LABELS["xgboost"], ("mae", 24)] == pytest.approx(5.7)


def test_tabular_renders_booktabs():
    tex = tabular(pd.DataFrame({"x": ["1"]}, index=["row"]), header=["Model", "X"],
                  align="lr")
    assert tex.startswith(r"\begin{tabular}{lr}")
    assert r"\toprule" in tex and "Model & X" in tex and "row & 1" in tex
    assert tex.rstrip().endswith(r"\end{tabular}")


def test_tabular_header_length_must_match():
    with pytest.raises(ValueError, match="header"):
        tabular(pd.DataFrame({"x": ["1"]}, index=["r"]), header=["only"], align="lr")


def _scores(by, values):
    rows = []
    for key in values:
        for model in ("persistence", "xgboost"):
            for lead in (1, 6, 24):
                rows.append({by: key, "model": model, "lead": lead, "mae": 1.0 + lead,
                             "rmse": 2.0, "alert_f1": 0.3})
    return pd.DataFrame(rows)


def test_overall_table_has_grouped_header():
    from pm25.evaluation.latex import overall_table

    df = _scores("dummy", ["x"]).drop(columns="dummy")
    tex = overall_table(df)
    assert r"\multicolumn{3}{c}{MAE}" in tex and r"\cmidrule(lr){2-4}" in tex
    assert "Persistence" in tex and "XGBoost" in tex


def test_season_and_region_tables():
    from pm25.evaluation.latex import region_table, season_table

    seasons = _scores("season", ["autumn_winter", "spring_summer"])
    assert "Episode hours" in season_table(seasons, seasons, lead=24)
    regions = _scores("region", ["northern", "central", "yunchianan", "kaoping"])
    assert "Kao-Ping" in region_table(regions, lead=24)


def test_dm_and_recovery_tables():
    from pm25.evaluation.latex import dm_table, recovery_table

    dm = pd.DataFrame({"model": ["xgboost"], "baseline": ["persistence"], "lead": [24],
                       "mean_diff": [-1.0], "dm_stat": [-5.0], "p_value": [1e-6], "n": [10],
                       "pct_change": [-16.9]})
    wins = pd.DataFrame({"model": ["xgboost"], "baseline": ["persistence"], "lead": [24],
                         "wins": [12], "stations": [12], "sig_wins": [11], "sig_losses": [0]})
    tex = dm_table(dm, wins, lead=24)
    assert r"-16.9\%" in tex and r"$<$0.001" in tex and "12/12 (11/0)" in tex
    recovery = pd.DataFrame({"model": ["xgboost"], "lead": [24], "median": [30.0],
                             "mean": [40.0], "count": [7]})
    assert "XGBoost & 30 & 7" in recovery_table(recovery, lead=24)
