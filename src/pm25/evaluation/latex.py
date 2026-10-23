"""LaTeX tables for the report, generated from results/tables so numbers never drift.

Run `uv run python -m pm25.evaluation.latex` after `report` and `analysis`.
"""

from pathlib import Path

import pandas as pd

from pm25.config import HORIZONS, REPORT_TABLES_DIR, STATIONS, TABLES_DIR

MODEL_LABELS = {
    "persistence": "Persistence",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "chronos_bolt_small": "Bolt-Small",
    "chronos_bolt_base": "Bolt-Base",
    "chronos2_uni": "C2 (univariate)",
    "chronos2_cov": "C2 + past cov.",
    "chronos2_wind_forecast": "C2 + NWP wind",
    "chronos2_wind_oracle": "C2 + oracle wind",
}
REGION_LABELS = {"northern": "North", "central": "Central", "yunchianan": "Yun-Chia-Nan",
                 "kaoping": "Kao-Ping"}
LOWER_IS_BETTER = {"mae": True, "rmse": True, "f1": False, "alert_f1": False,
                   "coverage80": None}


def bold_best(df: pd.DataFrame, lower_is_better: dict, digits: int) -> pd.DataFrame:
    """Format numbers; bold the best value of each column (None = no bolding)."""
    def column(name, values):
        direction = lower_is_better.get(name if not isinstance(name, tuple) else name[0])
        best = None if direction is None else (values.min() if direction else values.max())
        return [("--" if pd.isna(v) else
                 (rf"\textbf{{{v:.{digits}f}}}" if best is not None and v == best
                  else f"{v:.{digits}f}")) for v in values]
    return pd.DataFrame({c: column(c, df[c]) for c in df.columns}, index=df.index)


def metric_table(scores: pd.DataFrame, metrics, leads=HORIZONS,
                 models=tuple(MODEL_LABELS)) -> pd.DataFrame:
    """Rows = models in report order, columns = (metric, lead)."""
    wide = scores.pivot_table(index="model", columns="lead", values=list(metrics))
    order = [m for m in models if m in wide.index]
    wide = wide.loc[order, [(m, lead) for m in metrics for lead in leads]]
    wide.index = [MODEL_LABELS[m] for m in order]
    return wide


def tabular(body: pd.DataFrame, header: list[str], align: str,
            extra_header: str = "") -> str:
    if len(header) != body.shape[1] + 1:
        raise ValueError("header needs one label per column plus the row label")
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    if extra_header:
        lines.append(extra_header)
    lines += [" & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join([str(idx), *map(str, row)]) + r" \\"
              for idx, row in zip(body.index, body.itertuples(index=False))]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _lead_header(metrics: dict[str, str], leads) -> str:
    groups = " & ".join(rf"\multicolumn{{{len(leads)}}}{{c}}{{{label}}}"
                        for label in metrics.values())
    rules = " ".join(rf"\cmidrule(lr){{{2 + i * len(leads)}-{1 + (i + 1) * len(leads)}}}"
                     for i in range(len(metrics)))
    return f" & {groups} \\\\\n{rules}"


def overall_table(overall: pd.DataFrame, leads=HORIZONS) -> str:
    metrics = {"mae": "MAE", "rmse": "RMSE", "alert_f1": r"Alert F1 ($q_{0.9}$)"}
    body = bold_best(metric_table(overall, tuple(metrics), leads), LOWER_IS_BETTER, 2)
    header = ["Model", *[f"{lead}h" for _ in metrics for lead in leads]]
    return tabular(body, header, "l" + "r" * len(metrics) * len(leads),
                   _lead_header(metrics, leads))


def season_table(by_season: pd.DataFrame, episodes: pd.DataFrame, lead: int) -> str:
    cols = {}
    for name, frame in (("All", by_season), ("Episode", episodes)):
        for season, label in (("autumn_winter", "Aut.-Win."), ("spring_summer", "Spr.-Sum.")):
            sub = frame[(frame["season"] == season) & (frame["lead"] == lead)]
            cols[(name, label)] = sub.set_index("model")["mae"]
    wide = pd.DataFrame(cols)
    order = [m for m in MODEL_LABELS if m in wide.index]
    wide = wide.loc[order]
    wide.index = [MODEL_LABELS[m] for m in order]
    body = bold_best(wide, {c: True for c in wide.columns}, 2)
    extra = (r" & \multicolumn{2}{c}{All hours} & \multicolumn{2}{c}{Episode hours} \\"
             "\n" r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    return tabular(body, ["Model", "Aut.-Win.", "Spr.-Sum.", "Aut.-Win.", "Spr.-Sum."],
                   "lrrrr", extra)


def region_table(by_region: pd.DataFrame, lead: int) -> str:
    sub = by_region[by_region["lead"] == lead]
    wide = sub.pivot_table(index="model", columns="region", values="mae")
    order = [m for m in MODEL_LABELS if m in wide.index]
    wide = wide.loc[order, list(STATIONS)]
    wide.index = [MODEL_LABELS[m] for m in order]
    body = bold_best(wide, {c: True for c in wide.columns}, 2)
    return tabular(body, ["Model", *[REGION_LABELS[r] for r in STATIONS]], "lrrrr")


def _p(p: float) -> str:
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def dm_table(dm: pd.DataFrame, wins: pd.DataFrame, lead: int) -> str:
    merged = dm[dm["lead"] == lead].merge(wins, on=["model", "baseline", "lead"], how="left")
    body = pd.DataFrame({
        "pct": [f"{v:+.1f}\\%" for v in merged["pct_change"]],
        "dm": [f"{v:.2f}" for v in merged["dm_stat"]],
        "p": [_p(v) for v in merged["p_value"]],
        "wins": [f"{w}/{s} ({sw}/{sl})" for w, s, sw, sl in
                 zip(merged["wins"], merged["stations"], merged["sig_wins"],
                     merged["sig_losses"])],
    }, index=[f"{MODEL_LABELS[a]} vs.\\ {MODEL_LABELS[b]}"
              for a, b in zip(merged["model"], merged["baseline"])])
    return tabular(body, ["Comparison", r"$\Delta$MAE", "DM", "$p$", "Station wins"], "lrrrr")


def recovery_table(recovery: pd.DataFrame, lead: int) -> str:
    sub = recovery[recovery["lead"] == lead].set_index("model")
    order = [m for m in MODEL_LABELS if m in sub.index]
    body = pd.DataFrame({"median": [f"{sub.loc[m, 'median']:.0f}" for m in order],
                         "count": [f"{sub.loc[m, 'count']:.0f}" for m in order]},
                        index=[MODEL_LABELS[m] for m in order])
    return tabular(body, ["Model", "Median (h)", "Episodes"], "lrr")


def main() -> None:  # pragma: no cover - file IO entry point
    src, out = Path(TABLES_DIR), Path(REPORT_TABLES_DIR)
    out.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(src / f"{name}.csv")  # noqa: E731
    lead = max(HORIZONS)
    tables = {
        "overall": overall_table(read("overall")),
        "season": season_table(read("by_season"), read("episodes"), lead),
        "region": region_table(read("by_region"), lead),
        "dm": dm_table(read("dm_tests"), read("station_wins"), lead),
        "recovery": recovery_table(read("recovery"), lead),
    }
    for name, tex in tables.items():
        (out / f"{name}.tex").write_text(tex)
        print(out / f"{name}.tex")


if __name__ == "__main__":  # pragma: no cover
    main()
