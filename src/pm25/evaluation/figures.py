"""Report figures: overall MAE per model × lead, monthly MAE with ADWIN drift counts.

Run `uv run python -m pm25.evaluation.figures` after `pm25.evaluation.report`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pm25.config import (
    FIGURES_DIR,
    HORIZONS,
    PM25_AQI100_THRESHOLD,
    PREDICTIONS_DIR,
    STATION_EN,
    TABLES_DIR,
)

MODEL_ORDER = (
    "persistence", "xgboost", "xgboost_nwp", "lstm", "lstm_nwp", "chronos_bolt_small",
    "chronos_bolt_base", "chronos2_uni", "chronos2_cov", "chronos2_wind_nwp",
    "chronos2_wind_bestmatch", "chronos2_wind_oracle", "chronos2_ft_uni", "chronos2_ft_cov",
    "chronos2_ft_nwp",
)
# Fixed categorical order (validated palette, slots 1-8); colour follows the model family.
# NWP variants keep their base model's hue and are told apart by hatching / dashes.
BASE_COLORS = {
    "persistence": "#2a78d6", "xgboost": "#eb6834", "lstm": "#1baf7a",
    "chronos_bolt_small": "#eda100", "chronos_bolt_base": "#e87ba4", "chronos2_uni": "#008300",
    "chronos2_cov": "#6250d6", "chronos2_wind_oracle": "#e34948",
}
NWP_BASE = {"xgboost_nwp": "xgboost", "lstm_nwp": "lstm", "chronos2_wind_nwp": "chronos2_cov",
            "chronos2_wind_bestmatch": "chronos2_cov", "chronos2_ft_uni": "chronos2_uni",
            "chronos2_ft_cov": "chronos2_cov", "chronos2_ft_nwp": "chronos2_cov"}
MODEL_COLORS = {m: BASE_COLORS[NWP_BASE.get(m, m)] for m in MODEL_ORDER}
HATCHES = {"xgboost_nwp": "///", "lstm_nwp": "///", "chronos2_wind_nwp": "///",
           "chronos2_wind_bestmatch": "...", "chronos2_ft_uni": "xx", "chronos2_ft_cov": "xx",
           "chronos2_ft_nwp": "xx///"}
DASHED_MODELS = tuple(HATCHES)
EPISODE_MODELS = ("xgboost_nwp", "chronos2_uni", "chronos2_wind_nwp")
MONTHLY_MODELS = ("persistence", "xgboost", "chronos2_uni", "chronos2_wind_oracle")
INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _check_models(models) -> None:
    unknown = set(models) - set(MODEL_ORDER)
    if unknown:
        raise ValueError(f"no colour configured for models {sorted(unknown)}")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def plot_overall_mae(overall: pd.DataFrame, path: Path) -> Path:
    import matplotlib.pyplot as plt

    _check_models(overall["model"])
    leads = sorted(overall["lead"].unique())
    models = [m for m in MODEL_ORDER if m in set(overall["model"])]
    fig, axes = plt.subplots(1, len(leads), figsize=(4 * len(leads), 3.6), facecolor=SURFACE)
    for ax, lead in zip(np.atleast_1d(axes).ravel(), leads):
        values = overall[overall["lead"] == lead].set_index("model")["mae"].reindex(models)
        bars = ax.barh(range(len(models)), values, color=[MODEL_COLORS[m] for m in models],
                       height=0.7, edgecolor=SURFACE, linewidth=2)
        for bar, model in zip(bars, models):
            if model in HATCHES:
                bar.set_hatch(HATCHES[model])
        for i, v in enumerate(values):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(range(len(models)), models if lead == leads[0] else [])
        ax.invert_yaxis()
        ax.set_title(f"H = {lead} h", color=INK, fontsize=11)
        ax.set_xlabel("MAE (μg/m³)", color=INK_MUTED, fontsize=9)
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("PM2.5 forecast MAE, test year 2025 (12 stations)", color=INK, fontsize=12)
    return _save(fig, path)


def drift_counts_by_month(drift: pd.DataFrame, lead: int, months) -> pd.Series:
    """Number of ADWIN change points (all stations and models) per month for one lead."""
    subset = drift[drift["lead"] == lead]
    month = pd.to_datetime(subset["time"], utc=True).dt.tz_convert("Asia/Taipei")
    labels = month.dt.strftime("%Y-%m")
    return labels.value_counts().reindex(list(months), fill_value=0).astype(int)


def plot_monthly_with_drift(monthly: pd.DataFrame, drift: pd.DataFrame, lead: int,
                            path: Path, models=MONTHLY_MODELS) -> Path:
    import matplotlib.pyplot as plt

    _check_models(models)
    data = monthly[(monthly["lead"] == lead) & monthly["model"].isin(models)].copy()
    data["month"] = data["month"].astype(str)
    months = sorted(data["month"].unique())
    x = range(len(months))
    fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(9, 5.4), facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": (3, 1.2)})
    for model in models:
        series = data[data["model"] == model].set_index("month")["mae"].reindex(months)
        top.plot(x, series, color=MODEL_COLORS[model], linewidth=2, marker="o", markersize=4,
                 label=model, linestyle="--" if model in DASHED_MODELS else "-")
    top.set_ylabel("MAE (μg/m³)", color=INK_MUTED, fontsize=9)
    top.set_title(f"Monthly MAE at H = {lead} h", color=INK, fontsize=11, loc="left")
    top.legend(frameon=False, fontsize=8, ncol=2, labelcolor=INK)
    counts = drift_counts_by_month(drift, lead, months)
    bottom.bar(x, counts.to_numpy(), color=INK_MUTED, width=0.7)
    bottom.set_ylabel("ADWIN\nchanges", color=INK_MUTED, fontsize=9)
    bottom.set_xticks(list(x), [m[5:] for m in months])
    bottom.set_xlabel("Month of 2025", color=INK_MUTED, fontsize=9)
    for ax in (top, bottom):
        _style(ax)
    return _save(fig, path)


def peak_episode(preds: pd.DataFrame, model: str, lead: int,
                 window_hours: int = 72) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    """Station and a one-week window around its highest rolling-mean PM2.5 period."""
    sub = preds[(preds["model"] == model) & (preds["lead"] == lead)]
    best = None
    for station, group in sub.groupby("station"):
        rolling = group.set_index("target_time")["y"].sort_index().rolling(
            window_hours, min_periods=window_hours // 2).mean()
        if best is None or rolling.max() > best[1]:
            best = (station, rolling.max(), rolling.idxmax())
    station, _, peak = best
    return station, peak - pd.Timedelta(days=4), peak + pd.Timedelta(days=3)


def plot_episode(preds: pd.DataFrame, station: str, start, end, lead: int,
                 models=EPISODE_MODELS, path: Path = None) -> Path:
    import matplotlib.pyplot as plt

    _check_models(models)
    sub = preds[(preds["station"] == station) & (preds["lead"] == lead)
                & preds["target_time"].between(start, end)]
    fig, axes = plt.subplots(len(models), 1, sharex=True, sharey=True,
                             figsize=(9, 2.2 * len(models)), facecolor=SURFACE)
    for ax, model in zip(np.atleast_1d(axes).ravel(), models):
        m = sub[sub["model"] == model].sort_values("target_time")
        times = m["target_time"].dt.tz_localize(None)
        ax.fill_between(times, m["q10"], m["q90"], color=MODEL_COLORS[model], alpha=0.18,
                        linewidth=0, label="80% interval")
        ax.plot(times, m["q50"], color=MODEL_COLORS[model], linewidth=2, label="median")
        ax.plot(times, m["y"], color=INK, linewidth=1.2, label="observed")
        ax.axhline(PM25_AQI100_THRESHOLD, color=INK_MUTED, linewidth=1, linestyle=":")
        ax.set_title(model, color=INK, fontsize=10, loc="left")
        ax.set_ylabel("PM2.5", color=INK_MUTED, fontsize=9)
        _style(ax)
    first = np.atleast_1d(axes).ravel()[0]
    first.legend(frameon=False, fontsize=8, ncol=3, labelcolor=INK, loc="upper right")
    fig.suptitle(f"{STATION_EN.get(station, station)}, H = {lead} h forecasts (dotted: AQI 100 threshold)",
                 color=INK, fontsize=11)
    fig.autofmt_xdate()
    return _save(fig, path)


def main() -> None:  # pragma: no cover - file IO entry point
    import matplotlib

    matplotlib.use("Agg")
    tables, out = Path(TABLES_DIR), Path(FIGURES_DIR)
    overall = pd.read_csv(tables / "overall.csv")
    monthly = pd.read_csv(tables / "monthly.csv")
    drift = pd.read_csv(tables / "drift_points.csv")
    print(plot_overall_mae(overall, out / "overall_mae.png"))
    for lead in HORIZONS:
        print(plot_monthly_with_drift(monthly, drift, lead, out / f"monthly_drift_h{lead}.png"))
    from pm25.evaluation.report import load_predictions

    preds = load_predictions(Path(PREDICTIONS_DIR))
    lead = max(HORIZONS)
    station, start, end = peak_episode(preds, "xgboost", lead)
    models = tuple(m for m in EPISODE_MODELS if m in set(preds["model"]))
    print(plot_episode(preds, station, start, end, lead, models, out / "episode_case.png"))


if __name__ == "__main__":  # pragma: no cover
    main()
