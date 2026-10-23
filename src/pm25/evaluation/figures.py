"""Report figures: overall MAE per model × lead, monthly MAE with ADWIN drift counts.

Run `uv run python -m pm25.evaluation.figures` after `pm25.evaluation.report`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from pm25.config import FIGURES_DIR, HORIZONS, TABLES_DIR

MODEL_ORDER = (
    "persistence", "xgboost", "lstm", "chronos_bolt_small", "chronos_bolt_base",
    "chronos2_uni", "chronos2_cov", "chronos2_wind_oracle",
)
# Fixed categorical order (validated palette, slots 1-8); colour follows the model.
MODEL_COLORS = dict(zip(MODEL_ORDER, (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)))
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
        ax.barh(range(len(models)), values, color=[MODEL_COLORS[m] for m in models],
                height=0.7, edgecolor=SURFACE, linewidth=2)
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
                 label=model)
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


if __name__ == "__main__":  # pragma: no cover
    main()
