"""
Task 3 Sensitivity Analysis (UPDATED to match task3_model_sweep.py)

Reads:
  results/task3_hyperparam_sweep_and_effects.xlsx
    - hyperparam_sweep (required)
    - best_params (optional)

Outputs:
  results/figures_task3_sensitivity/
    - S3_lambda_sensitivity_<Target>.png
    - S3_topk_partner_<Target>.png
    - S3_topk_industry_<Target>.png
    - S3_topk_state_<Target>.png
    - S3_topk_country_<Target>.png
    - S3_best_config_table.png

  results/task3_sensitivity_summary.xlsx
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
RES = BASE_DIR / "results"
IN_XLSX = RES / "task3_hyperparam_sweep_and_effects.xlsx"

FIG_DIR = RES / "figures_task3_sensitivity"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_XLSX = RES / "task3_sensitivity_summary.xlsx"


# -------------------------
# Global style (Times New Roman + clean journal look)
# -------------------------
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.6,
        "grid.color": "#C9C9C9",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


# -------------------------
# Helpers
# -------------------------
def must_exist(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")


def savefig(path: Path, dpi: int = 350):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def col(df: pd.DataFrame, cands: list[str]) -> str:
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"Missing columns among {cands}. Existing={list(df.columns)}")


def fmt_num(x: float, nd: int = 3) -> str:
    try:
        x = float(x)
        if not np.isfinite(x):
            return "N/A"
        return f"{x:.{nd}f}"
    except Exception:
        return "N/A"


def best_by_rmse_then_r2(df: pd.DataFrame) -> pd.Series:
    """Pick best row by min CV_RMSE_mean, tie-break by max CV_R2_mean."""
    c_rmse = col(df, ["CV_RMSE_mean"])
    c_r2 = col(df, ["CV_R2_mean"])
    d = df.copy()
    d[c_rmse] = pd.to_numeric(d[c_rmse], errors="coerce")
    d[c_r2] = pd.to_numeric(d[c_r2], errors="coerce")
    d = d.dropna(subset=[c_rmse]).copy()
    if len(d) == 0:
        return df.iloc[0]
    d = d.sort_values([c_rmse, c_r2], ascending=[True, False])
    return d.iloc[0]


def make_best_table_figure(best_df: pd.DataFrame, outpath: Path):
    """Compact journal-style table figure for the best hyperparams by target."""
    if len(best_df) == 0:
        return

    cols_show = [
        "Target",
        "Best_Lambda",
        "Best_TopK_Partner",
        "Best_TopK_Industry",
        "Best_TopK_HomeState",
        "Best_TopK_HomeCountry",
        "Best_CV_RMSE_mean",
        "Best_CV_R2_mean",
        "NumSamples",
        "NumFeatures",
    ]
    cols_show = [c for c in cols_show if c in best_df.columns]
    tbl = best_df[cols_show].copy()

    for c in ["Best_Lambda", "Best_CV_RMSE_mean", "Best_CV_R2_mean"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].apply(lambda v: fmt_num(pd.to_numeric(v, errors="coerce"), 3))

    fig, ax = plt.subplots(figsize=(14.0, 2.8 + 0.55 * len(tbl)))
    ax.axis("off")

    table = ax.table(
        cellText=tbl.values,
        colLabels=list(tbl.columns),
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.4)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#D3D3D3")
        if r == 0:
            cell.set_facecolor("#F8F9FA")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    ax.set_title("Task 3 — Best Hyperparameters by Target", pad=10)
    savefig(outpath)


# -------------------------
# Sensitivity computations
# -------------------------
def lambda_frontier(df_t: pd.DataFrame) -> pd.DataFrame:
    """For each Lambda, take the best achievable (min RMSE) across TopK combos."""
    c_lam = col(df_t, ["Lambda"])
    c_rmse = col(df_t, ["CV_RMSE_mean"])
    c_r2 = col(df_t, ["CV_R2_mean"])

    d = df_t.copy()
    d[c_lam] = pd.to_numeric(d[c_lam], errors="coerce")
    d[c_rmse] = pd.to_numeric(d[c_rmse], errors="coerce")
    d[c_r2] = pd.to_numeric(d[c_r2], errors="coerce")
    d = d.dropna(subset=[c_lam, c_rmse]).copy()

    rows = []
    for lam, g in d.groupby(c_lam):
        best = best_by_rmse_then_r2(g)
        rows.append(
            {
                "Lambda": float(lam),
                "Best_RMSE": float(best[c_rmse]),
                "Best_R2": float(best[c_r2]) if np.isfinite(best[c_r2]) else np.nan,
                "Best_TopK_Partner": best.get("TopK_Partner", np.nan),
                "Best_TopK_Industry": best.get("TopK_Industry", np.nan),
                "Best_TopK_HomeState": best.get("TopK_HomeState", np.nan),
                "Best_TopK_HomeCountry": best.get("TopK_HomeCountry", np.nan),
                "NumFeatures": best.get("NumFeatures", np.nan),
                "NumSamples": best.get("NumSamples", np.nan),
            }
        )

    out = pd.DataFrame(rows).sort_values("Lambda")
    return out


def topk_frontier(df_t: pd.DataFrame, fixed_lambda: float, focus_col: str) -> pd.DataFrame:
    """Fix Lambda, then for each value of one TopK column, optimize the other TopKs by RMSE."""
    c_lam = col(df_t, ["Lambda"])
    c_rmse = col(df_t, ["CV_RMSE_mean"])
    c_r2 = col(df_t, ["CV_R2_mean"])

    d = df_t.copy()
    d[c_lam] = pd.to_numeric(d[c_lam], errors="coerce")
    d[c_rmse] = pd.to_numeric(d[c_rmse], errors="coerce")
    d[c_r2] = pd.to_numeric(d[c_r2], errors="coerce")

    d = d.dropna(subset=[c_lam, c_rmse]).copy()
    d = d[np.isclose(d[c_lam].astype(float), float(fixed_lambda))].copy()
    if len(d) == 0:
        return pd.DataFrame()

    if focus_col not in d.columns:
        raise KeyError(f"Missing column: {focus_col}")

    rows = []
    for v, g in d.groupby(focus_col):
        best = best_by_rmse_then_r2(g)
        rows.append(
            {
                focus_col: v,
                "Best_RMSE": float(best[c_rmse]),
                "Best_R2": float(best[c_r2]) if np.isfinite(best[c_r2]) else np.nan,
                "TopK_Partner": best.get("TopK_Partner", np.nan),
                "TopK_Industry": best.get("TopK_Industry", np.nan),
                "TopK_HomeState": best.get("TopK_HomeState", np.nan),
                "TopK_HomeCountry": best.get("TopK_HomeCountry", np.nan),
                "NumFeatures": best.get("NumFeatures", np.nan),
            }
        )

    out = pd.DataFrame(rows)
    out[focus_col] = pd.to_numeric(out[focus_col], errors="ignore")
    out = out.sort_values(focus_col)
    return out


# -------------------------
# Plotting
# -------------------------
def plot_lambda_sensitivity(front: pd.DataFrame, target: str, outpath: Path):
    if len(front) == 0:
        return

    fig, ax1 = plt.subplots(figsize=(10.8, 6.2))

    x = front["Lambda"].astype(float).values
    y = front["Best_RMSE"].astype(float).values

    ax1.plot(x, y, linewidth=2.2, marker="o", markersize=5)
    ax1.set_xlabel(r"Ridge strength $\lambda$")
    ax1.set_ylabel("Best CV RMSE (standardized y)")
    ax1.set_title(f"Task 3 — Lambda Sensitivity: {target}")

    # lambda has 0 -> use symlog
    if np.any(x == 0):
        ax1.set_xscale("symlog", linthresh=0.1)
    else:
        ax1.set_xscale("log")

    ax1.grid(True, alpha=0.18)

    ax2 = ax1.twinx()
    r2 = front["Best_R2"].astype(float).values
    ax2.plot(x, r2, linewidth=2.0, linestyle="--", marker="s", markersize=4, alpha=0.9)
    ax2.set_ylabel("Best CV R²")

    i_best = int(np.nanargmin(y))
    ax1.scatter([x[i_best]], [y[i_best]], s=70)
    ax1.text(
        0.02,
        0.98,
        f"Best: λ={x[i_best]:g}\nRMSE={y[i_best]:.3f}\nR²={r2[i_best]:.3f}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", fc="white", ec="#333333", lw=0.6, alpha=0.95),
    )

    savefig(outpath)


def plot_topk_sensitivity(front: pd.DataFrame, focus_col: str, target: str, fixed_lambda: float, outpath: Path):
    if len(front) == 0:
        return

    fig, ax = plt.subplots(figsize=(10.6, 6.0))

    x = pd.to_numeric(front[focus_col], errors="coerce").values
    y = front["Best_RMSE"].astype(float).values

    ax.plot(x, y, linewidth=2.2, marker="o", markersize=5)
    ax.set_xlabel(focus_col)
    ax.set_ylabel("Best CV RMSE (standardized y)")
    ax.set_title(f"Task 3 — TopK Sensitivity @ λ={fixed_lambda:g}: {target}\n(focus: {focus_col})")
    ax.grid(True, alpha=0.18)

    i_best = int(np.nanargmin(y))
    ax.scatter([x[i_best]], [y[i_best]], s=70)
    ax.text(
        0.02,
        0.98,
        f"Best {focus_col}={x[i_best]:g}\nRMSE={y[i_best]:.3f}\nR²={float(front['Best_R2'].iloc[i_best]):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", fc="white", ec="#333333", lw=0.6, alpha=0.95),
    )

    savefig(outpath)


# -------------------------
# Main
# -------------------------
def main():
    must_exist(IN_XLSX)

    xl = pd.ExcelFile(IN_XLSX)
    if "hyperparam_sweep" not in xl.sheet_names:
        raise KeyError(f"Sheet 'hyperparam_sweep' not found. Existing: {xl.sheet_names}")

    sweep = pd.read_excel(IN_XLSX, sheet_name="hyperparam_sweep")

    # Best params sheet is produced by task3_model_sweep.py; if missing, we derive it
    if "best_params" in xl.sheet_names:
        best_df = pd.read_excel(IN_XLSX, sheet_name="best_params")
    else:
        c_t = col(sweep, ["Target"])
        rows = []
        for t, g in sweep.groupby(c_t):
            best = best_by_rmse_then_r2(g)
            rows.append(
                {
                    "Target": t,
                    "Best_Lambda": best.get("Lambda", np.nan),
                    "Best_TopK_Partner": best.get("TopK_Partner", np.nan),
                    "Best_TopK_Industry": best.get("TopK_Industry", np.nan),
                    "Best_TopK_HomeState": best.get("TopK_HomeState", np.nan),
                    "Best_TopK_HomeCountry": best.get("TopK_HomeCountry", np.nan),
                    "Best_CV_RMSE_mean": best.get("CV_RMSE_mean", np.nan),
                    "Best_CV_R2_mean": best.get("CV_R2_mean", np.nan),
                    "NumSamples": best.get("NumSamples", np.nan),
                    "NumFeatures": best.get("NumFeatures", np.nan),
                }
            )
        best_df = pd.DataFrame(rows)

    c_target = col(sweep, ["Target"])

    lambda_tables = {}
    topk_tables = []

    for target, df_t in sweep.groupby(c_target):
        # Lambda frontier (min RMSE over TopK combos for each lambda)
        lam_front = lambda_frontier(df_t)
        lambda_tables[str(target)] = lam_front
        plot_lambda_sensitivity(lam_front, str(target), FIG_DIR / f"S3_lambda_sensitivity_{str(target)}.png")

        # Determine fixed lambda from best_params if possible
        fixed_lam = np.nan
        if "Target" in best_df.columns and "Best_Lambda" in best_df.columns:
            row = best_df[best_df["Target"].astype(str) == str(target)]
            if len(row) >= 1:
                fixed_lam = pd.to_numeric(row.iloc[0]["Best_Lambda"], errors="coerce")

        if not np.isfinite(fixed_lam) and len(lam_front) > 0:
            fixed_lam = float(lam_front.iloc[int(np.nanargmin(lam_front["Best_RMSE"].values))]["Lambda"])

        # TopK frontiers at fixed lambda
        for focus in ["TopK_Partner", "TopK_Industry", "TopK_HomeState", "TopK_HomeCountry"]:
            if focus not in df_t.columns:
                continue
            tk_front = topk_frontier(df_t, float(fixed_lam), focus)
            if len(tk_front) == 0:
                continue

            tk_front.insert(0, "Target", str(target))
            tk_front.insert(1, "Fixed_Lambda", float(fixed_lam))
            tk_front.insert(2, "Focus", focus)
            topk_tables.append(tk_front)

            fname = {
                "TopK_Partner": "S3_topk_partner",
                "TopK_Industry": "S3_topk_industry",
                "TopK_HomeState": "S3_topk_state",
                "TopK_HomeCountry": "S3_topk_country",
            }.get(focus, f"S3_topk_{focus}")

            plot_topk_sensitivity(
                tk_front,
                focus_col=focus,
                target=str(target),
                fixed_lambda=float(fixed_lam),
                outpath=FIG_DIR / f"{fname}_{str(target)}.png",
            )

    # Best config summary table figure
    make_best_table_figure(best_df, FIG_DIR / "S3_best_config_table.png")

    # Write XLSX summary
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        best_df.to_excel(w, index=False, sheet_name="best_params")
        for t, tab in lambda_tables.items():
            tab.to_excel(w, index=False, sheet_name=f"lambda_front_{t}"[:31])
        if len(topk_tables) > 0:
            pd.concat(topk_tables, ignore_index=True).to_excel(w, index=False, sheet_name="topk_frontiers")

    print("✅ Task3 sensitivity outputs written:")
    print("  Figures ->", FIG_DIR)
    print("  Summary ->", OUT_XLSX)


if __name__ == "__main__":
    main()