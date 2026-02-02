# plots_and_tables_task3.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# -------------------------
# Global style (Times New Roman + user-tunable palettes)
# -------------------------
FONT_FAMILY = "Times New Roman"

# Per-plot palettes (edit these freely)
# - grouped_bar: 3 target colors
# - partner/industry/state/country: 3 target colors
BAR_PALETTES = {
    "grouped_bar": ["#E0A7F5", "#DB6FF0", "#C802F9"],  # high-saturation (Set1-like)
    "partner": ["#E0A7F5", "#DB6FF0", "#C802F9"],
    "industry": ["#E0A7F5", "#DB6FF0", "#C802F9"],
    "homestate": ["#E0A7F5", "#DB6FF0", "#C802F9"],
    "homecountry": ["#E0A7F5", "#DB6FF0", "#C802F9"],
}

# Heatmap options (edit these freely)
HEATMAP_CMAP = "YlGnBu"  # or: "viridis", "mako", etc.
HEATMAP_VMIN = 0.00
HEATMAP_VMAX = 1.75

# Significance markers for |beta| (heuristic, not p-values)
# You can change thresholds if you want fewer/more stars.
SIG_THRESHOLDS = [
    (1.25, "**"),
    (0.75, "*"),
]

# Apply global font + clean style (no heavy background blocks)
plt.rcParams.update({
    # Force Times New Roman everywhere (titles/axes/legend/ticks)
    "font.family": FONT_FAMILY,
    "font.serif": [FONT_FAMILY, "Times"],
    "font.sans-serif": [FONT_FAMILY],
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "axes.unicode_minus": False,
    "axes.titleweight": "bold",
})
sns.set_theme(style="white", context="talk", font=FONT_FAMILY)

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
RES = BASE_DIR / "results"
IN_XLSX = RES / "task3_hyperparam_sweep_and_effects.xlsx"

FIG_DIR = RES / "figures_task3"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_TEX = RES / "tables_task3.tex"


# -------------------------
# Helpers
# -------------------------
def must_exist(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def safe_read(xlsx: Path, sheet: str) -> pd.DataFrame:
    try:
        return pd.read_excel(xlsx, sheet_name=sheet)
    except Exception as e:
        raise RuntimeError(f"Failed reading sheet '{sheet}' from {xlsx}: {e}")


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    if required:
        raise KeyError(f"Cannot find any of {candidates} in columns: {cols}")
    return None


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + "l" * len(cols) + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(cols) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}".rstrip("0").rstrip("."))
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def rolling_smooth(x: np.ndarray, y: np.ndarray, bins: int = 20):
    df = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if len(df) < 10:
        return df["x"].values, df["y"].values
    df["bin"] = pd.qcut(df["x"], q=min(bins, max(5, len(df) // 10)), duplicates="drop")
    g = df.groupby("bin", as_index=False).agg(x=("x", "mean"), y=("y", "mean"))
    return g["x"].values, g["y"].values


def lowess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.35):
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        df = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
        sm = lowess(df["y"].values, df["x"].values, frac=frac, return_sorted=True)
        return sm[:, 0], sm[:, 1]
    except Exception:
        return rolling_smooth(x, y, bins=20)


def one_row_to_series(df: pd.DataFrame) -> pd.Series:
    """Take first row beta columns as float series."""
    if len(df) == 0:
        raise ValueError("Empty beta sheet.")
    beta_cols = [c for c in df.columns if "beta" in str(c).lower()]
    if len(beta_cols) == 0:
        raise KeyError(f"No beta columns found in: {list(df.columns)}")
    return df.iloc[0][beta_cols].astype(float)


def pretty_feature_index(beta_series_index) -> list[str]:
    """Convert beta column names to feature labels."""
    out = []
    for c in beta_series_index:
        s = str(c).lower()
        s = s.replace("beta_", "").replace("beta", "").strip("_")
        # unify naming
        s = s.replace("home_state", "homestate").replace("homecountry_region", "homecountry")
        out.append(s.capitalize())
    return out


# -------------------------
# Main
# -------------------------
def main():
    must_exist(IN_XLSX)

    # sheets needed
    betas_perf = safe_read(IN_XLSX, "betas5_Perf_WeeksSurvived")
    betas_judge = safe_read(IN_XLSX, "betas5_Judge_AvgJudgeTotal")
    betas_fan = safe_read(IN_XLSX, "betas5_Fan_AvgFanShare")

    hs = safe_read(IN_XLSX, "hyperparam_sweep")
    bestp = safe_read(IN_XLSX, "best_params")
    model_data = safe_read(IN_XLSX, "model_data_used")

    print("\n[DEBUG] hyperparam_sweep columns:", list(hs.columns))
    print("[DEBUG] best_params columns:", list(bestp.columns))
    print("[DEBUG] model_data_used columns:", list(model_data.columns))

    # -------------------------
    # Build mat (THIS FIXES NameError)
    # -------------------------
    s_perf = one_row_to_series(betas_perf)
    s_jdg = one_row_to_series(betas_judge)
    s_fan = one_row_to_series(betas_fan)

    beta_index = pretty_feature_index(s_perf.index)

    mat = pd.DataFrame(
        {
            "Performance(WeeksSurvived)": s_perf.values,
            "Judges(AvgJudgeTotal)": s_jdg.values,
            "Fans(AvgFanShare)": s_fan.values,
        },
        index=beta_index,
    )

    # -------------------------
    # 1) Main Figure 1: Impact Heatmap (publication-style)
    #   Fixes:
    #   - stable 3-decimal formatting
    #   - optional star markers (heuristic, magnitude-based)
    #   - clearer cell boundaries + subtle horizontal grid
    #   - label rotation <= 15 deg
    #   - colorbar range fixed to [0.00, 1.75]
    # -------------------------
    def sig_stars(val: float) -> str:
        v = float(abs(val))
        for thr, sym in SIG_THRESHOLDS:
            if v >= thr:
                return sym
        return ""

    # Build annotation strings ourselves so formatting is always consistent
    annot = mat.copy().astype(float)
    annot_str = annot.copy().astype(object)
    for r in range(annot.shape[0]):
        for c in range(annot.shape[1]):
            v = float(annot.iat[r, c])
            annot_str.iat[r, c] = f"{v:.3f}{sig_stars(v)}"

    # 1:1 canvas + extra left margin so row labels are fully visible
    fig, ax = plt.subplots(figsize=(7.6, 7.6))

    hm = sns.heatmap(
        mat.astype(float),
        ax=ax,
        annot=annot_str,
        fmt="",
        cmap=HEATMAP_CMAP,
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        cbar=True,
        cbar_kws={
            "shrink": 0.92,
            "ticks": np.linspace(HEATMAP_VMIN, HEATMAP_VMAX, 8),
            "label": r"$|\beta|$ (standardized, absolute)",
        },
        square=True,
        linewidths=0.5,
        linecolor="#D3D3D3",
        annot_kws={"size": 8},
    )

    # Subtle horizontal guide lines (extra clarity; low alpha)
    x0, x1 = ax.get_xlim()
    for y in range(mat.shape[0] + 1):
        ax.hlines(y, x0, x1, colors="#E0E0E0", linewidth=0.25, alpha=0.15)

    # Title with key info + star legend (two-line to avoid clipping)
    star_note = "(* |β|≥0.75, ** |β|≥1.25)" if len(SIG_THRESHOLDS) else ""
    ax.set_title(
        "Impact Heatmap — Absolute Standardized Coefficients (5 features × 3 targets)\n" + star_note,
        fontsize=11,
        pad=12,
        fontfamily=FONT_FAMILY,
    )
    # Layout: keep a generous left/bottom margin (prevents label truncation)
    fig.subplots_adjust(left=0.30, right=0.90, bottom=0.20, top=0.86)
    ax.tick_params(axis="y", pad=8)

    # Axis label style
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Tick labels: right-align y labels, center x labels, slightly larger font
    ax.set_xticklabels(ax.get_xticklabels(), rotation=12, ha="center", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=9)

    # Save with padding to avoid cropping (title + colorbar)
    fig.savefig(
        FIG_DIR / "T3_main_1_impact_heatmap.png",
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.25,
    )
    plt.close(fig)

    # -------------------------
    # 2) Main Figure 2: Grouped Bar (high-saturation palette)
    # -------------------------
    df_bar = mat.reset_index().rename(columns={"index": "Feature"})
    df_bar_m = df_bar.melt(id_vars="Feature", var_name="Target", value_name="AbsBeta")

    plt.figure(figsize=(11.5, 5.5))
    palette = BAR_PALETTES.get("grouped_bar", sns.color_palette("Set1", n_colors=df_bar_m["Target"].nunique()))

    ax = sns.barplot(
        data=df_bar_m,
        x="Feature",
        y="AbsBeta",
        hue="Target",
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.set_title("Grouped Bar: Feature Impact Differences Across Targets", fontsize=16, weight="bold", pad=12)
    ax.set_xlabel("Feature", fontsize=13)
    ax.set_ylabel(r"Absolute Standardized Coefficient $|\beta|$", fontsize=13)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Target", frameon=True, fontsize=11, title_fontsize=12, loc="upper left")
    for lab in ax.get_xticklabels():
        lab.set_fontfamily(FONT_FAMILY)
    for lab in ax.get_yticklabels():
        lab.set_fontfamily(FONT_FAMILY)
    if ax.get_legend() is not None:
        for t in ax.get_legend().get_texts():
            t.set_fontfamily(FONT_FAMILY)
        ax.get_legend().get_title().set_fontfamily(FONT_FAMILY)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T3_main_2_grouped_bar.png", dpi=350)
    plt.close()

    # -------------------------
    # 3) Main Figure 3: λ vs CV_R²
    # -------------------------
    col_target = pick_col(hs, ["target", "Target", "objective", "Objective", "y", "Y", "y_name", "label"])
    col_lambda = pick_col(hs, ["lambda", "Lambda", "l2", "alpha", "ridge_lambda"])
    col_r2mean = pick_col(hs, ["cv_r2_mean", "CV_R2_mean", "mean_cv_r2", "cv_r2", "r2_mean"])
    col_r2std = pick_col(hs, ["cv_r2_std", "CV_R2_std", "std_cv_r2", "r2_std"], required=False)

    target_map = {
        "Perf_WeeksSurvived": "Performance_WeeksSurvived",
        "Judge_AvgJudgeTotal": "Judges_AvgJudgeTotal",
        "Fan_AvgFanShare": "Fans_AvgFanShare",
    }
    hs["TargetPretty"] = hs[col_target].astype(str).map(target_map).fillna(hs[col_target].astype(str))

    plt.figure(figsize=(11, 6))
    ax = sns.lineplot(data=hs, x=col_lambda, y=col_r2mean, hue="TargetPretty", marker="o")

    if col_r2std is not None:
        for t, g in hs.groupby("TargetPretty"):
            g = g.sort_values(col_lambda)
            plt.fill_between(
                g[col_lambda],
                g[col_r2mean] - g[col_r2std],
                g[col_r2mean] + g[col_r2std],
                alpha=0.15,
            )

    ax.set_xscale("log")
    ax.set_title("Ridge Regularization Sweep: λ vs Cross-Validated $R^2$")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("CV $R^2$ (mean ± std)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T3_main_3_lambda_vs_cvr2.png", dpi=300)
    plt.close()

    # -------------------------
    # 4) Appendix: Age scatter + smooth (3 targets)
    # -------------------------
    md = model_data.copy()

    col_age = pick_col(md, ["Age", "celebrity_age_during_season", "age"])
    col_perf = pick_col(md, ["WeeksSurvived", "weeks_survived", "Performance"], required=True)
    col_judge = pick_col(md, ["AvgJudgeTotal", "avg_judge_total", "Judges"], required=True)
    col_fan = pick_col(md, ["AvgFanShare", "avg_fan_share", "Fans"], required=True)

    md = md.dropna(subset=[col_age, col_perf, col_judge, col_fan])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    targets = [
        (col_perf, "Performance (Weeks Survived)"),
        (col_judge, "Judges (Avg Judge Total)"),
        (col_fan, "Fans (Avg Fan Share)"),
    ]
    for ax, (col, title) in zip(axes, targets):
        ax.scatter(md[col_age], md[col], alpha=0.35, s=20)
        xs, ys = lowess_smooth(md[col_age].values, md[col].values, frac=0.35)
        ax.plot(xs, ys, linewidth=3)
        ax.set_title(title)
        ax.set_xlabel("Celebrity Age")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Outcome")
    plt.suptitle("Age Effect (Scatter + Smooth) — Judges vs Fans vs Performance", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "T3_app_1_age_scatter_smooth.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------
    # 5) Appendix grouped bars: Partner/Industry/HomeState/HomeCountry
    # -------------------------
    def grouped_top_plot(cat_col_guess: list[str], title: str, outname: str, top_n: int = 12, min_count: int = 3):
        cat_col = pick_col(md, cat_col_guess, required=False)
        if cat_col is None:
            print(f"[WARN] skip {outname} (no column match for {cat_col_guess})")
            return

        tmp = md.copy()
        tmp[cat_col] = tmp[cat_col].fillna("Unknown").astype(str)

        agg = tmp.groupby(cat_col, as_index=False).agg(
            Count=(cat_col, "count"),
            Performance=(col_perf, "mean"),
            Judges=(col_judge, "mean"),
            Fans=(col_fan, "mean"),
        )
        agg = agg[agg["Count"] >= min_count].copy()
        if len(agg) == 0:
            return

        agg = agg.sort_values("Performance", ascending=False).head(top_n)

        melt = agg.melt(
            id_vars=[cat_col, "Count"],
            value_vars=["Performance", "Judges", "Fans"],
            var_name="Target",
            value_name="MeanOutcome",
        )

        plt.figure(figsize=(14, 6))
        # pick palette by chart type (partner/industry/homestate/homecountry)
        pal_key = outname.lower()
        if "partner" in pal_key:
            pal = BAR_PALETTES.get("partner")
        elif "industry" in pal_key:
            pal = BAR_PALETTES.get("industry")
        elif "homestate" in pal_key or "state" in pal_key:
            pal = BAR_PALETTES.get("homestate")
        elif "homecountry" in pal_key or "country" in pal_key:
            pal = BAR_PALETTES.get("homecountry")
        else:
            pal = None
        if pal is None:
            pal = sns.color_palette("Set1", 3)

        ax = sns.barplot(
            data=melt,
            y=cat_col,
            x="MeanOutcome",
            hue="Target",
            palette=pal,
            edgecolor="black",
            linewidth=0.7,
        )
        ax.set_title(title + f" (Top {top_n}, count≥{min_count})")
        ax.set_xlabel("Mean outcome (within category)")
        ax.set_ylabel(cat_col)
        ax.tick_params(axis="y", labelsize=8)
        for lab in ax.get_yticklabels():
            lab.set_fontfamily(FONT_FAMILY)
        for lab in ax.get_xticklabels():
            lab.set_fontfamily(FONT_FAMILY)
        plt.tight_layout()
        plt.savefig(FIG_DIR / outname, dpi=300)
        plt.close()

    grouped_top_plot(
        ["Partner", "ballroom_partner", "pro_dancer", "ProDancer"],
        "Pro Dancer Effect (Descriptive): Mean Outcomes by Partner",
        "T3_app_2_partner_grouped_bar.png",
    )
    grouped_top_plot(
        ["Industry", "celebrity_industry"],
        "Industry Effect (Descriptive): Mean Outcomes by Celebrity Industry",
        "T3_app_3_industry_grouped_bar.png",
    )
    grouped_top_plot(
        ["HomeState", "celebrity_homestate", "homestate"],
        "Home-State Effect (Descriptive): Mean Outcomes by Celebrity Home State",
        "T3_app_4_homestate_grouped_bar.png",
    )
    grouped_top_plot(
        ["HomeCountry", "celebrity_homecountry/region", "homecountry", "homecountry_region"],
        "Home-Country Effect (Descriptive): Mean Outcomes by Celebrity Home Country/Region",
        "T3_app_5_homecountry_grouped_bar.png",
    )

    # -------------------------
    # Bonus: Plotly 3D surface of CV_R2 (export-safe layout)
    #   Fixes:
    #   - add a real title that is not clipped
    #   - increase margins so the full scene + title render into PNG
    #   - use a slightly taller canvas
    # -------------------------
    try:
        piv = hs.pivot_table(index="TargetPretty", columns=col_lambda, values=col_r2mean, aggfunc="mean")
        piv = piv.sort_index().sort_index(axis=1)

        z = piv.values
        x = np.array(piv.columns.values, dtype=float)
        y = np.arange(len(piv.index))
        X, Y = np.meshgrid(x, y)

        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=z)])

        # NOTE: Plotly static image export (kaleido) does NOT render LaTeX in titles reliably.
        # Use plain unicode text instead of $R^2$.
        title_text = "3D Surface: CV R² over (λ, target)"

        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=title_text,
                x=0.5,
                y=0.985,
                xanchor="center",
                yanchor="top",
                font=dict(family=FONT_FAMILY, size=26),
            ),
            scene=dict(
                xaxis_title="λ",
                yaxis_title="Target",
                zaxis_title="CV R²",
                # Keep a stable aspect so it doesn't look squashed, but avoid cropping.
                aspectmode="manual",
                aspectratio=dict(x=1.15, y=1.0, z=0.85),
            ),
            # Generous margins prevent title/axes/colorbar from being cropped
            margin=dict(l=90, r=90, t=150, b=120),
        )

        # Export with a 3:4 portrait canvas so the title and full 3D scene are visible.
        fig.write_image(
            str(FIG_DIR / "T3_bonus_3d_surface_lambda_targets.png"),
            width=1200,
            height=1600,
            scale=2,
        )
    except Exception as e:
        print("[WARN] Plotly 3D export skipped:", e)

    # -------------------------
    # Tables (LaTeX)
    # -------------------------
    bp = bestp.copy()
    col_bt = pick_col(bp, ["target", "Target", "objective", "Objective", "y_name", "label"], required=False)
    if col_bt is not None:
        bp[col_bt] = bp[col_bt].astype(str).map(target_map).fillna(bp[col_bt].astype(str))

    # keep columns if exist
    keep = [c for c in [
        col_bt,
        "Best_Lambda", "Best_TopK_Partner", "Best_TopK_Industry", "Best_TopK_HomeState", "Best_TopK_HomeCountry",
        "Best_CV_R2_mean", "Best_CV_RMSE_mean",
        "NumSamples", "NumFeatures"
    ] if c is not None and c in bp.columns]
    best_tbl = bp[keep].copy() if len(keep) else bp.head(10)

    beta_tbl = mat.copy().reset_index().rename(columns={"index": "Feature"})

    tex_parts = []
    tex_parts.append("% Auto-generated tables for Task 3 (do not edit by hand)")
    tex_parts.append("% Recommended: \\usepackage{booktabs}")
    tex_parts.append("")

    tex_parts.append(
        to_latex_table(
            best_tbl,
            caption="Ridge hyperparameter selection via cross-validation (best $\\lambda$ and top-K truncation per target).",
            label="tab:task3_best_params",
        )
    )
    tex_parts.append(
        to_latex_table(
            beta_tbl,
            caption="Estimated feature impacts (absolute standardized coefficients) across three targets.",
            label="tab:task3_beta_impact",
        )
    )

    OUT_TEX.write_text("\n".join(tex_parts), encoding="utf-8")

    print("\n✅ Task 3 figures saved to:", FIG_DIR)
    print("✅ Task 3 LaTeX tables written to:", OUT_TEX)


if __name__ == "__main__":
    main()