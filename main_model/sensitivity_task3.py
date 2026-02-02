# task3_sensitivity_analysis.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# ======================================================
# Style / Palette (EDIT HERE)
# ======================================================
FONT_FAMILY = "Times New Roman"

# Paper-style palette (editable)
PALETTE = {
    "perf": "#202EEB",      # deep blue
    "judge": "#DA2F43",     # warm orange/gray
    "fan": "#10DF4A",       # green
    "neutral": "#5B4848",
    "grid": "#E0E0E0",
}

# Bar-chart palette (editable). Keys must match Target names used in the report.
BAR_COLORS = {
    "Performance_WeeksSurvived": "#58A9FF",  # deep blue
    "Judges_AvgJudgeTotal": "#787AFF",       # warm sand
    "Fans_AvgFanShare": "#EA61F9",          # green
}
BAR_ALPHA = 0.75
SCATTER_ALPHA = 0.55
SCATTER_SIZE = 0

# Seed-stability boxplot colors (editable)
# Keys must match the `Target` strings in the sensitivity report.
SEED_BOX_COLORS = {
    "Performance_WeeksSurvived": "#58A9FF",
    "Judges_AvgJudgeTotal": "#787AFF",
    "Fans_AvgFanShare": "#EA61F9",
}
SEED_BOX_ALPHA = 0.55          # box face transparency
SEED_STRIP_COLOR = "#BA98E0"   # jitter points color
SEED_STRIP_ALPHA = 0.55        # jitter points transparency
SEED_STRIP_SIZE = 0          # jitter points size

ALPHA_FILL = 0.25
LINEWIDTH = 1.6
DPI = 320

# Sensitivity figures you want to keep (set False to skip)
KEEP_FIG_SEED_BOX = True
KEEP_FIG_BETA_CI = True
KEEP_FIG_LAMBDA_CURVE = True
KEEP_FIG_TOPK = False  # usually too many panels
KEEP_FIG_BETA_VIOLIN = False


# ======================================================
# Paths
# ======================================================
BASE = Path(__file__).resolve().parent
RES = BASE / "results"
IN_XLSX = RES / "task3_hyperparam_sweep_and_effects.xlsx"
OUT_XLSX = RES / "task3_sensitivity_report.xlsx"
FIG_DIR = RES / "figures_task3_sensitivity"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_TEX = RES / "tables_task3_sensitivity.tex"
OUT_MD = RES / "task3_sensitivity_summary.md"


# ======================================================
# Helpers (match your Task3 logic)
# ======================================================
def must_exist(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")


def ridge_beta(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    if lam == 0.0:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return b
    XtX = X.T @ X
    I = np.eye(X.shape[1])
    I[0, 0] = 0.0  # do not penalize intercept
    return np.linalg.solve(XtX + lam * I, X.T @ y)


def rmse_r2(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, r2


def standardize_train_apply(y_train: np.ndarray, y_test: np.ndarray):
    mu = float(np.mean(y_train))
    sd = float(np.std(y_train))
    if not np.isfinite(sd) or sd <= 1e-12:
        sd = 1.0
    return (y_train - mu) / sd, (y_test - mu) / sd, mu, sd


def kfold_indices(n: int, k: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        yield tr, te


def cap_top_k(series: pd.Series, k: int, other: str = "Other") -> pd.Series:
    series = series.fillna("Unknown").astype(str)
    if k is None or k <= 0:
        return series
    vc = series.value_counts(dropna=False)
    keep = set(vc.head(k).index.tolist())
    return series.where(series.isin(keep), other=other)


def build_design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    # numeric
    age = pd.to_numeric(df["Age"], errors="coerce").astype(float)
    age_z = (age - age.mean()) / (age.std() if age.std() != 0 else 1.0)

    X_parts = [age_z.to_numpy()[:, None]]
    feat_names = ["Age_z"]

    groups = [
        ("Partner", "Partner_c"),
        ("Industry", "Industry_c"),
        ("HomeState", "HomeState_c"),
        ("HomeCountry", "HomeCountry_c"),
    ]
    for prefix, col in groups:
        oh = pd.get_dummies(df[col].astype(str), prefix=prefix, drop_first=True)
        X_parts.append(oh.to_numpy())
        feat_names.extend(list(oh.columns))

    X = np.concatenate(X_parts, axis=1)
    X = np.column_stack([np.ones(len(df)), X])  # intercept
    feat_names = ["Intercept"] + feat_names
    return X, feat_names


def group_beta_strength(coefs: pd.Series) -> dict:
    def l2_of_prefix(prefix: str) -> float:
        v = coefs[coefs.index.str.startswith(prefix + "_")].to_numpy(dtype=float)
        return float(np.sqrt(np.sum(v * v))) if v.size else 0.0

    return {
        "beta_age": float(abs(coefs.get("Age_z", 0.0))),
        "beta_partner": l2_of_prefix("Partner"),
        "beta_industry": l2_of_prefix("Industry"),
        "beta_homestate": l2_of_prefix("HomeState"),
        "beta_homecountry": l2_of_prefix("HomeCountry"),
    }


def bootstrap_ci(vals: np.ndarray, B: int = 300, alpha: float = 0.05, seed: int = 123):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 3:
        return np.nan, np.nan, np.nan
    boots = []
    n = len(vals)
    for _ in range(B):
        sample = rng.choice(vals, size=n, replace=True)
        boots.append(float(np.mean(sample)))
    boots = np.array(boots, dtype=float)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(np.mean(vals)), lo, hi


# ======================================================
# Plot helpers (post-analysis of the generated report)
# ======================================================
def set_times_new_roman():
    """Global plotting style: Times New Roman + journal-friendly minimalism."""
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.serif": [FONT_FAMILY, "Times", "DejaVu Serif"],
        # Make mathtext closer to Times New Roman
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        "axes.unicode_minus": False,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.edgecolor": PALETTE["neutral"],
        "axes.linewidth": 0.6,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
    })

    # seaborn: no background blocks, thin axes, inherit font
    sns.set_theme(style="white", context="paper", font=FONT_FAMILY)


def savefig(path: Path, dpi: int | None = None):
    plt.tight_layout()
    plt.savefig(path, dpi=(DPI if dpi is None else dpi), bbox_inches="tight")
    plt.close()


# ---------------------------------------------
# Helper: Resolve bar colors for bar+scatter plots
# ---------------------------------------------
def resolve_bar_colors(targets: list[str], user_colors: dict[str, str] | None = None) -> list[str]:
    """Resolve bar colors in a stable order; user_colors overrides defaults."""
    cmap = dict(BAR_COLORS)
    if user_colors:
        cmap.update(user_colors)
    return [cmap.get(str(t), PALETTE["neutral"]) for t in targets]


# ---------------------------------------------
# Helper: Render stats table as a figure
# ---------------------------------------------
def render_stats_table(df_stats: pd.DataFrame, title: str, outpath: Path):
    """Render a clean journal-style stats table as a figure."""
    fig, ax = plt.subplots(figsize=(8.6, 2.6))
    ax.axis("off")

    # Format numbers
    fmt = df_stats.copy()
    for c in fmt.columns:
        if c.lower() in {"target"}:
            continue
        fmt[c] = fmt[c].map(lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))

    table = ax.table(
        cellText=fmt.values,
        colLabels=fmt.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    # Light borders
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#F8F9FA")
            cell.set_text_props(weight="bold", color=PALETTE["neutral"])
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color=PALETTE["neutral"])

    ax.set_title(title, weight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


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
            if isinstance(v, (float, np.floating)):
                vals.append(f"{float(v):.3f}".rstrip("0").rstrip("."))
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


def analyze_generated_excel(report_xlsx: Path):
    """Analyze `task3_sensitivity_report.xlsx` and generate figures + LaTeX tables."""
    set_times_new_roman()

    xls = pd.ExcelFile(report_xlsx)
    needed = {
        "seed_stability",
        "beta_stability_by_seed",
        "beta_CI_bootstrap",
        "topk_sensitivity",
        "lambda_curve_sensitivity",
    }
    missing = sorted(list(needed - set(xls.sheet_names)))
    if missing:
        raise KeyError(f"Sensitivity report is missing sheets: {missing}. Found: {xls.sheet_names}")

    seed = pd.read_excel(report_xlsx, sheet_name="seed_stability")
    beta_seed = pd.read_excel(report_xlsx, sheet_name="beta_stability_by_seed")
    beta_ci = pd.read_excel(report_xlsx, sheet_name="beta_CI_bootstrap")
    topk = pd.read_excel(report_xlsx, sheet_name="topk_sensitivity")
    lamc = pd.read_excel(report_xlsx, sheet_name="lambda_curve_sensitivity")

    # -------------------------
    # A) Seed stability: enhanced boxplot + inference note + summary table
    # -------------------------
    if KEEP_FIG_SEED_BOX and "Target" in seed.columns and "CV_R2_mean" in seed.columns:
        plot_df = seed[["Target", "CV_R2_mean"]].dropna().copy()

        # Stats summary per target
        stats = (plot_df.groupby("Target", as_index=False)
                 .agg(
                     N=("CV_R2_mean", "count"),
                     Mean=("CV_R2_mean", "mean"),
                     SD=("CV_R2_mean", "std"),
                     Median=("CV_R2_mean", "median"),
                     Q1=("CV_R2_mean", lambda x: float(np.quantile(x, 0.25))),
                     Q3=("CV_R2_mean", lambda x: float(np.quantile(x, 0.75))),
                 ))
        stats["IQR"] = stats["Q3"] - stats["Q1"]

        # Inference: Kruskal-Wallis across targets (robust, small n)
        p_kw = np.nan
        try:
            from scipy.stats import kruskal
            groups = [g["CV_R2_mean"].values for _, g in plot_df.groupby("Target")]
            if len(groups) >= 2:
                _, p_kw = kruskal(*groups)
        except Exception:
            p_kw = np.nan

        plt.figure(figsize=(8.8, 4.6))
        # User-tunable palette for this figure
        box_palette = {t: SEED_BOX_COLORS.get(str(t), PALETTE["neutral"]) for t in plot_df["Target"].astype(str).unique()}

        ax = sns.boxplot(
            data=plot_df,
            x="Target",
            y="CV_R2_mean",
            width=0.55,
            fliersize=1.2,
            palette=box_palette,
            saturation=1.0,
            boxprops=dict(edgecolor=PALETTE["neutral"], linewidth=0.9),
            medianprops=dict(color=PALETTE["neutral"], linewidth=1.2),
            whiskerprops=dict(color=PALETTE["neutral"], linewidth=0.9),
            capprops=dict(color=PALETTE["neutral"], linewidth=0.9),
        )

        # Apply transparency to box faces (seaborn returns matplotlib artists)
        for patch in ax.patches:
            try:
                patch.set_alpha(SEED_BOX_ALPHA)
            except Exception:
                pass

        # Add jittered points (distribution detail)
        sns.stripplot(
            data=plot_df,
            x="Target",
            y="CV_R2_mean",
            color=SEED_STRIP_COLOR,
            size=SEED_STRIP_SIZE,
            alpha=SEED_STRIP_ALPHA,
            jitter=0.18,
            ax=ax,
        )

        title = "Seed Stability of CV $R^2$ (5-fold CV)"
        if np.isfinite(p_kw):
            star = "*" if p_kw < 0.05 else ""
            title += f"  |  Kruskal–Wallis p={p_kw:.3f}{star}"
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("CV $R^2$")
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        plt.xticks(rotation=10, ha="right")
        ax.grid(True, axis="y")
        savefig(FIG_DIR / "T3S_1_seed_stability_box.png")

        # Separate stats table figure
        render_stats_table(
            stats[["Target", "N", "Mean", "SD", "Median", "IQR"]],
            title="Seed Stability Summary (CV $R^2$) — Mean/SD/Median/IQR",
            outpath=FIG_DIR / "T3S_1b_seed_stability_stats_table.png",
        )

    # -------------------------
    # B) Beta stability across seeds (violin)
    # -------------------------
    beta_cols = [c for c in beta_seed.columns if c.startswith("beta_")]
    if KEEP_FIG_BETA_VIOLIN and "Target" in beta_seed.columns and beta_cols:
        b_long = beta_seed.melt(id_vars=["Target", "Seed"], value_vars=beta_cols,
                               var_name="BetaGroup", value_name="Value")
        plt.figure(figsize=(10.5, 5.2))
        ax = sns.violinplot(data=b_long, x="BetaGroup", y="Value", hue="Target",
                            cut=0, inner="quartile", linewidth=0.8)
        ax.set_title("Stability of Group-Level Effect Sizes (|β|) Across Random Seeds")
        ax.set_xlabel("Feature group")
        ax.set_ylabel("Group effect size (L2 / |Age|)")
        ax.legend(title="Target", frameon=True, loc="upper right")
        plt.xticks(rotation=12, ha="right")
        savefig(FIG_DIR / "T3S_2_beta_stability_violin.png")

    # -------------------------
    # C) Bootstrap CI of beta means: CI dot plot + effect labels + zero line
    # -------------------------
    if KEEP_FIG_BETA_CI and set(["Target", "BetaGroup", "Mean", "CI_low", "CI_high"]).issubset(beta_ci.columns):
        order = ["beta_age", "beta_partner", "beta_industry", "beta_homestate", "beta_homecountry"]
        beta_ci2 = beta_ci.copy()
        beta_ci2["BetaGroup"] = pd.Categorical(beta_ci2["BetaGroup"], categories=order, ordered=True)
        beta_ci2 = beta_ci2.sort_values(["BetaGroup", "Target"]).copy()

        plt.figure(figsize=(10.6, 5.4))
        ax = plt.gca()

        targets = list(beta_ci2["Target"].astype(str).unique())
        offsets = np.linspace(-0.22, 0.22, num=max(1, len(targets)))
        xs = np.arange(len(order), dtype=float)

        color_map = {
            "Performance_WeeksSurvived": PALETTE["perf"],
            "Judges_AvgJudgeTotal": PALETTE["judge"],
            "Fans_AvgFanShare": PALETTE["fan"],
        }

        # zero line (reference)
        ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0, zorder=0)

        for t, off in zip(targets, offsets):
            sub = beta_ci2[beta_ci2["Target"].astype(str) == str(t)].copy()
            sub = sub.set_index("BetaGroup").reindex(order)
            y = sub["Mean"].to_numpy(float)
            lo = sub["CI_low"].to_numpy(float)
            hi = sub["CI_high"].to_numpy(float)

            c = color_map.get(str(t), PALETTE["neutral"])

            # CI segments + dots
            for i in range(len(order)):
                if not (np.isfinite(y[i]) and np.isfinite(lo[i]) and np.isfinite(hi[i])):
                    continue
                xi = xs[i] + off
                ax.plot([xi, xi], [lo[i], hi[i]], color=c, linewidth=1.6, alpha=0.95)
                ax.scatter([xi], [y[i]], s=42, color=c, edgecolor=PALETTE["neutral"], linewidth=0.6, zorder=3)

                # effect label
                ax.annotate(
                    f"{y[i]:.3f}",
                    xy=(xi, y[i]),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=PALETTE["neutral"],
                )

        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=12, ha="right")
        ax.set_title("Bootstrap 95% CI of Mean Group Effects (CI Dot Plot)")
        ax.set_xlabel("Feature group")
        ax.set_ylabel("Mean effect size")
        ax.grid(True, axis="y")
        ax.legend(
            handles=[plt.Line2D([0], [0], color=color_map.get(t, PALETTE["neutral"]), marker='o', linestyle='') for t in targets],
            labels=targets,
            title="Target",
            frameon=True,
            loc="upper right",
        )
        savefig(FIG_DIR / "T3S_3_beta_ci_dotplot.png")

        # ---- Additional figure: Bar chart + per-seed scatter (journal-style) ----
        try:
            # Prepare CI means in long form
            ci_long = beta_ci2.copy()
            ci_long["BetaGroup"] = ci_long["BetaGroup"].astype(str)
            ci_long["Target"] = ci_long["Target"].astype(str)

            # Prepare per-seed beta values to overlay as scatter
            beta_seed_long = beta_seed.copy()
            beta_seed_long["Target"] = beta_seed_long["Target"].astype(str)
            keep_cols = [c for c in ["beta_age", "beta_partner", "beta_industry", "beta_homestate", "beta_homecountry"] if c in beta_seed_long.columns]
            if keep_cols:
                beta_seed_long = beta_seed_long.melt(
                    id_vars=["Target", "Seed"],
                    value_vars=keep_cols,
                    var_name="BetaGroup",
                    value_name="Value",
                )

            # Order + aesthetics
            ci_long["BetaGroup"] = pd.Categorical(ci_long["BetaGroup"], categories=order, ordered=True)
            ci_long = ci_long.sort_values(["BetaGroup", "Target"]).copy()
            targets2 = list(ci_long["Target"].unique())
            colors = resolve_bar_colors(targets2)

            fig, ax = plt.subplots(figsize=(11.2, 5.6))

            x = np.arange(len(order), dtype=float)
            nT = max(1, len(targets2))
            width = 0.22 if nT >= 3 else 0.28
            offsets2 = np.linspace(-width, width, num=nT)

            for t_idx, (t, off, colr) in enumerate(zip(targets2, offsets2, colors)):
                sub = ci_long[ci_long["Target"] == t].set_index("BetaGroup").reindex(order)
                means = sub["Mean"].to_numpy(float)
                lo = sub["CI_low"].to_numpy(float)
                hi = sub["CI_high"].to_numpy(float)
                yerr = np.vstack([means - lo, hi - means])

                ax.bar(
                    x + off,
                    means,
                    width=width,
                    color=colr,
                    alpha=BAR_ALPHA,
                    edgecolor=PALETTE["neutral"],
                    linewidth=0.6,
                    label=t,
                    zorder=2,
                )
                # CI error bars
                ax.errorbar(
                    x + off,
                    means,
                    yerr=yerr,
                    fmt="none",
                    ecolor=PALETTE["neutral"],
                    elinewidth=1.0,
                    capsize=3,
                    capthick=1.0,
                    alpha=0.95,
                    zorder=3,
                )

                # Scatter overlay (per-seed): jitter around each bar
                if keep_cols and len(beta_seed_long) > 0:
                    pts = beta_seed_long[beta_seed_long["Target"] == t].copy()
                    pts["BetaGroup"] = pd.Categorical(pts["BetaGroup"], categories=order, ordered=True)
                    for i_bg, bg in enumerate(order):
                        v = pts.loc[pts["BetaGroup"] == bg, "Value"].to_numpy(float)
                        v = v[np.isfinite(v)]
                        if v.size == 0:
                            continue
                        rng = np.random.default_rng(2026 + t_idx * 31 + i_bg)
                        jitter = rng.uniform(-width * 0.25, width * 0.25, size=v.size)
                        ax.scatter(
                            np.full(v.size, x[i_bg] + off) + jitter,
                            v,
                            s=SCATTER_SIZE,
                            color=PALETTE["neutral"],
                            alpha=SCATTER_ALPHA,
                            linewidths=0,
                            zorder=4,
                        )

            ax.axhline(0.0, color="#888888", linestyle="--", linewidth=1.0, zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels(order, rotation=12, ha="right")
            ax.set_title("Group Effects: Mean ± 95% CI with Per-Seed Scatter")
            ax.set_xlabel("Feature group")
            ax.set_ylabel("Effect size")
            ax.grid(True, axis="y")
            ax.legend(title="Target", frameon=True, loc="upper right")

            savefig(FIG_DIR / "T3S_3b_beta_bar_scatter.png")
        except Exception as e:
            print("[WARN] Skipped bar+scatter beta figure:", e)

    # -------------------------
    # D) Lambda curve sensitivity: LOESS + CI band + best-λ marker
    # -------------------------
    if KEEP_FIG_LAMBDA_CURVE and set(["Target", "Lambda", "CV_R2_mean"]).issubset(lamc.columns):
        has_std = "CV_R2_std" in lamc.columns

        # Best λ from best_params (if available)
        best_lambda_map = {}
        try:
            bestp = pd.read_excel(report_xlsx, sheet_name="seed_stability")  # fallback (not best params)
        except Exception:
            bestp = None

        # Prefer reading from the original task3 file if present next to report
        # We infer it from directory
        try:
            task3_xlsx = RES / "task3_hyperparam_sweep_and_effects.xlsx"
            if task3_xlsx.exists():
                bp = pd.read_excel(task3_xlsx, sheet_name="best_params")
                if set(["Target", "Best_Lambda"]).issubset(bp.columns):
                    best_lambda_map = dict(zip(bp["Target"].astype(str), bp["Best_Lambda"].astype(float)))
        except Exception:
            best_lambda_map = {}

        plt.figure(figsize=(10.2, 5.4))
        ax = plt.gca()

        color_map = {
            "Performance_WeeksSurvived": PALETTE["perf"],
            "Judges_AvgJudgeTotal": PALETTE["judge"],
            "Fans_AvgFanShare": PALETTE["fan"],
        }

        def lowess_smooth(x, y, frac=0.45):
            from statsmodels.nonparametric.smoothers_lowess import lowess
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) < 3:
                return x, y
            loess = lowess(y, x, frac=frac, return_sorted=True)
            return loess[:, 0], loess[:, 1]

        for t, g in lamc.groupby("Target"):
            g = g.sort_values("Lambda").copy()
            x = g["Lambda"].to_numpy(float)
            y = g["CV_R2_mean"].to_numpy(float)
            ystd = g["CV_R2_std"].to_numpy(float) if has_std else None

            # LOESS smooth (on log-lambda axis)
            xlog = np.log10(np.clip(x, 1e-12, None))
            xs, ys = lowess_smooth(xlog, y, frac=0.45)
            x_smooth = 10 ** xs

            c = color_map.get(str(t), PALETTE["neutral"])
            ax.plot(x_smooth, ys, linewidth=1.8, color=c, label=str(t) + " (LOESS)")

            # CI band: use ±1.96*std if available, otherwise a conservative ±0.02 band
            if ystd is not None and np.all(np.isfinite(ystd)):
                # Smooth std on same grid by interpolation in log-space
                ystd_i = np.interp(xs, xlog, ystd)
                lo = ys - 1.96 * ystd_i
                hi = ys + 1.96 * ystd_i
            else:
                lo = ys - 0.02
                hi = ys + 0.02
            ax.fill_between(x_smooth, lo, hi, color=c, alpha=ALPHA_FILL)

            # Mark best λ
            bl = best_lambda_map.get(str(t))
            if bl is not None and np.isfinite(bl):
                # evaluate smooth at bl
                y_bl = float(np.interp(np.log10(bl), xs, ys))
                ax.scatter([bl], [y_bl], s=45, color=c, edgecolor=PALETTE["neutral"], zorder=5)
                ax.annotate(
                    f"best λ={bl:.3g}",
                    xy=(bl, y_bl),
                    xytext=(6, 8),
                    textcoords="offset points",
                    fontsize=9,
                    color=PALETTE["neutral"],
                )

        ax.set_xscale("log")
        ax.set_title("Ridge Sensitivity: $\lambda$ vs CV $R^2$ (LOESS + 95% CI)")
        ax.set_xlabel(r"$\lambda$ (log scale)")
        ax.set_ylabel("CV $R^2$")
        ax.grid(True, axis="y")
        ax.legend(title="Target", frameon=True, loc="best")
        savefig(FIG_DIR / "T3S_4_lambda_curve.png")

    # -------------------------
    # E) TopK sensitivity (partner/industry only in this script)
    # -------------------------
    if KEEP_FIG_TOPK and set(["Target", "Varied", "Value"]).issubset(topk.columns) and beta_cols:
        # choose the same 5 beta groups if present
        keep = [c for c in ["beta_age", "beta_partner", "beta_industry", "beta_homestate", "beta_homecountry"] if c in topk.columns]
        if keep:
            for varied in sorted(topk["Varied"].astype(str).unique().tolist()):
                sub = topk[topk["Varied"].astype(str) == varied].copy()
                # plot one panel per target
                for t in sorted(sub["Target"].astype(str).unique().tolist()):
                    st = sub[sub["Target"].astype(str) == t].sort_values("Value")
                    plt.figure(figsize=(10.0, 4.8))
                    ax = plt.gca()
                    for b in keep:
                        ax.plot(st["Value"].values, st[b].values, marker="o", linewidth=LINEWIDTH, label=b)
                    ax.set_title(f"TopK Sensitivity ({varied}) — Target: {t}")
                    ax.set_xlabel(varied)
                    ax.set_ylabel("Group effect size")
                    ax.grid(True, axis="y")
                    ax.legend(frameon=True, ncol=2)
                    savefig(FIG_DIR / f"T3S_5_topk_{varied}_{t}.png")

    # -------------------------
    # Summary tables (LaTeX) + Markdown notes
    # -------------------------
    # Table 1: seed stability summary (mean±std across seeds)
    t_seed = (seed.groupby("Target", as_index=False)
                  .agg(CV_R2_mean=("CV_R2_mean", "mean"), CV_R2_std=("CV_R2_mean", "std"),
                       CV_RMSE_mean=("CV_RMSE_mean", "mean"), CV_RMSE_std=("CV_RMSE_mean", "std")))

    # Table 2: beta CI (only 5 BetaGroups and 3 targets)
    order = ["beta_age", "beta_partner", "beta_industry", "beta_homestate", "beta_homecountry"]
    beta_ci["BetaGroup"] = pd.Categorical(beta_ci["BetaGroup"], categories=order, ordered=True)
    t_ci = beta_ci[beta_ci["BetaGroup"].isin(order)].copy()
    t_ci = t_ci.sort_values(["Target", "BetaGroup"])

    tex_parts = []
    tex_parts.append("% Auto-generated Task3 sensitivity tables (do not edit by hand)")
    tex_parts.append("% Recommended: \\usepackage{booktabs}")
    tex_parts.append("")
    tex_parts.append(to_latex_table(t_seed, "Seed stability summary of cross-validated performance.", "tab:t3_seed_stability"))
    tex_parts.append(to_latex_table(t_ci, "Bootstrap 95\\% confidence intervals for mean group effects.", "tab:t3_beta_ci"))
    OUT_TEX.write_text("\n".join(tex_parts), encoding="utf-8")

    # markdown summary (short, reproducible)
    lines = []
    lines.append("# Task 3 Sensitivity Analysis Summary")
    lines.append("")
    lines.append("## What we tested")
    lines.append("- Random-seed stability of 5-fold CV metrics (RMSE, R^2).")
    lines.append("- Stability of 5 grouped effect sizes (Age/Partner/Industry/HomeState/HomeCountry).")
    lines.append("- Bootstrap CIs for mean grouped effects.")
    lines.append("- Sensitivity to ridge penalty λ and TopK category caps.")
    lines.append("")
    lines.append("## Key takeaways")
    if len(t_seed) > 0:
        for _, r in t_seed.iterrows():
            lines.append(f"- **{r['Target']}**: CV $R^2$ = {r['CV_R2_mean']:.3f} ± {r['CV_R2_std']:.3f} (across seeds).")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- Figures: `{FIG_DIR}`")
    lines.append(f"- LaTeX tables: `{OUT_TEX}`")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    kept = []
    if KEEP_FIG_SEED_BOX: kept.append("Seed stability box")
    if KEEP_FIG_BETA_CI: kept.append("Beta CI errorbars")
    if KEEP_FIG_LAMBDA_CURVE: kept.append("Lambda curve")
    print("\n[INFO] Sensitivity figures generated:", ", ".join(kept) if kept else "(none)")

    print("✅ Sensitivity post-analysis saved:")
    print(" - Figures:", FIG_DIR)
    print(" - LaTeX:", OUT_TEX)
    print(" - Summary:", OUT_MD)


# ======================================================
# Main
# ======================================================
def main():
    must_exist(IN_XLSX)

    model_df = pd.read_excel(IN_XLSX, sheet_name="model_data_used")
    hyper = pd.read_excel(IN_XLSX, sheet_name="hyperparam_sweep")
    bestp = pd.read_excel(IN_XLSX, sheet_name="best_params")

    # --------------------------------------------------
    # IMPORTANT (sync with task3_model_sweep.py):
    # The sweep chooses best params using PER-TARGET filtering:
    #   model_df = base_model_df.dropna(subset=["Age", ycol])
    # If we don't mirror that here, NaNs leak into CV and betas.
    # --------------------------------------------------
    base_model_df = model_df.copy()
    for c in ["Partner", "Industry", "HomeState", "HomeCountry"]:
        if c in base_model_df.columns:
            base_model_df[c] = base_model_df[c].fillna("Unknown").astype(str)

    # Targets mapping from your Task3
    targets = {
        "Performance_WeeksSurvived": "WeeksSurvived",
        "Judges_AvgJudgeTotal": "AvgJudgeTotal",
        "Fans_AvgFanShare": "AvgFanShare",
    }

    # make sure needed columns exist
    needed = ["Partner", "Industry", "HomeState", "HomeCountry", "Age"] + list(targets.values())
    for c in needed:
        if c not in model_df.columns:
            raise KeyError(f"model_data_used missing column: {c}")

    # NOTE: task3_model_sweep.py does per-target filtering on ["Age", ycol].
    # This script now mirrors that to avoid NaN leakage into CV metrics.

    # ------------------------------
    # A) SEED stability (CV metrics)
    # ------------------------------
    SEEDS = [1, 7, 21, 42, 77, 123, 2026]
    K = 5

    seed_rows = []
    beta_rows = []

    best_map = bestp.set_index("Target").to_dict(orient="index")

    for tname, ycol in targets.items():
        if tname not in best_map:
            print(f"[WARN] best_params missing target: {tname}")
            continue

        lam = float(best_map[tname]["Best_Lambda"])
        kp = int(best_map[tname]["Best_TopK_Partner"])
        ki = int(best_map[tname]["Best_TopK_Industry"])
        ks = int(best_map[tname]["Best_TopK_HomeState"])
        kc = int(best_map[tname]["Best_TopK_HomeCountry"])

        # Per-target filtering (MUST match task3_model_sweep.py)
        tmp = base_model_df.dropna(subset=["Age", ycol]).copy()
        if len(tmp) < 10:
            print(f"[WARN] Too few samples after filtering for {tname}: n={len(tmp)}")
            continue

        # cap categories (same as task3_model_sweep.py)
        tmp["Partner_c"] = cap_top_k(tmp["Partner"], kp)
        tmp["Industry_c"] = cap_top_k(tmp["Industry"], ki)
        tmp["HomeState_c"] = cap_top_k(tmp["HomeState"], ks)
        tmp["HomeCountry_c"] = cap_top_k(tmp["HomeCountry"], kc)

        X, feat = build_design_matrix(tmp)
        y_all_raw = tmp[ycol].to_numpy(dtype=float)

        for seed in SEEDS:
            rmses, r2s = [], []
            for tr, te in kfold_indices(len(tmp), k=K, seed=seed):
                y_tr, y_te, *_ = standardize_train_apply(y_all_raw[tr], y_all_raw[te])
                b = ridge_beta(X[tr], y_tr, lam)
                yhat = X[te] @ b
                rmse, r2 = rmse_r2(y_te, yhat)
                rmses.append(rmse)
                r2s.append(r2)

            seed_rows.append({
                "Target": tname,
                "Lambda": lam,
                "TopK_Partner": kp,
                "TopK_Industry": ki,
                "TopK_HomeState": ks,
                "TopK_HomeCountry": kc,
                "Seed": seed,
                "CV_RMSE_mean": float(np.mean(rmses)),
                "CV_RMSE_std": float(np.std(rmses)),
                "CV_R2_mean": float(np.mean(r2s)),
                "CV_R2_std": float(np.std(r2s)),
            })

            # fit once on full standardized y to extract betas
            y_std = (y_all_raw - y_all_raw.mean()) / (y_all_raw.std() if y_all_raw.std() != 0 else 1.0)
            b_full = ridge_beta(X, y_std, lam)
            coef_s = pd.Series(b_full, index=feat, dtype=float)
            betas5 = group_beta_strength(coef_s)

            beta_rows.append({
                "Target": tname,
                "Seed": seed,
                **betas5
            })

    seed_stability = pd.DataFrame(seed_rows)
    beta_stability = pd.DataFrame(beta_rows)

    # ------------------------------
    # B) Bootstrap CI for betas
    # ------------------------------
    # Here we bootstrap over seeds (lightweight) + show CI of beta means
    ci_rows = []
    for tname in beta_stability["Target"].unique():
        sub = beta_stability[beta_stability["Target"] == tname].copy()
        for b in ["beta_age","beta_partner","beta_industry","beta_homestate","beta_homecountry"]:
            mu, lo, hi = bootstrap_ci(sub[b].values, B=500, seed=2026)
            ci_rows.append({
                "Target": tname,
                "BetaGroup": b,
                "Mean": mu,
                "CI_low": lo,
                "CI_high": hi,
            })
    beta_ci = pd.DataFrame(ci_rows)

    # ------------------------------
    # C) TopK sensitivity (one dimension at a time)
    # ------------------------------
    topk_rows = []
    for tname, ycol in targets.items():
        if tname not in best_map:
            continue

        lam = float(best_map[tname]["Best_Lambda"])
        kp0 = int(best_map[tname]["Best_TopK_Partner"])
        ki0 = int(best_map[tname]["Best_TopK_Industry"])
        ks0 = int(best_map[tname]["Best_TopK_HomeState"])
        kc0 = int(best_map[tname]["Best_TopK_HomeCountry"])

        grid_partner = sorted(set([10, 20, 40, kp0]))
        grid_industry = sorted(set([6, 12, 20, ki0]))
        grid_state = sorted(set([10, 20, 40, ks0]))
        grid_country = sorted(set([5, 10, 20, kc0]))

        # Per-target filtering (match task3_model_sweep.py)
        base_t = base_model_df.dropna(subset=["Age", ycol]).copy()
        if len(base_t) < 10:
            print(f"[WARN] Too few samples after filtering for {tname}: n={len(base_t)}")
            continue

        y_all_raw = base_t[ycol].to_numpy(dtype=float)

        for kp in grid_partner:
            tmp = base_t.copy()
            tmp["Partner_c"] = cap_top_k(tmp["Partner"], kp)
            tmp["Industry_c"] = cap_top_k(tmp["Industry"], ki0)
            tmp["HomeState_c"] = cap_top_k(tmp["HomeState"], ks0)
            tmp["HomeCountry_c"] = cap_top_k(tmp["HomeCountry"], kc0)
            X, feat = build_design_matrix(tmp)
            y_std = (y_all_raw - y_all_raw.mean()) / (y_all_raw.std() if y_all_raw.std() != 0 else 1.0)
            b = ridge_beta(X, y_std, lam)
            betas5 = group_beta_strength(pd.Series(b, index=feat))
            topk_rows.append({"Target": tname, "Varied": "TopK_Partner", "Value": kp, **betas5})

        for ki in grid_industry:
            tmp = base_t.copy()
            tmp["Partner_c"] = cap_top_k(tmp["Partner"], kp0)
            tmp["Industry_c"] = cap_top_k(tmp["Industry"], ki)
            tmp["HomeState_c"] = cap_top_k(tmp["HomeState"], ks0)
            tmp["HomeCountry_c"] = cap_top_k(tmp["HomeCountry"], kc0)
            X, feat = build_design_matrix(tmp)
            y_std = (y_all_raw - y_all_raw.mean()) / (y_all_raw.std() if y_all_raw.std() != 0 else 1.0)
            b = ridge_beta(X, y_std, lam)
            betas5 = group_beta_strength(pd.Series(b, index=feat))
            topk_rows.append({"Target": tname, "Varied": "TopK_Industry", "Value": ki, **betas5})

    topk_sensitivity = pd.DataFrame(topk_rows)

    # ------------------------------
    # D) Lambda curve sensitivity (fixed best topKs)
    # ------------------------------
    lam_rows = []
    for tname, ycol in targets.items():
        if tname not in best_map:
            continue

        kp = int(best_map[tname]["Best_TopK_Partner"])
        ki = int(best_map[tname]["Best_TopK_Industry"])
        ks = int(best_map[tname]["Best_TopK_HomeState"])
        kc = int(best_map[tname]["Best_TopK_HomeCountry"])

        # Per-target filtering (match task3_model_sweep.py)
        tmp = base_model_df.dropna(subset=["Age", ycol]).copy()
        if len(tmp) < 10:
            print(f"[WARN] Too few samples after filtering for {tname}: n={len(tmp)}")
            continue

        tmp["Partner_c"] = cap_top_k(tmp["Partner"], kp)
        tmp["Industry_c"] = cap_top_k(tmp["Industry"], ki)
        tmp["HomeState_c"] = cap_top_k(tmp["HomeState"], ks)
        tmp["HomeCountry_c"] = cap_top_k(tmp["HomeCountry"], kc)

        X, feat = build_design_matrix(tmp)
        y_all_raw = tmp[ycol].to_numpy(dtype=float)

        for lam in sorted(hyper[hyper["Target"] == tname]["Lambda"].unique().tolist()):
            rmses, r2s = [], []
            for tr, te in kfold_indices(len(tmp), k=5, seed=42):
                y_tr, y_te, *_ = standardize_train_apply(y_all_raw[tr], y_all_raw[te])
                b = ridge_beta(X[tr], y_tr, float(lam))
                yhat = X[te] @ b
                rmse, r2 = rmse_r2(y_te, yhat)
                rmses.append(rmse)
                r2s.append(r2)

            lam_rows.append({
                "Target": tname,
                "Lambda": float(lam),
                "CV_RMSE_mean": float(np.mean(rmses)),
                "CV_RMSE_std": float(np.std(rmses)),
                "CV_R2_mean": float(np.mean(r2s)),
                "CV_R2_std": float(np.std(r2s)),
            })

    lambda_curve = pd.DataFrame(lam_rows)

    # ------------------------------
    # Export
    # ------------------------------
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        seed_stability.to_excel(w, index=False, sheet_name="seed_stability")
        beta_stability.to_excel(w, index=False, sheet_name="beta_stability_by_seed")
        beta_ci.to_excel(w, index=False, sheet_name="beta_CI_bootstrap")
        topk_sensitivity.to_excel(w, index=False, sheet_name="topk_sensitivity")
        lambda_curve.to_excel(w, index=False, sheet_name="lambda_curve_sensitivity")

    print("✅ Wrote sensitivity report:", OUT_XLSX)

    # Post-analysis: generate figures + tables from the generated Excel
    analyze_generated_excel(OUT_XLSX)


if __name__ == "__main__":
    main()