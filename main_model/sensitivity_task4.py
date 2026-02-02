# sensitivity_task4.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# Optional stats
try:
    from scipy.stats import bootstrap
except Exception:
    bootstrap = None


# ======================================================
# Paths
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
RES = BASE_DIR / "results"

IN_REPORT = RES / "fairvote_backtest_report.xlsx"

OUT_DIR = RES / "figures_task4_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_XLSX = RES / "task4_sensitivity_report.xlsx"


# ======================================================
# Global style (Times New Roman everywhere)
# ======================================================
FONT = "Times New Roman"
plt.rcParams.update(
    {
        # Force Times New Roman for all text
        "font.family": "Times New Roman",
        "font.serif": [FONT],
        "font.sans-serif": [FONT],
        "font.monospace": [FONT],
        # Make sure vector exports keep real text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Sizes
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        # Avoid unicode minus rendering quirks
        "axes.unicode_minus": False,
    }
)

# Seaborn must also be told which font to use
sns.set_theme(style="white", context="talk", font=FONT, rc={"font.family": FONT})

# Warn if Times New Roman is missing
try:
    import matplotlib.font_manager as fm
    _has_tnr = any("Times New Roman" in f.name for f in fm.fontManager.ttflist)
    if not _has_tnr:
        print("[WARN] Times New Roman not found in system fonts; matplotlib may fall back to a different serif.")
except Exception:
    pass


# ======================================================
# Color / alpha options (edit freely)
# ======================================================
COLORMAP_MAIN = "viridis"  # heatmaps
HEATMAP_ALPHA = 0.98

LINE_COLOR = "#2E5A88"
BAND_COLOR = "#6BA67C"
BAND_ALPHA = 0.22

POINT_COLOR = "#111827"
POINT_ALPHA = 0.35
POINT_SIZE = 14

GRID_COLOR = "#E0E0E0"
GRID_ALPHA = 0.15
GRID_LW = 0.25

# For bootstrap CI display
CI_ALPHA = 0.25
CI_LINE_LW = 1.6


# ======================================================
# Utilities
# ======================================================
def must_exist(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def safe_read_sheet(xlsx: Path, sheet: str) -> pd.DataFrame:
    try:
        return pd.read_excel(xlsx, sheet_name=sheet)
    except Exception as e:
        raise RuntimeError(f"Failed to read {sheet} from {xlsx}: {e}")


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()
    print("✅ saved:", path)


def bootstrap_ci(arr: np.ndarray, stat=np.nanmean, n_resamples: int = 2000, alpha=0.05) -> tuple[float, float, float]:
    """
    Returns (center, low, high).
    If scipy bootstrap unavailable, uses simple normal approx fallback.
    """
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float(stat(x)) if len(x) else np.nan, np.nan, np.nan

    center = float(stat(x))

    if bootstrap is None:
        # fallback: normal approx via std/sqrt(n)
        se = float(np.nanstd(x, ddof=1) / np.sqrt(len(x)))
        z = 1.96
        return center, center - z * se, center + z * se

    # scipy bootstrap
    res = bootstrap((x,), stat, confidence_level=1 - alpha, n_resamples=n_resamples, method="basic")
    low = float(res.confidence_interval.low)
    high = float(res.confidence_interval.high)
    return center, low, high


def no_offset_ticks(ax):
    # Avoid "3 0.425" type offset text surprises
    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.get_offset_text().set_visible(False)


# ======================================================
# Main
# ======================================================
def main():
    must_exist(IN_REPORT)

    # Sheets expected in report:
    # ['param_sweep','best_params','metrics_report','weekly_predictions','season_summary','bsi','regression_coefs']
    sweep = safe_read_sheet(IN_REPORT, "param_sweep")
    bestp = safe_read_sheet(IN_REPORT, "best_params")
    metrics = safe_read_sheet(IN_REPORT, "metrics_report")
    weekly = safe_read_sheet(IN_REPORT, "weekly_predictions")
    season = safe_read_sheet(IN_REPORT, "season_summary")
    bsi = safe_read_sheet(IN_REPORT, "bsi")

    # ------------------------------
    # Sanity: required columns
    # ------------------------------
    req_sweep = ["k", "tau", "Consistency", "ExcitementRate", "DeltaFanShareElim(FV-Actual)"]
    missing = [c for c in req_sweep if c not in sweep.columns]
    if missing:
        raise KeyError(f"param_sweep missing required columns: {missing}. Found: {list(sweep.columns)}")

    # Optional extra surfaces if available
    has_bsi = "BSI_FV_sumAbsGamma" in sweep.columns
    has_r2 = "R2_FV" in sweep.columns

    # best parameters from sheet
    best_k = float(bestp.iloc[0].get("k", bestp.iloc[0].get("Best_k", np.nan)))
    best_tau = float(bestp.iloc[0].get("tau", bestp.iloc[0].get("Best_tau", np.nan)))

    # ======================================================
    # Sensitivity metric 1: Param surface (Consistency)
    # ======================================================
    piv_cons = sweep.pivot_table(index="k", columns="tau", values="Consistency", aggfunc="mean").sort_index().sort_index(axis=1)
    piv_exc = sweep.pivot_table(index="k", columns="tau", values="ExcitementRate", aggfunc="mean").sort_index().sort_index(axis=1)
    piv_fan = sweep.pivot_table(index="k", columns="tau", values="DeltaFanShareElim(FV-Actual)", aggfunc="mean").sort_index().sort_index(axis=1)
    piv_bsi = None
    piv_r2 = None
    if has_bsi:
        piv_bsi = sweep.pivot_table(index="k", columns="tau", values="BSI_FV_sumAbsGamma", aggfunc="mean").sort_index().sort_index(axis=1)
    if has_r2:
        piv_r2 = sweep.pivot_table(index="k", columns="tau", values="R2_FV", aggfunc="mean").sort_index().sort_index(axis=1)

    # ---- FIG S1: Consistency heatmap + best marker
    plt.figure(figsize=(9.4, 7.6))
    ax = sns.heatmap(
        piv_cons,
        cmap=COLORMAP_MAIN,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="#D3D3D3",
        cbar=True,
        cbar_kws={"label": "Consistency (match rate)"},
        alpha=HEATMAP_ALPHA,
    )
    ax.set_title("Task4 Sensitivity: Consistency over (k, τ)\n(best point marked)", weight="bold", pad=12)
    ax.set_xlabel("τ (gate threshold)")
    ax.set_ylabel("k (uncertainty penalty)")

    # mark best
    if np.isfinite(best_k) and np.isfinite(best_tau):
        # find nearest grid cell
        ks = piv_cons.index.values.astype(float)
        taus = piv_cons.columns.values.astype(float)
        ik = int(np.argmin(np.abs(ks - best_k)))
        it = int(np.argmin(np.abs(taus - best_tau)))
        ax.scatter(it + 0.5, ik + 0.5, s=220, marker="o", facecolors="none", edgecolors="#111827", linewidths=2.2)

    ax.grid(False)
    savefig(OUT_DIR / "T4S_1_consistency_surface_heatmap.png")

    # ---- FIG S2: Excitement surface (optional but useful)
    plt.figure(figsize=(9.4, 7.6))
    ax = sns.heatmap(
        piv_exc,
        cmap="plasma",
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="#D3D3D3",
        cbar=True,
        cbar_kws={"label": "ExcitementRate (gate frequency)"},
        alpha=HEATMAP_ALPHA,
    )
    ax.set_title("Task4 Sensitivity: ExcitementRate over (k, τ)", weight="bold", pad=12)
    ax.set_xlabel("τ (gate threshold)")
    ax.set_ylabel("k (uncertainty penalty)")
    savefig(OUT_DIR / "T4S_2_excitement_surface_heatmap.png")

    # ---- FIG S3: Fan-protection surface (Δ<0 means FV eliminates LOWER fan share than actual)
    plt.figure(figsize=(9.4, 7.6))
    ax = sns.heatmap(
        piv_fan,
        cmap="coolwarm",
        center=0.0,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        linecolor="#D3D3D3",
        cbar=True,
        cbar_kws={"label": r"Δ fan share eliminated (FV − Actual)"},
        alpha=HEATMAP_ALPHA,
    )
    ax.set_title("Task4 Sensitivity: Fan-Favoring (Δ fan share eliminated)\n(negative = more fan-protective)", weight="bold", pad=12)
    ax.set_xlabel("τ (gate threshold)")
    ax.set_ylabel("k (uncertainty penalty)")
    savefig(OUT_DIR / "T4S_3_fan_favoring_surface_heatmap.png")

    # ---- FIG S3b: BSI surface (lower = less feature-driven bias)
    if piv_bsi is not None:
        plt.figure(figsize=(9.4, 7.6))
        ax = sns.heatmap(
            piv_bsi,
            cmap="mako",
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            linecolor="#D3D3D3",
            cbar=True,
            cbar_kws={"label": "BSI_FV_sumAbsGamma (lower = fairer)"},
            alpha=HEATMAP_ALPHA,
        )
        ax.set_title("Task4 Sensitivity: Bias Stability Index over (k, τ)\n(lower = less feature-driven bias)", weight="bold", pad=12)
        ax.set_xlabel("τ (gate threshold)")
        ax.set_ylabel("k (uncertainty penalty)")

        if np.isfinite(best_k) and np.isfinite(best_tau):
            ks = piv_bsi.index.values.astype(float)
            taus = piv_bsi.columns.values.astype(float)
            ik = int(np.argmin(np.abs(ks - best_k)))
            it = int(np.argmin(np.abs(taus - best_tau)))
            ax.scatter(it + 0.5, ik + 0.5, s=220, marker="o", facecolors="none", edgecolors="#111827", linewidths=2.2)

        savefig(OUT_DIR / "T4S_3b_bsi_surface_heatmap.png")

    # ---- FIG S3c: R2 surface (higher = more explainable PredElimWeek)
    if piv_r2 is not None:
        plt.figure(figsize=(9.4, 7.6))
        ax = sns.heatmap(
            piv_r2,
            cmap="YlGnBu",
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            linecolor="#D3D3D3",
            cbar=True,
            cbar_kws={"label": "R2_FV (higher = better fit)"},
            alpha=HEATMAP_ALPHA,
        )
        ax.set_title("Task4 Sensitivity: R² over (k, τ)\n(higher = better fit)", weight="bold", pad=12)
        ax.set_xlabel("τ (gate threshold)")
        ax.set_ylabel("k (uncertainty penalty)")

        if np.isfinite(best_k) and np.isfinite(best_tau):
            ks = piv_r2.index.values.astype(float)
            taus = piv_r2.columns.values.astype(float)
            ik = int(np.argmin(np.abs(ks - best_k)))
            it = int(np.argmin(np.abs(taus - best_tau)))
            ax.scatter(it + 0.5, ik + 0.5, s=220, marker="o", facecolors="none", edgecolors="#111827", linewidths=2.2)

        savefig(OUT_DIR / "T4S_3c_r2_surface_heatmap.png")

    # ======================================================
    # Sensitivity metric 2: Bootstrap stability for BEST parameters (weekly_predictions)
    #   - Consistency
    #   - Fan favoring (Δ)
    #   - Excitement rate
    # ======================================================
    # Use the best_weekly rows already produced by your backtest.
    # Here we add CI bands using bootstrap resampling across weeks.
    dfw = weekly.copy()
    # seasons list needed for titles / reporting
    seasons = sorted(dfw["Season"].dropna().astype(int).unique().tolist())
    n_seasons = len(seasons)

    if "FV_Match" in dfw.columns:
        cons_arr = dfw["FV_Match"].astype(float).values
    else:
        cons_arr = (dfw["FV_Pred"].astype(str) == dfw["ActualEliminated"].astype(str)).astype(float).values

    exc_arr = dfw["FV_Gate"].astype(float).values if "FV_Gate" in dfw.columns else np.zeros(len(dfw))
    delta_arr = (dfw["FanShare_Elim_FV"].astype(float) - dfw["FanShare_Elim_Actual"].astype(float)).values

    cons_c, cons_lo, cons_hi = bootstrap_ci(cons_arr, np.nanmean)
    exc_c, exc_lo, exc_hi = bootstrap_ci(exc_arr, np.nanmean)
    del_c, del_lo, del_hi = bootstrap_ci(delta_arr, np.nanmean)

    metrics_ci = pd.DataFrame(
        [
            {"Metric": "Consistency", "Estimate": cons_c, "CI_low": cons_lo, "CI_high": cons_hi},
            {"Metric": "ExcitementRate", "Estimate": exc_c, "CI_low": exc_lo, "CI_high": exc_hi},
            {"Metric": "DeltaFanShareElim(FV-Actual)", "Estimate": del_c, "CI_low": del_lo, "CI_high": del_hi},
        ]
    )

    # ---- FIG S4: Forest plot — Best-parameter estimates with 95% CI
    # We report three key tuning/behavior parameters:
    #   ω̄ : mean adaptive fan weight across weeks (bootstrap CI from weekly ω)
    #   τ  : gate threshold selected by tuning (CI from near-optimal region in param_sweep)
    #   k  : uncertainty penalty selected by tuning (CI from near-optimal region in param_sweep)

    # (A) ω̄ from weekly predictions
    omega_col = "FV_Omega" if "FV_Omega" in dfw.columns else ("Omega" if "Omega" in dfw.columns else None)
    if omega_col is not None:
        omega_arr = dfw[omega_col].astype(float).values
        omega_c, omega_lo, omega_hi = bootstrap_ci(omega_arr, np.nanmean, n_resamples=10000)
    else:
        omega_c, omega_lo, omega_hi = np.nan, np.nan, np.nan

    # (B) τ and k from tuning stability: take "near-optimal" region in param_sweep
    # Define near-optimal as within 1% of the best Consistency (robust, avoids over-claiming).
    best_cons = float(np.nanmax(sweep["Consistency"].astype(float).values))
    thr = best_cons - 0.01 * max(best_cons, 1e-12)
    near = sweep[sweep["Consistency"].astype(float) >= thr].copy()
    if len(near) < 5:
        # fallback: take top-10 by Consistency
        near = sweep.sort_values("Consistency", ascending=False).head(10).copy()

    k_vals = near["k"].astype(float).values
    tau_vals = near["tau"].astype(float).values

    # Use quantiles as an empirical CI for tuning stability
    def qci(x: np.ndarray):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return np.nan, np.nan, np.nan
        return float(np.nanmedian(x)), float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975))

    k_c, k_lo, k_hi = qci(k_vals)
    tau_c, tau_lo, tau_hi = qci(tau_vals)

    param_tbl = pd.DataFrame(
        [
            {"Parameter": r"$\bar{\omega}$", "Name": "fan weight (mean)", "Estimate": omega_c, "CI_low": omega_lo, "CI_high": omega_hi},
            {"Parameter": r"$\tau$", "Name": "close-call gate", "Estimate": float(best_tau) if np.isfinite(best_tau) else tau_c, "CI_low": tau_lo, "CI_high": tau_hi},
            {"Parameter": r"$k$", "Name": "uncertainty penalty", "Estimate": float(best_k) if np.isfinite(best_k) else k_c, "CI_low": k_lo, "CI_high": k_hi},
        ]
    )

    # Save this table into the output xlsx later

    # Forest plot
    plt.figure(figsize=(10.8, 5.6))
    ax = plt.gca()

    # Colors by parameter type (color-blind friendly)
    colors = ["#2E5A88", "#6BA67C", "#D4A76A"]  # blue/green/orange

    # y positions top-to-bottom
    y = np.arange(len(param_tbl))[::-1]

    for i, (_, r) in enumerate(param_tbl.iterrows()):
        est, lo, hi = float(r["Estimate"]), float(r["CI_low"]), float(r["CI_high"])
        col = colors[i % len(colors)]
        # CI line
        ax.hlines(y[i], lo, hi, color=col, linewidth=1.8, alpha=0.95)
        # Point
        ax.scatter([est], [y[i]], c=col, s=70, zorder=5, edgecolors="#111827", linewidths=0.6)
        # Label next to point
        txt = f"{r['Parameter']} = {est:.3f} [{lo:.3f}, {hi:.3f}]"
        ax.text(hi + 0.02 * (np.nanmax(param_tbl['CI_high']) - np.nanmin(param_tbl['CI_low']) + 1e-9), y[i], txt,
                va="center", ha="left", fontsize=10)

    # Left-side names column
    ax.set_yticks(y)
    ax.set_yticklabels([f"{param_tbl.loc[i,'Parameter']}  ({param_tbl.loc[i,'Name']})" for i in range(len(param_tbl))][::-1])

    # Reference line (null / baseline)
    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(0.0, y.max() + 0.6, "Reference: null / baseline", color="#888888", ha="center", va="bottom", fontsize=9)

    # Axis limits with padding
    xmin = float(np.nanmin(param_tbl["CI_low"].values))
    xmax = float(np.nanmax(param_tbl["CI_high"].values))
    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
        pad = 0.12 * (xmax - xmin)
        ax.set_xlim(xmin - pad, xmax + 2.2 * pad)

    ax.set_xlabel("Estimate with 95% CI", fontstyle="italic")
    ax.set_title(
        "Task4 Robustness: Best-Parameter Estimates (Bootstrap B=10,000, n_weeks={})".format(len(dfw)),
        weight="bold",
        pad=10,
    )

    ax.grid(True, axis="x", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=GRID_LW)
    ax.grid(False, axis="y")
    no_offset_ticks(ax)

    savefig(OUT_DIR / "T4S_4_forest_best_parameters.png")

    # ======================================================
    # Sensitivity metric 3: Leave-one-season-out (LOSO) stability
    #   - recompute metrics per season excluded (no re-backtest needed)
    #   - gives stability evidence: not driven by one weird season
    # ======================================================
    loso_rows = []

    for s in seasons:
        sub = dfw[dfw["Season"].astype(int) != int(s)].copy()
        if len(sub) < 30:
            continue
        c = float(np.nanmean(sub["FV_Match"].astype(float).values))
        e = float(np.nanmean(sub["FV_Gate"].astype(float).values)) if "FV_Gate" in sub.columns else np.nan
        d = float(np.nanmean((sub["FanShare_Elim_FV"].astype(float) - sub["FanShare_Elim_Actual"].astype(float)).values))
        loso_rows.append({"LeaveOutSeason": int(s), "Consistency": c, "ExcitementRate": e, "DeltaFanShare": d, "NumWeeks": int(len(sub))})

    loso = pd.DataFrame(loso_rows).sort_values("LeaveOutSeason")

    # ---- FIG S5: LOSO stability — jittered points + moving median + outlier marks
    if len(loso) > 0:
        plt.figure(figsize=(12.2, 5.4))
        ax = plt.gca()

        x = loso["LeaveOutSeason"].astype(int).values
        yv = loso["Consistency"].astype(float).values

        # Jitter to avoid overlap
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.18, 0.18, size=len(x))

        ax.scatter(
            x + jitter,
            yv,
            c="#111827",
            s=22,
            alpha=0.70,
            zorder=3,
        )

        # 7-point moving median (centered)
        s = pd.Series(yv, index=x).sort_index()
        med = s.rolling(window=7, center=True, min_periods=3).median()
        ax.plot(med.index.values, med.values, color=LINE_COLOR, linewidth=1.8, label="7-season moving median")

        # Residuals and outliers (> 2σ)
        resid = s.values - med.reindex(s.index).values
        resid = resid[np.isfinite(resid)]
        sigma = float(np.nanstd(resid)) if len(resid) else np.nan
        if np.isfinite(sigma) and sigma > 1e-12:
            out_idx = np.where(np.abs(s.values - med.reindex(s.index).values) > 2.0 * sigma)[0]
        else:
            out_idx = np.array([], dtype=int)

        for oi in out_idx:
            xs = int(s.index.values[oi])
            ys = float(s.values[oi])
            ax.scatter(xs, ys, c="#DC2626", s=90, marker="*", zorder=6)
            ax.text(xs + 0.35, ys, f"S{xs}*", color="#DC2626", fontsize=9, weight="bold", va="center")

        # Bootstrap CI band (global, over LOSO values) to communicate stability
        c_center, c_lo, c_hi = bootstrap_ci(yv, np.nanmean, n_resamples=10000)
        if np.isfinite(c_lo) and np.isfinite(c_hi):
            ax.fill_between(
                [x.min() - 0.5, x.max() + 0.5],
                [c_lo, c_lo],
                [c_hi, c_hi],
                color="#66B2FF",
                alpha=0.20,
                label="95% CI band (bootstrap)",
                zorder=1,
            )

        # Mean reference line
        ax.axhline(c_center, color="#888888", linestyle=":", linewidth=1.0, label=f"Mean = {c_center:.3f}")

        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        # Focus range: tighten if possible
        ymin = float(np.nanmin(yv))
        ymax = float(np.nanmax(yv))
        if np.isfinite(ymin) and np.isfinite(ymax) and (ymax - ymin) < 0.25:
            pad = 0.18 * (ymax - ymin + 1e-9)
            ax.set_ylim(max(0.0, ymin - pad), min(1.05, ymax + pad))
        else:
            ax.set_ylim(0, 1.05)

        ax.set_xlabel("Excluded Season", fontstyle="italic")
        ax.set_ylabel("Consistency", fontstyle="italic")
        ax.set_title("LOSO Stability: Consistency when Excluding Each Season (n={})".format(len(seasons)), weight="bold")

        ax.grid(True, axis="y", color=GRID_COLOR, alpha=GRID_ALPHA, linewidth=GRID_LW)
        ax.grid(False, axis="x")
        ax.legend(loc="lower right", frameon=True)
        no_offset_ticks(ax)

        savefig(OUT_DIR / "T4S_5_loso_stability_jitter_median.png")

    # ======================================================
    # Collect report tables
    # ======================================================
    # BSI table already in report; we just re-export with context.
    bsi_tbl = bsi.copy()
    bsi_tbl["Note"] = "Lower BSI_sumAbsGamma => less driven by non-performance features"

    # Write output report
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        sweep.to_excel(w, index=False, sheet_name="param_sweep_raw")
        piv_cons.reset_index().to_excel(w, index=False, sheet_name="pivot_consistency")
        piv_exc.reset_index().to_excel(w, index=False, sheet_name="pivot_excitement")
        piv_fan.reset_index().to_excel(w, index=False, sheet_name="pivot_fan_favoring")
        if piv_bsi is not None:
            piv_bsi.reset_index().to_excel(w, index=False, sheet_name="pivot_bsi")
        if piv_r2 is not None:
            piv_r2.reset_index().to_excel(w, index=False, sheet_name="pivot_r2")
        metrics_ci.to_excel(w, index=False, sheet_name="best_metrics_CI")
        param_tbl.to_excel(w, index=False, sheet_name="best_parameter_estimates")
        loso.to_excel(w, index=False, sheet_name="loso_stability")
        bsi_tbl.to_excel(w, index=False, sheet_name="bsi_reference")

    print("\n✅ Task4 sensitivity complete.")
    print("Figures:", OUT_DIR)
    print("Report :", OUT_XLSX)


if __name__ == "__main__":
    main()