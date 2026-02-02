# plots_and_tables_task4_fairvote.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
RES = BASE_DIR / "results"
IN_XLSX = RES / "fairvote_backtest_report.xlsx"

FIG_DIR = RES / "figures_task4"
FIG_DIR.mkdir(parents=True, exist_ok=True)


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
        "axes.labelsize": 14,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 20,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.5,
        "grid.color": "#C9C9C9",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

sns.set_theme(style="white", context="talk")


# -------------------------
# Helpers
# -------------------------
def must_exist(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")


def read_sheet(xlsx: Path, name: str) -> pd.DataFrame:
    return pd.read_excel(xlsx, sheet_name=name)


def pick_first_existing_sheet(xlsx: Path, candidates: list[str]) -> str:
    xl = pd.ExcelFile(xlsx)
    for s in candidates:
        if s in xl.sheet_names:
            return s
    raise KeyError(f"None of sheets {candidates} found. Existing: {xl.sheet_names}")


def savefig(path: Path, dpi: int = 350):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def col(df: pd.DataFrame, cands: list[str]) -> str:
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"Missing columns among {cands}. Existing={list(df.columns)}")


def _as_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def fmt_num(x: float, nd: int = 3) -> str:
    if not np.isfinite(x):
        return "N/A"
    return f"{x:.{nd}f}"


def find_metric(metrics_row: pd.Series, keys: list[str], default=np.nan):
    for k in keys:
        if k in metrics_row.index:
            return metrics_row[k]
    return default


def bootstrap_ci(x: np.ndarray, B: int = 2000, alpha: float = 0.05, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan, np.nan
    n = len(x)
    means = np.empty(B, dtype=float)
    for b in range(B):
        samp = rng.choice(x, size=n, replace=True)
        means[b] = float(np.mean(samp))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def ks_pvalue(a: np.ndarray, b: np.ndarray) -> float | None:
    """Two-sample KS p-value if SciPy exists; otherwise None."""
    try:
        from scipy.stats import ks_2samp

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 5 or len(b) < 5:
            return None
        return float(ks_2samp(a, b).pvalue)
    except Exception:
        return None


def metric_table_figure(
    metrics_row: pd.Series,
    weekly: pd.DataFrame,
    bestp: pd.DataFrame,
    outpath: Path,
):
    """
    A compact journal-style Metric/Value table figure.

    Updated for NEW task4 backtest report:
      - Consistency/Excitement/Δ FanShare/BSI come from metrics_report
      - Δ median, P(Δ>0), ω mean/median, CI are computed from weekly_predictions
    """

    # --- weekly columns
    c_match = col(weekly, ["FV_Match"])
    c_gate = col(weekly, ["FV_Gate"])
    c_omega = col(weekly, ["FV_Omega"])
    c_fv = col(weekly, ["FanShare_Elim_FV"])
    c_ac = col(weekly, ["FanShare_Elim_Actual"])

    # --- derived stats from weekly
    delta = (weekly[c_fv].astype(float) - weekly[c_ac].astype(float)).values
    delta = delta[np.isfinite(delta)]
    delta_med = float(np.median(delta)) if len(delta) else np.nan
    delta_pos = float(np.mean(delta > 0)) if len(delta) else np.nan

    omega = weekly[c_omega].astype(float).values
    omega = omega[np.isfinite(omega)]
    omega_mean = float(np.mean(omega)) if len(omega) else np.nan
    omega_mdn = float(np.median(omega)) if len(omega) else np.nan

    # Consistency CI from weekly FV_Match
    m = weekly[c_match].astype(float).values
    m = m[np.isfinite(m)]
    c_lo, c_hi = bootstrap_ci(m, B=2000, alpha=0.05, seed=7)
    c_mean = float(np.mean(m)) if len(m) else np.nan

    # Gate rate from weekly
    g = weekly[c_gate].astype(float).values
    g = g[np.isfinite(g)]
    excite_weekly = float(np.mean(g)) if len(g) else np.nan

    # --- from metrics_report (single row)
    consistency_report = _as_float(find_metric(metrics_row, ["Consistency"]))
    excite_report = _as_float(find_metric(metrics_row, ["ExcitementRate"]))
    delta_mean = _as_float(find_metric(metrics_row, ["DeltaFanShareElim(FV-Actual)"]))
    delta_gate = _as_float(find_metric(metrics_row, ["DeltaFanShareElim_GateOnly(FV-Actual)"]))
    gate_weeks = find_metric(metrics_row, ["GateWeeks"], default=np.nan)

    bsi_fv = _as_float(find_metric(metrics_row, ["BSI_FairVote(best)_sumAbsGamma"]))
    bsi_rank = _as_float(find_metric(metrics_row, ["BSI_Rank_sumAbsGamma"]))
    bsi_pct = _as_float(find_metric(metrics_row, ["BSI_Percent_sumAbsGamma"]))
    bsi_b2 = _as_float(find_metric(metrics_row, ["BSI_Bottom2_sumAbsGamma"]))

    # params: prefer metrics_report, fallback best_params
    k = _as_float(find_metric(metrics_row, ["Best_k"]))
    tau = _as_float(find_metric(metrics_row, ["Best_tau"]))
    wmin = _as_float(find_metric(metrics_row, ["Wmin"]))
    wmax = _as_float(find_metric(metrics_row, ["Wmax"]))

    if (not np.isfinite(k) or not np.isfinite(tau)) and len(bestp):
        k = _as_float(bestp.iloc[0].get("k", k))
        tau = _as_float(bestp.iloc[0].get("tau", tau))

    def fmt_ci(mean, lo, hi):
        if np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi):
            return f"{fmt_num(mean)} [{fmt_num(lo)}, {fmt_num(hi)}]"
        return fmt_num(mean)

    # Consistency: prefer weekly-derived (it matches your actual evaluation set)
    consistency_show = c_mean if np.isfinite(c_mean) else consistency_report
    excite_show = excite_report if np.isfinite(excite_report) else excite_weekly

    rows = [
        ("Consistency (match rate)", fmt_ci(consistency_show, c_lo, c_hi)),
        ("Excitement rate (gate)", fmt_num(excite_show)),
        ("Fan-favoring Δ mean (all)", fmt_num(delta_mean)),
        ("Fan-favoring Δ median (all)", fmt_num(delta_med)),
        ("P(Δ > 0) (all)", fmt_num(delta_pos)),
        ("Fan-favoring Δ mean (gate-only)", fmt_num(delta_gate)),
        ("Gate weeks (count)", str(int(gate_weeks)) if np.isfinite(_as_float(gate_weeks)) else "N/A"),
        ("ω (mean / median)", f"{fmt_num(omega_mean)} / {fmt_num(omega_mdn)}"),
        ("BSI (FairVote)  Σ|γk|", fmt_num(bsi_fv)),
        ("BSI baseline (Rank/Percent/Bottom2)", f"{fmt_num(bsi_rank)} / {fmt_num(bsi_pct)} / {fmt_num(bsi_b2)}"),
        ("Best params (k, τ)", f"{fmt_num(k)} , {fmt_num(tau)}"),
        ("ω bounds (ωmin, ωmax)", f"{fmt_num(wmin)} , {fmt_num(wmax)}"),
    ]

    # Build table-like figure
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.axis("off")

    ax.add_patch(
        plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", edgecolor="none", zorder=0)
    )

    cell_text = [[m, v] for (m, v) in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.25)
        cell.set_edgecolor("#D3D3D3")
        if r == 0:
            cell.set_facecolor("#F8F9FA")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    try:
        table[(1, 1)].set_text_props(weight="bold")
    except Exception:
        pass

    ax.set_title("FairVote Backtest Summary", pad=10)
    savefig(outpath)


# -------------------------
# Main
# -------------------------
def main():
    must_exist(IN_XLSX)

    # match your workbook sheets
    s_metrics = pick_first_existing_sheet(IN_XLSX, ["metrics_report"])
    s_weekly = pick_first_existing_sheet(IN_XLSX, ["weekly_predictions"])
    s_season = pick_first_existing_sheet(IN_XLSX, ["season_summary"])
    s_best = pick_first_existing_sheet(IN_XLSX, ["best_params"])

    metrics = read_sheet(IN_XLSX, s_metrics)
    weekly = read_sheet(IN_XLSX, s_weekly)
    season = read_sheet(IN_XLSX, s_season)
    bestp = read_sheet(IN_XLSX, s_best)

    # -------------------------
    # Main Fig 0: metric table
    # -------------------------
    metric_table_figure(metrics.iloc[0], weekly, bestp, FIG_DIR / "T4_main_0_metric_cards.png")

    # -------------------------
    # Main Fig 1: fan-favoring hexbin (FV vs Actual eliminated fan share)
    # -------------------------
    c_fv = col(weekly, ["FanShare_Elim_FV"])
    c_ac = col(weekly, ["FanShare_Elim_Actual"])

    df = weekly[[c_fv, c_ac]].dropna().copy()
    df["Delta"] = df[c_fv] - df[c_ac]

    fig, ax = plt.subplots(figsize=(10.2, 7.8))
    hb = ax.hexbin(
        df[c_ac],
        df[c_fv],
        gridsize=38,
        mincnt=1,
        cmap="viridis",
        linewidths=0.15,
        edgecolors="none",
    )

    lim = float(max(df[c_ac].max(), df[c_fv].max()))
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=0.8, color="#111827", alpha=0.85)

    p_pos = float((df["Delta"] > 0).mean())
    ax.text(
        0.02,
        0.97,
        rf"$\mathbb{{P}}(\Delta>0)={p_pos:.3f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", fc="white", ec="#333333", lw=0.5, alpha=0.95),
    )

    ax.set_title("Fan-Favoring Check: Eliminated Fan Share (FairVote vs Actual)")
    ax.set_xlabel("Actual eliminated fan share")
    ax.set_ylabel("FairVote eliminated fan share")

    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("Points per hexbin")

    try:
        hb.set_clim(0, 30)
    except Exception:
        pass

    ax.grid(True, alpha=0.18)
    savefig(FIG_DIR / "T4_main_1_fan_favoring_hexbin.png")

    # -------------------------
    # Main Fig 2: Δ distribution (overall vs gate-only) + mean lines + KS p-value
    # -------------------------
    c_gate = col(weekly, ["FV_Gate"])
    d0 = df["Delta"].values
    d1 = (weekly.loc[weekly[c_gate] == 1, c_fv].values - weekly.loc[weekly[c_gate] == 1, c_ac].values).astype(float)
    d1 = d1[np.isfinite(d1)]

    fig, ax = plt.subplots(figsize=(11.2, 6.6))

    sns.kdeplot(d0, fill=True, alpha=0.25, linewidth=1.6, ax=ax, label="All weeks")
    if len(d1) > 5:
        sns.kdeplot(d1, fill=True, alpha=0.25, linewidth=1.8, ax=ax, label="Gate weeks only")

    ax.axvline(0, linestyle="--", linewidth=1.0, color="#333333", alpha=0.9)

    mu0 = float(np.mean(d0[np.isfinite(d0)])) if np.isfinite(d0).any() else np.nan
    ax.axvline(mu0, linestyle="--", linewidth=1.4, color="#4975EE", alpha=0.9)
    ax.text(
        0.02,
        0.92,
        rf"$\mu_\mathrm{{all}}={mu0:.3f}$",
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", fc="white", ec="#333333", lw=0.5, alpha=0.95),
    )

    p = ks_pvalue(d0, d1) if len(d1) > 5 else None
    if p is not None:
        star = "*" if p < 0.05 else ""
        ax.text(
            0.02,
            0.82,
            f"KS test p = {p:.3f}{star}",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", fc="white", ec="#333333", lw=0.5, alpha=0.95),
        )

    ax.set_title("Distribution of Δ FanShare Eliminated (FairVote − Actual)")
    ax.set_xlabel("Δ fan share")
    ax.set_ylabel("Density")

    leg = ax.legend(loc="lower right", frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("#333333")
    leg.get_frame().set_linewidth(0.5)

    ax.grid(True, alpha=0.18)
    savefig(FIG_DIR / "T4_main_2_delta_fanshare_kde.png")

    # -------------------------
    # Appendix: consistency by season (RIDGE / stacked-area style)
    # -------------------------
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    c_season = col(season, ["Season"])
    c_fvcons = col(season, ["FV_Consistency"])
    base_cols = [c for c in ["Rank", "Percent", "Bottom2"] if c in season.columns]

    tmp = season[[c_season, c_fvcons] + base_cols].dropna(subset=[c_season]).copy()
    tmp = tmp.sort_values(c_season)
    tmp[c_season] = tmp[c_season].astype(int)

    method_cols = [("FairVote", c_fvcons)] + [(m, m) for m in base_cols]

    mondrian = ["#a667c4", "#804DA3", "#86308A", "#681268", "#1e0221"]
    cmap = LinearSegmentedColormap.from_list("mondrian_map", mondrian, N=256)

    means = {name: float(np.nanmean(tmp[colname].astype(float).values)) for name, colname in method_cols}
    vmin, vmax = min(means.values()), max(means.values())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmin, vmax = 0.0, 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    x = tmp[c_season].values
    n_methods = len(method_cols)

    band_gap = 1.3
    band_height = 1.0

    fig, ax = plt.subplots(figsize=(16.2, 7.4))
    ax.set_facecolor("white")

    for i, (name, colname) in enumerate(method_cols):
        y = tmp[colname].astype(float).values
        y = np.clip(y, 0.0, 1.05)

        base = (n_methods - 1 - i) * band_gap
        color = cmap(norm(means[name]))

        ax.fill_between(
            x,
            base - 0.04,
            base - 0.04 + band_height * y,
            facecolor="#111827",
            edgecolor="none",
            alpha=0.10,
            zorder=1,
        )

        ax.fill_between(
            x,
            base,
            base + band_height * y,
            facecolor=color,
            edgecolor="#111827",
            linewidth=0.9,
            alpha=0.78,
            zorder=2,
        )

        ax.plot(x, base + band_height * y, color="#ffffff", linewidth=1.3, alpha=0.55, zorder=3)
        ax.plot(x, base + band_height * y, color="#111827", linewidth=0.8, alpha=0.8, zorder=4)

        ax.text(
            x.min() - 0.7,
            base + 0.52 * band_height,
            f"{name}",
            ha="right",
            va="center",
            fontsize=12,
            weight="bold",
            color="#111827",
        )

    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.set_ylim(-0.4, (n_methods - 1) * band_gap + band_height * 1.12)
    ax.set_yticks([])
    ax.set_xlabel("Season")
    ax.set_title("Consistency by Season — Ridgeline Stacked-Area (color = method mean consistency)", pad=12)

    ax.grid(True, axis="x", alpha=0.16)
    ax.grid(False, axis="y")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Average consistency (method mean)")

    if len(x) > 18:
        step = 2 if len(x) <= 34 else 3
        ax.set_xticks(x[::step])

    savefig(FIG_DIR / "T4_app_1_consistency_ridge_mondrian.png")

    # -------------------------
    # Appendix: gate rate by season + rolling mean
    # -------------------------
    c_gate_rate = col(season, ["FV_Excitement"])
    tmp2 = season[[c_season, c_gate_rate]].dropna().sort_values(c_season).copy()
    tmp2["roll5"] = tmp2[c_gate_rate].rolling(5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(tmp2[c_season].astype(int), tmp2[c_gate_rate], alpha=0.65)
    ax.plot(tmp2[c_season].astype(int), tmp2["roll5"], linewidth=2.6, color="#455F96")
    ax.set_title("Excitement Rate (Gate Trigger) by Season with 5-Season Rolling Mean")
    ax.set_xlabel("Season")
    ax.set_ylabel("Gate trigger rate")
    ax.grid(True, axis="y", alpha=0.18)
    savefig(FIG_DIR / "T4_app_2_gate_rate_by_season.png")

    # -------------------------
    # Appendix: omega violin + academic annotation + 95% CI
    # -------------------------
    c_omega = col(weekly, ["FV_Omega"])
    w2 = weekly[c_omega].dropna().astype(float).values
    w2 = w2[np.isfinite(w2)]

    mu = float(np.mean(w2)) if len(w2) else np.nan
    mdn = float(np.median(w2)) if len(w2) else np.nan
    ci_lo, ci_hi = bootstrap_ci(w2, B=2000, alpha=0.05, seed=7)

    fig, ax = plt.subplots(figsize=(9.2, 6))

    # If omega is (near) constant, seaborn's violin collapses and the plot looks empty.
    # In that case, draw a simple point+line and widen the y-limits for readability.
    if len(w2) == 0:
        ax.text(0.5, 0.5, "No ω values available", transform=ax.transAxes, ha="center", va="center")
    else:
        rng = float(np.nanmax(w2) - np.nanmin(w2))
        if (not np.isfinite(rng)) or (rng < 1e-6):
            y0 = float(mu)
            ax.scatter([0], [y0], s=80, zorder=3)
            ax.axhline(y0, linestyle="-", linewidth=1.6, alpha=0.7, zorder=2)
            ax.set_xlim(-0.6, 0.6)
            ax.set_xticks([])
            pad = 0.015
            ax.set_ylim(y0 - pad, y0 + pad)
        else:
            sns.violinplot(y=w2, inner=None, color="#A8D0F8", linewidth=0.0, cut=0, ax=ax)
            for coll in ax.collections:
                try:
                    coll.set_alpha(0.35)
                except Exception:
                    pass

            sns.violinplot(y=w2, inner="quartile", color="#087FF7", linewidth=0.8, cut=0, ax=ax)
            try:
                # make the second layer slightly transparent
                for coll in ax.collections[-1:]:
                    coll.set_alpha(0.35)
            except Exception:
                pass

    ax.set_title("Distribution of Adaptive Fan Weight ω")
    ax.set_ylabel("ω")
    ax.set_xlabel("")

    # Turn off scientific notation / offsets (prevents stray offset text like a lone '3')
    from matplotlib.ticker import ScalarFormatter

    fmt = ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.yaxis.get_offset_text().set_visible(False)

    # CI lines
    if np.isfinite(ci_lo):
        ax.axhline(ci_lo, linestyle="--", linewidth=1.2, color="#333333", alpha=0.85)
    if np.isfinite(ci_hi):
        ax.axhline(ci_hi, linestyle="--", linewidth=1.2, color="#333333", alpha=0.85)

    # Use a normal f-string so newlines render correctly (avoid raw-string '\\n' showing up)
    note = (
        f"$\\mu={mu:.3f}$ (mean)\n"
        f"$\\mathrm{{Mdn}}={mdn:.3f}$ (median)\n"
        + (f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]" if np.isfinite(ci_lo) else "95% CI: N/A")
    )
    ax.text(
        0.02,
        0.97,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(boxstyle="round", fc="white", ec="#333333", lw=0.5, alpha=0.95),
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#333333")
    ax.grid(False)

    savefig(FIG_DIR / "T4_app_3_omega_violin.png")

    # -------------------------
    # Appendix: margin distribution + tau line + KDE styling + p-value star
    # -------------------------
    c_margin = col(weekly, ["FV_Margin"])
    tau_val = np.nan
    if len(bestp):
        if "tau" in bestp.columns:
            tau_val = _as_float(bestp.iloc[0]["tau"])
        elif "Best_tau" in bestp.columns:
            tau_val = _as_float(bestp.iloc[0]["Best_tau"])

    m = weekly[c_margin].dropna().astype(float).values

    fig, ax = plt.subplots(figsize=(11.2, 6))

    ax.hist(
        m,
        bins=35,
        color="#80B3FF",
        edgecolor="#0066CC",
        linewidth=0.8,
        alpha=0.55,
    )

    sns.kdeplot(m, ax=ax, color="#111827", linewidth=1.8)

    star = ""
    if np.isfinite(tau_val) and c_gate in weekly.columns:
        mg = weekly.loc[weekly[c_gate] == 1, c_margin].dropna().astype(float).values
        mng = weekly.loc[weekly[c_gate] == 0, c_margin].dropna().astype(float).values
        p_tau = ks_pvalue(mg, mng)
        if p_tau is not None and p_tau < 0.05:
            star = "*"

    if np.isfinite(tau_val):
        ax.axvline(tau_val, linestyle="--", linewidth=2.2, color="#333333")
        ax.text(
            0.98,
            0.95,
            f"τ = {tau_val:.3f}{star}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round", fc="white", ec="#333333", lw=0.5, alpha=0.95),
        )

    ax.set_title("Close-call Margin Distribution")
    ax.set_xlabel(r"$S_{(n-1),w} - S_{(n),w}$")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", alpha=0.18)

    savefig(FIG_DIR / "T4_app_4_margin_hist.png")

    print("✅ Task4 figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()