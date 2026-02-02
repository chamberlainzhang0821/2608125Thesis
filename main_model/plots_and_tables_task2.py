# plots_and_tables_task2.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# seaborn + matplotlib (static, paper-ready)
import matplotlib.pyplot as plt
import seaborn as sns

# plotly (interactive 3D)

import plotly.graph_objects as go

# -------------------------
# Global typography + user-tunable colors
# -------------------------
FONT_FAMILY = "Times New Roman"

# Force Times New Roman everywhere (matplotlib + seaborn)
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.serif": [FONT_FAMILY],
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",
    "axes.unicode_minus": False,
})

# Centralized style knobs (edit these to tune colors/alpha per figure type)
STYLE = {
    # heatmaps
    "heatmap_linecolor": "#FFFFFF",
    "heatmap_linewidth": 0.2,

    # violin
    "violin_color": "#CD9CEA",
    "violin_alpha": 0.35,
    "violin_line": "#7223B8",

    # kde density (Task2 replacement for ECDF)
    "kde_line": "#2553BE",
    "kde_lw": 2.0,
    "kde_fill": "#6BA67C",
    "kde_fill_alpha": 0.25,
    "kde_norm": "#888888",
    "kde_norm_lw": 1.6,

    # generic line plots
    "line_alpha": 0.95,
    "line_lw": 2.0,

    # hist / kde fallback
    "hist_fill": "#80B3FF",
    "hist_edge": "#0066CC",
    "hist_alpha": 0.50,

    # plotly 3D (PNG-only)
    "surface_colorscale": "Viridis",
    "surface_opacity": 0.95,

    # typography
    "surface_title_size": 50,
    "surface_axis_size": 18,
    "surface_ticks_size": 13,

    # layout / geometry
    "surface_bg": "white",
    "surface_width": 1400,
    "surface_height": 1400,   # 1:1 output
    "surface_scale": 2,

    # camera + aspect ratio to avoid clipped surfaces
    "surface_camera_eye": (1.55, 1.35, 0.95),   # (x,y,z)
    "surface_aspectratio": (1.25, 1.00, 0.75),  # (x,y,z)

    # margins (increase top for title; give right room for colorbar)
    "surface_margin": (40, 80, 140, 40),  # (l,r,t,b)
}


# -------------------------
# Style (paper-like)
# -------------------------
sns.set_theme(style="whitegrid", context="talk", font=FONT_FAMILY)


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
RES = BASE_DIR / "results"
FIG_DIR = RES / "figures_task2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WEEKLY = RES / "weekly_uncertainty.xlsx"
MW = RES / "method_outcomes_by_week.xlsx"
SUMMARY = RES / "method_comparison_summary.xlsx"
BENEFIT = RES / "contestant_benefit_analysis.xlsx"

WEEK_DIFF = RES / "task2_week_consistency_diff.xlsx"
CAND_DIFF = RES / "task2_candidate_consistency_diff.xlsx"

OUT_TEX = RES / "tables_task2.tex"


def must_exist(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


def pivot_grid(df: pd.DataFrame, value_col: str, idx: str = "Season", col: str = "Week") -> pd.DataFrame:
    g = df.pivot_table(index=idx, columns=col, values=value_col, aggfunc="mean")
    g = g.sort_index().sort_index(axis=1)
    return g


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap_seaborn(grid: pd.DataFrame, title: str, outpath: Path):
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(
        grid,
        linewidths=float(STYLE["heatmap_linewidth"]),
        linecolor=str(STYLE["heatmap_linecolor"]),
        cbar_kws={"label": grid.columns.name if grid.columns.name else ""},
    )
    ax.set_title(title, fontname=FONT_FAMILY)
    ax.set_xlabel("Week", fontname=FONT_FAMILY)
    ax.set_ylabel("Season", fontname=FONT_FAMILY)
    ax.tick_params(labelsize=11)
    savefig(outpath)


def plot_violin_seaborn(values: np.ndarray, title: str, ylabel: str, outpath: Path):
    values = pd.Series(values).dropna()
    df = pd.DataFrame({"Shift": values})
    plt.figure(figsize=(8, 5))
    ax = sns.violinplot(
        data=df,
        y="Shift",
        inner="box",
        cut=0,
        color=str(STYLE["violin_color"]),
    )

    # Make the violin body semi-transparent (seaborn doesn't expose alpha directly)
    for coll in ax.collections:
        try:
            coll.set_alpha(float(STYLE["violin_alpha"]))
            coll.set_edgecolor(str(STYLE["violin_line"]))
        except Exception:
            pass

    ax.axhline(0, linewidth=1.5, linestyle="--", color=str(STYLE["violin_line"]))
    ax.set_title(title, fontname=FONT_FAMILY)
    ax.set_ylabel(ylabel, fontname=FONT_FAMILY)
    ax.set_xlabel("")
    savefig(outpath)



def plot_ecdf_seaborn(values: np.ndarray, title: str, xlabel: str, outpath: Path):
    values = pd.Series(values).dropna()
    plt.figure(figsize=(8, 5))
    ax = sns.ecdfplot(values)
    for line in ax.lines:
        line.set_alpha(float(STYLE["line_alpha"]))
        line.set_linewidth(float(STYLE["line_lw"]))
    ax.set_title(title, fontname=FONT_FAMILY)
    ax.set_xlabel(xlabel, fontname=FONT_FAMILY)
    ax.set_ylabel("ECDF", fontname=FONT_FAMILY)
    ax.grid(True, alpha=0.3)
    savefig(outpath)


# --- KDE dual-density plot ---
def plot_kde_dual_density(values: np.ndarray, title: str, xlabel: str, outpath: Path, bandwidth: float = 0.5):
    """Paper-style density plot (KDE + normal reference + skewness annotation).

    Requirements (per user):
      - KDE bandwidth h = 0.5
      - KDE line: #2E5A88, linewidth=2.0
      - fill: #6BA67C, alpha=0.25
      - normal ref: grey dashed #888888
      - annotate skewness (with optional * marker)
      - axes labels: x='Survival-week shift (Δ)', y='Density'
    """
    vals = pd.Series(values).dropna().astype(float).to_numpy()
    if len(vals) < 3:
        return

    # --- KDE via sklearn (exact bandwidth control) ---
    try:
        from sklearn.neighbors import KernelDensity

        x_min, x_max = float(np.min(vals)), float(np.max(vals))
        pad = 0.08 * (x_max - x_min + 1e-9)
        grid = np.linspace(x_min - pad, x_max + pad, 600)[:, None]

        kde = KernelDensity(kernel="gaussian", bandwidth=float(bandwidth))
        kde.fit(vals[:, None])
        log_dens = kde.score_samples(grid)
        dens = np.exp(log_dens)

        xg = grid[:, 0]

    except Exception:
        # Fallback: seaborn's KDE (bandwidth becomes approximate)
        xg = np.linspace(float(np.min(vals)), float(np.max(vals)), 600)
        # crude gaussian_kde fallback
        try:
            from scipy.stats import gaussian_kde
            gk = gaussian_kde(vals)
            dens = gk(xg)
        except Exception:
            # last-resort: histogram density
            dens, edges = np.histogram(vals, bins=40, density=True)
            xg = (edges[:-1] + edges[1:]) / 2

    # --- Normal reference (same mean/std as data) ---
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float(np.std(vals))
    sd = sd if sd > 1e-12 else 1e-12
    normal = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xg - mu) / sd) ** 2)

    # --- Skewness ---
    try:
        from scipy.stats import skew
        sk = float(skew(vals, bias=False))
    except Exception:
        m3 = float(np.mean((vals - mu) ** 3))
        sk = m3 / (sd ** 3)

    # Heuristic “significance” marker (avoid claiming formal p-values)
    star = "*" if (len(vals) >= 30 and abs(sk) >= 1.0) else ""

    plt.figure(figsize=(9.5, 5.8))

    # Filled KDE
    plt.fill_between(xg, 0, dens, color=str(STYLE["kde_fill"]), alpha=float(STYLE["kde_fill_alpha"]), linewidth=0)
    # KDE curve
    plt.plot(xg, dens, color=str(STYLE["kde_line"]), linewidth=float(STYLE["kde_lw"]), label=f"KDE (h={bandwidth:g})")
    # Normal reference
    plt.plot(xg, normal, color=str(STYLE["kde_norm"]), linestyle="--", linewidth=float(STYLE["kde_norm_lw"]), label="Normal ref.")

    plt.title(title, fontname=FONT_FAMILY)
    plt.xlabel(xlabel, fontname=FONT_FAMILY)
    plt.ylabel("Density", fontname=FONT_FAMILY)

    # Skewness annotation (top-left, paper-friendly)
    plt.annotate(
        f"Skewness = {sk:.2f}{star}",
        xy=(0.02, 0.92),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(boxstyle="round", fc="white", ec="#333333", alpha=0.95),
    )

    plt.legend(frameon=True)
    plt.grid(True, axis="y", alpha=0.25)
    plt.grid(False, axis="x")

    savefig(outpath)



# --- Helper utilities for consistency/flip plots ---
def _try_import_scipy():
    try:
        from scipy.interpolate import UnivariateSpline  # type: ignore
        from scipy.stats import ks_2samp, ttest_rel  # type: ignore
        return UnivariateSpline, ks_2samp, ttest_rel
    except Exception:
        return None, None, None


def spline_smooth_ci(
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    smooth_factor: float | None = None,
    n_boot: int = 400,
    seed: int = 7,
):
    """Cubic-spline smoothing + bootstrap CI.

    Notes:
    - With only one observation per season, we bootstrap seasons (pairs) to approximate uncertainty.
    - If SciPy is unavailable, falls back to a low-degree polynomial fit with bootstrap CI.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # guard
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        # too few points; return raw interpolation
        return np.interp(x_grid, x, y), None, None

    UnivariateSpline, _, _ = _try_import_scipy()

    rng = np.random.default_rng(seed)

    def _fit_predict(xb, yb):
        # sort for stable fit
        o = np.argsort(xb)
        xb, yb = xb[o], yb[o]
        if UnivariateSpline is not None:
            # k=3 cubic spline
            s = float(smooth_factor) if smooth_factor is not None else max(1e-8, 0.6 * len(xb))
            sp = UnivariateSpline(xb, yb, k=3, s=s)
            return sp(x_grid)
        # fallback: cubic polynomial
        deg = 3 if len(xb) >= 4 else max(1, len(xb) - 1)
        coefs = np.polyfit(xb, yb, deg=deg)
        return np.polyval(coefs, x_grid)

    y_hat = _fit_predict(x, y)

    # bootstrap CI
    boots = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(x), size=len(x))
        boots.append(_fit_predict(x[idx], y[idx]))
    boots = np.asarray(boots, dtype=float)
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return y_hat, lo, hi


def plot_consistency_spline_ci(season_df: pd.DataFrame, outpath: Path):
    """Main Fig C1: spline-smoothed trends with 95% CI + small points."""
    d = season_df.copy()
    d = d.sort_values("Season")
    x = d["Season"].astype(float).to_numpy()

    series = {
        "Rank": ("Consistency_Rank", "#1478E3"),
        "Percent": ("Consistency_Percent", "#F742E4"),
        "Bottom2": ("Consistency_Bottom2", "#FF6B6B"),
    }

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)

    plt.figure(figsize=(13.5, 5.6))
    ax = plt.gca()

    # lighter, journal-style grid
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax.grid(False, axis="x")

    # Compute a KS-test between Rank and Percent season distributions (display only; no overclaim)
    UnivariateSpline, ks_2samp, ttest_rel = _try_import_scipy()
    p_ks_txt = ""
    p_star = ""
    if ks_2samp is not None:
        try:
            p_ks = float(ks_2samp(d["Consistency_Rank"].dropna(), d["Consistency_Percent"].dropna()).pvalue)
            p_ks_txt = f"KS test (Rank vs Percent): p={p_ks:.3f}{'*' if p_ks < 0.05 else ''}"
        except Exception:
            p_ks_txt = ""

    # Paired test for a conservative star marker
    p_pair = None
    if ttest_rel is not None:
        try:
            rr = d[["Consistency_Rank", "Consistency_Percent"]].dropna()
            if len(rr) >= 6:
                p_pair = float(ttest_rel(rr["Consistency_Rank"], rr["Consistency_Percent"]).pvalue)
        except Exception:
            p_pair = None

    # plot each smoothed line + CI + small points
    for name, (col, color) in series.items():
        if col not in d.columns:
            continue
        y = d[col].astype(float).to_numpy()
        yhat, lo, hi = spline_smooth_ci(x, y, x_grid, smooth_factor=None, n_boot=400, seed=7)
        ax.plot(x_grid, yhat, color=color, linewidth=1.8, alpha=0.95, label=name)
        if lo is not None and hi is not None:
            ax.fill_between(x_grid, lo, hi, color=color, alpha=0.25, linewidth=0)
        ax.scatter(x, y, color=color, s=18, alpha=0.40, edgecolors="none")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Season", fontname=FONT_FAMILY)
    ax.set_ylabel("Match rate", fontname=FONT_FAMILY)
    ax.set_title("Consistency by Season Under Different Combination Rules", fontname=FONT_FAMILY, weight="bold")
    ax.legend(ncol=3, frameon=True)

    # annotate KS test p-value
    if p_ks_txt:
        ax.text(
            0.01,
            1.02,
            p_ks_txt,
            transform=ax.transAxes,
            fontsize=11,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="#333333", alpha=0.95),
        )

    # add a * marker at the largest Rank-Percent divergence if paired p<0.05
    try:
        diff = np.abs(d["Consistency_Rank"].astype(float) - d["Consistency_Percent"].astype(float))
        j = int(np.nanargmax(diff.values))
        x_star = float(d.iloc[j]["Season"])
        y_star = float(max(d.iloc[j]["Consistency_Rank"], d.iloc[j]["Consistency_Percent"]))
        if p_pair is not None and p_pair < 0.05:
            ax.text(x_star, min(1.03, y_star + 0.03), "*", fontsize=16, ha="center", va="center")
    except Exception:
        pass

    savefig(outpath)


def plot_flip_kde_and_diff(season_df: pd.DataFrame, outpath: Path):
    """Main Fig C2: dual-density (KDE) + difference curve with CI.

    We interpret 'Rank flip rate' and 'Percent flip rate' as their flip relative to Bottom2.
    - Rank flip: Flip_Rank_vs_Bottom2
    - Percent flip: Flip_Percent_vs_Bottom2
    Difference curve: |RankFlip - PercentFlip| across seasons.
    """
    d = season_df.copy().sort_values("Season")
    x = d["Season"].astype(float).to_numpy()

    # required columns
    if ("Flip_Rank_vs_Bottom2" not in d.columns) or ("Flip_Percent_vs_Bottom2" not in d.columns):
        # fallback to original single-series if needed
        plt.figure(figsize=(12, 5))
        ax = sns.lineplot(data=d, x="Season", y="Flip_Rank_vs_Percent", marker="o")
        plt.title("Flip Rate (Rank vs Percent) by Season")
        plt.ylabel("Flip Rate")
        plt.xlabel("Season")
        savefig(outpath)
        return

    r = d["Flip_Rank_vs_Bottom2"].astype(float).to_numpy()
    p = d["Flip_Percent_vs_Bottom2"].astype(float).to_numpy()
    diff = np.abs(r - p)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13.8, 7.0))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.25], hspace=0.20)

    # --- (A) dual density ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax1.grid(False, axis="x")

    # KDE (seaborn)
    sns.kdeplot(r, ax=ax1, color="#4D47EC", linewidth=2.0, fill=True, alpha=0.18, label="Rank flip (vs Bottom2)")
    sns.kdeplot(p, ax=ax1, color="#C46AD4", linewidth=2.0, fill=True, alpha=0.18, label="Percent flip (vs Bottom2)")

    ax1.set_xlabel("")
    ax1.set_ylabel("Density", fontname=FONT_FAMILY)
    ax1.set_title("Flip Behavior Across Seasons: Density + Difference", fontname=FONT_FAMILY, weight="bold")
    ax1.legend(frameon=True, ncol=2, loc="upper right")

    # --- (B) difference curve with CI ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax2.grid(False, axis="x")

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 300)
    yhat, lo, hi = spline_smooth_ci(x, diff, x_grid, smooth_factor=None, n_boot=400, seed=9)

    ax2.axhline(0, color="#888888", linestyle="--", linewidth=1.2, alpha=0.9)
    ax2.plot(x_grid, yhat, color="#FF6B6B", linewidth=1.2, alpha=0.95, label=r"$|\Delta|$ difference")
    if lo is not None and hi is not None:
        ax2.fill_between(x_grid, lo, hi, color="#FF6B6B", alpha=0.18, linewidth=0)

    ax2.scatter(x, diff, color="#FF6B6B", s=18, alpha=0.40, edgecolors="none")

    ax2.set_xlabel("Season", fontname=FONT_FAMILY)
    ax2.set_ylabel("Flip Rate Difference", fontname=FONT_FAMILY)

    # star on peak if KS between r and p is significant
    UnivariateSpline, ks_2samp, _ = _try_import_scipy()
    p_ks = None
    if ks_2samp is not None:
        try:
            p_ks = float(ks_2samp(pd.Series(r).dropna(), pd.Series(p).dropna()).pvalue)
        except Exception:
            p_ks = None

    try:
        j = int(np.nanargmax(diff))
        if p_ks is not None and p_ks < 0.05:
            ax2.text(float(d.iloc[j]["Season"]), float(diff[j]) + 0.01, "*", fontsize=16, ha="center", va="bottom")
            ax1.text(
                0.01,
                0.88,
                f"KS test (RankFlip vs PercentFlip): p={p_ks:.3f}*",
                transform=ax1.transAxes,
                fontsize=11,
                bbox=dict(boxstyle="round", fc="white", ec="#333333", alpha=0.95),
            )
        elif p_ks is not None:
            ax1.text(
                0.01,
                0.88,
                f"KS test (RankFlip vs PercentFlip): p={p_ks:.3f}",
                transform=ax1.transAxes,
                fontsize=11,
                bbox=dict(boxstyle="round", fc="white", ec="#333333", alpha=0.95),
            )
    except Exception:
        pass

    savefig(outpath)


def plot_surface_plotly(grid: pd.DataFrame, title: str, out_png: Path):
    """Plotly 3D surface. Export PNG only (square 1:1). Requires kaleido.

    Fixes:
      - enforce Times New Roman via layout.font + axis fonts
      - enlarge title and add top margin so it is never clipped
      - set camera + aspect ratio so the surface is fully visible (no cut corners)
      - do NOT write any html files
    """
    z = grid.values.astype(float)
    x = grid.columns.astype(int).to_numpy()
    y = grid.index.astype(int).to_numpy()

    # user-tunable knobs
    W = int(STYLE.get("surface_width", 1400))
    H = int(STYLE.get("surface_height", 1400))
    SCALE = int(STYLE.get("surface_scale", 2))
    cam_eye = STYLE.get("surface_camera_eye", (1.55, 1.35, 0.95))
    ar = STYLE.get("surface_aspectratio", (1.25, 1.00, 0.75))
    m = STYLE.get("surface_margin", (40, 80, 140, 40))
    ml, mr, mt, mb = int(m[0]), int(m[1]), int(m[2]), int(m[3])

    fig = go.Figure(
        data=[
            go.Surface(
                z=z,
                x=x,
                y=y,
                colorscale=str(STYLE["surface_colorscale"]),
                opacity=float(STYLE["surface_opacity"]),
                showscale=True,
                contours={
                    "z": {"show": False}
                },
            )
        ]
    )

    fig.update_layout(
        # global font (important for kaleido export)
        font=dict(family=FONT_FAMILY, size=int(STYLE["surface_ticks_size"])),
        title=dict(
            text=title,
            x=0.5,
            y=0.99,
            xanchor="center",
            yanchor="top",
            font=dict(family=FONT_FAMILY, size=int(STYLE["surface_title_size"])),
        ),
        scene=dict(
            xaxis_title="Week",
            yaxis_title="Season",
            zaxis_title="Value",
            xaxis=dict(
                title_font=dict(family=FONT_FAMILY, size=int(STYLE["surface_axis_size"])),
                tickfont=dict(family=FONT_FAMILY, size=int(STYLE["surface_ticks_size"])),
                showbackground=True,
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(0,0,0,0.10)",
                zerolinecolor="rgba(0,0,0,0.10)",
            ),
            yaxis=dict(
                title_font=dict(family=FONT_FAMILY, size=int(STYLE["surface_axis_size"])),
                tickfont=dict(family=FONT_FAMILY, size=int(STYLE["surface_ticks_size"])),
                showbackground=True,
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(0,0,0,0.10)",
                zerolinecolor="rgba(0,0,0,0.10)",
            ),
            zaxis=dict(
                title_font=dict(family=FONT_FAMILY, size=int(STYLE["surface_axis_size"])),
                tickfont=dict(family=FONT_FAMILY, size=int(STYLE["surface_ticks_size"])),
                showbackground=True,
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(0,0,0,0.10)",
                zerolinecolor="rgba(0,0,0,0.10)",
            ),
            bgcolor=str(STYLE["surface_bg"]),
            camera=dict(eye=dict(x=float(cam_eye[0]), y=float(cam_eye[1]), z=float(cam_eye[2]))),
            aspectmode="manual",
            aspectratio=dict(x=float(ar[0]), y=float(ar[1]), z=float(ar[2])),
        ),
        margin=dict(l=ml, r=mr, t=mt, b=mb),
        width=W,
        height=H,
    )

    # PNG export only (no html)
    try:
        fig.write_image(str(out_png), width=W, height=H, scale=SCALE)
    except Exception:
        # If kaleido is not available, silently skip PNG export.
        pass


def latex_escape(s: str) -> str:
    """
    Escape special LaTeX chars in strings (names like 'Bobby Bones' ok, but keep safe).
    """
    if s is None:
        return ""
    s = str(s)
    repl = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def to_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + "l" * len(cols) + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join([latex_escape(c) for c in cols]) + r" \\")
    lines.append(r"\midrule")

    for _, row in df.iterrows():
        row_vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                row_vals.append("NA")
            elif isinstance(v, (int, np.integer)):
                row_vals.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                row_vals.append(f"{float(v):.3f}".rstrip("0").rstrip("."))
            else:
                row_vals.append(latex_escape(str(v)))
        lines.append(" & ".join(row_vals) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{latex_escape(caption)}}}")
    lines.append(rf"\label{{{latex_escape(label)}}}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    # --- load required ---
    for p in [WEEKLY, MW, SUMMARY, BENEFIT]:
        must_exist(p)

    weekly = pd.read_excel(WEEKLY)
    mw = pd.read_excel(MW)
    summary = pd.read_excel(SUMMARY)
    benefit = pd.read_excel(BENEFIT)

    week_diff = pd.read_excel(WEEK_DIFF) if WEEK_DIFF.exists() else None
    cand_diff = pd.read_excel(CAND_DIFF) if CAND_DIFF.exists() else None

    # -------------------------
    # 1) Landscapes (heatmap + plotly 3D)
    # -------------------------
    # expected columns in weekly:
    # Season, Week, AvgVoteUncertainty, AvgAcceptanceRate, NumCouples
    acc_col = "AvgAcceptanceRate"
    unc_col = "AvgVoteUncertainty"
    if acc_col not in weekly.columns:
        raise KeyError(f"{acc_col} not found in weekly_uncertainty.xlsx")
    if unc_col not in weekly.columns:
        raise KeyError(f"{unc_col} not found in weekly_uncertainty.xlsx")

    acc_grid = pivot_grid(weekly, acc_col)
    unc_grid = pivot_grid(weekly, unc_col)

    plot_heatmap_seaborn(
        acc_grid,
        title="Weekly Acceptance Rate Landscape",
        outpath=FIG_DIR / "H1_acceptance_heatmap_seaborn.png",
    )
    plot_heatmap_seaborn(
        unc_grid,
        title="Weekly Uncertainty Landscape",
        outpath=FIG_DIR / "H2_uncertainty_heatmap_seaborn.png",
    )

    # plotly 3D surfaces: export PNG only (square 1:1)
    plot_surface_plotly(
        acc_grid,
        title="Acceptance Rate Landscape (3D Surface)",
        out_png=FIG_DIR / "S1_acceptance_surface_plotly.png",
    )
    plot_surface_plotly(
        unc_grid,
        title="Uncertainty Landscape (3D Surface)",
        out_png=FIG_DIR / "S2_uncertainty_surface_plotly.png",
    )

    # -------------------------
    # 2) Benefit distribution (violin + ECDF)
    # -------------------------
    b_pr = benefit["Benefit_Percent_minus_Rank"].dropna().astype(float).to_numpy()
    b_bp = benefit["Benefit_Bottom2_minus_Percent"].dropna().astype(float).to_numpy()

    plot_violin_seaborn(
        b_pr,
        title="Benefit Distribution (Percent − Rank)",
        ylabel="Survival-week shift",
        outpath=FIG_DIR / "V1_violin_percent_minus_rank_seaborn.png",
    )
    plot_violin_seaborn(
        b_bp,
        title="Impact Distribution (Bottom2 − Percent)",
        ylabel="Survival-week shift",
        outpath=FIG_DIR / "V2_violin_bottom2_minus_percent_seaborn.png",
    )
    plot_kde_dual_density(
        b_pr,
        title="Distribution of Survival-week Shift (Percent − Rank)",
        xlabel="Survival-week shift (Δ)",
        outpath=FIG_DIR / "E1_ecdf_percent_minus_rank_seaborn.png",
        bandwidth=0.5,
    )

    # -------------------------
    # 3) Season-level curves (consistency + flip)
    # -------------------------
    season_summary = summary[summary["Season"].astype(str) != "ALL"].copy()
    if len(season_summary) > 0:
        season_summary["Season"] = season_summary["Season"].astype(int)
        season_summary = season_summary.sort_values("Season")

        # --- C1: spline-smoothed consistency with 95% CI + small points + KS annotation
        plot_consistency_spline_ci(
            season_summary,
            outpath=FIG_DIR / "C1_consistency_by_season_seaborn.png",
        )

        # --- C2: dual density + difference curve (interpreting flip vs Bottom2)
        plot_flip_kde_and_diff(
            season_summary,
            outpath=FIG_DIR / "C2_flip_rank_vs_percent_seaborn.png",
        )

    # -------------------------
    # 4) Tables for LaTeX
    # -------------------------
    # (A) Given 4 cases
    given_cases = [
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones"),
    ]
    rows = []
    for s, name in given_cases:
        sub = benefit[benefit["Season"] == s].copy()
        hit = sub[sub["CoupleID"].str.contains(name, case=False, na=False)]
        if len(hit) == 0:
            rows.append({
                "Season": s,
                "Celebrity": name,
                "Rank": "NA",
                "Percent": "NA",
                "Bottom2": "NA",
                "Bottom2-Percent": "NA",
            })
        else:
            r = hit.iloc[0]
            rows.append({
                "Season": int(s),
                "Celebrity": name,
                "Rank": int(r["PredElimWeek_Rank"]),
                "Percent": int(r["PredElimWeek_Percent"]),
                "Bottom2": int(r["PredElimWeek_Bottom2"]),
                "Bottom2-Percent": int(r["Benefit_Bottom2_minus_Percent"]),
            })
    cases_df = pd.DataFrame(rows)

    # (B) Top5 weirdest (from cand_diff)
    top5_df = pd.DataFrame()
    if cand_diff is not None and len(cand_diff) > 0:
        score_col = None
        for c in ["WeirdScore", "ControversyScore", "Score"]:
            if c in cand_diff.columns:
                score_col = c
                break
        if score_col is None:
            score_col = "MaxMethodDiff" if "MaxMethodDiff" in cand_diff.columns else cand_diff.columns[-1]

        cand_sorted = cand_diff.sort_values(score_col, ascending=False).head(5).copy()

        keep = []
        for c in ["Season", "Celebrity", "ProDancer", score_col,
                  "PredElimWeek_Rank", "PredElimWeek_Percent", "PredElimWeek_Bottom2"]:
            if c in cand_sorted.columns:
                keep.append(c)

        top5_df = cand_sorted[keep].copy()
        if score_col in top5_df.columns and score_col != "WeirdScore":
            top5_df = top5_df.rename(columns={score_col: "WeirdScore"})

        # simplify column names
        rename_map = {
            "PredElimWeek_Rank": "Rank",
            "PredElimWeek_Percent": "Percent",
            "PredElimWeek_Bottom2": "Bottom2",
        }
        top5_df = top5_df.rename(columns=rename_map)

    # (C) Flip weeks table (appendix)
    flip6 = pd.DataFrame()
    if "Pred_Rank" in mw.columns and "Pred_Percent" in mw.columns:
        tmp = mw.copy()
        tmp["Flip_RP"] = (tmp["Pred_Rank"] != tmp["Pred_Percent"]).astype(int)
        flip6 = tmp[tmp["Flip_RP"] == 1].sort_values(["Season", "Week"]).copy()
        keep = [c for c in ["Season", "Week", "ActualEliminated", "Pred_Rank", "Pred_Percent"] if c in flip6.columns]
        flip6 = flip6[keep].head(10)

    # write LaTeX
    tex_parts = []
    tex_parts.append("% Auto-generated tables for Task 2 (do not edit by hand)")
    tex_parts.append("% Add in preamble: \\usepackage{booktabs}")
    tex_parts.append("")

    tex_parts.append(
        to_latex_table(
            cases_df,
            caption="Given controversy cases: predicted elimination week under different rules.",
            label="tab:given_controversy_cases",
        )
    )

    if len(top5_df) > 0:
        tex_parts.append(
            to_latex_table(
                top5_df,
                caption="Top 5 most method-sensitive contestants (automatically detected).",
                label="tab:top5_weirdest",
            )
        )

    if len(flip6) > 0:
        tex_parts.append(
            to_latex_table(
                flip6,
                caption="Weeks where Rank and Percent predict different eliminations (first rows shown).",
                label="tab:flip_weeks_rank_percent",
            )
        )

    OUT_TEX.write_text("\n".join(tex_parts), encoding="utf-8")

    print("✅ Static figures (PNG) saved to:", FIG_DIR)
    print("✅ 3D surfaces exported as square PNG to:", FIG_DIR)
    print("✅ LaTeX tables written to:", OUT_TEX)
    print("Tip: in LaTeX, use \\input{results/tables_task2.tex}")


if __name__ == "__main__":
    main()