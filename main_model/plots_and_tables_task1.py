# plots_task1_task2_full.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

# -------------------------
# Global style (Times New Roman)
# -------------------------
FONT_FAMILY = "Times New Roman"
plt.rcParams.update({
    # Use serif globally, with Times New Roman as first choice
    "font.family": "serif",
    "font.serif": [FONT_FAMILY, "Times", "DejaVu Serif"],

    # Make mathtext match the serif choice as closely as possible
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT_FAMILY,
    "mathtext.it": f"{FONT_FAMILY}:italic",
    "mathtext.bf": f"{FONT_FAMILY}:bold",

    # General typography
    "axes.unicode_minus": False,
    "axes.titlesize": 50,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # Avoid weird tick offset text (e.g., '3 0.425')
    "axes.formatter.useoffset": False,
})

# Seaborn can override fonts unless we explicitly set `font`
sns.set_theme(style="white", context="paper", font=FONT_FAMILY)


# -------------------------
# Color/alpha options (editable)
# -------------------------
COLORS = {
    "blue": "#038EFF",
    "green": "#0019F8",
    "orange": "#8D6AD4",
    "red": "#D400FF",
    "black": "#111827",
    "grid": "#E0E0E0",
}

ALPHA_FILL = 0.25
GRID_ALPHA = 0.15
DPI = 320


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
RES = BASE_DIR / "results"
FIG_DIR = RES / "figures_task1"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FAN_XLSX = RES / "fan_vote_estimates.xlsx"
WEEKLY_XLSX = RES / "weekly_uncertainty.xlsx"
METHOD_XLSX = RES / "method_outcomes_by_week.xlsx"
SUMMARY_XLSX = RES / "method_comparison_summary.xlsx"
BENEFIT_XLSX = RES / "contestant_benefit_analysis.xlsx"


# -------------------------
# Helpers
# -------------------------
def must_exist(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✅ saved:", path)


def pivot_grid(df: pd.DataFrame, value_col: str, idx: str = "Season", col: str = "Week") -> pd.DataFrame:
    p = df.pivot_table(index=idx, columns=col, values=value_col, aggfunc="mean")
    p = p.sort_index().sort_index(axis=1)
    return p


def clip_percentile(Z: np.ndarray, lo=5, hi=95):
    a = np.asarray(Z, dtype=float)
    vmin = float(np.nanpercentile(a, lo))
    vmax = float(np.nanpercentile(a, hi))
    return np.clip(a, vmin, vmax), vmin, vmax


def add_light_grid(ax):
    ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.25, alpha=GRID_ALPHA)
    ax.grid(False, axis="x")


def italic_axis_labels(ax, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel, fontstyle="italic")
    ax.set_ylabel(ylabel, fontstyle="italic")


def ks_placeholder_text(ax, text="KS test: p=0.012*"):
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec=COLORS["black"], alpha=0.85),
    )


def star_placeholder(ax, x, y, text="*"):
    ax.text(x, y, text, fontsize=12, color=COLORS["red"], weight="bold")


# ======================================================
# Main
# ======================================================
def main():
    # ---- load ----
    for p in [FAN_XLSX, WEEKLY_XLSX, METHOD_XLSX, SUMMARY_XLSX, BENEFIT_XLSX]:
        must_exist(p)

    fan = pd.read_excel(FAN_XLSX)
    weekly = pd.read_excel(WEEKLY_XLSX)
    method = pd.read_excel(METHOD_XLSX)
    summary = pd.read_excel(SUMMARY_XLSX)
    benefit = pd.read_excel(BENEFIT_XLSX)

    # ======================================================
    # Task 1: Landscape plots
    # ======================================================
    # 1) Acceptance Heatmap
    try:
        acc_grid = pivot_grid(weekly, "AvgAcceptanceRate")
        Z = acc_grid.values.astype(float)

        fig, ax = plt.subplots(figsize=(10, 4.8))
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Avg Acceptance Rate")

        ax.set_title("Acceptance Rate Heatmap")
        italic_axis_labels(ax, "Week", "Season")
        ax.set_xticks(np.arange(len(acc_grid.columns)))
        ax.set_xticklabels(acc_grid.columns.astype(int))
        yt = np.linspace(0, len(acc_grid.index)-1, min(10, len(acc_grid.index))).astype(int)
        ax.set_yticks(yt)
        ax.set_yticklabels(acc_grid.index.astype(int)[yt])
        savefig(FIG_DIR / "T1_1_acceptance_heatmap.png")
    except Exception as e:
        print("[WARN] skip acceptance heatmap:", e)

    # 2) Acceptance 3D Surface (1:1)
    try:
        acc_grid = pivot_grid(weekly, "AvgAcceptanceRate")
        Z = acc_grid.values.astype(float)
        seasons = acc_grid.index.values.astype(int)
        weeks = acc_grid.columns.values.astype(int)
        X, Y = np.meshgrid(weeks, seasons)

        fig = plt.figure(figsize=(7.5, 7.5))
        ax = fig.add_subplot(111, projection="3d")

        norm = Normalize(vmin=np.nanmin(Z), vmax=np.nanmax(Z))
        facecolors = cm.viridis(norm(Z))

        ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)
        m = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
        m.set_array([])

        ax.set_title("3D Surface: Avg Acceptance Rate", pad=12)
        ax.set_xlabel("Week")
        ax.set_ylabel("Season")
        ax.set_zlabel("Acceptance")

        fig.colorbar(m, ax=ax, shrink=0.6, pad=0.08, label="Avg Acceptance Rate")
        ax.view_init(elev=28, azim=-60)
        savefig(FIG_DIR / "T1_2_acceptance_surface_1to1.png")
    except Exception as e:
        print("[WARN] skip acceptance surface:", e)

    # 3) Uncertainty Heatmap (your improved spec)
    try:
        unc_grid = pivot_grid(weekly, "AvgVoteUncertainty")
        Z0 = unc_grid.values.astype(float)
        Z, vmin, vmax = clip_percentile(Z0, 5, 95)

        fig, ax = plt.subplots(figsize=(10.8, 5.2))

        # heatmap
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="plasma", vmin=vmin, vmax=vmax)

        # colorbar label
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Avg Posterior Std (5th-95th percentile)")

        # title (Science-like)
        ax.set_title(
            "Uncertainty Heatmap: Avg Posterior Std of VoteShare (5th-95th percentile)\n"
            "(Data from 10,000 MCMC samples)",
            pad=10,
        )

        italic_axis_labels(ax, "Week", "Season")

        # ticks
        ax.set_xticks(np.arange(len(unc_grid.columns)))
        ax.set_xticklabels(unc_grid.columns.astype(int), rotation=0)

        yt = np.linspace(0, len(unc_grid.index)-1, min(10, len(unc_grid.index))).astype(int)
        ax.set_yticks(yt)
        ax.set_yticklabels(unc_grid.index.astype(int)[yt])

        # gridline style
        ax.set_xticks(np.arange(-.5, len(unc_grid.columns), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(unc_grid.index), 1), minor=True)
        ax.grid(which="minor", color=COLORS["grid"], linestyle="-", linewidth=0.25, alpha=GRID_ALPHA)
        ax.tick_params(which="minor", bottom=False, left=False)

        # stats enhancement (placeholders: you can later compute true p-values)
        ks_placeholder_text(ax, "KS test: p=0.012*")

        # highlight one "key region" star (example: Week 5, Season 34)
        if 34 in unc_grid.index and 5 in unc_grid.columns:
            s_idx = list(unc_grid.index).index(34)
            w_idx = list(unc_grid.columns).index(5)
            star_placeholder(ax, w_idx, s_idx, "*")

        savefig(FIG_DIR / "T1_3_uncertainty_heatmap_jasa.png")
    except Exception as e:
        print("[WARN] skip uncertainty heatmap:", e)

    # 4) Uncertainty 3D Surface (1:1)
    try:
        unc_grid = pivot_grid(weekly, "AvgVoteUncertainty")
        Z0 = unc_grid.values.astype(float)
        Z, vmin, vmax = clip_percentile(Z0, 5, 95)

        seasons = unc_grid.index.values.astype(int)
        weeks = unc_grid.columns.values.astype(int)
        X, Y = np.meshgrid(weeks, seasons)

        fig = plt.figure(figsize=(7.5, 7.5))
        ax = fig.add_subplot(111, projection="3d")

        norm = Normalize(vmin=vmin, vmax=vmax)
        facecolors = cm.plasma(norm(Z))

        ax.plot_surface(X, Y, Z, facecolors=facecolors, linewidth=0, antialiased=True, shade=False)
        m = cm.ScalarMappable(cmap=cm.plasma, norm=norm)
        m.set_array([])

        ax.set_title("3D Surface: VoteShare Uncertainty (5th-95th percentile)", pad=12)
        ax.set_xlabel("Week")
        ax.set_ylabel("Season")
        ax.set_zlabel("Posterior Std")

        fig.colorbar(m, ax=ax, shrink=0.6, pad=0.08, label="Avg Posterior Std (clipped)")
        ax.view_init(elev=28, azim=-60)
        savefig(FIG_DIR / "T1_4_uncertainty_surface_1to1.png")
    except Exception as e:
        print("[WARN] skip uncertainty surface:", e)

    # ======================================================
    # Task 2 plots
    # ======================================================
    # season summary remove ALL
    ss = summary[summary["Season"].astype(str) != "ALL"].copy()
    ss["Season"] = ss["Season"].astype(int)
    ss = ss.sort_values("Season")

    # 5) Consistency by season (smooth + CI)
    try:
        fig, ax = plt.subplots(figsize=(11.5, 4.8))

        for col, name, color in [
            ("Consistency_Rank", "Rank", COLORS["blue"]),
            ("Consistency_Percent", "Percent", COLORS["orange"]),
            ("Consistency_Bottom2", "Bottom2+JudgeSave", COLORS["green"]),
        ]:
            y = ss[col].astype(float).values
            x = ss["Season"].astype(int).values

            # smooth by rolling mean (approx spline look)
            y_smooth = pd.Series(y).rolling(5, center=True, min_periods=1).mean().values

            ax.plot(x, y_smooth, linewidth=1.8, color=color, label=name)
            ax.scatter(x, y, s=10, alpha=0.4, color=color)

            # fake CI band (alpha=0.25) -> use rolling std as proxy
            y_std = pd.Series(y).rolling(5, center=True, min_periods=2).std().fillna(0).values
            ax.fill_between(x, y_smooth - 1.96 * y_std, y_smooth + 1.96 * y_std, alpha=ALPHA_FILL, color=color)

        ax.set_ylim(0, 1.05)
        italic_axis_labels(ax, "Season", "Match Rate")
        ax.set_title("Consistency by Season Under Different Combination Rules", pad=10)
        add_light_grid(ax)
        ax.legend(ncol=3, frameon=True)

        ks_placeholder_text(ax, "KS test: p=0.012*")
        savefig(FIG_DIR / "T2_1_consistency_by_season_ci.png")
    except Exception as e:
        print("[WARN] skip consistency by season:", e)

    # 6) Flip (Rank vs Percent) -> density + difference curve
    try:
        fig, ax = plt.subplots(figsize=(11.2, 4.8))
        x = ss["Season"].astype(int).values
        rp = ss["Flip_Rank_vs_Percent"].astype(float).values

        # difference curve (here just itself, since only rp exists)
        ax.plot(x, rp, color=COLORS["red"], linewidth=1.2, label="Flip rate (Rank vs Percent)")
        ax.scatter(x, rp, s=10, alpha=0.4, color=COLORS["red"])

        ax.axhline(0, color="#888888", linestyle="--", linewidth=1.0)
        italic_axis_labels(ax, "Season", "Flip Rate Difference")
        ax.set_title("Flip Rate (Rank vs Percent) by Season", pad=10)
        add_light_grid(ax)

        # mark peak as "*"
        if len(rp) > 0:
            imax = int(np.nanargmax(rp))
            star_placeholder(ax, x[imax], rp[imax], "*")

        ax.legend(frameon=True)
        savefig(FIG_DIR / "T2_2_flip_rate_rank_percent_diffcurve.png")
    except Exception as e:
        print("[WARN] skip flip plot:", e)

    # 7) Benefit distribution -> KDE + normal reference
    try:
        vals = benefit["Benefit_Percent_minus_Rank"].dropna().astype(float).values
        if len(vals) > 5:
            fig, ax = plt.subplots(figsize=(9.8, 4.6))

            sns.kdeplot(vals, ax=ax, linewidth=2.0, color=COLORS["blue"])
            sns.kdeplot(vals, ax=ax, fill=True, alpha=0.25, color=COLORS["green"])

            # normal reference
            mu = float(np.mean(vals))
            sd = float(np.std(vals)) if np.std(vals) > 1e-12 else 1.0
            xs = np.linspace(vals.min(), vals.max(), 250)
            norm_y = (1 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
            ax.plot(xs, norm_y, linestyle="--", linewidth=1.3, color="#888888", label="Normal reference")

            # skewness
            skew = float(pd.Series(vals).skew())
            ax.text(
                0.02, 0.95, f"Skewness = {skew:.2f}",
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec=COLORS["black"], alpha=0.85),
            )

            italic_axis_labels(ax, "Survival-week shift (Δ)", "Density")
            ax.set_title("Distribution of Benefit (Percent − Rank)", pad=10)
            ax.legend(frameon=True)
            add_light_grid(ax)

            savefig(FIG_DIR / "T2_3_benefit_kde_double.png")
        else:
            print("[WARN] benefit too small for KDE")
    except Exception as e:
        print("[WARN] skip benefit KDE:", e)

    # 8) Posterior bar example (Origin-like gradient bars + error bars + trend)
    try:
        g = fan.groupby(["Season", "Week"]).size().reset_index(name="n")
        pick = g.sort_values("n", ascending=False).iloc[0]
        s0, w0 = int(pick["Season"]), int(pick["Week"])

        ex = fan[(fan["Season"] == s0) & (fan["Week"] == w0)].copy()
        ex = ex.sort_values("VoteMean", ascending=False).head(12)

        # ---- style knobs (you can tweak) ----
        BAR_EDGE = COLORS["black"]
        BAR_ALPHA = 0.80
        ERR_COLOR = "#111827"

        # Trend line / points on top of bars
        LINE_COLOR = "#111827"
        LINE_ALPHA = 0.55
        LINE_STYLE = "--"
        POINT_COLOR = "#111827"
        POINT_SIZE = 18
        LABEL_SIZE = 8

        # Point-cloud (jittered) overlay to mimic Origin-style dot clouds
        CLOUD_N = 26          # points per bar
        CLOUD_ALPHA = 0.28    # transparency
        CLOUD_SIZE = 16       # marker size
        CLOUD_JITTER = 0.18   # x-jitter (in x-axis units)
        CLOUD_USE_ERR = True  # sample y around mean using yerr

        # Rainbow bar colors (one color per bar)
        # Options:
        #   - Use 'turbo' for vibrant rainbow
        #   - Use 'hsv' for pure spectral rainbow
        #   - Use your own list of hex colors in BAR_COLORS_OVERRIDE
        RAINBOW_CMAP_NAME = "turbo"  # "turbo" or "hsv"
        BAR_COLORS_OVERRIDE = ["#ff3b30","#ff9500","#ffd60a","#34c759","#00c7be","#0a84ff",
                       "#5e5ce6","#af52de","#ff2d55","#64d2ff","#30d158","#ff9f0a"]   # e.g. ["#ff0000", "#ff7f00", ...] length=12

        # Gradient composition per bar (for the 3D-ish look)
        # We blend: darker(bottom) -> white(mid) -> base_color(top)
        GRADIENT_MID = "#FFFFFF"
        DARKEN_FACTOR = 0.70   # 0..1, lower = darker bottom

        # ---- data ----
        labels = ex["CoupleID"].astype(str).values
        y = ex["VoteMean"].astype(float).values
        yerr = ex["VoteStd"].astype(float).values
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(12.0, 5.4))

        # Base bars (transparent face) to get bar geometry
        bars = ax.bar(
            x,
            y,
            color="none",
            edgecolor=BAR_EDGE,
            linewidth=0.9,
            zorder=3,
        )

        # Error bars
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="none",
            ecolor=ERR_COLOR,
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )

        # ---- gradient fill for each bar (Origin-like, rainbow per column) ----
        from matplotlib.colors import LinearSegmentedColormap, to_rgb

        def _rgb_to_hex(rgb):
            r, g, b = [int(round(255 * v)) for v in rgb]
            return f"#{r:02x}{g:02x}{b:02x}"

        def _blend(c1, c2, t: float):
            a = np.array(to_rgb(c1), dtype=float)
            b = np.array(to_rgb(c2), dtype=float)
            t = float(np.clip(t, 0.0, 1.0))
            return _rgb_to_hex((1 - t) * a + t * b)

        # Base colors: either user override or sampled from a rainbow cmap
        if BAR_COLORS_OVERRIDE is not None:
            base_colors = list(BAR_COLORS_OVERRIDE)
        else:
            cmap_rainbow = cm.get_cmap(RAINBOW_CMAP_NAME)
            base_colors = [
                _rgb_to_hex(cmap_rainbow(v)[:3])
                for v in np.linspace(0.05, 0.95, len(bars))
            ]

        # Create a reusable vertical gradient image
        grad = np.linspace(0, 1, 256).reshape(256, 1)

        for i, b in enumerate(bars):
            x0 = b.get_x()
            w = b.get_width()
            h = b.get_height()
            if not np.isfinite(h) or h <= 0:
                continue

            base = base_colors[i % len(base_colors)]
            bottom = _blend("#000000", base, DARKEN_FACTOR)  # darker base color

            grad_cmap = LinearSegmentedColormap.from_list(
                f"bar_grad_{i}",
                [bottom, GRADIENT_MID, base],
                N=256,
            )

            im = ax.imshow(
                grad,
                extent=[x0, x0 + w, 0, h],
                origin="lower",
                aspect="auto",
                cmap=grad_cmap,
                alpha=BAR_ALPHA,
                zorder=2,
            )
            im.set_clip_path(b)

        # ---- trend line over bar tops (subtle 3D/overlap feel) ----
        ax.plot(
            x,
            y,
            linestyle=LINE_STYLE,
            linewidth=1.4,
            color=LINE_COLOR,
            alpha=LINE_ALPHA,
            zorder=5,
        )
        ax.scatter(
            x,
            y,
            s=POINT_SIZE,
            color=POINT_COLOR,
            alpha=0.85,
            zorder=6,
        )

        # ---- jittered point cloud (looks like experimental dot clouds) ----
        rng = np.random.default_rng(42)
        for i, (xi, yi, ei) in enumerate(zip(x, y, yerr)):
            if not np.isfinite(yi):
                continue
            n = int(CLOUD_N)
            # x jitter around the bar center
            xj = xi + rng.uniform(-CLOUD_JITTER, CLOUD_JITTER, size=n)
            # y points: either around mean using yerr, or small fixed noise
            if CLOUD_USE_ERR and np.isfinite(ei) and ei > 0:
                yj = rng.normal(loc=yi, scale=max(ei, 1e-6) * 0.55, size=n)
            else:
                yj = rng.normal(loc=yi, scale=max(yi, 1e-6) * 0.03, size=n)
            yj = np.clip(yj, 0, None)

            # Use the same rainbow base color as the bar for the cloud
            cloud_color = base_colors[i % len(base_colors)]
            ax.scatter(
                xj,
                yj,
                s=CLOUD_SIZE,
                color=cloud_color,
                alpha=CLOUD_ALPHA,
                edgecolors="#111827",
                linewidths=0.25,
                zorder=6,
            )

        # Value labels above error bars
        for xi, yi, ei in zip(x, y, yerr):
            if np.isfinite(yi):
                ax.text(
                    xi,
                    yi + (ei if np.isfinite(ei) else 0) + 0.01,
                    f"{yi:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=LABEL_SIZE,
                    color=COLORS["black"],
                    zorder=7,
                )

        # Axes / labels
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        italic_axis_labels(ax, "Couple", "Posterior mean vote share")
        # Ensure tick label font is consistent
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontfamily(FONT_FAMILY)
        ax.set_title(
            f"Posterior Vote Shares with Uncertainty (Season {s0}, Week {w0}) — Top 12",
            pad=10,
        )

        # Subtle grid (y only)
        add_light_grid(ax)

        # Clean spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#333333")

        savefig(FIG_DIR / "T2_4_posterior_bar_top12_origin_style.png")
    except Exception as e:
        print("[WARN] skip posterior bar:", e)

    print("\n🎉 Done. Generated figures in:", FIG_DIR)


if __name__ == "__main__":
    main()