# sensitivity_task1.py
"""\
Sensitivity analysis for Task 1 (Stable ABC inference, Jan 2026 patch).

What this script does
---------------------
It re-runs the Task-1 *per-week* Bayesian fan-vote inference under different
hyperparameters and summarizes how inference stability changes.

We mirror the exact inference logic used in main.py:
  - Phase 0: exact rejection ABC (match actual eliminated)
  - Phase 1: relaxed ABC (actual eliminated in bottom-2 risk set)
  - Phase 2: soft importance resampling (always returns MIN_ACCEPT)

Inputs (same folder):
  - 2026_MCM_Problem_C_Data.xlsx

Outputs (results/):
  - sensitivity_summary.xlsx
  - figures/S1_sensitivity_alpha.png
  - figures/S2_sensitivity_budget.png

Notes
-----
- Compared with the old sensitivity script (legacy infer_week_votes), this one
  is aligned with the new stable inference used by main.py.
- "MC budget" is interpreted as MAX_DRAWS (hard cap for rejection phases).
  If the run falls back to soft resampling, MAX_DRAWS has limited effect.
"""

from __future__ import annotations

from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import dirichlet

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "2026_MCM_Problem_C_Data.xlsx"
OUT_DIR = BASE_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

assert DATA_FILE.exists(), f"Missing {DATA_FILE}"


# -------------------------
# Voting rules (match main.py)
# -------------------------
class VotingRule(Enum):
    RANK = "rank"
    PERCENT = "percent"
    BOTTOM2_JUDGE_SAVE = "bottom2_judge_save"


def era_rule(season: int) -> VotingRule:
    if season <= 2:
        return VotingRule.RANK
    elif season <= 27:
        return VotingRule.PERCENT
    else:
        return VotingRule.BOTTOM2_JUDGE_SAVE


# -------------------------
# Parsers + utilities (match main.py)
# -------------------------
def parse_elim_week(result) -> int | None:
    if isinstance(result, str) and "Eliminated Week" in result:
        try:
            return int(str(result).split()[-1])
        except Exception:
            return None
    return None


def detect_max_week_from_columns(df: pd.DataFrame, hard_cap: int = 20) -> int:
    max_w = 0
    for c in df.columns:
        if isinstance(c, str) and c.lower().startswith("week") and "_judge" in c.lower() and "_score" in c.lower():
            wtag = c.lower().split("_")[0]  # week{n}
            try:
                w = int(wtag.replace("week", ""))
                max_w = max(max_w, w)
            except Exception:
                pass
    if max_w <= 0:
        max_w = 12
    return min(max_w, hard_cap)


def judge_total(row: pd.Series, week: int) -> float | None:
    """Unified judge_total: judges 1..5, positive only."""
    scores = []
    for j in range(1, 6):
        col = f"week{week}_judge{j}_score"
        if col in row and pd.notna(row[col]):
            try:
                v = float(row[col])
                if v > 0:
                    scores.append(v)
            except Exception:
                pass
    return float(np.sum(scores)) if scores else None


def rank_desc(values: np.ndarray) -> np.ndarray:
    """Rank descending: best gets rank 1. Ties averaged (deterministic)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    # average ties
    groups: dict[float, list[int]] = {}
    for i, v in enumerate(values):
        groups.setdefault(float(v), []).append(i)
    for _, idxs in groups.items():
        if len(idxs) > 1:
            ranks[idxs] = float(np.mean(ranks[idxs]))
    return ranks


def eliminate_index(judge_scores: np.ndarray, fan_shares: np.ndarray, rule: VotingRule) -> int:
    judge_scores = np.asarray(judge_scores, dtype=float)
    fan_shares = np.asarray(fan_shares, dtype=float)

    if rule == VotingRule.PERCENT:
        js = judge_scores / judge_scores.sum()
        combined = js + fan_shares
        return int(np.argmin(combined))

    if rule == VotingRule.RANK:
        combined = rank_desc(judge_scores) + rank_desc(fan_shares)
        return int(np.argmax(combined))

    # bottom2 judge-save
    combined = rank_desc(judge_scores) + rank_desc(fan_shares)
    bottom2 = np.argsort(-combined)[:2]
    loser = bottom2[np.argmin(judge_scores[bottom2])]
    return int(loser)


def bottom2_risk_set(judge_scores: np.ndarray, fan_shares: np.ndarray, rule: VotingRule) -> np.ndarray:
    judge_scores = np.asarray(judge_scores, dtype=float)
    fan_shares = np.asarray(fan_shares, dtype=float)

    if rule == VotingRule.PERCENT:
        js = judge_scores / judge_scores.sum()
        combined = js + fan_shares
        return np.argsort(combined)[:2]

    combined = rank_desc(judge_scores) + rank_desc(fan_shares)
    return np.argsort(-combined)[:2]


# -------------------------
# Stable ABC inference (parameterized)
# -------------------------

def infer_week_votes_abc_stable(
    judge_scores: np.ndarray,
    eliminated_idx: int,
    likelihood_rule: VotingRule,
    *,
    alpha: float,
    batch_size: int,
    min_accept: int,
    early_stop_accept: int,
    max_draws: int,
    soft_pool_draws: int,
    soft_temperature: float,
    rng: np.random.Generator,
) -> dict:
    """A parameterized copy of infer_week_votes_abc_stable from main.py."""
    judge_scores = np.asarray(judge_scores, dtype=float)
    n = len(judge_scores)

    target = max(int(min_accept), 1)
    early_target = max(int(early_stop_accept), target)

    # Phase 0: exact
    accepted: list[np.ndarray] = []
    draws = 0
    while draws < max_draws and len(accepted) < early_target:
        m = min(batch_size, max_draws - draws)
        batch = dirichlet.rvs([alpha] * n, size=m, random_state=rng)
        draws += m
        for v in batch:
            if eliminate_index(judge_scores, v, likelihood_rule) == int(eliminated_idx):
                accepted.append(v)
                if len(accepted) >= early_target:
                    break

    if len(accepted) >= target:
        acc = np.asarray(accepted, dtype=float)
        return {
            "samples": acc,
            "acceptance_rate": float(len(acc) / max(draws, 1)),
            "draws": int(draws),
            "fallback_level": 0,
            "status": "exact",
        }

    # Phase 1: bottom2 membership
    accepted = []
    draws = 0
    while draws < max_draws and len(accepted) < early_target:
        m = min(batch_size, max_draws - draws)
        batch = dirichlet.rvs([alpha] * n, size=m, random_state=rng)
        draws += m
        for v in batch:
            risk = bottom2_risk_set(judge_scores, v, likelihood_rule)
            if int(eliminated_idx) in set(map(int, risk)):
                accepted.append(v)
                if len(accepted) >= early_target:
                    break

    if len(accepted) >= target:
        acc = np.asarray(accepted, dtype=float)
        return {
            "samples": acc,
            "acceptance_rate": float(len(acc) / max(draws, 1)),
            "draws": int(draws),
            "fallback_level": 1,
            "status": "bottom2_membership",
        }

    # Phase 2: soft resampling (always returns MIN_ACCEPT)
    pool_draws = max(int(soft_pool_draws), target)
    pool = dirichlet.rvs([alpha] * n, size=pool_draws, random_state=rng)

    weights = np.zeros(pool_draws, dtype=float)
    elim = int(eliminated_idx)

    if likelihood_rule == VotingRule.PERCENT:
        js = judge_scores / judge_scores.sum()
        for i in range(pool_draws):
            v = pool[i]
            combined = js + v
            gap = float(combined[elim] - np.min(combined))
            weights[i] = float(np.exp(-soft_temperature * gap))
    else:
        jr = rank_desc(judge_scores)
        for i in range(pool_draws):
            v = pool[i]
            combined = jr + rank_desc(v)
            gap = float(np.max(combined) - combined[elim])
            weights[i] = float(np.exp(-soft_temperature * gap))

    if (not np.isfinite(weights).all()) or float(weights.sum()) <= 0.0:
        weights = np.ones(pool_draws, dtype=float)
    weights = weights / weights.sum()

    idxs = rng.choice(np.arange(pool_draws), size=target, replace=True, p=weights)
    acc = pool[idxs]

    return {
        "samples": acc,
        "acceptance_rate": float(target / pool_draws),
        "draws": int(pool_draws),
        "fallback_level": 2,
        "status": "soft_resample",
    }


# -------------------------
# Sensitivity runner
# -------------------------

def run_sensitivity(
    seasons_focus: list[int],
    weeks_max: int,
    alpha_list: list[float],
    budget_list: list[int],
    *,
    batch_size: int = 1000,
    min_accept: int = 500,
    early_stop_accept: int = 800,
    soft_pool_draws: int = 120000,
    soft_temperature: float = 35.0,
    seed: int = 7,
) -> pd.DataFrame:
    data = pd.read_excel(DATA_FILE)
    data["ElimWeek"] = data["results"].apply(parse_elim_week)
    data["CoupleID"] = data["celebrity_name"].astype(str) + "_" + data["ballroom_partner"].astype(str)

    rows: list[dict] = []

    rng_master = np.random.default_rng(seed)

    for season in seasons_focus:
        season_df = data[data["season"] == season].copy()
        if season_df.empty:
            continue

        for week in range(1, weeks_max + 1):
            elim_row = season_df[season_df["ElimWeek"] == week]
            if len(elim_row) != 1:
                continue

            week_df = season_df[season_df["ElimWeek"].isna() | (season_df["ElimWeek"] >= week)]

            judge_scores: list[float] = []
            couples: list[str] = []
            for _, r in week_df.iterrows():
                s = judge_total(r, week)
                if s is not None:
                    judge_scores.append(float(s))
                    couples.append(str(r["CoupleID"]))

            if len(judge_scores) < 2:
                continue

            elim_cid = str(elim_row.iloc[0]["CoupleID"])
            if elim_cid not in couples:
                continue

            eliminated_idx = int(couples.index(elim_cid))
            rule = era_rule(int(season))
            judge_scores_arr = np.asarray(judge_scores, dtype=float)

            # --- sweep alpha (fix budget at max) ---
            budget_fixed = int(max(budget_list))
            for alpha in alpha_list:
                # deterministic per (season,week,alpha) while still varying across trials
                rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                out = infer_week_votes_abc_stable(
                    judge_scores_arr,
                    eliminated_idx,
                    rule,
                    alpha=float(alpha),
                    batch_size=int(batch_size),
                    min_accept=int(min_accept),
                    early_stop_accept=int(early_stop_accept),
                    max_draws=int(budget_fixed),
                    soft_pool_draws=int(soft_pool_draws),
                    soft_temperature=float(soft_temperature),
                    rng=rng,
                )

                samples = np.asarray(out["samples"], dtype=float)
                std = samples.std(axis=0)

                rows.append(
                    {
                        "Season": int(season),
                        "Week": int(week),
                        "Rule": rule.value,
                        "Alpha": float(alpha),
                        "MAX_DRAWS": int(budget_fixed),
                        "FallbackLevel": int(out["fallback_level"]),
                        "AcceptanceRate": float(out["acceptance_rate"]),
                        "AvgVoteStd": float(np.nanmean(std)) if len(std) else np.nan,
                    }
                )

            # --- sweep budget (fix alpha at median) ---
            alpha_fixed = float(np.median(alpha_list))
            for budget in budget_list:
                rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                out = infer_week_votes_abc_stable(
                    judge_scores_arr,
                    eliminated_idx,
                    rule,
                    alpha=float(alpha_fixed),
                    batch_size=int(batch_size),
                    min_accept=int(min_accept),
                    early_stop_accept=int(early_stop_accept),
                    max_draws=int(budget),
                    soft_pool_draws=int(soft_pool_draws),
                    soft_temperature=float(soft_temperature),
                    rng=rng,
                )

                samples = np.asarray(out["samples"], dtype=float)
                std = samples.std(axis=0)

                rows.append(
                    {
                        "Season": int(season),
                        "Week": int(week),
                        "Rule": rule.value,
                        "Alpha": float(alpha_fixed),
                        "MAX_DRAWS": int(budget),
                        "FallbackLevel": int(out["fallback_level"]),
                        "AcceptanceRate": float(out["acceptance_rate"]),
                        "AvgVoteStd": float(np.nanmean(std)) if len(std) else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def plot_sensitivity(df: pd.DataFrame):
    # 1) alpha sensitivity (aggregate)
    alpha_df = (
        df.groupby(["Alpha"], as_index=False)
        .agg(
            MeanAcceptance=("AcceptanceRate", "mean"),
            MeanStd=("AvgVoteStd", "mean"),
            MeanFallback=("FallbackLevel", "mean"),
        )
        .sort_values("Alpha")
    )

    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(alpha_df["Alpha"], alpha_df["MeanAcceptance"], marker="o", linewidth=2, label="Mean Acceptance")
    ax.set_xlabel("DIRICHLET_ALPHA")
    ax.set_ylabel("Mean Acceptance Rate")
    ax.grid(True, alpha=0.25)
    ax.set_title("Sensitivity to Prior Concentration (Alpha) — Stable ABC")

    ax2 = ax.twinx()
    ax2.plot(alpha_df["Alpha"], alpha_df["MeanFallback"], marker="s", linestyle="--", linewidth=1.6, label="Mean Fallback", alpha=0.85)
    ax2.set_ylabel("Mean fallback level (0=exact, 2=soft)")

    # combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "S1_sensitivity_alpha.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # 2) budget sensitivity (aggregate)
    b_df = (
        df.groupby(["MAX_DRAWS"], as_index=False)
        .agg(
            MeanAcceptance=("AcceptanceRate", "mean"),
            MeanStd=("AvgVoteStd", "mean"),
            MeanFallback=("FallbackLevel", "mean"),
        )
        .sort_values("MAX_DRAWS")
    )

    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(b_df["MAX_DRAWS"], b_df["MeanAcceptance"], marker="o", linewidth=2, label="Mean Acceptance")
    ax.set_xscale("log")
    ax.set_xlabel("MAX_DRAWS (log scale)")
    ax.set_ylabel("Mean Acceptance Rate")
    ax.grid(True, alpha=0.25)
    ax.set_title("Sensitivity to Rejection-ABC Budget (MAX_DRAWS) — Stable ABC")

    ax2 = ax.twinx()
    ax2.plot(b_df["MAX_DRAWS"], b_df["MeanFallback"], marker="s", linestyle="--", linewidth=1.6, label="Mean Fallback", alpha=0.85)
    ax2.set_ylabel("Mean fallback level (0=exact, 2=soft)")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "S2_sensitivity_budget.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    # representative seasons across eras
    seasons_focus = [2, 11, 27, 31]

    # use detected max week, but keep a modest cap for runtime
    data = pd.read_excel(DATA_FILE)
    weeks_max = min(detect_max_week_from_columns(data, hard_cap=20), 11)

    # sensitivity grids
    alpha_list = [0.1, 0.3, 0.6, 1.0]
    # interpret as the rejection-ABC draw cap for phases 0/1
    budget_list = [20000, 50000, 100000, 250000]

    df = run_sensitivity(
        seasons_focus,
        weeks_max,
        alpha_list,
        budget_list,
        batch_size=1000,
        min_accept=500,
        early_stop_accept=800,
        soft_pool_draws=120000,
        soft_temperature=35.0,
        seed=7,
    )

    out_xlsx = OUT_DIR / "sensitivity_summary.xlsx"
    df.to_excel(out_xlsx, index=False)
    print("saved:", out_xlsx)

    plot_sensitivity(df)
    print("saved figures into:", FIG_DIR)


if __name__ == "__main__":
    main()