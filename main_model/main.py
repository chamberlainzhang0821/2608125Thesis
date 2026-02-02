"""
MCM Problem C
Task 1–3: Unified Bayesian Fan Vote Inference + Voting Rule Comparison
XLSX-only, single-input-file implementation

Input (same folder):
  - 2026_MCM_Problem_C_Data.xlsx

Outputs (results/):
  - fan_vote_estimates.xlsx
  - weekly_uncertainty.xlsx
  - method_outcomes_by_week.xlsx
  - method_comparison_summary.xlsx
  - contestant_benefit_analysis.xlsx

Additional output (results/):
  - task1_inference_log.xlsx   (per season-week diagnostics)

Notes (Jan 2026 patch):
  - Stabilized Task-1 ABC inference:
      * min accepted samples
      * early stop
      * max draw cap
      * fallback mechanisms (exact -> bottom2-membership -> soft importance resampling)
"""

from __future__ import annotations

from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from scipy.stats import dirichlet


# ======================================================
# Paths & Parameters (EDIT ONLY IF NEEDED)
# ======================================================

BASE_DIR = Path(__file__).resolve().parent
MAIN_DATA_FILE = BASE_DIR / "2026_MCM_Problem_C_Data.xlsx"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------------------
# Task-1 Stable ABC settings
# -------------------------------
# We will *try* to collect at least MIN_ACCEPT accepted samples.
# We stop early once we reach EARLY_STOP_ACCEPT (>= MIN_ACCEPT by default).
# If exact matching is too hard, we progressively relax:
#   fallback_level=0: exact elimination match
#   fallback_level=1: actual eliminated appears in bottom-2 risk set
#   fallback_level=2: soft importance resampling (always returns MIN_ACCEPT samples)

MC_SAMPLES = 20000          # legacy: used as a target scale for reporting
BATCH_SIZE = 1000
DIRICHLET_ALPHA = 0.3

MIN_ACCEPT = 500            # minimum accepted samples to stabilize mean/std
EARLY_STOP_ACCEPT = 800     # optional: stop early once this many accepts are collected
MAX_DRAWS = 250000          # hard cap on number of prior draws for exact/bottom2 rejection

# Soft fallback parameters
SOFT_POOL_DRAWS = 120000    # pool size for soft importance resampling
SOFT_TEMPERATURE = 35.0     # larger => sharper preference for matching actual elimination

# If your data has more weeks, this is auto-expanded by scanning columns,
# but we keep a safe upper bound to avoid accidental huge loops.
MAX_WEEKS_HARD_CAP = 20


# ======================================================
# Voting Rules (Methods)
# ======================================================

class VotingRule(Enum):
    RANK = "rank"
    PERCENT = "percent"
    BOTTOM2_JUDGE_SAVE = "bottom2_judge_save"


def era_rule(season: int) -> VotingRule:
    """Piecewise likelihood (historical rule by season)."""
    if season <= 2:
        return VotingRule.RANK
    elif season <= 27:
        return VotingRule.PERCENT
    else:
        return VotingRule.BOTTOM2_JUDGE_SAVE


# ======================================================
# Data Loading (XLSX ONLY)
# ======================================================

def load_main_data() -> pd.DataFrame:
    assert MAIN_DATA_FILE.exists(), f"Missing: {MAIN_DATA_FILE}"
    df = pd.read_excel(MAIN_DATA_FILE)
    return df


# ======================================================
# Utilities
# ======================================================

def parse_elim_week(result) -> int | None:
    if isinstance(result, str) and "Eliminated Week" in result:
        try:
            return int(result.split()[-1])
        except Exception:
            return None
    return None


def detect_max_week_from_columns(df: pd.DataFrame) -> int:
    """
    Detect max week number from columns like week11_judge3_score.
    Falls back to MAX_WEEKS_HARD_CAP if not found.
    """
    max_w = 0
    for c in df.columns:
        if isinstance(c, str) and c.lower().startswith("week") and "_judge" in c.lower() and "_score" in c.lower():
            # extract leading "week{n}"
            s = c.lower().split("_")[0]  # week{n}
            try:
                w = int(s.replace("week", ""))
                max_w = max(max_w, w)
            except Exception:
                pass

    if max_w <= 0:
        max_w = 12  # sensible default
    return min(max_w, MAX_WEEKS_HARD_CAP)


def judge_total(row: pd.Series, week: int) -> float | None:
    """Sum available judges' scores for this week. (Unified: judges 1..5, positive only.)"""
    scores = []
    for j in range(1, 6):  # allow up to 5 judges just in case
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
    """Rank in descending order: best gets rank 1. Ties averaged (deterministic)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)

    # tie smoothing (average ranks for equal values)
    # (Optional: rounding could be used; kept as-is for backward compatibility)
    unique_vals = {}
    for idx, val in enumerate(values):
        unique_vals.setdefault(val, []).append(idx)
    for val, idxs in unique_vals.items():
        if len(idxs) > 1:
            avg = float(np.mean(ranks[idxs]))
            ranks[idxs] = avg
    return ranks


# ======================================================
# Rule Mechanics (Given judge scores + fan vote shares)
# ======================================================

def eliminate_index(judge_scores: np.ndarray, fan_shares: np.ndarray, rule: VotingRule) -> int:
    """
    Returns the index of the eliminated contestant among active couples this week.
    Notes:
      - For RANK: lower is better; we eliminate worst (max rank sum)
      - For PERCENT: higher is better; we eliminate lowest combined percent-like score
      - For BOTTOM2_JUDGE_SAVE: find bottom2 by rank-sum then eliminate the one with lower judge score
    """
    judge_scores = np.asarray(judge_scores, dtype=float)
    fan_shares = np.asarray(fan_shares, dtype=float)

    if rule == VotingRule.PERCENT:
        js = judge_scores / judge_scores.sum()
        combined = js + fan_shares
        return int(np.argmin(combined))

    if rule == VotingRule.RANK:
        combined = rank_desc(judge_scores) + rank_desc(fan_shares)
        return int(np.argmax(combined))

    # bottom-2 judge save (approx)
    combined = rank_desc(judge_scores) + rank_desc(fan_shares)
    # worst = largest combined rank sum
    bottom2 = np.argsort(-combined)[:2]
    # judges save the better judge-score among bottom2 => eliminate lower judge-score
    loser = bottom2[np.argmin(judge_scores[bottom2])]
    return int(loser)


def bottom2_risk_set(judge_scores: np.ndarray, fan_shares: np.ndarray, rule: VotingRule) -> np.ndarray:
    """Return indices of the bottom-2 'risk set' contestants under a rule."""
    judge_scores = np.asarray(judge_scores, dtype=float)
    fan_shares = np.asarray(fan_shares, dtype=float)

    if rule == VotingRule.PERCENT:
        js = judge_scores / judge_scores.sum()
        combined = js + fan_shares
        return np.argsort(combined)[:2]  # two smallest combined

    # For both RANK and BOTTOM2_JUDGE_SAVE, risk-set is bottom-2 by rank-sum
    combined = rank_desc(judge_scores) + rank_desc(fan_shares)
    return np.argsort(-combined)[:2]  # two largest rank-sum (worst)


# ======================================================
# Stable Bayesian / ABC-Rejection Inference (Task 1 Core)
# ======================================================

def _weighted_mean_std(samples: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean/std along axis 0; weights must be nonnegative and sum>0."""
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    sw = float(w.sum())
    if sw <= 0:
        # fallback to unweighted
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        return mean, std

    w = w / sw
    mean = (samples * w[:, None]).sum(axis=0)
    var = (w[:, None] * (samples - mean) ** 2).sum(axis=0)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std


def infer_week_votes_abc_stable(
    judge_scores: np.ndarray,
    eliminated_idx: int,
    likelihood_rule: VotingRule,
) -> dict:
    """
    Stable ABC inference.

    Returns dict with keys:
      - samples: (k x n) accepted or resampled fan share samples
      - acceptance_rate: accepted / draws (for the phase used)
      - draws: total draws used in the phase used
      - fallback_level: 0/1/2
      - status: string

    Behavior:
      1) Exact rejection ABC until MIN_ACCEPT (early stop at EARLY_STOP_ACCEPT), capped by MAX_DRAWS.
      2) If insufficient, relax to accepting samples where actual eliminated is in bottom2 risk set.
      3) If still insufficient, do soft importance resampling to ALWAYS output MIN_ACCEPT samples.
    """
    judge_scores = np.asarray(judge_scores, dtype=float)
    n = len(judge_scores)

    # -----------------
    # Phase 0: exact
    # -----------------
    accepted: list[np.ndarray] = []
    draws = 0
    target = max(MIN_ACCEPT, 1)
    early_target = max(EARLY_STOP_ACCEPT, target)

    while draws < MAX_DRAWS and len(accepted) < early_target:
        m = min(BATCH_SIZE, MAX_DRAWS - draws)
        batch = dirichlet.rvs([DIRICHLET_ALPHA] * n, size=m)
        draws += m
        for v in batch:
            if eliminate_index(judge_scores, v, likelihood_rule) == eliminated_idx:
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

    # -----------------
    # Phase 1: bottom-2 membership (risk set)
    # -----------------
    accepted = []
    draws = 0

    while draws < MAX_DRAWS and len(accepted) < early_target:
        m = min(BATCH_SIZE, MAX_DRAWS - draws)
        batch = dirichlet.rvs([DIRICHLET_ALPHA] * n, size=m)
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

    # -----------------
    # Phase 2: soft importance resampling (always returns MIN_ACCEPT)
    # -----------------
    pool_draws = max(SOFT_POOL_DRAWS, target)
    pool = dirichlet.rvs([DIRICHLET_ALPHA] * n, size=pool_draws)

    # Define a nonnegative weight that favors samples that eliminate the actual eliminated.
    # For stability across rules, we use a score gap: smaller gap => larger weight.
    #
    # Percent:
    #   combined = judge_share + fan_share; eliminated = argmin(combined)
    #   gap = combined[elim] - min(combined)  (0 if it is the min)
    # Rank & Bottom2:
    #   combined_rank_sum = rank_desc(judge) + rank_desc(fan)
    #   eliminated = argmax(combined_rank_sum)
    #   gap = max(combined) - combined[elim]  (0 if it is the max)

    weights = np.zeros(pool_draws, dtype=float)

    if likelihood_rule == VotingRule.PERCENT:
        js = judge_scores / judge_scores.sum()
        for i in range(pool_draws):
            v = pool[i]
            combined = js + v
            gap = float(combined[int(eliminated_idx)] - np.min(combined))
            weights[i] = np.exp(-SOFT_TEMPERATURE * gap)
    else:
        jr = rank_desc(judge_scores)
        for i in range(pool_draws):
            v = pool[i]
            combined = jr + rank_desc(v)
            gap = float(np.max(combined) - combined[int(eliminated_idx)])
            weights[i] = np.exp(-SOFT_TEMPERATURE * gap)

    # If weights collapse to ~0 (numerical), fall back to uniform
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(pool_draws, dtype=float)

    weights = weights / weights.sum()

    # Resample MIN_ACCEPT samples according to weights
    idxs = np.random.choice(np.arange(pool_draws), size=target, replace=True, p=weights)
    acc = pool[idxs]

    return {
        "samples": acc,
        "acceptance_rate": float(target / pool_draws),
        "draws": int(pool_draws),
        "fallback_level": 2,
        "status": "soft_resample",
    }


# ======================================================
# Main Pipeline
# ======================================================

def run():
    data = load_main_data()

    # --- preprocess ---
    data["ElimWeek"] = data["results"].apply(parse_elim_week)
    data["CoupleID"] = data["celebrity_name"].astype(str) + "_" + data["ballroom_partner"].astype(str)

    max_week = detect_max_week_from_columns(data)

    # We treat "active in week w" as: ElimWeek is NaN (never eliminated => winner/finalist)
    # or ElimWeek >= w
    # Actual eliminated of week w is the unique row with ElimWeek == w (if it exists).
    # Some seasons may have weeks without elimination -> we skip those weeks.
    inference_rows = []
    log_rows = []

    # For layer-2: store per season-week active roster + judge totals + actual eliminated
    week_context = {}

    seasons = sorted([int(x) for x in data["season"].dropna().unique().tolist()])

    for season in seasons:
        season_df = data[data["season"] == season].copy()

        for week in range(1, max_week + 1):
            active_df = season_df[season_df["ElimWeek"].isna() | (season_df["ElimWeek"] >= week)].copy()
            elim_row = season_df[season_df["ElimWeek"] == week].copy()

            if len(elim_row) != 1:
                continue

            judge_scores = []
            couples = []

            for _, r in active_df.iterrows():
                s = judge_total(r, week)
                if s is not None:
                    judge_scores.append(s)
                    couples.append(r["CoupleID"])

            if len(judge_scores) < 2:
                continue

            # actual eliminated must be among active couples
            elim_couple = str(elim_row.iloc[0]["CoupleID"])
            if elim_couple not in couples:
                continue

            eliminated_idx = couples.index(elim_couple)
            like_rule = era_rule(season)

            # store context for layer-2 simulation later
            week_context[(season, week)] = {
                "season": season,
                "week": week,
                "couples": couples,
                "judge_scores": np.array(judge_scores, dtype=float),
                "actual_elim_couple": elim_couple,
                "actual_elim_idx": eliminated_idx,
                "likelihood_rule": like_rule,
            }

            # ----- layer-1 stable inference -----
            out = infer_week_votes_abc_stable(
                judge_scores=np.array(judge_scores, dtype=float),
                eliminated_idx=eliminated_idx,
                likelihood_rule=like_rule,
            )

            acc_samples = out["samples"]
            acc_rate = float(out["acceptance_rate"])
            draws_used = int(out["draws"])
            fallback_level = int(out["fallback_level"])
            status = str(out["status"])

            # compute mean/std
            mean = acc_samples.mean(axis=0)
            std = acc_samples.std(axis=0)

            # log week-level diagnostics
            log_rows.append(
                {
                    "Season": season,
                    "Week": week,
                    "LikelihoodRule": like_rule.value,
                    "NumCouples": int(len(couples)),
                    "AcceptedSamples": int(acc_samples.shape[0]),
                    "DrawsUsed": int(draws_used),
                    "AcceptanceRate": float(acc_rate),
                    "FallbackLevel": fallback_level,
                    "Status": status,
                }
            )

            for i, cid in enumerate(couples):
                inference_rows.append(
                    {
                        "Season": season,
                        "Week": week,
                        "CoupleID": cid,
                        "LikelihoodRule": like_rule.value,
                        "VoteMean": float(mean[i]),
                        "VoteStd": float(std[i]),
                        "AcceptedSamples": int(acc_samples.shape[0]),
                        "AcceptanceRate": float(acc_rate),
                        # extra diagnostics (should not break downstream code)
                        "DrawsUsed": int(draws_used),
                        "FallbackLevel": int(fallback_level),
                        "InferenceStatus": status,
                    }
                )

    # =============================
    # Save layer-1 inference outputs
    # =============================
    fan_df = pd.DataFrame(inference_rows)
    fan_out = OUTPUT_DIR / "fan_vote_estimates.xlsx"
    fan_df.to_excel(fan_out, index=False)

    # Weekly uncertainty summary (same as before, plus optional extra cols if desired)
    if len(fan_df) > 0:
        weekly_df = (
            fan_df.groupby(["Season", "Week", "LikelihoodRule"], as_index=False)
            .agg(
                AvgVoteUncertainty=("VoteStd", "mean"),
                AvgAcceptanceRate=("AcceptanceRate", "mean"),
                NumCouples=("CoupleID", "count"),
                FallbackLevel=("FallbackLevel", "max"),
            )
        )
    else:
        weekly_df = pd.DataFrame()

    weekly_out = OUTPUT_DIR / "weekly_uncertainty.xlsx"
    weekly_df.to_excel(weekly_out, index=False)

    # Save inference log
    log_df = pd.DataFrame(log_rows)
    log_out = OUTPUT_DIR / "task1_inference_log.xlsx"
    log_df.to_excel(log_out, index=False)

    # ======================================================
    # Layer-2: Apply ALL methods to ALL seasons using inferred votes
    # ======================================================
    # Use posterior mean VoteMean as our point estimate of fan shares each week.
    # For each season-week, compute predicted eliminated under each method and compare.
    method_rows = []

    # Build fast lookup: (season, week) -> {couple -> vote_mean}
    vote_mean_lookup = {}
    for (season, week), g in fan_df.groupby(["Season", "Week"]):
        vote_mean_lookup[(int(season), int(week))] = dict(zip(g["CoupleID"], g["VoteMean"]))

    all_methods = [VotingRule.RANK, VotingRule.PERCENT, VotingRule.BOTTOM2_JUDGE_SAVE]

    for _, ctx in week_context.items():
        season = ctx["season"]
        week = ctx["week"]
        couples = ctx["couples"]
        js = ctx["judge_scores"]
        actual = ctx["actual_elim_couple"]

        vm = vote_mean_lookup.get((season, week), {})
        # ensure order matches couples
        fan_shares = np.array([float(vm.get(cid, np.nan)) for cid in couples], dtype=float)
        if np.any(np.isnan(fan_shares)):
            # if some missing, skip this week
            continue

        # normalize fan shares (small numerical drift)
        s = fan_shares.sum()
        if s > 0:
            fan_shares = fan_shares / s

        preds = {}
        for method in all_methods:
            idx = eliminate_index(js, fan_shares, method)
            preds[method.value] = couples[idx]

        method_rows.append(
            {
                "Season": season,
                "Week": week,
                "ActualEliminated": actual,
                "Pred_Rank": preds["rank"],
                "Pred_Percent": preds["percent"],
                "Pred_Bottom2": preds["bottom2_judge_save"],
                "Match_Rank": int(preds["rank"] == actual),
                "Match_Percent": int(preds["percent"] == actual),
                "Match_Bottom2": int(preds["bottom2_judge_save"] == actual),
            }
        )

    method_df = pd.DataFrame(method_rows)
    method_out = OUTPUT_DIR / "method_outcomes_by_week.xlsx"
    method_df.to_excel(method_out, index=False)

    # ======================================================
    # Task-2 summary: flip rates + consistency by season
    # ======================================================
    def flip(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a != b).astype(int)

    if len(method_df) > 0:
        method_df["Flip_Rank_vs_Percent"] = flip(method_df["Pred_Rank"], method_df["Pred_Percent"])
        method_df["Flip_Rank_vs_Bottom2"] = flip(method_df["Pred_Rank"], method_df["Pred_Bottom2"])
        method_df["Flip_Percent_vs_Bottom2"] = flip(method_df["Pred_Percent"], method_df["Pred_Bottom2"])

        summary_season = (
            method_df.groupby("Season", as_index=False)
            .agg(
                Weeks=("Week", "count"),
                Consistency_Rank=("Match_Rank", "mean"),
                Consistency_Percent=("Match_Percent", "mean"),
                Consistency_Bottom2=("Match_Bottom2", "mean"),
                Flip_Rank_vs_Percent=("Flip_Rank_vs_Percent", "mean"),
                Flip_Rank_vs_Bottom2=("Flip_Rank_vs_Bottom2", "mean"),
                Flip_Percent_vs_Bottom2=("Flip_Percent_vs_Bottom2", "mean"),
            )
        )

        summary_overall = pd.DataFrame(
            [
                {
                    "Season": "ALL",
                    "Weeks": int(method_df["Week"].count()),
                    "Consistency_Rank": float(method_df["Match_Rank"].mean()),
                    "Consistency_Percent": float(method_df["Match_Percent"].mean()),
                    "Consistency_Bottom2": float(method_df["Match_Bottom2"].mean()),
                    "Flip_Rank_vs_Percent": float(method_df["Flip_Rank_vs_Percent"].mean()),
                    "Flip_Rank_vs_Bottom2": float(method_df["Flip_Rank_vs_Bottom2"].mean()),
                    "Flip_Percent_vs_Bottom2": float(method_df["Flip_Percent_vs_Bottom2"].mean()),
                }
            ]
        )

        summary_df = pd.concat([summary_season, summary_overall], ignore_index=True)
    else:
        summary_df = pd.DataFrame()

    summary_out = OUTPUT_DIR / "method_comparison_summary.xlsx"
    summary_df.to_excel(summary_out, index=False)

    # ======================================================
    # "Who benefits / who suffers": predicted elimination week under each method
    # ======================================================
    # Build predicted eliminated-per-week sequences, then infer each couple's predicted elim week.
    # If never eliminated in observed weeks -> treat as max_week+1 (survives longer).
    benefit_rows = []

    if len(method_df) > 0:
        for season in method_df["Season"].unique():
            sdf = method_df[method_df["Season"] == season].copy()
            sdf = sdf.sort_values("Week")

            # collect all couples that appear in this season’s inference
            couples_in_season = sorted(set(fan_df[fan_df["Season"] == season]["CoupleID"].unique().tolist()))

            pred_elim_week = {m.value: {} for m in all_methods}
            for cid in couples_in_season:
                for m in all_methods:
                    pred_elim_week[m.value][cid] = max_week + 1  # default: survives

            # fill predicted elimination week for each method (first time eliminated)
            for _, r in sdf.iterrows():
                w = int(r["Week"])
                for m in all_methods:
                    col = {"rank": "Pred_Rank", "percent": "Pred_Percent", "bottom2_judge_save": "Pred_Bottom2"}[m.value]
                    elim_cid = str(r[col])
                    if pred_elim_week[m.value].get(elim_cid, max_week + 1) == max_week + 1:
                        pred_elim_week[m.value][elim_cid] = w

            # compare methods: positive delta => eliminated later => benefited
            for cid in couples_in_season:
                e_rank = pred_elim_week["rank"][cid]
                e_pct = pred_elim_week["percent"][cid]
                e_b2 = pred_elim_week["bottom2_judge_save"][cid]

                benefit_rows.append(
                    {
                        "Season": int(season),
                        "CoupleID": cid,
                        "PredElimWeek_Rank": e_rank,
                        "PredElimWeek_Percent": e_pct,
                        "PredElimWeek_Bottom2": e_b2,
                        "Benefit_Percent_minus_Rank": e_pct - e_rank,
                        "Benefit_Bottom2_minus_Rank": e_b2 - e_rank,
                        "Benefit_Bottom2_minus_Percent": e_b2 - e_pct,
                    }
                )

    benefit_df = pd.DataFrame(benefit_rows)
    benefit_out = OUTPUT_DIR / "contestant_benefit_analysis.xlsx"
    benefit_df.to_excel(benefit_out, index=False)

    print("✅ Completed:")
    print(f"  - {fan_out.name}")
    print(f"  - {weekly_out.name}")
    print(f"  - {method_out.name}")
    print(f"  - {summary_out.name}")
    print(f"  - {benefit_out.name}")
    print(f"  - {log_out.name}")


if __name__ == "__main__":
    run()