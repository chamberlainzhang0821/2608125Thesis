"""
Task 2 Post-processing (Two XLSX outputs)

Output #1: results/task2_week_consistency_diff.xlsx
  - week_level: each season-week, matches for Rank/Percent/Bottom2 + difference flags
  - season_level: per-season averages + difference magnitudes
  - overall: overall averages

Output #2: results/task2_candidate_consistency_diff.xlsx
  - candidate_level: each contestant, actual elim week + predicted elim week under 3 methods
                    + per-method consistency + method differences + error to actual
  - top_diff: top-K contestants with largest method disagreement (and/or biggest error)
  - given_4_cases: the 4 named controversy cases in prompt (sanity check)

Inputs (from results/):
  - method_outcomes_by_week.xlsx
  - method_comparison_summary.xlsx (optional; we recompute anyway)
  - contestant_benefit_analysis.xlsx
  - fan_vote_estimates.xlsx (optional; used for mapping)
Optional (recommended, in same folder as this script):
  - 2026_MCM_Problem_C_Data.xlsx (for actual elimination week + celebrity names)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------
# Paths
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
RES = BASE_DIR / "results"

MW_XLSX = RES / "method_outcomes_by_week.xlsx"
BENEFIT_XLSX = RES / "contestant_benefit_analysis.xlsx"
FAN_XLSX = RES / "fan_vote_estimates.xlsx"  # optional

MAIN_DATA_XLSX = BASE_DIR / "2026_MCM_Problem_C_Data.xlsx"  # strongly recommended

OUT_WEEK = RES / "task2_week_consistency_diff.xlsx"
OUT_CAND = RES / "task2_candidate_consistency_diff.xlsx"


# ---------------------------
# Roster alignment (fair universe)
# ---------------------------
def build_full_roster_by_season(df_main: pd.DataFrame) -> dict[int, list[str]]:
    """
    Universe roster per season from MAIN DATA (full season roster).
    Stable order: first appearance in the main file.
    """
    df = df_main.copy()

    if "CoupleID" not in df.columns:
        df["CoupleID"] = df["celebrity_name"].astype(str) + "_" + df["ballroom_partner"].astype(str)

    rosters: dict[int, list[str]] = {}
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())

    for s in seasons:
        sub = df[df["season"].astype(int) == int(s)]
        seen = set()
        ordered: list[str] = []
        for cid in sub["CoupleID"].astype(str).tolist():
            cid = str(cid).strip()
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
        rosters[int(s)] = ordered

    return rosters


def align_share_to_roster(share: pd.Series, roster: list[str]) -> pd.Series:
    """
    Reindex to full roster; missing = 0; renormalize if sum > 0.
    Useful for any probability/share vectors in fairness metrics.
    """
    s = share.reindex(roster).fillna(0.0).astype(float)
    tot = float(s.sum())
    if tot > 0:
        s = s / tot
    return s


def align_rank_to_roster(rank: pd.Series, roster: list[str]) -> pd.Series:
    """
    Reindex to full roster; missing = worst rank (=len(roster)).
    Useful for any rank-based fairness metrics.
    """
    r = rank.reindex(roster).astype(float)
    r = r.fillna(float(len(roster)))
    return r


# ---------------------------
# Helpers
# ---------------------------
def must_exist(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")


def parse_elim_week(result) -> int | None:
    if isinstance(result, str) and "Eliminated Week" in result:
        try:
            return int(result.split()[-1])
        except Exception:
            return None
    return None


def detect_max_week_from_columns(df: pd.DataFrame, hard_cap: int = 30) -> int:
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


def safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def norm_couple_id(cid: str) -> str:
    return str(cid).strip()


# ---------------------------
# Output #1: week-level / season-level consistency differences
# ---------------------------
def build_week_consistency_outputs(mw: pd.DataFrame):
    df = mw.copy()

    # Ensure columns exist
    required_cols = [
        "Season", "Week", "ActualEliminated",
        "Pred_Rank", "Pred_Percent", "Pred_Bottom2",
        "Match_Rank", "Match_Percent", "Match_Bottom2",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"method_outcomes_by_week missing column: {c}")

    # Flip / difference flags
    df["Flip_Rank_vs_Percent"] = (df["Pred_Rank"] != df["Pred_Percent"]).astype(int)
    df["Flip_Rank_vs_Bottom2"] = (df["Pred_Rank"] != df["Pred_Bottom2"]).astype(int)
    df["Flip_Percent_vs_Bottom2"] = (df["Pred_Percent"] != df["Pred_Bottom2"]).astype(int)

    # Signed week-level match advantage
    df["DiffCons_Rank_minus_Percent"] = df["Match_Rank"].astype(int) - df["Match_Percent"].astype(int)
    df["DiffCons_Rank_minus_Bottom2"] = df["Match_Rank"].astype(int) - df["Match_Bottom2"].astype(int)
    df["DiffCons_Percent_minus_Bottom2"] = df["Match_Percent"].astype(int) - df["Match_Bottom2"].astype(int)

    week_level = df[[
        "Season", "Week",
        "ActualEliminated",
        "Pred_Rank", "Pred_Percent", "Pred_Bottom2",
        "Match_Rank", "Match_Percent", "Match_Bottom2",
        "Flip_Rank_vs_Percent", "Flip_Rank_vs_Bottom2", "Flip_Percent_vs_Bottom2",
        "DiffCons_Rank_minus_Percent", "DiffCons_Rank_minus_Bottom2", "DiffCons_Percent_minus_Bottom2",
    ]].copy()

    season_level = (
        df.groupby("Season", as_index=False)
        .agg(
            Weeks=("Week", "count"),
            Consistency_Rank=("Match_Rank", "mean"),
            Consistency_Percent=("Match_Percent", "mean"),
            Consistency_Bottom2=("Match_Bottom2", "mean"),
            Flip_Rank_vs_Percent=("Flip_Rank_vs_Percent", "mean"),
            Flip_Rank_vs_Bottom2=("Flip_Rank_vs_Bottom2", "mean"),
            Flip_Percent_vs_Bottom2=("Flip_Percent_vs_Bottom2", "mean"),
            WinWeeks_Rank_over_Percent=("DiffCons_Rank_minus_Percent", lambda x: int((x > 0).sum())),
            WinWeeks_Percent_over_Rank=("DiffCons_Rank_minus_Percent", lambda x: int((x < 0).sum())),
            WinWeeks_Rank_over_Bottom2=("DiffCons_Rank_minus_Bottom2", lambda x: int((x > 0).sum())),
            WinWeeks_Bottom2_over_Rank=("DiffCons_Rank_minus_Bottom2", lambda x: int((x < 0).sum())),
            WinWeeks_Percent_over_Bottom2=("DiffCons_Percent_minus_Bottom2", lambda x: int((x > 0).sum())),
            WinWeeks_Bottom2_over_Percent=("DiffCons_Percent_minus_Bottom2", lambda x: int((x < 0).sum())),
        )
    )

    overall = pd.DataFrame([{
        "Weeks_Total": int(len(df)),
        "Consistency_Rank": float(df["Match_Rank"].mean()) if len(df) else np.nan,
        "Consistency_Percent": float(df["Match_Percent"].mean()) if len(df) else np.nan,
        "Consistency_Bottom2": float(df["Match_Bottom2"].mean()) if len(df) else np.nan,
        "Flip_Rank_vs_Percent": float(df["Flip_Rank_vs_Percent"].mean()) if len(df) else np.nan,
        "Flip_Rank_vs_Bottom2": float(df["Flip_Rank_vs_Bottom2"].mean()) if len(df) else np.nan,
        "Flip_Percent_vs_Bottom2": float(df["Flip_Percent_vs_Bottom2"].mean()) if len(df) else np.nan,
        "WinWeeks_Rank_over_Percent": int((df["DiffCons_Rank_minus_Percent"] > 0).sum()),
        "WinWeeks_Percent_over_Rank": int((df["DiffCons_Rank_minus_Percent"] < 0).sum()),
        "WinWeeks_Rank_over_Bottom2": int((df["DiffCons_Rank_minus_Bottom2"] > 0).sum()),
        "WinWeeks_Bottom2_over_Rank": int((df["DiffCons_Rank_minus_Bottom2"] < 0).sum()),
        "WinWeeks_Percent_over_Bottom2": int((df["DiffCons_Percent_minus_Bottom2"] > 0).sum()),
        "WinWeeks_Bottom2_over_Percent": int((df["DiffCons_Percent_minus_Bottom2"] < 0).sum()),
    }])

    return week_level, season_level, overall


# ---------------------------
# Output #2: contestant-level consistency differences
# ---------------------------
def build_candidate_outputs(benefit: pd.DataFrame, main_data: pd.DataFrame | None):
    b = benefit.copy()

    req = [
        "Season", "CoupleID",
        "PredElimWeek_Rank", "PredElimWeek_Percent", "PredElimWeek_Bottom2",
    ]
    for c in req:
        if c not in b.columns:
            raise KeyError(f"contestant_benefit_analysis missing column: {c}")

    b["CoupleID"] = b["CoupleID"].astype(str).map(norm_couple_id)
    b["Season"] = b["Season"].apply(safe_int)

    # Attach actual elimination week + names (if main data provided)
    if main_data is not None:
        md = main_data.copy()

        # build roster universe per season (fair, full participants)
        full_roster_by_season = build_full_roster_by_season(md)

        md["ElimWeek"] = md["results"].apply(parse_elim_week)
        md["CoupleID"] = md["celebrity_name"].astype(str) + "_" + md["ballroom_partner"].astype(str)
        md["CoupleID"] = md["CoupleID"].astype(str).map(norm_couple_id)
        md["Season"] = md["season"].apply(safe_int)

        max_week = detect_max_week_from_columns(md, hard_cap=30)
        md["ActualElimWeek"] = md["ElimWeek"].fillna(max_week + 1).astype(int)

        map_cols = ["Season", "CoupleID", "celebrity_name", "ballroom_partner", "ActualElimWeek"]
        md_map = md[map_cols].drop_duplicates(subset=["Season", "CoupleID"])

        # merge names + actual
        b = b.merge(md_map, how="left", on=["Season", "CoupleID"])
        b.rename(columns={"celebrity_name": "Celebrity", "ballroom_partner": "ProDancer"}, inplace=True)

        # --- CRITICAL FIX ---
        # Ensure candidate sheet covers FULL roster per season (not only those in benefit file)
        # Add missing roster members with NaNs, so fairness comparisons are universe-consistent.
        extra_rows = []
        for season, roster in full_roster_by_season.items():
            roster_set = set(roster)
            have = set(b.loc[b["Season"] == season, "CoupleID"].astype(str).tolist())
            missing = sorted(list(roster_set - have))
            if not missing:
                continue

            # pull name+actual from md_map
            sub = md_map[md_map["Season"] == season].set_index("CoupleID")
            for cid in missing:
                row = {
                    "Season": season,
                    "CoupleID": cid,
                    "PredElimWeek_Rank": np.nan,
                    "PredElimWeek_Percent": np.nan,
                    "PredElimWeek_Bottom2": np.nan,
                    "Celebrity": sub.loc[cid, "celebrity_name"] if cid in sub.index else cid.split("_")[0],
                    "ProDancer": sub.loc[cid, "ballroom_partner"] if cid in sub.index else "_".join(cid.split("_")[1:]),
                    "ActualElimWeek": int(sub.loc[cid, "ActualElimWeek"]) if cid in sub.index else np.nan,
                }
                extra_rows.append(row)

        if extra_rows:
            b = pd.concat([b, pd.DataFrame(extra_rows)], ignore_index=True)

        # stable ordering: by season, then roster order
        sort_key = []
        for season, roster in full_roster_by_season.items():
            for i, cid in enumerate(roster):
                sort_key.append((season, cid, i))
        key_df = pd.DataFrame(sort_key, columns=["Season", "CoupleID", "_RosterOrder"])
        b = b.merge(key_df, how="left", on=["Season", "CoupleID"])
        b["_RosterOrder"] = b["_RosterOrder"].fillna(1e9)
        b = b.sort_values(["Season", "_RosterOrder", "CoupleID"]).drop(columns=["_RosterOrder"])

    else:
        b["Celebrity"] = b["CoupleID"].str.split("_").str[0]
        b["ProDancer"] = b["CoupleID"].str.split("_").str[1:].apply(
            lambda x: "_".join(x) if isinstance(x, list) else ""
        )

    # Consistency per method (requires ActualElimWeek)
    if "ActualElimWeek" in b.columns:
        # Make sure prediction columns are numeric; keep missing as NaN
        pred_cols = ["PredElimWeek_Rank", "PredElimWeek_Percent", "PredElimWeek_Bottom2"]
        for col in pred_cols:
            b[col] = pd.to_numeric(b[col], errors="coerce")

        b["ActualElimWeek"] = pd.to_numeric(b["ActualElimWeek"], errors="coerce")

        def _consistent(pred: pd.Series, actual: pd.Series) -> pd.Series:
            mask = pred.notna() & actual.notna()
            out = pd.Series(np.nan, index=pred.index, dtype="float")
            # compare on masked rows only (safe to cast there)
            out.loc[mask] = (pred.loc[mask].round().astype(int) == actual.loc[mask].round().astype(int)).astype(int)
            return out

        def _abserr(pred: pd.Series, actual: pd.Series) -> pd.Series:
            # keep NaN if either side missing
            return (pred - actual).abs()

        b["Consistent_Rank"] = _consistent(b["PredElimWeek_Rank"], b["ActualElimWeek"])
        b["Consistent_Percent"] = _consistent(b["PredElimWeek_Percent"], b["ActualElimWeek"])
        b["Consistent_Bottom2"] = _consistent(b["PredElimWeek_Bottom2"], b["ActualElimWeek"])

        b["AbsError_Rank"] = _abserr(b["PredElimWeek_Rank"], b["ActualElimWeek"])
        b["AbsError_Percent"] = _abserr(b["PredElimWeek_Percent"], b["ActualElimWeek"])
        b["AbsError_Bottom2"] = _abserr(b["PredElimWeek_Bottom2"], b["ActualElimWeek"])
    else:
        b["Consistent_Rank"] = np.nan
        b["Consistent_Percent"] = np.nan
        b["Consistent_Bottom2"] = np.nan
        b["AbsError_Rank"] = np.nan
        b["AbsError_Percent"] = np.nan
        b["AbsError_Bottom2"] = np.nan

    # Method disagreement
    # keep NaN if any side NaN (newly-added roster rows)
    for col in ["PredElimWeek_Rank", "PredElimWeek_Percent", "PredElimWeek_Bottom2"]:
        b[col] = pd.to_numeric(b[col], errors="coerce")
    b["Diff_Rank_vs_Percent"] = (b["PredElimWeek_Rank"] - b["PredElimWeek_Percent"]).abs()
    b["Diff_Rank_vs_Bottom2"] = (b["PredElimWeek_Rank"] - b["PredElimWeek_Bottom2"]).abs()
    b["Diff_Percent_vs_Bottom2"] = (b["PredElimWeek_Percent"] - b["PredElimWeek_Bottom2"]).abs()
    b["MaxMethodDiff"] = b[["Diff_Rank_vs_Percent", "Diff_Rank_vs_Bottom2", "Diff_Percent_vs_Bottom2"]].max(axis=1)

    # WeirdScore
    if "ActualElimWeek" in b.columns:
        b["MaxAbsErrorToActual"] = b[["AbsError_Rank", "AbsError_Percent", "AbsError_Bottom2"]].max(axis=1)
        b["WeirdScore"] = 1.0 * b["MaxMethodDiff"] + 0.25 * b["MaxAbsErrorToActual"]
    else:
        b["WeirdScore"] = b["MaxMethodDiff"]

    # Candidate-level sheet
    actual_col = "ActualElimWeek" if "ActualElimWeek" in b.columns else "CoupleID"
    benefit_rank = "Benefit_Percent_minus_Rank" if "Benefit_Percent_minus_Rank" in b.columns else None
    benefit_b2 = "Benefit_Bottom2_minus_Percent" if "Benefit_Bottom2_minus_Percent" in b.columns else None

    cols = [
        "Season", "Celebrity", "ProDancer", "CoupleID",
        actual_col,
        "PredElimWeek_Rank", "PredElimWeek_Percent", "PredElimWeek_Bottom2",
        "Consistent_Rank", "Consistent_Percent", "Consistent_Bottom2",
        "AbsError_Rank", "AbsError_Percent", "AbsError_Bottom2",
        "Diff_Rank_vs_Percent", "Diff_Rank_vs_Bottom2", "Diff_Percent_vs_Bottom2",
        "MaxMethodDiff", "WeirdScore",
    ]
    if benefit_rank is not None:
        cols.append(benefit_rank)
    if benefit_b2 is not None:
        cols.append(benefit_b2)

    candidate_level = b[cols].copy()

    # Top-K weirdest (ignore rows with no predictions)
    top_diff = (
        candidate_level[candidate_level["WeirdScore"].notna()]
        .sort_values(["WeirdScore", "MaxMethodDiff"], ascending=False)
        .head(30)
        .copy()
    )

    # Given four cases
    cases = [
        (2, "Jerry Rice"),
        (4, "Billy Ray Cyrus"),
        (11, "Bristol Palin"),
        (27, "Bobby Bones"),
    ]
    given_rows = []
    for s, name in cases:
        sub = candidate_level[
            (candidate_level["Season"] == s)
            & (candidate_level["Celebrity"].astype(str).str.contains(name, case=False, na=False))
        ]
        if len(sub) == 0:
            given_rows.append({"Season": s, "Name": name, "Found": 0})
        else:
            r = sub.iloc[0].to_dict()
            r["Name"] = name
            r["Found"] = 1
            given_rows.append(r)
    given_4 = pd.DataFrame(given_rows)

    return candidate_level, top_diff, given_4


# ---------------------------
# Main
# ---------------------------
def main():
    must_exist(MW_XLSX)
    must_exist(BENEFIT_XLSX)

    mw = pd.read_excel(MW_XLSX)
    benefit = pd.read_excel(BENEFIT_XLSX)

    main_data = None
    if MAIN_DATA_XLSX.exists():
        main_data = pd.read_excel(MAIN_DATA_XLSX)

    # Build outputs
    week_level, season_level, overall = build_week_consistency_outputs(mw)
    candidate_level, top_diff, given_4 = build_candidate_outputs(benefit, main_data)

    # Write OUT_WEEK
    RES.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_WEEK, engine="openpyxl") as writer:
        week_level.to_excel(writer, index=False, sheet_name="week_level")
        season_level.to_excel(writer, index=False, sheet_name="season_level")
        overall.to_excel(writer, index=False, sheet_name="overall")

    # Write OUT_CAND
    with pd.ExcelWriter(OUT_CAND, engine="openpyxl") as writer:
        candidate_level.to_excel(writer, index=False, sheet_name="candidate_level")
        top_diff.to_excel(writer, index=False, sheet_name="top_diff")
        given_4.to_excel(writer, index=False, sheet_name="given_4_cases")

    # Console summary
    print(f"✅ Wrote: {OUT_WEEK}")
    print(f"✅ Wrote: {OUT_CAND}")

    print("\n=== Top 10 Weirdest (by WeirdScore) ===")
    if len(top_diff):
        show = top_diff.head(10)[[
            "Season", "Celebrity", "ProDancer",
            "MaxMethodDiff", "WeirdScore",
            "PredElimWeek_Rank", "PredElimWeek_Percent", "PredElimWeek_Bottom2",
        ]]
        print(show.to_string(index=False))
    else:
        print("(no candidates with predictions found)")


if __name__ == "__main__":
    main()