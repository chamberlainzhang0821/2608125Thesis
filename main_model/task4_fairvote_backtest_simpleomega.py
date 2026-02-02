# task4_fairvote_backtest_simpleomega.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Optional but recommended for BSI regression
try:
    from sklearn.linear_model import LinearRegression
except Exception:
    LinearRegression = None


# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
RES = BASE_DIR / "results"

MAIN_DATA = BASE_DIR / "2026_MCM_Problem_C_Data.xlsx"
FAN_XLSX = RES / "fan_vote_estimates.xlsx"
WEEKLY_XLSX = RES / "weekly_uncertainty.xlsx"
METHOD_WEEK_XLSX = RES / "method_outcomes_by_week.xlsx"
TASK3_XLSX = RES / "task3_hyperparam_sweep_and_effects.xlsx"

OUT_REPORT = RES / "fairvote_backtest_report.xlsx"


# -------------------------
# Helpers
# -------------------------
def must_exist(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")


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
            wtag = c.lower().split("_")[0]
            try:
                w = int(wtag.replace("week", ""))
                max_w = max(max_w, w)
            except Exception:
                pass
    return min(max_w if max_w > 0 else 12, hard_cap)


def judge_total(row: pd.Series, week: int) -> float | None:
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


def zscore_within_week(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd <= 1e-12:
        return x - mu
    return (x - mu) / sd


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def load_task3_betas(task3_xlsx: Path) -> dict[str, pd.Series]:
    """
    Reads 3 sheets:
      - betas5_Perf_WeeksSurvived
      - betas5_Judge_AvgJudgeTotal
      - betas5_Fan_AvgFanShare
    """
    betas_perf = pd.read_excel(task3_xlsx, sheet_name="betas5_Perf_WeeksSurvived")
    betas_judge = pd.read_excel(task3_xlsx, sheet_name="betas5_Judge_AvgJudgeTotal")
    betas_fan = pd.read_excel(task3_xlsx, sheet_name="betas5_Fan_AvgFanShare")

    def one_row(df):
        beta_cols = [c for c in df.columns if "beta" in str(c).lower()]
        if len(beta_cols) == 0:
            raise KeyError(f"No beta columns in sheet: {list(df.columns)}")
        return df.iloc[0][beta_cols].astype(float)

    return {"perf": one_row(betas_perf), "judge": one_row(betas_judge), "fan": one_row(betas_fan)}


def beta_get(beta_series: pd.Series, key: str) -> float:
    """
    key: 'partner','industry','state','country','age' (substring match)
    """
    for k in beta_series.index:
        lk = str(k).lower()
        if key in lk:
            return float(beta_series.loc[k])
    return 0.0


def build_feature_map_from_task3(task3_xlsx: Path) -> tuple[dict[str, dict[str, float]], list[str]]:
    """
    Use task3 sheet model_data_used to build 5 standardized features for each CoupleID.
    Handles duplicate CoupleID by aggregating to one row per CoupleID.
    Returns:
      X_map: CoupleID -> dict of X cols
      Xcols: list of feature col names (order fixed)
    """
    md = pd.read_excel(task3_xlsx, sheet_name="model_data_used").copy()

    need_cols = ["CoupleID", "Partner", "Industry", "HomeState", "HomeCountry", "Age"]
    for c in need_cols:
        if c not in md.columns:
            raise KeyError(f"model_data_used missing column: {c}")

    # ---- Build stable numeric encodings (no leakage, no extra fitting)
    def freq_log(series: pd.Series):
        vc = series.fillna("Unknown").astype(str).value_counts()
        return series.fillna("Unknown").astype(str).map(lambda x: np.log1p(vc.get(x, 1)))

    md["X_partner"] = freq_log(md["Partner"])
    md["X_industry"] = freq_log(md["Industry"])
    md["X_state"] = freq_log(md["HomeState"])
    md["X_country"] = freq_log(md["HomeCountry"])
    md["X_age"] = pd.to_numeric(md["Age"], errors="coerce")

    Xcols = ["X_partner", "X_industry", "X_state", "X_country", "X_age"]

    # ---- KEY FIX: aggregate duplicate CoupleID to one row
    md_u = (
        md.groupby("CoupleID", as_index=False)[Xcols]
          .mean(numeric_only=True)
          .copy()
    )

    # ---- global standardize
    for c in Xcols:
        arr = md_u[c].astype(float).values
        mu = np.nanmean(arr)
        sd = np.nanstd(arr)
        md_u[c] = (md_u[c] - mu) / (sd if sd > 1e-12 else 1.0)

    X_map = md_u.set_index("CoupleID")[Xcols].to_dict(orient="index")
    return X_map, Xcols


def compute_bias_terms(
    all_couples: list[str],
    X_map: dict[str, dict[str, float]],
    betas: dict[str, pd.Series],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    biasJ[cid] = beta_J^T X_i
    biasF[cid] = beta_F^T X_i
    """
    bJ = {
        "partner": beta_get(betas["judge"], "partner"),
        "industry": beta_get(betas["judge"], "industry"),
        "state": beta_get(betas["judge"], "state"),
        "country": beta_get(betas["judge"], "country"),
        "age": beta_get(betas["judge"], "age"),
    }
    bF = {
        "partner": beta_get(betas["fan"], "partner"),
        "industry": beta_get(betas["fan"], "industry"),
        "state": beta_get(betas["fan"], "state"),
        "country": beta_get(betas["fan"], "country"),
        "age": beta_get(betas["fan"], "age"),
    }

    def bias_term(cid: str, which: str) -> float:
        x = X_map.get(cid)
        if x is None:
            return 0.0
        if which == "judge":
            return (
                bJ["partner"] * x["X_partner"]
                + bJ["industry"] * x["X_industry"]
                + bJ["state"] * x["X_state"]
                + bJ["country"] * x["X_country"]
                + bJ["age"] * x["X_age"]
            )
        else:
            return (
                bF["partner"] * x["X_partner"]
                + bF["industry"] * x["X_industry"]
                + bF["state"] * x["X_state"]
                + bF["country"] * x["X_country"]
                + bF["age"] * x["X_age"]
            )

    biasJ = {cid: float(bias_term(cid, "judge")) for cid in all_couples}
    biasF = {cid: float(bias_term(cid, "fan")) for cid in all_couples}
    return biasJ, biasF


def build_season_roster_map(df_main: pd.DataFrame) -> dict[int, list[str]]:
    """
    P0#3 FIX: roster universe per season from MAIN_DATA (full season roster).
    Stable order: first appearance in main file.
    """
    df = df_main.copy()
    if "CoupleID" not in df.columns:
        df["CoupleID"] = df["celebrity_name"].astype(str) + "_" + df["ballroom_partner"].astype(str)

    rosters: dict[int, list[str]] = {}
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())

    for s in seasons:
        sub = df[df["season"].astype(int) == int(s)]
        seen = set()
        ordered = []
        for cid in sub["CoupleID"].astype(str).tolist():
            cid = str(cid)
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
        rosters[int(s)] = ordered

    return rosters


# -------------------------
# FV week rule (simple omega)
# -------------------------
def omega_simple(unc: float, k: float, wmin: float, wmax: float) -> float:
    # omega_w = clip(0.5 - k * Unc_w, wmin, wmax)
    if not np.isfinite(unc):
        unc = 0.0
    return clip(0.5 - k * float(unc), wmin, wmax)


def fairvote_predict_one_week(
    couples: list[str],
    judge_scores: np.ndarray,
    fan_shares: np.ndarray,
    biasJ: dict[str, float],
    biasF: dict[str, float],
    unc_week: float,
    k: float,
    wmin: float,
    wmax: float,
    tau: float,
) -> tuple[str, float, int, float, list[str]]:
    """
    Returns:
      pred_elim,
      omega,
      gate(0/1),
      margin = S(second_worst) - S(worst),
      bottom2_ids (for debug/reporting)
    """
    js = np.asarray(judge_scores, dtype=float)
    fs = np.asarray(fan_shares, dtype=float)

    # 1) debias residuals
    Jp = np.array([js[i] - float(biasJ.get(couples[i], 0.0)) for i in range(len(couples))], dtype=float)
    Fp = np.array([fs[i] - float(biasF.get(couples[i], 0.0)) for i in range(len(couples))], dtype=float)

    # 2) standardize within week
    Jt = zscore_within_week(Jp)
    Ft = zscore_within_week(Fp)

    # 3) simple adaptive omega
    omega = omega_simple(unc=unc_week, k=k, wmin=wmin, wmax=wmax)

    # 4) combine score
    S = (1 - omega) * Jt + omega * Ft

    order = np.argsort(S)  # worst first
    worst = int(order[0])
    second = int(order[1]) if len(order) > 1 else int(order[0])
    margin = float(S[second] - S[worst])
    bottom2_ids = [couples[worst], couples[second]]

    gate = 1 if margin < tau else 0
    if gate == 1:
        # bottom2 judge-save: eliminate lower raw judge score within bottom2
        idxs = np.array([worst, second], dtype=int)
        loser = int(idxs[np.argmin(js[idxs])])
        return couples[loser], float(omega), 1, margin, bottom2_ids
    else:
        return couples[worst], float(omega), 0, margin, bottom2_ids


# -------------------------
# Backtest + Metrics
# -------------------------
def build_pred_elim_week_from_weekly(
    df_week: pd.DataFrame,
    pred_col: str,
    max_week: int,
    season_roster: dict[int, list[str]],
) -> pd.DataFrame:
    """
    P0#3 FIX:
    From weekly predictions (one row per season-week), compute predicted elimination week per CoupleID.
    Must include ALL couples in that season (roster universe), else BSI is biased.

    Rule:
      - initialize PredElimWeek = max_week+1 for every couple in roster (winners/finalists survive)
      - iterate weeks in order; first time a couple is predicted eliminated -> set its PredElimWeek = week
    """
    out_rows = []
    for season, g in df_week.groupby("Season"):
        season_i = int(season)
        g = g.sort_values("Week")

        roster = season_roster.get(season_i, [])
        # fallback: if roster missing, use union to avoid crash (still better than nothing)
        if not roster:
            roster = sorted(set(g[pred_col].astype(str).tolist()) | set(g["ActualEliminated"].astype(str).tolist()))

        pred_week = {cid: max_week + 1 for cid in roster}

        for _, r in g.iterrows():
            w = int(r["Week"])
            elim = str(r[pred_col])
            if elim in pred_week and pred_week[elim] == max_week + 1:
                pred_week[elim] = w
            elif elim not in pred_week:
                # if a prediction contains an unseen id, include it anyway (defensive)
                pred_week[elim] = w

        for cid, wk in pred_week.items():
            out_rows.append({"Season": season_i, "CoupleID": str(cid), "PredElimWeek": int(wk)})

    return pd.DataFrame(out_rows)


def bsi_regression(
    y_df: pd.DataFrame,
    X_map: dict[str, dict[str, float]],
    Xcols: list[str],
    label: str,
) -> tuple[float, float, pd.DataFrame]:
    """
    Fit linear regression: y ~ X (5 features), return:
      sum_abs_gamma, R2, coef_table
    """
    if LinearRegression is None:
        return np.nan, np.nan, pd.DataFrame({"Feature": Xcols, "Gamma": [np.nan]*len(Xcols), "Model": label})

    xs = []
    ys = []
    for _, r in y_df.iterrows():
        cid = str(r["CoupleID"])
        x = X_map.get(cid)
        if x is None:
            continue
        y = float(r["PredElimWeek"])
        if not np.isfinite(y):
            continue
        xs.append([float(x[c]) for c in Xcols])
        ys.append(y)

    if len(xs) < 20:
        return np.nan, np.nan, pd.DataFrame({"Feature": Xcols, "Gamma": [np.nan]*len(Xcols), "Model": label})

    X = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)

    reg = LinearRegression()
    reg.fit(X, y)
    r2 = float(reg.score(X, y))
    gamma = reg.coef_.astype(float)

    coef_tbl = pd.DataFrame({"Feature": Xcols, "Gamma": gamma, "AbsGamma": np.abs(gamma)})
    coef_tbl["Model"] = label

    return float(np.sum(np.abs(gamma))), r2, coef_tbl


def main():
    for p in [MAIN_DATA, FAN_XLSX, WEEKLY_XLSX, METHOD_WEEK_XLSX, TASK3_XLSX]:
        must_exist(p)

    data = pd.read_excel(MAIN_DATA)
    data["ElimWeek"] = data["results"].apply(parse_elim_week)
    data["CoupleID"] = data["celebrity_name"].astype(str) + "_" + data["ballroom_partner"].astype(str)

    # P0#3: full roster per season
    season_roster = build_season_roster_map(data)

    fan = pd.read_excel(FAN_XLSX)
    weekly = pd.read_excel(WEEKLY_XLSX)
    mw = pd.read_excel(METHOD_WEEK_XLSX)

    # lookups
    fan_lookup = fan.set_index(["Season", "Week", "CoupleID"])["VoteMean"]

    def get_fan(season: int, week: int, cid: str) -> float:
        try:
            return float(fan_lookup.loc[(int(season), int(week), str(cid))])
        except Exception:
            return np.nan

    wk_unc = weekly.set_index(["Season", "Week"])["AvgVoteUncertainty"].to_dict()

    max_week = detect_max_week_from_columns(data)
    seasons = sorted([int(x) for x in data["season"].dropna().unique().tolist()])

    # task3-based debias
    betas = load_task3_betas(TASK3_XLSX)
    X_map, Xcols = build_feature_map_from_task3(TASK3_XLSX)
    all_couples = sorted(set(data["CoupleID"].astype(str).tolist()))
    biasJ, biasF = compute_bias_terms(all_couples, X_map, betas)

    # -------------------------
    # Param grid
    # -------------------------
    K_GRID = [0.0, 0.5, 1.0, 2.0, 3.0]            # strength of uncertainty penalty
    TAU_GRID = [0.0, 0.05, 0.10, 0.15, 0.20]      # close-call gate threshold
    WMIN, WMAX = 0.35, 0.65

    sweep_rows = []
    best = None
    best_weekly = None
    best_coef_fv = None

    # -------------------------
    # Backtest loop
    # -------------------------
    for k in K_GRID:
        for tau in TAU_GRID:
            rows = []

            for season in seasons:
                sdf = data[data["season"] == season].copy()

                for week in range(1, max_week + 1):
                    elim_row = sdf[sdf["ElimWeek"] == week]
                    if len(elim_row) != 1:
                        continue
                    actual = str(elim_row.iloc[0]["CoupleID"])

                    active = sdf[sdf["ElimWeek"].isna() | (sdf["ElimWeek"] >= week)].copy()

                    couples, js, fs = [], [], []
                    for _, r in active.iterrows():
                        cid = str(r["CoupleID"])
                        jt = judge_total(r, week)
                        if jt is None:
                            continue
                        f = get_fan(season, week, cid)
                        if not np.isfinite(f):
                            continue
                        couples.append(cid)
                        js.append(float(jt))
                        fs.append(float(f))

                    if len(couples) < 2:
                        continue
                    if actual not in couples:
                        continue

                    fs = np.asarray(fs, dtype=float)
                    s = fs.sum()
                    if s > 0:
                        fs = fs / s

                    unc = float(wk_unc.get((season, week), np.nan))
                    if not np.isfinite(unc):
                        unc = float(np.nanmean(list(wk_unc.values()))) if len(wk_unc) else 0.1

                    pred, omega, gate, margin, bottom2_ids = fairvote_predict_one_week(
                        couples=couples,
                        judge_scores=np.asarray(js, dtype=float),
                        fan_shares=fs,
                        biasJ=biasJ,
                        biasF=biasF,
                        unc_week=unc,
                        k=float(k),
                        wmin=WMIN,
                        wmax=WMAX,
                        tau=float(tau),
                    )

                    rows.append({
                        "Season": int(season),
                        "Week": int(week),
                        "ActualEliminated": actual,
                        "FV_Pred": pred,
                        "FV_Match": int(pred == actual),
                        "FV_Omega": float(omega),
                        "FV_Gate": int(gate),
                        "FV_Margin": float(margin),
                        "Unc": float(unc),
                        "Bottom2_1": bottom2_ids[0],
                        "Bottom2_2": bottom2_ids[1],
                        "FanShare_Elim_FV": float(get_fan(season, week, pred)),
                        "FanShare_Elim_Actual": float(get_fan(season, week, actual)),
                    })

            if len(rows) == 0:
                continue

            dfw = pd.DataFrame(rows)

            # (1) Consistency
            consistency = float(dfw["FV_Match"].mean())

            # (4) Excitement rate
            excitement = float(dfw["FV_Gate"].mean())

            # (2) Fan-favoring
            delta_fan_overall = float(np.nanmean(dfw["FanShare_Elim_FV"]) - np.nanmean(dfw["FanShare_Elim_Actual"]))

            gate_df = dfw[dfw["FV_Gate"] == 1]
            if len(gate_df) > 0:
                delta_fan_gate = float(np.nanmean(gate_df["FanShare_Elim_FV"]) - np.nanmean(gate_df["FanShare_Elim_Actual"]))
                gate_count = int(len(gate_df))
            else:
                delta_fan_gate = np.nan
                gate_count = 0

            # (3) BSI: include ALL couples in season roster (P0#3)
            fv_pred_week_df = build_pred_elim_week_from_weekly(
                dfw, pred_col="FV_Pred", max_week=max_week, season_roster=season_roster
            )
            bsi_fv, r2_fv, coef_fv = bsi_regression(fv_pred_week_df, X_map, Xcols, label="FairVote")

            # Baselines
            base_week = mw[["Season", "Week", "ActualEliminated", "Pred_Rank", "Pred_Percent", "Pred_Bottom2"]].copy()

            rank_pred_week_df = build_pred_elim_week_from_weekly(
                base_week.rename(columns={"Pred_Rank": "Pred"}),
                pred_col="Pred",
                max_week=max_week,
                season_roster=season_roster,
            )
            pct_pred_week_df = build_pred_elim_week_from_weekly(
                base_week.rename(columns={"Pred_Percent": "Pred"}),
                pred_col="Pred",
                max_week=max_week,
                season_roster=season_roster,
            )
            b2_pred_week_df = build_pred_elim_week_from_weekly(
                base_week.rename(columns={"Pred_Bottom2": "Pred"}),
                pred_col="Pred",
                max_week=max_week,
                season_roster=season_roster,
            )

            bsi_rank, r2_rank, _ = bsi_regression(rank_pred_week_df, X_map, Xcols, label="Rank")
            bsi_pct, r2_pct, _ = bsi_regression(pct_pred_week_df, X_map, Xcols, label="Percent")
            bsi_b2, r2_b2, _ = bsi_regression(b2_pred_week_df, X_map, Xcols, label="Bottom2")

            sweep_rows.append({
                "k": float(k),
                "tau": float(tau),
                "wmin": WMIN,
                "wmax": WMAX,
                "Consistency": consistency,
                "ExcitementRate": excitement,
                "DeltaFanShareElim(FV-Actual)": delta_fan_overall,
                "DeltaFanShareElim_GateOnly(FV-Actual)": delta_fan_gate,
                "GateWeeks": gate_count,
                "BSI_FV_sumAbsGamma": bsi_fv,
                "BSI_Rank_sumAbsGamma": bsi_rank,
                "BSI_Percent_sumAbsGamma": bsi_pct,
                "BSI_Bottom2_sumAbsGamma": bsi_b2,
                "R2_FV": r2_fv,
                "R2_Rank": r2_rank,
                "R2_Percent": r2_pct,
                "R2_Bottom2": r2_b2,
                "NumWeeks": int(len(dfw)),
            })

            # select best by (1) Consistency then (4) ExcitementRate (tiebreak)
            if best is None:
                best = {"Consistency": consistency, "ExcitementRate": excitement, "k": k, "tau": tau}
                best_weekly = dfw.copy()
                best_coef_fv = coef_fv.copy()
            else:
                if (consistency > best["Consistency"]) or (consistency == best["Consistency"] and excitement > best["ExcitementRate"]):
                    best = {"Consistency": consistency, "ExcitementRate": excitement, "k": k, "tau": tau}
                    best_weekly = dfw.copy()
                    best_coef_fv = coef_fv.copy()

    if len(sweep_rows) == 0:
        raise RuntimeError("No weeks evaluated. Check inputs / fan estimates alignment.")

    sweep_df = pd.DataFrame(sweep_rows).sort_values(["Consistency", "ExcitementRate"], ascending=False)
    best_row = sweep_df.iloc[0].to_dict()

    # Season summary for best
    season_sum = (
        best_weekly.groupby("Season", as_index=False)
        .agg(
            Weeks=("Week", "count"),
            FV_Consistency=("FV_Match", "mean"),
            FV_Excitement=("FV_Gate", "mean"),
            FV_Omega_mean=("FV_Omega", "mean"),
            FV_Omega_min=("FV_Omega", "min"),
            FV_Omega_max=("FV_Omega", "max"),
        )
        .sort_values("Season")
    )

    # ---- Baseline season summary (Rank/Percent/Bottom2) + Flip_RP (robust)
    mw2 = mw.copy()

    if "Flip_Rank_vs_Percent" not in mw2.columns:
        if ("Pred_Rank" in mw2.columns) and ("Pred_Percent" in mw2.columns):
            mw2["Flip_Rank_vs_Percent"] = (mw2["Pred_Rank"].astype(str) != mw2["Pred_Percent"].astype(str)).astype(int)
        else:
            mw2["Flip_Rank_vs_Percent"] = np.nan

    for c in ["Match_Rank", "Match_Percent", "Match_Bottom2"]:
        if c not in mw2.columns:
            mw2[c] = np.nan

    base_sum = (
        mw2.groupby("Season", as_index=False)
        .agg(
            Rank=("Match_Rank", "mean"),
            Percent=("Match_Percent", "mean"),
            Bottom2=("Match_Bottom2", "mean"),
            Flip_RP=("Flip_Rank_vs_Percent", "mean"),
        )
    )
    season_sum = season_sum.merge(base_sum, on="Season", how="left")

    # ---- BSI for BEST params (with coef tables)
    fv_pred_week_df_best = build_pred_elim_week_from_weekly(
        best_weekly, pred_col="FV_Pred", max_week=max_week, season_roster=season_roster
    )
    bsi_fv, r2_fv, coef_fv = bsi_regression(fv_pred_week_df_best, X_map, Xcols, label="FairVote(best)")

    base_week = mw[["Season", "Week", "ActualEliminated", "Pred_Rank", "Pred_Percent", "Pred_Bottom2"]].copy()
    rank_pred_week_df = build_pred_elim_week_from_weekly(
        base_week.rename(columns={"Pred_Rank": "Pred"}), pred_col="Pred", max_week=max_week, season_roster=season_roster
    )
    pct_pred_week_df = build_pred_elim_week_from_weekly(
        base_week.rename(columns={"Pred_Percent": "Pred"}), pred_col="Pred", max_week=max_week, season_roster=season_roster
    )
    b2_pred_week_df = build_pred_elim_week_from_weekly(
        base_week.rename(columns={"Pred_Bottom2": "Pred"}), pred_col="Pred", max_week=max_week, season_roster=season_roster
    )

    bsi_rank, r2_rank, coef_rank = bsi_regression(rank_pred_week_df, X_map, Xcols, label="Rank")
    bsi_pct, r2_pct, coef_pct = bsi_regression(pct_pred_week_df, X_map, Xcols, label="Percent")
    bsi_b2, r2_b2, coef_b2 = bsi_regression(b2_pred_week_df, X_map, Xcols, label="Bottom2")

    bsi_tbl = pd.DataFrame([
        {"Model": "FairVote(best)", "BSI_sumAbsGamma": bsi_fv, "R2": r2_fv},
        {"Model": "Rank",           "BSI_sumAbsGamma": bsi_rank, "R2": r2_rank},
        {"Model": "Percent",        "BSI_sumAbsGamma": bsi_pct, "R2": r2_pct},
        {"Model": "Bottom2",        "BSI_sumAbsGamma": bsi_b2, "R2": r2_b2},
    ])

    reg_coefs = pd.concat([coef_fv, coef_rank, coef_pct, coef_b2], ignore_index=True)

    # Overall metrics report (single-row)
    gate_df = best_weekly[best_weekly["FV_Gate"] == 1]
    metrics_report = pd.DataFrame([{
        "Best_k": best_row["k"],
        "Best_tau": best_row["tau"],
        "Wmin": WMIN,
        "Wmax": WMAX,
        "Consistency": float(best_weekly["FV_Match"].mean()),
        "ExcitementRate": float(best_weekly["FV_Gate"].mean()),
        "DeltaFanShareElim(FV-Actual)": float(np.nanmean(best_weekly["FanShare_Elim_FV"]) - np.nanmean(best_weekly["FanShare_Elim_Actual"])),
        "DeltaFanShareElim_GateOnly(FV-Actual)": float(np.nanmean(gate_df["FanShare_Elim_FV"]) - np.nanmean(gate_df["FanShare_Elim_Actual"])) if len(gate_df) else np.nan,
        "GateWeeks": int(len(gate_df)),
        "BSI_FairVote(best)_sumAbsGamma": bsi_fv,
        "BSI_Percent_sumAbsGamma": bsi_pct,
        "BSI_Rank_sumAbsGamma": bsi_rank,
        "BSI_Bottom2_sumAbsGamma": bsi_b2,
    }])

    # -------------------------
    # Write report XLSX
    # -------------------------
    with pd.ExcelWriter(OUT_REPORT, engine="openpyxl") as writer:
        sweep_df.to_excel(writer, index=False, sheet_name="param_sweep")
        pd.DataFrame([best_row]).to_excel(writer, index=False, sheet_name="best_params")
        metrics_report.to_excel(writer, index=False, sheet_name="metrics_report")
        best_weekly.to_excel(writer, index=False, sheet_name="weekly_predictions")
        season_sum.to_excel(writer, index=False, sheet_name="season_summary")
        bsi_tbl.to_excel(writer, index=False, sheet_name="bsi")
        reg_coefs.to_excel(writer, index=False, sheet_name="regression_coefs")

    print("✅ FairVote backtest report written to:", OUT_REPORT)
    print("Best params:", best_row)


if __name__ == "__main__":
    main()