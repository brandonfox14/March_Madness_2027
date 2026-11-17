
"""March Metrics 2027 — daily matchup predictor.

Save as APP/Pages/3_Todays_Games.py.
APP/App.py already sets Streamlit page configuration.
"""

from pathlib import Path
from math import erf, sqrt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)

# ============================================================
# FILE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "27_March_Madness_Data"

FILES = {
    "date": "Current_Day-Table 1.csv",
    "schedule": "Schedule-Table 1.csv",
    "historical": "PREVIOUS_YEARS-Table 1.csv",
    "season": "COMP_STAT-THE_TABLE.csv",
    "teams": "All_Stats-THE_TABLE.csv",
    "coaches": "Coach-Table 1.csv",
}

# ============================================================
# MODEL SETTINGS
# ============================================================

CURRENT_YEAR_MULTIPLIER = 3.0
CONFERENCE_MULTIPLIER = 0.20
# Historical component: 90% team/coach, at most 10% opponent-conference.
TEAM_COACH_CONTEXT_SHARE = 0.95
CONFERENCE_CONTEXT_SHARE = 0.05
PRIOR_GAMES = 8.0
MIN_SD = 6.0
TEAM_SCORE_SHARE = 0.75
OPP_DEFENSE_SHARE = 0.25
ML_CONTEXT_SHARE = 0.65
ML_CONFERENCE_SHARE = 0.05

ML_START_DATE = pd.Timestamp(2026, 11, 15)
ML_INFLUENCE = 0.30
ML_MIN_ROWS = 40

OUTCOMES = [
    "1H", "OPP_1H",
    "2H", "OPP_2H",
    "OT_IND",
    "PTS", "OPP_PTS",
]

# Pregame game-log field -> current All_Stats field
ML_FEATURE_MAP = {
    "GP": None,
    "WIN_PCT": "Win_Perc",
    "PTS_FOR": "PTS",
    "PTS_ALLOW": "OPP_PTS",
    "1H_FOR": "1H",
    "1H_ALLOW": "OPP_1H",
    "2H_FOR": "2H",
    "2H_ALLOW": "OPP_2H",
    "FG_PCT_FOR": "FGPerc",
    "FG_PCT_ALLOW": "OPP_FGPerc",
    "2P_PCT_FOR": "FG2Perc",
    "2P_PCT_ALLOW": "OPP_FG2Perc",
    "3P_PCT_FOR": "FG3Perc",
    "3P_PCT_ALLOW": "OPP_FG3Perc",
    "FT_PCT_FOR": "FTPerc",
    "EFG_FOR": "EFG_PCT",
    "EFG_ALLOW": "OPP_EFG_PCT",
    "TS_FOR": "TS_PCT",
    "TS_ALLOW": "OPP_TS_PCT",
    "OFF_EFF": "OFF_EFF",
    "DEF_EFF": "DEF_EFF",
    "NET_EFF": "NET_EFF",
    "OREB_RATE": "OREB_RATE",
    "DREB_RATE": "DREB_RATE",
    "REB_SHARE": "REB_SHARE",
    "TURN_RATE": "TURN_RATE",
    "FORCED_TURN_RATE": "FORCED_TURN_RATE",
    "EXTRA_CREATED": "EXTRA_OPP_CREATED",
    "EXTRA_ALLOWED": "EXTRA_OPP_ALLOWED",
    "NET_EXTRA_OPP": "NET_EXTRA_OPP",
    "NET_EXTRA_RATE": "NET_EXTRA_OPP_RATE",
    "AST_TO": "AST_TO_RATIO",
    "AST_TO_ALLOW": "OPP_AST_TO_RATIO",
    "AST_FGM_RATE": "AST_FGM_RATE",
    "AST_FGM_RATE_ALLOW": "OPP_AST_FGM_RATE",
}

ML_PER_GAME_FIELDS = {
    "PTS_FOR", "PTS_ALLOW",
    "1H_FOR", "1H_ALLOW",
    "2H_FOR", "2H_ALLOW",
}


# ============================================================
# DATA LOADING
# ============================================================

def read_csv(path):
    if not path.is_file():
        return None

    for encoding in ("utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False
            )
            df.columns = [
                str(c).strip() for c in df.columns
            ]
            return df
        except UnicodeDecodeError:
            continue

    return None


@st.cache_data(show_spinner=False)
def read_sources(file_signatures):
    return {
        key: read_csv(DATA / name)
        for key, name, _ in file_signatures
    }


def n(value):
    return pd.to_numeric(value, errors="coerce")


def clean_text(series):
    return series.fillna("").astype(str).str.strip()


def key_text(value):
    if pd.notna(value):
        return str(value).strip().casefold()
    return ""


def date_from_row(row):
    return pd.Timestamp(
        year=int(n(row["YYYY"])),
        month=int(n(row["MM"])),
        day=int(n(row["DD"]))
    )


def find_date(df):
    date_col = (
        "GAME_DATE" if "GAME_DATE" in df.columns
        else "Date" if "Date" in df.columns
        else None
    )

    if date_col:
        parsed = pd.to_datetime(
            df[date_col],
            errors="coerce",
            format="%m/%d/%y"
        )

        missing = parsed.isna()

        if missing.any():
            parsed.loc[missing] = pd.to_datetime(
                df.loc[missing, date_col],
                errors="coerce",
                format="%m/%d/%Y"
            )
    else:
        parsed = pd.Series(pd.NaT, index=df.index)

    if {"MM", "DD", "YYYY"}.issubset(df.columns):
        parts = pd.to_datetime(
            pd.DataFrame({
                "year": n(df["YYYY"]),
                "month": n(df["MM"]),
                "day": n(df["DD"])
            }),
            errors="coerce"
        )
        parsed = parsed.fillna(parts)

    return parsed


def team_catalog(df):
    required = {"Teams", "Conference", "Acronym"}

    if df is None or not required.issubset(df.columns):
        raise ValueError(
            "All_Stats must contain Teams, Conference, and Acronym."
        )

    t = df.drop_duplicates(
        subset=["Acronym"]
    ).copy()

    t["ac_key"] = (
        clean_text(t["Acronym"]).str.upper()
    )

    return t.set_index("ac_key", drop=False)


# ============================================================
# PREPARE HISTORICAL GAME DATA
# ============================================================

def prepare_games(df, source, current_date, catalog):
    if df is None or df.empty:
        return pd.DataFrame()

    needed = {
        "TEAM", "OPP", "Coach",
        "CONF", "OPP_CONF",
        "HAN", "PTS", "OPP_PTS"
    }

    if not needed.issubset(df.columns):
        missing = sorted(needed - set(df.columns))
        raise ValueError(
            f"{source} missing columns: {missing}"
        )

    out = df.copy()
    out["_DATE"] = find_date(out)

    # Only use games completed BEFORE the selected date.
    out = out.loc[
        out["_DATE"].notna()
        & (out["_DATE"] < current_date)
    ].copy()

    for col in OUTCOMES + ["SPREAD", "TOTAL_LINE"]:
        if col in out:
            out[col] = n(out[col])
        else:
            out[col] = np.nan

    out = out.loc[
        out["PTS"].notna()
        & out["OPP_PTS"].notna()
    ].copy()

    # Historical market results
    out["WIN_RESULT"] = (
        out["PTS"] > out["OPP_PTS"]
    ).astype(float)

    out["COVER_RESULT"] = np.where(
        out["SPREAD"].notna(),
        (
            out["PTS"] + out["SPREAD"] > out["OPP_PTS"]
        ).astype(float),
        np.nan
    )

    out["OVER_RESULT"] = np.where(
        out["TOTAL_LINE"].notna(),
        (
            out["PTS"] + out["OPP_PTS"]
            > out["TOTAL_LINE"]
        ).astype(float),
        np.nan
    )

    # Final-result margins used only for retrospective diagnostics.
    out["WIN_MARGIN"] = out["PTS"] - out["OPP_PTS"]
    out["COVER_MARGIN_CALC"] = out["WIN_MARGIN"] + out["SPREAD"]
    out["OVER_MARGIN_CALC"] = out["PTS"] + out["OPP_PTS"] - out["TOTAL_LINE"]

    if out.empty:
        return out

    for col in [
        "TEAM", "OPP", "Coach",
        "CONF", "OPP_CONF", "HAN", "FAVORITE"
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = clean_text(out[col]).str.casefold()

    out["HAN"] = out["HAN"].str.upper()
    out["_YEAR"] = out["_DATE"].dt.year.astype(int)
    out["_SOURCE"] = source

    age = np.maximum(
        0,
        current_date.year - out["_YEAR"].to_numpy()
    )

    if source == "season":
        out["_WEIGHT"] = CURRENT_YEAR_MULTIPLIER
    else:
        out["_WEIGHT"] = 1.0 / (2.0 + age)

    if "GAME_ID" in out.columns:
        out["_KEY"] = (
            out["GAME_ID"].astype(str)
            + "|" + out["TEAM"]
            + "|" + out["_DATE"].astype(str)
        )
    else:
        out["_KEY"] = (
            out["TEAM"]
            + "|" + out["OPP"]
            + "|" + out["_DATE"].astype(str)
        )

    return out


# ============================================================
# WEIGHTED STATISTICS
# ============================================================

def weighted_stats(values, weights):
    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not mask.any():
        return np.nan, np.nan, 0.0

    x = values[mask]
    w = weights[mask]

    mean = np.average(x, weights=w)

    variance = np.average(
        (x - mean) ** 2,
        weights=w
    )

    effective = (
        w.sum() ** 2 / np.square(w).sum()
    )

    return (
        float(mean),
        float(np.sqrt(variance)),
        float(effective)
    )


def normal_cdf(z):
    return 0.5 * (
        1.0 + erf(z / sqrt(2.0))
    )


# ============================================================
# BUILD MATCHUP CONTEXT
# ============================================================

def make_context(
    team, opponent, home_team, loc, catalog
):
    a = catalog.loc[str(team).strip().upper()]
    b = catalog.loc[str(opponent).strip().upper()]

    h = str(home_team).strip().casefold()

    if h in {"neutral", "neither", "none"}:
        han = "NEUTRAL"

    elif h in {
        key_text(a["Teams"]),
        key_text(a["Acronym"])
    }:
        han = "HOME"

    elif h in {
        key_text(b["Teams"]),
        key_text(b["Acronym"])
    }:
        han = "AWAY"

    else:
        han = "UNKNOWN"

    return {
        "TEAM": key_text(a["Teams"]),
        "Coach": key_text(a.get("Coach Name", "")),
        "CONF": key_text(a["Conference"]),
        "OPP_CONF": key_text(b["Conference"]),
        "HAN": han,
        "NAME": str(a["Teams"]),
        "OPP_NAME": str(b["Teams"])
    }


# ============================================================
# HISTORICAL CONTEXT GROUPS
# ============================================================

def group_definitions(ctx):
    definitions = [
        (
            "Team",
            {"TEAM": ctx["TEAM"]},
            2.0
        ),
        (
            "Coach",
            {"Coach": ctx["Coach"]},
            1.1
        ),
        (
            "Team + Coach",
            {
                "TEAM": ctx["TEAM"],
                "Coach": ctx["Coach"]
            },
            3.0
        ),
        (
            "Conference vs Opp Conference",
            {
                "CONF": ctx["CONF"],
                "OPP_CONF": ctx["OPP_CONF"]
            },
            0.20
        ),
        (
            "Team + Opp Conference",
            {
                "TEAM": ctx["TEAM"],
                "OPP_CONF": ctx["OPP_CONF"]
            },
            1.5
        ),
        (
            "Coach + Opp Conference",
            {
                "Coach": ctx["Coach"],
                "OPP_CONF": ctx["OPP_CONF"]
            },
            0.8
        ),
    ]

    if ctx["HAN"] != "UNKNOWN":
        definitions.extend([
            (
                "Team + HAN",
                {
                    "TEAM": ctx["TEAM"],
                    "HAN": ctx["HAN"]
                },
                2.8
            ),
            (
                "Coach + HAN",
                {
                    "Coach": ctx["Coach"],
                    "HAN": ctx["HAN"]
                },
                1.4
            ),
            (
                "Team + Coach + HAN",
                {
                    "TEAM": ctx["TEAM"],
                    "Coach": ctx["Coach"],
                    "HAN": ctx["HAN"]
                },
                4.0
            ),
            (
                "Team + Opp Conf + HAN",
                {
                    "TEAM": ctx["TEAM"],
                    "OPP_CONF": ctx["OPP_CONF"],
                    "HAN": ctx["HAN"]
                },
                2.0
            ),
            (
                "Coach + Opp Conf + HAN",
                {
                    "Coach": ctx["Coach"],
                    "OPP_CONF": ctx["OPP_CONF"],
                    "HAN": ctx["HAN"]
                },
                1.0
            ),
            (
                "Conference + Opp Conf + HAN",
                {
                    "CONF": ctx["CONF"],
                    "OPP_CONF": ctx["OPP_CONF"],
                    "HAN": ctx["HAN"]
                },
                0.20
            ),
            (
                "Team + Coach + Opp Conf + HAN",
                {
                    "TEAM": ctx["TEAM"],
                    "Coach": ctx["Coach"],
                    "OPP_CONF": ctx["OPP_CONF"],
                    "HAN": ctx["HAN"]
                },
                4.5
            ),
        ])

    if ctx["CONF"] == ctx["OPP_CONF"]:
        definitions = [
            (
                name,
                criteria,
                importance * (
                    1.2 if "Opp Conf" in name else 1.0
                )
            )
            for name, criteria, importance in definitions
        ]

    return definitions


# ============================================================
# HISTORICAL FORECAST FOR ONE TEAM
# ============================================================

def forecast_side(games, ctx):
    if games.empty:
        return None

    weights = games["_WEIGHT"].to_numpy(dtype=float)

    global_mean = {}
    global_sd = {}
    team_history = games.loc[games["TEAM"] == ctx["TEAM"]]

    targets = OUTCOMES + [
        "WIN_RESULT",
        "COVER_RESULT",
        "OVER_RESULT"
    ]

    for outcome in targets:
        mu, sd, _ = weighted_stats(
            games[outcome].to_numpy(dtype=float),
            weights
        )

        global_mean[outcome] = mu
        global_sd[outcome] = sd

    descriptions = []

    for title, criteria, importance in group_definitions(ctx):
        if not all(value for value in criteria.values()):
            continue

        subset = games

        for col, value in criteria.items():
            subset = subset.loc[
                subset[col] == value
            ]

        if subset.empty:
            continue

        descriptions.append(
            (title, subset, importance)
        )

    if not descriptions:
        return None

    predictions = {}
    spreads = {}
    pool_means = {}
    info = []

    for title, sub, imp in descriptions:
        w = sub["_WEIGHT"].to_numpy(dtype=float)

        effective = (
            w.sum() ** 2 / np.square(w).sum()
        )

        info.append({
            "Context": title,
            "Games": len(sub),
            "Effective sample": round(effective, 1),
            "Influence": imp
        })

    # Separate team/coach contexts from any contexts involving opponent
    # conference. This prevents the numerous opponent-conference combinations
    # from unintentionally overpowering the team/coach contexts.
    for target in targets:
        pool_results = {}
        for pool_name in ("team_coach", "conference"):
            estimates, masses, variances = [], [], []
            for title, sub, imp in descriptions:
                involves_conference = "Opp Conf" in title
                if (pool_name == "conference") != involves_conference:
                    continue

                vals = sub[target].to_numpy(dtype=float)
                ws = sub["_WEIGHT"].to_numpy(dtype=float)
                mu, sd, effective = weighted_stats(vals, ws)
                if not np.isfinite(mu):
                    continue
                reliability = effective / (effective + PRIOR_GAMES)
                # Prefer this team's own past, not the national scoring mean.
                team_prior, _, _ = weighted_stats(
                    team_history[target].to_numpy(dtype=float),
                    team_history["_WEIGHT"].to_numpy(dtype=float)
                ) if not team_history.empty else (np.nan, np.nan, 0.0)
                prior = team_prior if np.isfinite(team_prior) else mu
                estimates.append(reliability * mu + (1 - reliability) * prior)
                masses.append(imp * reliability)
                variances.append((sd if np.isfinite(sd) else 0.0) ** 2)

            if masses and sum(masses) > 0:
                pool_mean = float(np.average(estimates, weights=masses))
                within = float(np.average(variances, weights=masses))
                between = float(np.average(
                    (np.array(estimates) - pool_mean) ** 2,
                    weights=masses
                ))
                pool_results[pool_name] = (pool_mean, within + between)

        pool_means[target] = {
            key: pair[0] for key, pair in pool_results.items()
        }

        if "team_coach" in pool_results and "conference" in pool_results:
            shares = {"team_coach": TEAM_COACH_CONTEXT_SHARE,
                      "conference": CONFERENCE_CONTEXT_SHARE}
        elif "team_coach" in pool_results:
            shares = {"team_coach": 1.0}
        elif "conference" in pool_results:
            # Only a conference signal exists: do not pretend it represents
            # 90% of the evidence; fall back to the broad historical mean.
            shares = {"conference": CONFERENCE_CONTEXT_SHARE}
            if np.isfinite(global_mean[target]):
                pool_results["global_fallback"] = (
                    global_mean[target],
                    (global_sd[target] if np.isfinite(global_sd[target]) else MIN_SD) ** 2
                )
                shares["global_fallback"] = 1 - CONFERENCE_CONTEXT_SHARE
            else:
                shares = {"conference": 1.0}
        else:
            predictions[target] = global_mean[target]
            spreads[target] = global_sd[target]
            continue

        mu = sum(shares[k] * pool_results[k][0] for k in shares)
        variance = sum(
            shares[k] * (pool_results[k][1] + (pool_results[k][0] - mu) ** 2)
            for k in shares
        )
        predictions[target] = float(mu)
        spreads[target] = float(sqrt(max(0.0, variance)))

    for probability in [
        "OT_IND",
        "WIN_RESULT",
        "COVER_RESULT",
        "OVER_RESULT"
    ]:
        if np.isfinite(predictions[probability]):
            predictions[probability] = float(
                np.clip(
                    predictions[probability],
                    0,
                    1
                )
            )

    return {
        "mean": predictions,
        "sd": spreads,
        "pools": pool_means,
        "detail": pd.DataFrame(info),
        "team_games": int(
            (games["TEAM"] == ctx["TEAM"]).sum()
        )
    }


# ============================================================
# COMBINE BOTH TEAM PERSPECTIVES
# ============================================================

def blend(a, b, a_key, b_key, share=TEAM_SCORE_SHARE):
    """Use the scoring team's history primarily; defense <= 25%."""
    x = a["mean"].get(a_key, np.nan)
    y = b["mean"].get(b_key, np.nan)
    if np.isfinite(x) and np.isfinite(y):
        return float(share * x + (1 - share) * y)
    return float(x if np.isfinite(x) else y)


def context_component(side, target, pool):
    return side.get("pools", {}).get(target, {}).get(pool, np.nan)


def context_score(offense, defense, pool):
    p_for = context_component(offense, "PTS", pool)
    p_allowed = context_component(defense, "OPP_PTS", pool)
    if np.isfinite(p_for) and np.isfinite(p_allowed):
        return TEAM_SCORE_SHARE * p_for + OPP_DEFENSE_SHARE * p_allowed
    if np.isfinite(p_for):
        return p_for
    if np.isfinite(p_allowed):
        return p_allowed
    return np.nan


def scoring_dispersion(offense, defense):
    """Predictive score SD estimated from matched scoring/allowing histories.

    Does NOT artificially force a 75-point lower bound; 95% coverage
    has to be evaluated on held-out results later.
    """
    s_for = offense["sd"].get("PTS", np.nan)
    s_against = defense["sd"].get("OPP_PTS", np.nan)
    if np.isfinite(s_for) and np.isfinite(s_against):
        return float(max(MIN_SD, sqrt(
            TEAM_SCORE_SHARE * s_for**2 + OPP_DEFENSE_SHARE * s_against**2
        )))
    if np.isfinite(s_for):
        return float(max(MIN_SD, s_for))
    if np.isfinite(s_against):
        return float(max(MIN_SD, s_against))
    return float(MIN_SD)


def weighted_quantile(values, weights, q):
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    v = values[valid]
    w = weights[valid]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cumulative, v))


def score_interval(offense, defense, games, predicted_mean, fallback_sd):
    """Recency-weighted 2.5%-97.5% empirical score residual interval.

    Only the scoring team's OWN points and opponent's OWN points allowed
    enter, at 75/25. National or conference score distributions don't
    inflate it. This is still uncalibrated without held-out testing.
    """
    components = []
    for ctx, score_col, share in (
        (offense, "PTS", TEAM_SCORE_SHARE),
        (defense, "OPP_PTS", OPP_DEFENSE_SHARE),
    ):
        sub = games.loc[games["TEAM"] == ctx["TEAM"]]
        if sub.empty:
            continue
        values = sub[score_col].to_numpy(dtype=float)
        weights = sub["_WEIGHT"].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            continue
        values, weights = values[valid], weights[valid]
        effective = weights.sum() ** 2 / np.sum(weights ** 2)
        if effective < 8:
            continue
        resid = values - np.average(values, weights=weights)
        components.append((resid, weights / weights.sum(), share))
    if not components or not np.isfinite(predicted_mean):
        return (max(0, predicted_mean - 1.96 * fallback_sd),
                predicted_mean + 1.96 * fallback_sd)
    total_share = sum(item[2] for item in components)
    residuals = np.concatenate([c[0] for c in components])
    weights = np.concatenate([c[1] * c[2] / total_share for c in components])
    lower = weighted_quantile(residuals, weights, 0.025)
    upper = weighted_quantile(residuals, weights, 0.975)
    # Minimum ±1 standard deviation when small/degenerate samples.
    lower = min(lower, -fallback_sd)
    upper = max(upper, fallback_sd)
    return (max(0.0, predicted_mean + lower), predicted_mean + upper)


def model_matchup(first, second):
    # The two contexts are combined explicitly below, not double-counted.
    t1 = context_score(first, second, "team_coach")
    t2 = context_score(second, first, "team_coach")
    c1 = context_score(first, second, "conference")
    c2 = context_score(second, first, "conference")

    # Before ML starts: 95% team/coach + 5% conference, if present.
    # Once ML exists: 65% team/coach + 5% conference + 30% ML.
    def combine(team, conf):
        if np.isfinite(team) and np.isfinite(conf):
            return TEAM_COACH_CONTEXT_SHARE * team + CONFERENCE_CONTEXT_SHARE * conf
        return team if np.isfinite(team) else conf

    p1, p2 = combine(t1, c1), combine(t2, c2)
    h1 = blend(first, second, "1H", "OPP_1H")
    h2 = blend(second, first, "1H", "OPP_1H")
    s1 = blend(first, second, "2H", "OPP_2H")
    s2 = blend(second, first, "2H", "OPP_2H")
    ot_values = [first["mean"].get("OT_IND", np.nan),
                 second["mean"].get("OT_IND", np.nan)]
    ot_values = [x for x in ot_values if np.isfinite(x)]
    ot_prob = float(np.mean(ot_values)) if ot_values else np.nan
    sd1 = scoring_dispersion(first, second)
    sd2 = scoring_dispersion(second, first)

    return {
        "p1": p1, "p2": p2, "h1": h1, "h2": h2,
        "s1": s1, "s2": s2, "ot_prob": ot_prob,
        "sd1": sd1, "sd2": sd2,
        "sd_margin": sqrt(sd1**2 + sd2**2),
        "sd_total": sqrt(sd1**2 + sd2**2),
        "team1_context": t1, "team2_context": t2,
        "team1_conference": c1, "team2_conference": c2,
    }


# ============================================================
# CURRENT-SEASON MACHINE LEARNING
# ============================================================
# Three methods:
# 1. Ridge Regression
# 2. Extra Trees Regression
# 3. Histogram Gradient Boosting
#
# No team/coach/conference identifiers.
# Only COMP_STAT completed games.
# Only pregame columns used as predictors.
# PTS and OPP_PTS are separate targets.
# ============================================================

def ml_feature_names():
    return (
        ["TEAM_PRE_" + k for k in ML_FEATURE_MAP]
        + ["OPP_PRE_" + k for k in ML_FEATURE_MAP]
    )


def ml_train_data(current_games):
    names = ml_feature_names()

    result = pd.DataFrame(
        index=current_games.index
    )

    for feature in names:
        if feature in current_games.columns:
            result[feature] = n(
                current_games[feature]
            )
        else:
            result[feature] = np.nan

    selected = [
        f for f in names
        if result[f].notna().sum() >= 15
        and result[f].nunique(dropna=True) > 1
    ]

    return (
        result[selected].replace(
            [np.inf, -np.inf],
            np.nan
        ),
        selected
    )


@st.cache_resource(show_spinner=False)
def train_season_ml(training_signature, training_frame):
    if len(training_frame) < ML_MIN_ROWS:
        return (
            None,
            f"Only {len(training_frame)} training rows; "
            f"need {ML_MIN_ROWS}."
        )

    X, selected = ml_train_data(training_frame)

    if len(selected) < 4:
        return (
            None,
            "Too few populated, varying pregame "
            "statistic columns."
        )

    targets = ["PTS", "OPP_PTS"]

    y = training_frame[targets].apply(
        pd.to_numeric,
        errors="coerce"
    )

    keep = y.notna().all(axis=1)

    X = X.loc[keep]
    y = y.loc[keep]

    if len(X) < ML_MIN_ROWS:
        return (
            None,
            "Not enough games with valid scoring targets."
        )

    models = {
        "Ridge": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=25.0)
        ),

        "Extra Trees": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=55,
                max_depth=11,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=42
            )
        ),

        "Histogram Boosting": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=65,
                max_leaf_nodes=12,
                learning_rate=0.07,
                min_samples_leaf=15,
                random_state=42
            )
        ),
    }

    fitted = {}

    for name, factory in models.items():
        fitted[name] = []

        for target in targets:
            model = factory()
            model.fit(X, y[target])
            fitted[name].append(model)

    return {
        "models": fitted,
        "features": selected,
        "rows": len(X)
    }, None


# ============================================================
# CURRENT TEAM STATISTICAL PROFILE
# ============================================================

def current_profile(catalog, acronym):
    row = catalog.loc[
        str(acronym).strip().upper()
    ]

    gp = (
        n(row.get("Win", np.nan))
        + n(row.get("Loss", np.nan))
    )

    if pd.notna(gp) and gp > 0:
        gp = float(gp)
    else:
        gp = np.nan

    record = {}

    for prefix in ("TEAM_PRE_", "OPP_PRE_"):
        for name, column in ML_FEATURE_MAP.items():
            feature = prefix + name

            if name == "GP":
                value = gp

            elif column is None:
                value = np.nan

            else:
                value = n(
                    row.get(column, np.nan)
                )

                if (
                    name in ML_PER_GAME_FIELDS
                    and pd.notna(value)
                    and np.isfinite(gp)
                ):
                    value = value / gp

            record[feature] = value

    return record


# ============================================================
# ML MATCHUP PREDICTIONS
# ============================================================

def ml_predict_matchup(ml, catalog, acr1, acr2):
    profile1 = current_profile(
        catalog, acr1
    )

    profile2 = current_profile(
        catalog, acr2
    )

    def row_for(a, b):
        return pd.DataFrame([{
            f: (
                a.get(f, np.nan) if f.startswith("TEAM_PRE_")
                else b.get("TEAM_PRE_" + f[len("OPP_PRE_"):], np.nan)
            )
            for f in ml["features"]
        }])

    x1 = row_for(profile1, profile2)
    x2 = row_for(profile2, profile1)

    individual = {}

    for name, (
        points_model,
        allowed_model
    ) in ml["models"].items():

        t1 = float(
            points_model.predict(x1)[0]
        )

        t2_allow = float(
            allowed_model.predict(x2)[0]
        )

        t2 = float(
            points_model.predict(x2)[0]
        )

        t1_allow = float(
            allowed_model.predict(x1)[0]
        )

        individual[name] = (
            TEAM_SCORE_SHARE * t1 + OPP_DEFENSE_SHARE * t2_allow,
            TEAM_SCORE_SHARE * t2 + OPP_DEFENSE_SHARE * t1_allow
        )

    return {
        "p1": float(
            np.mean([
                v[0] for v in individual.values()
            ])
        ),
        "p2": float(
            np.mean([
                v[1] for v in individual.values()
            ])
        ),
        "models": individual
    }


def show_probability(label, p):
    if np.isfinite(p):
        return f"{label}: {p:.1%}"
    return f"{label}: N/A"


# ============================================================
# BETTING ODDS AND PROBABILITY TABLES
# ============================================================

def valid_odds(value):
    """American odds, including missing NA / blank cells."""
    try:
        x = float(value)
        return x if np.isfinite(x) and x != 0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def american_implied(odds):
    odds = valid_odds(odds)
    if not np.isfinite(odds):
        return np.nan
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (-odds + 100.0)


def edge_background(value):
    """Difference in percentage points. +/-3 is white; +/-20 saturates."""
    if not np.isfinite(value) or abs(value) <= 3:
        return "background-color: #ffffff; color: #111111"
    strength = min(1.0, (abs(value) - 3.0) / 17.0)
    base = (255, 255, 255)
    target = (34, 153, 84) if value > 0 else (207, 52, 52)
    rgb = tuple(round(base[j] + strength * (target[j] - base[j])) for j in range(3))
    font = "#ffffff" if strength > 0.72 else "#111111"
    return f"background-color: rgb{rgb}; color: {font}; font-weight: bold"


def model_win_background(probability):
    """0% deep red, 50% gray, 100% deep green."""
    if not np.isfinite(probability):
        return ""
    p = float(np.clip(probability, 0, 1))
    neutral = (215, 215, 215)
    target = (35, 155, 85) if p >= 0.5 else (205, 65, 65)
    t = abs(p - 0.5) * 2.0
    color = tuple(round(neutral[k] + t * (target[k] - neutral[k])) for k in range(3))
    fg = "#ffffff" if t > 0.75 else "#111111"
    return f"background-color: rgb{color}; color: {fg}; font-weight: bold"


def betting_table(rows, shade_model=False):
    """Rows contain selection, odds, market probability, model probability."""
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df["Difference (pp)"] = (
        df["Model Probability"] - df["Implied Probability"]
    ) * 100
    fmt = {
        "Implied Probability": lambda x: f"{x:.1%}" if pd.notna(x) else "N/A",
        "Model Probability": lambda x: f"{x:.1%}" if pd.notna(x) else "N/A",
        "Difference (pp)": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
    }
    styled = df.style.format(fmt, na_rep="N/A").map(
        edge_background, subset=["Difference (pp)"]
    )
    if shade_model:
        styled = styled.map(model_win_background, subset=["Model Probability"])
    st.dataframe(styled, hide_index=True, use_container_width=True)


# ============================================================
# HISTORICAL +5/+10, -5/-10 MARGIN ANALYSIS
# ============================================================

def matchup_margin_history(games, side, ctx, column):
    """Current-season-weighted margins for this team/coach/context.

    Repeated contextual rows are not duplicated. Team and coach matching
    provide extra importance; all calculations are retrospective, not ML features.
    """
    if games.empty or column not in games.columns:
        return None
    mask_team = games["TEAM"].eq(ctx["TEAM"])
    mask_coach = games["Coach"].eq(ctx["Coach"]) if ctx["Coach"] else pd.Series(False, index=games.index)
    chosen = games.loc[mask_team | mask_coach].copy()
    if chosen.empty:
        return None
    team_match = chosen["TEAM"].eq(ctx["TEAM"]).to_numpy()
    coach_match = chosen["Coach"].eq(ctx["Coach"]).to_numpy() if ctx["Coach"] else np.zeros(len(chosen), dtype=bool)
    weights = chosen["_WEIGHT"].to_numpy(dtype=float) * (
        1.0 + 1.5 * team_match + 0.4 * coach_match + 1.2 * (team_match & coach_match)
    )
    if ctx["HAN"] != "UNKNOWN":
        weights *= np.where(chosen["HAN"].eq(ctx["HAN"]).to_numpy(), 1.3, 1.0)
    values = n(chosen[column]).to_numpy(dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(good):
        return None
    values, weights = values[good], weights[good]
    effective = float(weights.sum()**2 / np.sum(weights**2))
    return values, weights, effective


def margin_thresholds(history):
    if history is None:
        return None
    values, weights, effective = history
    return {
        "Hit +5": float(np.average(values >= 5.0, weights=weights)),
        "Miss -5": float(np.average(values <= -5.0, weights=weights)),
        "Hit +10": float(np.average(values >= 10.0, weights=weights)),
        "Miss -10": float(np.average(values <= -10.0, weights=weights)),
        "Effective games": effective,
        "Rows": len(values)
    }


def historical_cover_rate_near_line(games, ctx, signed_spread):
    """Time-weighted ATS performance near today's spread, if available."""
    if games.empty:
        return np.nan, 0.0
    sub = games.loc[games["TEAM"] == ctx["TEAM"]].copy()
    if sub.empty:
        return np.nan, 0.0
    sub["SPREAD"] = n(sub["SPREAD"])
    sub = sub.loc[
        sub["COVER_RESULT"].notna()
        & sub["SPREAD"].notna()
        & ((sub["SPREAD"] - signed_spread).abs() <= 5)
    ]
    if sub.empty:
        return np.nan, 0.0
    weights = sub["_WEIGHT"].to_numpy(dtype=float)
    if ctx["HAN"] != "UNKNOWN":
        weights *= np.where(sub["HAN"].eq(ctx["HAN"]).to_numpy(), 1.3, 1.0)
    outcome = sub["COVER_RESULT"].to_numpy(dtype=float)
    effective = weights.sum() ** 2 / np.sum(weights**2)
    return float(np.average(outcome, weights=weights)), float(effective)


def empirical_cover_probability(first, second, ctx1, ctx2, margin, signed, sd_margin, games):
    """Model-margin-aligned historical tails, with a normal fallback.

    Historic win margins and historic cover margins capture asymmetry. To
    avoid confusing win and cover, the actual upcoming spread enters this
    threshold separately. Historical bet performance only adjusts the tail.
    """
    baseline = normal_cdf((margin + signed) / sd_margin)
    hist1 = matchup_margin_history(games, first, ctx1, "WIN_MARGIN")
    hist2 = matchup_margin_history(games, second, ctx2, "WIN_MARGIN")
    if hist1 is None and hist2 is None:
        return baseline

    # Align each past margin distribution to the predicted matchup margin.
    residual_values, residual_weights = [], []
    for hist, reverse in ((hist1, False), (hist2, True)):
        if hist is None:
            continue
        v, w, _ = hist
        v = -v if reverse else v
        centered = v - np.average(v, weights=w)
        residual_values.append(centered)
        residual_weights.append(w / w.sum())
    if not residual_values:
        return baseline
    v = np.concatenate(residual_values)
    w = np.concatenate(residual_weights)
    # Logistic smoothing prevents impossible 0%/100% from tiny samples.
    bandwidth = max(2.0, 0.25 * sd_margin)
    z = np.clip((margin + signed + v) / bandwidth, -30, 30)
    empirical = float(np.average(1.0 / (1.0 + np.exp(-z)), weights=w))
    samples = sum(h[2] for h in (hist1, hist2) if h is not None)
    trust = 0.65 * samples / (samples + 15.0)
    result = (1 - trust) * baseline + trust * empirical

    # Historical ATS success is separately informative from winning outright.
    # Limit this calibration term to 15% and only when similar historical
    # spreads exist. This preserves the matchup margin as the primary driver.
    ats1, effective1 = historical_cover_rate_near_line(games, ctx1, signed)
    if np.isfinite(ats1):
        ats_share = 0.15 * effective1 / (effective1 + 12.0)
        result = (1 - ats_share) * result + ats_share * ats1
    return float(np.clip(result, 0.005, 0.995))


# ============================================================
# CURRENT-SEASON RESULTS (DISPLAY ONLY)
# ============================================================

def current_year_result_table(current_games, team_name, acronym):
    """Completed COMP_STAT games for one team, with betting result colors."""
    if current_games is None or current_games.empty:
        return pd.DataFrame()

    team_key = key_text(team_name)
    acr_key = str(acronym).strip().upper()
    selected = current_games.loc[current_games["TEAM"].eq(team_key)].copy()
    if "TEAM_ACR" in current_games.columns:
        by_acronym = current_games["TEAM_ACR"].astype(str).str.strip().str.upper().eq(acr_key)
        selected = current_games.loc[current_games["TEAM"].eq(team_key) | by_acronym].copy()
    if selected.empty:
        return pd.DataFrame()

    selected = selected.sort_values("_DATE", ascending=False)
    rows = []

    def number_text(value, signed=False):
        value = n(value)
        if pd.isna(value):
            return "N/A"
        return f"{float(value):+g}" if signed else f"{float(value):g}"

    for _, game in selected.iterrows():
        points = n(game.get("PTS", np.nan))
        opponent_points = n(game.get("OPP_PTS", np.nan))
        if pd.isna(points) or pd.isna(opponent_points):
            continue

        margin = float(points - opponent_points)
        final_score = f"{float(points):g}–{float(opponent_points):g}"
        moneyline = game.get("ML", np.nan)
        spread = n(game.get("SPREAD", np.nan))
        total_line = n(game.get("TOTAL_LINE", np.nan))
        moneyline_available = np.isfinite(valid_odds(moneyline))

        # For ML, the game score determines whether the team won.
        # A missing historical ML remains uncolored / N/A.
        ml_result = ("W" if margin > 0 else "L") if moneyline_available else "N/A"
        if margin == 0 and moneyline_available:
            ml_result = "Push"

        if pd.notna(spread):
            spread_margin = margin + float(spread)
            spread_result = "W" if spread_margin > 0 else "L" if spread_margin < 0 else "Push"
        else:
            spread_result = "N/A"

        # O/U is explicitly shown from the perspective of an OVER bet.
        # Green means OVER, red means UNDER, gray means PUSH.
        if pd.notna(total_line):
            total_margin = float(points + opponent_points - total_line)
            over_result = "W" if total_margin > 0 else "L" if total_margin < 0 else "Push"
        else:
            over_result = "N/A"

        game_date = game.get("_DATE", pd.NaT)
        date_label = game_date.strftime("%m/%d/%y") if pd.notna(game_date) else "N/A"
        rows.append({
            "Date": date_label,
            "Opponent": str(game.get("OPP", "")).strip().title(),
            "HAN": str(game.get("HAN", "")).upper(),
            "Final (Team–Opp)": final_score,
            "ML Odds": number_text(moneyline, signed=True) if moneyline_available else "N/A",
            "ML Result": ml_result,
            "Spread": number_text(spread, signed=True),
            "Spread Result": spread_result,
            "O/U Line": number_text(total_line),
            "Over Result": over_result,
        })

    return pd.DataFrame(rows)


def color_current_result(value):
    if value == "W":
        return "background-color: #c9efd2; color: #14532d; font-weight: bold"
    if value == "L":
        return "background-color: #f7c8c8; color: #7f1d1d; font-weight: bold"
    if value == "Push":
        return "background-color: #e5e7eb; color: #374151; font-weight: bold"
    return ""


def show_current_year_results(current_games, team_name, acronym):
    st.markdown(f"**{team_name}**")
    table = current_year_result_table(current_games, team_name, acronym)
    if table.empty:
        st.caption("No completed 2026–27 games before the databook date.")
        return
    def shade_result_cells(row):
        styles = ["" for _ in row.index]
        for result_col, value_col in (
            ("ML Result", "ML Odds"),
            ("Spread Result", "Spread"),
            ("Over Result", "O/U Line"),
        ):
            color = color_current_result(row[result_col])
            styles[row.index.get_loc(result_col)] = color
            styles[row.index.get_loc(value_col)] = color
        return styles

    styled = table.style.apply(shade_result_cells, axis=1)
    st.dataframe(styled, hide_index=True, use_container_width=True)


# ============================================================
# V3 — TEN-OBJECTIVE PREGAME REGRESSIONS / JOINT SIMULATION
# ============================================================
# The ten heads fit two predicted scores jointly using different smooth
# objectives.  Binary-event heads use logistic loss, not a postgame feature.
# Every feature is known before tipoff. This is NOT an ROI guarantee.

from scipy.optimize import minimize
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
import zlib

V3_WEIGHTS = {
    "Regression objectives": 0.42,
    "Team/coach history": 0.33,
    "Comparable odds": 0.10,
    "Supervised correction": 0.10,
    "Betting tendencies": 0.05,
}
V3_OBJECTIVES = [
    ("ML outcome", ("ML",), 7),
    ("Spread outcome", ("ATS",), 7),
    ("Total outcome", ("OU",), 7),
    ("ML + Spread", ("ML", "ATS"), 10),
    ("ML + Total", ("ML", "OU"), 10),
    ("Spread + Total", ("ATS", "OU"), 10),
    ("All 3", ("ML", "ATS", "OU"), 15),
    ("Score error", ("SCORE",), 17),
    ("Exact rounded score", ("EXACT",), 5),
    ("Avoid 5+ point misses", ("LARGE",), 12),
]
V3_MIN_TRAIN = 90
V3_MAX_TRAIN = 5500  # cap regression optimization to keep Streamlit responsive
V3_MAX_FEATURES = 55
V3_SIMULATIONS = 7000

# Only allow pregame fields from game-level data; no game box-score columns.
V3_CORE_FIELDS = [
    "PTS", "FGM", "FGA", "FG_PERC", "FG3M", "FG3A", "FG3_PERC",
    "FTM", "FTA", "FT_PERC", "REB", "AST", "TO", "STL", "BLKS",
    "OREB", "DREB", "PF", "FOUL_TROUBLE", "PTS_PERC", "FGA_PERC",
    "REB_PERC", "AST_PERC", "TO_PERC", "OREB_PERC",
]
V3_CLUTCH_FIELDS = [
    "FGM", "FGA", "FGPERC", "FG3M", "FG3A", "FG3PERC", "FTM",
    "FTA", "FTPERC", "OPP_FGM", "OPP_FGA", "OPP_FGPERC",
    "SM", "REB", "OPP_REB", "OREB", "TURN", "OPP_TURN", "STLS",
]


def v3_training_frame(games):
    """Completed current-season team-game observations with valid pregame pace."""
    if games is None or games.empty:
        return pd.DataFrame()
    needed = ("TEAM_PRE_POSS", "OPP_PRE_POSS", "PTS", "OPP_PTS")
    if any(k not in games.columns for k in needed):
        return pd.DataFrame()
    work = games.copy()
    for k in needed:
        work[k] = n(work[k])
    work = work.loc[(work.TEAM_PRE_POSS > 0) & (work.OPP_PRE_POSS > 0)
                    & (work.PTS > 0) & (work.OPP_PTS > 0)].copy()
    if work.empty:
        return work
    work["_SPREAD_V3"] = n(work["SPREAD"]) if "SPREAD" in work else np.nan
    work["_TOTAL_V3"] = n(work["TOTAL_LINE"]) if "TOTAL_LINE" in work else np.nan
    work["_ML_V3"] = n(work["ML"]) if "ML" in work else np.nan
    # Game-level CORE and CLUTCH columns must be pregame snapshots in COMP_STAT.
    # Neither team can contribute clutch features unless both have played clutch possessions.
    team_cl = n(work.get("TEAM_CLUTCH_FGM", pd.Series(np.nan, index=work.index)))
    opp_cl = n(work.get("OPP_PRE_CLUTCH_FGM", pd.Series(np.nan, index=work.index)))
    work["_CLUTCH_OK"] = (team_cl > 0) & (opp_cl > 0)
    return work.sort_values("_DATE").copy()


def v3_feature_columns(frame):
    columns = []
    allowed_prefixes = ("TEAM_PRE_", "OPP_PRE_", "CORE_", "OPP_CORE_",
                        "TEAM_CLUTCH_", "OPP_PRE_CLUTCH_")
    for col in frame.columns:
        if not col.startswith(allowed_prefixes):
            continue
        # Exclude outcome/event indicators, narrative or name fields if present.
        if col in {"TEAM_PRE_CLUTCH_GAMES", "OPP_PRE_CLUTCH_GAMES"}:
            pass
        if "CLUTCH" in col and col.split("_")[-1] not in V3_CLUTCH_FIELDS and not col.endswith("CLUTCH_GAMES"):
            continue
        if col.startswith("CORE_") and col[5:] not in V3_CORE_FIELDS:
            continue
        if col.startswith("OPP_CORE_") and col[9:] not in V3_CORE_FIELDS:
            continue
        arr = n(frame[col]).replace([np.inf, -np.inf], np.nan)
        if arr.notna().sum() < max(30, int(len(frame) * 0.12)) or arr.nunique(dropna=True) < 3:
            continue
        columns.append(col)
    # Restrict redundant variables to maintain computational performance.
    priorities = ["POSS", "PTS_FOR", "PTS_ALLOW", "OFF_EFF", "DEF_EFF", "NET_EFF",
                  "EFG", "TS_", "TURN_RATE", "3P_PCT", "OREB", "DREB", "REB_SHARE",
                  "AST_TO", "EXTRA", "WIN_PCT", "CORE_PTS", "CORE_FGA", "CLUTCH"]
    def priority(c):
        try:
            return min(i for i, term in enumerate(priorities) if term in c)
        except ValueError:
            return len(priorities)
    columns.sort(key=lambda c: (priority(c), c))
    # Reserve slots for opponent, core and eligible clutch context. Otherwise
    # 50 correlated team shooting/efficiency columns can crowd them all out.
    buckets = {
        "team": [c for c in columns if c.startswith("TEAM_PRE_")],
        "opp": [c for c in columns if c.startswith("OPP_PRE_") and "CLUTCH" not in c],
        "core": [c for c in columns if c.startswith(("CORE_", "OPP_CORE_"))],
        "clutch": [c for c in columns if "CLUTCH" in c],
    }
    picked = []
    for bucket, limit in (("team", 20), ("opp", 18), ("core", 10), ("clutch", 7)):
        picked.extend(c for c in buckets[bucket][:limit] if c not in picked)
    picked.extend(c for c in columns if c not in picked)
    # Market and location features belong to the pregame data, never the target.
    return picked[:V3_MAX_FEATURES]


def v3_feature_matrix(frame, columns):
    data = pd.DataFrame(index=frame.index)
    for col in columns:
        x = n(frame[col]) if col in frame else pd.Series(np.nan, index=frame.index)
        if "CLUTCH" in col and "_CLUTCH_OK" in frame:
            x = x.where(frame["_CLUTCH_OK"])
        data[col] = x
    for col in ("_SPREAD_V3", "_TOTAL_V3", "_ML_V3"):
        data[col] = n(frame[col]) if col in frame else np.nan
    if "HAN" in frame:
        for v in ("HOME", "AWAY", "NEUTRAL"):
            data["VENUE_" + v] = frame["HAN"].astype(str).str.upper().eq(v).astype(float)
    else:
        for v in ("HOME", "AWAY", "NEUTRAL"):
            data["VENUE_" + v] = np.nan
    return data.replace([np.inf, -np.inf], np.nan)


def v3_profile_value(row, suffix, gp):
    """Translate All_Stats season-to-date into game-level PRE features."""
    if suffix == "GP":
        return gp
    mapped = ML_FEATURE_MAP.get(suffix)
    if mapped is None:
        # Named pregame fields not present in ML_FEATURE_MAP are unavailable,
        # rather than incorrectly assuming a similar box-score column.
        if suffix == "POSS":
            mapped = "EST_POSS"
        elif suffix == "CLUTCH_GAMES":
            mapped = "Clutch_Games"
        else:
            return np.nan
    value = n(row.get(mapped, np.nan))
    if pd.isna(value):
        return np.nan
    if suffix in ML_PER_GAME_FIELDS and np.isfinite(gp) and gp > 0:
        value = value / gp
    if suffix == "POSS" and np.isfinite(gp) and gp > 0 and value > 140:
        value = value / gp
    return float(value)


def v3_current_features(catalog, acr1, acr2, columns, spread1, total_line, ml1, han):
    first, second = catalog.loc[acr1.upper()], catalog.loc[acr2.upper()]
    def gp_of(row):
        w, l = n(row.get("Win", np.nan)), n(row.get("Loss", np.nan))
        return float(w + l) if pd.notna(w) and pd.notna(l) and w + l > 0 else np.nan
    gp1, gp2 = gp_of(first), gp_of(second)
    cl1 = n(first.get("CLUTCH_FGM", np.nan))
    cl2 = n(second.get("CLUTCH_FGM", np.nan))
    clutch_ok = pd.notna(cl1) and pd.notna(cl2) and cl1 > 0 and cl2 > 0
    result = {}
    for col in columns:
        if col.startswith("TEAM_PRE_"):
            result[col] = v3_profile_value(first, col[9:], gp1)
        elif col.startswith("OPP_PRE_CLUTCH_"):
            field = col[len("OPP_PRE_CLUTCH_"):]
            result[col] = n(second.get("CLUTCH_" + field, np.nan)) if clutch_ok else np.nan
        elif col.startswith("OPP_PRE_"):
            result[col] = v3_profile_value(second, col[8:], gp2)
        elif col.startswith("OPP_CORE_"):
            result[col] = n(second.get("CORE_" + col[len("OPP_CORE_"):], np.nan))
        elif col.startswith("CORE_"):
            result[col] = n(first.get(col, np.nan))
        elif col.startswith("TEAM_CLUTCH_"):
            result[col] = n(first.get("CLUTCH_" + col[len("TEAM_CLUTCH_"):], np.nan)) if clutch_ok else np.nan
    result.update({"_SPREAD_V3": spread1, "_TOTAL_V3": total_line,
                   "_ML_V3": ml1, "VENUE_HOME": float(han == "HOME"),
                   "VENUE_AWAY": float(han == "AWAY"),
                   "VENUE_NEUTRAL": float(han == "NEUTRAL")})
    return result


def v3_head_loss(theta, X, y, spread, total, recency, objectives, ridge=0.035):
    """Differentiable two-score objective with exact analytical gradient."""
    d = X.shape[1]
    w = theta.reshape(d, 2)
    pred = X @ w + 70.0  # intercept captured by first X column
    residual = pred - y
    grad_pred = np.zeros_like(pred)
    # Score anchor keeps all event-focused models anchored to realistic scores.
    scale = 20.0
    if "SCORE" in objectives:
        alpha = 1.0
    elif "EXACT" in objectives:
        alpha = 1.4
    elif "LARGE" in objectives:
        alpha = 0.9
    else:
        alpha = 0.42
    r = residual / scale
    if "SCORE" in objectives:
        # Pseudo-Huber is robust to lopsided blowouts.
        q = np.sqrt(1.0 + r * r)
        loss_rows = alpha * np.sum(q - 1.0, axis=1)
        grad_pred += alpha * r / (scale * q)
    elif "EXACT" in objectives:
        # Smooth near-exact-score surrogate. Rounding itself is discontinuous.
        q = np.sqrt(1.0 + (residual / 5.0)**2)
        loss_rows = alpha * np.sum(q - 1, axis=1)
        grad_pred += alpha * residual / (25.0 * q)
    else:
        loss_rows = 0.5 * alpha * np.sum(r * r, axis=1)
        grad_pred += alpha * residual / (scale**2)
    margin = pred[:, 0] - pred[:, 1]
    actual_margin = y[:, 0] - y[:, 1]
    actual_total = y.sum(axis=1)
    for category in objectives:
        if category not in ("ML", "ATS", "OU"):
            continue
        if category == "ML":
            actual = actual_margin
            estimated = margin
            denom = 10.0
            valid = np.ones(len(y), dtype=bool)
        elif category == "ATS":
            actual = actual_margin + spread
            estimated = margin + spread
            denom = 11.0
            valid = np.isfinite(spread)
        else:
            actual = actual_total - total
            estimated = pred.sum(axis=1) - total
            denom = 13.0
            valid = np.isfinite(total)
        sign = np.sign(actual)
        valid = valid & (sign != 0) & np.isfinite(actual)
        signed_logit = np.where(valid, sign * estimated / denom, 0.0)
        # BCE loss, stable even for an extreme projected score.
        loss_rows += 4.0 * np.where(valid, np.logaddexp(0.0, -signed_logit), 0.0)
        derivative = np.where(valid, -4.0 * sign * expit(-signed_logit) / denom, 0.0)
        if category == "OU":
            grad_pred[:, 0] += derivative
            grad_pred[:, 1] += derivative
        else:
            grad_pred[:, 0] += derivative
            grad_pred[:, 1] -= derivative
    if "LARGE" in objectives:
        # Stronger cost for mistakes that change predicted margin/total by >=5.
        dif_m = margin - actual_margin
        dif_t = pred.sum(axis=1) - actual_total
        pm = np.maximum(0.0, np.abs(dif_m) - 5.0)
        pt = np.maximum(0.0, np.abs(dif_t) - 5.0)
        loss_rows += 0.004 * (pm**2 + pt**2)
        gm = 0.008 * pm * np.sign(dif_m)
        gt = 0.008 * pt * np.sign(dif_t)
        grad_pred[:, 0] += gm + gt
        grad_pred[:, 1] += -gm + gt
    norm = max(float(recency.sum()), 1e-9)
    loss = float(np.dot(loss_rows, recency) / norm)
    grad = X.T @ (grad_pred * (recency / norm)[:, None])
    # Do not regularize the intercept.
    penalty = w[1:, :]
    loss += ridge * float(np.sum(penalty**2))
    grad[1:, :] += 2.0 * ridge * penalty
    return loss, grad.ravel().astype(float)


def v3_chronological_validation(work, feature_cols):
    """Out-of-time check of the ten-head consensus, no later games in training.

    This validates the regression component, not the entire blended model.
    Periodic true walk-forward evaluation of all five components is still needed.
    """
    dates = sorted(work["_DATE"].dropna().unique())
    if len(work) < 240 or len(dates) < 8:
        return None
    cutoff = dates[int(len(dates) * .78)]
    train = work.loc[work["_DATE"] < cutoff].copy()
    test = work.loc[work["_DATE"] >= cutoff].copy()
    if len(train) < 150 or len(test) < 35:
        return None
    feature_train = v3_feature_matrix(train, feature_cols)
    feature_test = v3_feature_matrix(test, feature_cols)
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    scale = StandardScaler()
    X = np.column_stack([np.ones(len(train)), np.clip(scale.fit_transform(imp.fit_transform(feature_train)), -8, 8)])
    Xt = np.column_stack([np.ones(len(test)), np.clip(scale.transform(imp.transform(feature_test)), -8, 8)])
    y = train[["PTS", "OPP_PTS"]].to_numpy(float)
    spreads = train["_SPREAD_V3"].to_numpy(float)
    totals = train["_TOTAL_V3"].to_numpy(float)
    days = (train["_DATE"].max() - train["_DATE"]).dt.days.fillna(0).to_numpy(float)
    weight = 0.40 + 0.60 * np.exp(-np.maximum(days, 0) / 40.0)
    init = np.zeros((X.shape[1], 2))
    init[0] = y.mean(axis=0) - 70.0
    guess = np.zeros((len(test), 2))
    for name, objectives, importance in V3_OBJECTIVES:
        result = minimize(v3_head_loss, init.ravel(),
                          args=(X, y, spreads, totals, weight, objectives),
                          method="L-BFGS-B", jac=True,
                          options={"maxiter": 36, "ftol": 5e-5, "maxls": 12})
        guess += (importance / 100.0) * (Xt @ result.x.reshape(X.shape[1], 2) + 70.0)
    truth = test[["PTS", "OPP_PTS"]].to_numpy(float)
    pred_margin = guess[:, 0] - guess[:, 1]
    actual_margin = truth[:, 0] - truth[:, 1]
    valid_ml = actual_margin != 0
    evaluation = {
        "Train rows": len(train), "Holdout rows": len(test),
        "Holdout from": pd.Timestamp(cutoff).strftime("%Y-%m-%d"),
        "Score MAE": round(float(np.abs(guess - truth).mean()), 2),
        "ML accuracy": round(float(np.mean((pred_margin[valid_ml] > 0) ==
                                               (actual_margin[valid_ml] > 0))), 3) if valid_ml.any() else np.nan,
    }
    sp = test["_SPREAD_V3"].to_numpy(float)
    mask = np.isfinite(sp) & (actual_margin + sp != 0)
    evaluation["Spread accuracy"] = (
        round(float(np.mean((pred_margin[mask] + sp[mask] > 0) ==
                            (actual_margin[mask] + sp[mask] > 0))), 3)
        if mask.any() else np.nan)
    line = test["_TOTAL_V3"].to_numpy(float)
    mask = np.isfinite(line) & (truth.sum(axis=1) != line)
    evaluation["Total accuracy"] = (
        round(float(np.mean((guess[mask].sum(axis=1) > line[mask]) ==
                            (truth[mask].sum(axis=1) > line[mask]))), 3)
        if mask.any() else np.nan)
    return evaluation


@st.cache_resource(show_spinner=False)
def v3_train_engine(signature, frame):
    work = v3_training_frame(frame)
    if len(work) < V3_MIN_TRAIN:
        return None, f"{len(work)} eligible rows (need at least {V3_MIN_TRAIN})."
    if len(work) > V3_MAX_TRAIN:
        # Most recent examples win when the training sample is capped.
        work = work.tail(V3_MAX_TRAIN).copy()
    feat = v3_feature_columns(work)
    if len(feat) < 8:
        return None, "Not enough populated pregame features for regression."
    # Date-held-out check is fitted only on earlier games; displayed separately.
    validation = v3_chronological_validation(work, feat)
    xf = v3_feature_matrix(work, feat)
    features = list(xf.columns)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(xf))
    X = np.column_stack([np.ones(len(X)), np.clip(X, -8, 8)])
    y = work[["PTS", "OPP_PTS"]].to_numpy(float)
    spread = work["_SPREAD_V3"].to_numpy(float)
    total = work["_TOTAL_V3"].to_numpy(float)
    dates = work["_DATE"].max() - work["_DATE"]
    age_days = dates.dt.days.fillna(0).clip(lower=0).to_numpy(float)
    recency = 0.40 + 0.60 * np.exp(-age_days / 40.0)
    # Initial coefficients approximate the unconditional scoring mean.
    x0 = np.zeros((X.shape[1], 2))
    x0[0] = np.mean(y, axis=0) - 70.0
    heads, diagnostics = {}, []
    for label, obj, importance in V3_OBJECTIVES:
        result = minimize(v3_head_loss, x0.ravel(),
                          args=(X, y, spread, total, recency, obj),
                          method="L-BFGS-B", jac=True,
                          options={"maxiter": 48, "ftol": 2e-5, "maxls": 12})
        heads[label] = result.x.reshape(X.shape[1], 2).copy()
        diagnostics.append({"Objective": label, "Share of 10 heads": importance / 100,
                            "Converged": bool(result.success)})
    # A separate limited-complexity supervised correction learner.
    meta_models = []
    # Fit residual targets against the mean objective prediction; the resulting
    # forecast is anchored to primary regressions and not another independent score.
    primary = sum((imp / 100.0) * (X @ heads[name] + 70.0)
                  for name, _, imp in V3_OBJECTIVES)
    for side in range(2):
        meta = HistGradientBoostingRegressor(
            max_iter=45, learning_rate=0.055, max_leaf_nodes=9,
            min_samples_leaf=max(12, int(len(X) * 0.008)),
            l2_regularization=12.0, random_state=42)
        meta.fit(X[:, 1:], np.clip(y[:, side] - primary[:, side], -28, 28))
        meta_models.append(meta)
    residuals = y - primary
    # Empirical residuals from fitted training data provide provisional
    # variability, NOT calibrated out-of-sample 95% coverage.
    spread_sd = np.std(residuals, axis=0, ddof=1)
    spread_sd = np.clip(spread_sd, 9.0, 24.0)
    corr = np.corrcoef(residuals.T)[0, 1] if len(residuals) > 5 else 0.15
    corr = float(np.clip(corr if np.isfinite(corr) else 0.15, -0.25, 0.55))
    return {
        "features": feat, "matrix_columns": features,
        "imputer": imputer, "scaler": scaler, "heads": heads,
        "meta": meta_models, "sd": spread_sd, "corr": corr,
        "rows": len(work), "diagnostics": pd.DataFrame(diagnostics),
        "validation": validation,
    }, None


def v3_predict_regressions(engine, catalog, acr1, acr2, ctx1, ctx2, fixture):
    def side(acr, opp, ctx, ml_col, signed):
        src = v3_current_features(catalog, acr, opp, engine["features"],
                                  signed, n(fixture.get("OverUnder", np.nan)),
                                  n(fixture.get(ml_col, np.nan)), ctx["HAN"])
        xf = pd.DataFrame([{col: src.get(col, np.nan)
                            for col in engine["matrix_columns"]}])
        X = engine["scaler"].transform(engine["imputer"].transform(xf))
        X = np.column_stack([np.ones(len(X)), np.clip(X, -8, 8)])
        forecasts = {}
        for name, _, _ in V3_OBJECTIVES:
            forecasts[name] = (X @ engine["heads"][name] + 70.0)[0]
        primary = sum((imp / 100.0) * forecasts[name]
                      for name, _, imp in V3_OBJECTIVES)
        correction = np.array([m.predict(X[:, 1:])[0] for m in engine["meta"]])
        return primary, correction, forecasts
    favorite = key_text(fixture.get("Favorite", ""))
    magnitude = n(fixture.get("Spread", np.nan))
    if pd.isna(magnitude):
        signed1 = signed2 = np.nan
    elif favorite in (ctx1["TEAM"], acr1.casefold()):
        signed1, signed2 = -abs(float(magnitude)), abs(float(magnitude))
    elif favorite in (ctx2["TEAM"], acr2.casefold()):
        signed1, signed2 = abs(float(magnitude)), -abs(float(magnitude))
    else:
        signed1 = signed2 = np.nan
    a, ma, heads_a = side(acr1, acr2, ctx1, "ML1", signed1)
    b, mb, heads_b = side(acr2, acr1, ctx2, "ML2", signed2)
    # A team's own scoring model gets 75%, opposing allowed-score gets 25%.
    p1 = 0.75 * a[0] + 0.25 * b[1]
    p2 = 0.75 * b[0] + 0.25 * a[1]
    meta1 = 0.75 * ma[0] + 0.25 * mb[1]
    meta2 = 0.75 * mb[0] + 0.25 * ma[1]
    detail = []
    for name, _, importance in V3_OBJECTIVES:
        h1, h2 = heads_a[name], heads_b[name]
        detail.append({"Objective": name, "Weight (%)": importance,
                       ctx1["NAME"]: round(float(.75 * h1[0] + .25 * h2[1]), 1),
                       ctx2["NAME"]: round(float(.75 * h2[0] + .25 * h1[1]), 1)})
    return {"regression": np.array([p1, p2]),
            "meta": np.array([meta1, meta2]), "detail": pd.DataFrame(detail)}


def v3_odds_estimate(current_games, fixture, ctx1, ctx2):
    """Within-five comparable spread/total markets, recency-smoothed."""
    if current_games is None or current_games.empty:
        return None, pd.DataFrame()
    spread = n(fixture.get("Spread", np.nan))
    total = n(fixture.get("OverUnder", np.nan))
    favorite = key_text(fixture.get("Favorite", ""))
    if pd.isna(spread) or pd.isna(total) or favorite not in {
        ctx1["TEAM"], ctx2["TEAM"], key_text(ctx1["NAME"]), key_text(ctx2["NAME"])
    }:
        return None, pd.DataFrame()
    favorite_is_first = favorite in {ctx1["TEAM"], key_text(ctx1["NAME"])}
    magnitude = abs(float(spread))
    # Comparables are games, not team-game rows: do not count both team sides
    # twice when COMP_STAT contains one observation per team.
    src = current_games.drop_duplicates("GAME_ID") if "GAME_ID" in current_games else current_games
    if "SPREAD" not in src or "TOTAL_LINE" not in src:
        return None, pd.DataFrame()
    ss = n(src["SPREAD"])
    tt = n(src["TOTAL_LINE"])
    close_spread = (ss.abs() - magnitude).abs() <= 5.0
    close_total = (tt - float(total)).abs() <= 5.0
    subsets = [("Similar spread", close_spread), ("Similar total", close_total),
               ("Both within 5", close_spread & close_total)]
    rows = []
    comps = []
    for name, mask in subsets:
        g = src.loc[mask & ss.notna() & tt.notna()].copy()
        if g.empty:
            rows.append({"Comparison": name, "Games": 0})
            continue
        ws = g["_WEIGHT"].to_numpy(float)
        diff = (n(g["PTS"]) - n(g["OPP_PTS"])).to_numpy(float)
        actual_market = (-n(g["SPREAD"])).to_numpy(float)
        total_resid = (n(g["PTS"]) + n(g["OPP_PTS"]) - n(g["TOTAL_LINE"])).to_numpy(float)
        mask_valid = np.isfinite(diff) & np.isfinite(actual_market) & np.isfinite(total_resid)
        if not mask_valid.any():
            continue
        ws, diff, actual_market, total_resid = (z[mask_valid] for z in (ws, diff, actual_market, total_resid))
        # Team orientation: favorite's true margin, accounting for signed historical spread.
        fav_margins = np.where(actual_market >= 0, diff, -diff)
        spread_resid = fav_margins - np.abs(actual_market)
        cover = np.average(spread_resid > 0, weights=ws)
        over = np.average(total_resid > 0, weights=ws)
        es = ws.sum()**2 / np.square(ws).sum()
        rows.append({"Comparison": name, "Games": len(ws),
                     "Favorite win": round(float(np.average(fav_margins > 0, weights=ws)), 3),
                     "Underdog win": round(float(np.average(fav_margins < 0, weights=ws)), 3),
                     "Favorite cover": round(float(cover), 3),
                     "Underdog cover": round(float(np.average(spread_resid < 0, weights=ws)), 3),
                     "Over": round(float(over), 3),
                     "Under": round(float(np.average(total_resid < 0, weights=ws)), 3),
                     "Spread residual": round(float(np.average(spread_resid, weights=ws)), 1),
                     "Total residual": round(float(np.average(total_resid, weights=ws)), 1)})
        # Shrink residual effects strongly when only a handful of comparables exist.
        weight = (1.5 if name == "Both within 5" else 0.7) * es / (es + 30.0)
        comps.append((weight, float(np.average(spread_resid, weights=ws)),
                      float(np.average(total_resid, weights=ws))))
    if not comps:
        return None, pd.DataFrame(rows)
    mass = sum(q[0] for q in comps)
    adj_spread = np.clip(sum(q[0] * q[1] for q in comps) / mass, -6, 6)
    adj_total = np.clip(sum(q[0] * q[2] for q in comps) / mass, -8, 8)
    # Market anchor adjusts by demonstrated average historical differences.
    m = magnitude + .35 * adj_spread
    if not favorite_is_first:
        m *= -1
    t = float(total) + .35 * adj_total
    return np.array([(t + m) / 2, (t - m) / 2]), pd.DataFrame(rows)


def v3_tendency_adjustment(games, ctx1, ctx2, fixture):
    """Small sample-size-adjusted score shift from performance versus lines."""
    shifts = []
    for ctx in (ctx1, ctx2):
        sub = games.loc[games["TEAM"].eq(ctx["TEAM"])] if not games.empty else pd.DataFrame()
        if sub.empty:
            shifts.append(0.0)
            continue
        w = sub["_WEIGHT"].to_numpy(float)
        spread_delta = n(sub.get("COVER_MARGIN_CALC", pd.Series(np.nan, index=sub.index))).to_numpy(float)
        total_delta = n(sub.get("OVER_MARGIN_CALC", pd.Series(np.nan, index=sub.index))).to_numpy(float)
        effective = w.sum()**2 / np.sum(w**2)
        trust = effective / (effective + 25.0)
        vals = []
        for delta in (spread_delta, total_delta):
            good = np.isfinite(delta)
            if np.any(good):
                vals.append(float(np.average(np.clip(delta[good], -12, 12), weights=w[good])))
        shifts.append(float(np.clip(trust * np.mean(vals) * .25, -4, 4)) if vals else 0.0)
    return np.array(shifts)


def v3_blend_forecasts(historical, modeled, odds, tendency):
    available = {"Team/coach history": np.asarray(historical, dtype=float)}
    if modeled is not None:
        available["Regression objectives"] = modeled["regression"]
        available["Supervised correction"] = modeled["regression"] + np.clip(modeled["meta"], -8, 8)
    if odds is not None:
        available["Comparable odds"] = odds
    if tendency is not None:
        available["Betting tendencies"] = available.get("Team/coach history", historical) + tendency
    valid = {k: v for k, v in available.items() if np.isfinite(v).all()}
    if not valid:
        return np.array([np.nan, np.nan]), {}
    total = sum(V3_WEIGHTS[k] for k in valid)
    shares = {k: V3_WEIGHTS[k] / total for k in valid}
    final = sum(shares[k] * valid[k] for k in valid)
    return np.maximum(0.0, final), shares


def v3_simulate(p1, p2, first, second, engine, games, seed_key):
    """One joint distribution powers ML, ATS, totals and predictive ranges."""
    fallback_sd = np.array([scoring_dispersion(first, second),
                            scoring_dispersion(second, first)])
    if engine is not None:
        model_sd = np.asarray(engine["sd"], dtype=float)
        sd = np.clip(.5 * fallback_sd + .5 * model_sd, 8, 22)
        corr = engine["corr"]
    else:
        sd = np.clip(fallback_sd, 8, 23)
        corr = 0.17
    cov = np.array([[sd[0]**2, corr * sd[0] * sd[1]],
                    [corr * sd[0] * sd[1], sd[1]**2]])
    seed = zlib.crc32(str(seed_key).encode("utf-8"))
    rng = np.random.default_rng(seed)
    scores = rng.multivariate_normal([p1, p2], cov, size=V3_SIMULATIONS)
    # Basketball is scored in whole points, but round only after simulation.
    scores = np.rint(np.maximum(0, scores)).astype(int)
    ties = np.flatnonzero(scores[:, 0] == scores[:, 1])
    if ties.size:
        # NCAA games cannot end tied. An overtime adds scoring to both teams;
        # assign a random 1–7 point OT margin to resolve each simulated tie.
        extra = rng.integers(4, 12, size=ties.size)
        margin = rng.integers(1, 8, size=ties.size)
        winner_first = rng.random(ties.size) < 0.5
        scores[ties, 0] += extra + winner_first * margin
        scores[ties, 1] += extra + (~winner_first) * margin
    return scores


def v3_event_probabilities(scores, spread_signed, ou_line):
    margin = scores[:, 0] - scores[:, 1]
    total = scores.sum(axis=1)
    result = {"ml1": float(np.mean(margin > 0)),
              "ml2": float(np.mean(margin < 0)),
              "ml_push": float(np.mean(margin == 0))}
    if np.isfinite(spread_signed):
        cover = margin + spread_signed
        result.update({"cover1": float(np.mean(cover > 0)),
                       "cover2": float(np.mean(cover < 0)),
                       "spread_push": float(np.mean(cover == 0))})
    if np.isfinite(ou_line):
        over = total - ou_line
        result.update({"over": float(np.mean(over > 0)),
                       "under": float(np.mean(over < 0)),
                       "total_push": float(np.mean(over == 0))})
    return result


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.title("Today's Games — March Metrics")

signatures = tuple(
    (
        key,
        name,
        (DATA / name).stat().st_mtime_ns
        if (DATA / name).is_file()
        else None
    )
    for key, name in FILES.items()
)

sources = read_sources(signatures)

for mandatory in ("date", "schedule", "teams"):
    if sources[mandatory] is None:
        st.error(
            f"Missing required file: "
            f"{DATA / FILES[mandatory]}"
        )
        st.stop()

if (
    sources["historical"] is None
    and sources["season"] is None
):
    st.error(
        "Both historical and current-season "
        "game files are missing."
    )
    st.stop()

try:
    date_df = sources["date"]

    if (
        not {"MM", "DD", "YYYY"}.issubset(date_df.columns)
        or date_df.empty
    ):
        raise ValueError(
            "Current_Day requires MM, DD, YYYY "
            "in its first data row."
        )

    today = date_from_row(
        date_df.iloc[0]
    )

    schedule = sources["schedule"].copy()

    required_schedule = {
        "MM", "DD", "YYYY",
        "Team1", "Team2",
        "Team1Acronym", "Team2Acronym",
        "Home_Team"
    }

    if not required_schedule.issubset(schedule.columns):
        raise ValueError(
            "Schedule is missing date, teams, "
            "acronyms, or Home_Team columns."
        )

    for col in ["MM", "DD", "YYYY"]:
        schedule[col] = n(schedule[col])

    todays = schedule.loc[
        (schedule["MM"] == today.month)
        & (schedule["DD"] == today.day)
        & (schedule["YYYY"] == today.year)
    ].copy()

    catalog = team_catalog(
        sources["teams"]
    )

    past = prepare_games(
        sources["historical"],
        "historical",
        today,
        catalog
    )

    current = prepare_games(
        sources["season"],
        "season",
        today,
        catalog
    )

    if not past.empty or not current.empty:
        games = pd.concat(
            [past, current],
            ignore_index=True
        )

        games["_SOURCE_PRIORITY"] = (
            games["_SOURCE"]
            .eq("season")
            .astype(int)
        )

        games = (
            games.sort_values(
                "_SOURCE_PRIORITY",
                ascending=False
            )
            .drop_duplicates("_KEY")
            .reset_index(drop=True)
        )
    else:
        games = pd.DataFrame()

except (
    ValueError,
    KeyError,
    IndexError,
    TypeError
) as exc:
    st.error(
        f"Unable to prepare game data: {exc}"
    )
    st.stop()


st.caption(
    f"Databook date: {today:%m/%d/%Y} · "
    f"{len(todays)} scheduled games · "
    f"{len(games)} historical team-game rows"
)

st.info(
    "V3: 10 objective-weighted score regressions (42%), team/coach history "
    "(33%), odds comparables (10%), correction ML (10%), tendencies (5%) "
    "when all components are eligible. Only past results are used."
)


# ============================================================
# TRAIN ML ONCE PER DATE / DATA UPDATE
# ============================================================

ml_engine = None
ml_note = "V3 regression system inactive until November 20, 2026."
if today >= ML_START_DATE:
    if current.empty:
        ml_note = "V3 unavailable: no completed COMP_STAT games before databook date."
    else:
        ml_engine, problem = v3_train_engine(
            ((DATA / FILES["season"]).stat().st_mtime_ns, today.isoformat()),
            current,
        )
        if ml_engine is None:
            ml_note = f"V3 regression unavailable: {problem} Historical model remains active."
        else:
            ml_note = (f"V3 active: {ml_engine['rows']} eligible COMP_STAT team-game rows; "
                       "10 regression objectives + a lightweight correction model.")
st.caption(ml_note)

if todays.empty:
    st.info(
        "No games scheduled for the selected date."
    )
    st.stop()


# ============================================================
# ### FUTURE GAME ORDERING BY TEAM RATINGS ###
# ============================================================

# Once Rankings exposes a numeric Team Rating
# keyed by Acronym, sort matchups by:
#
# Team1_Rating + Team2_Rating
#
# Highest combined rating first.
#
# Rankings are not yet built, so sorting remains off.


# ============================================================
# CACHE MATCHUP CONTEXT FORECASTS
# ============================================================

forecast_cache = {}


def predict_team(
    acronym, opp_acronym, home, location
):
    cache_key = (
        str(acronym),
        str(opp_acronym),
        str(home),
        str(location)
    )

    if cache_key not in forecast_cache:
        ctx = make_context(
            acronym,
            opp_acronym,
            home,
            location,
            catalog
        )

        if not games.empty:
            forecast_cache[cache_key] = forecast_side(
                games, ctx
            )
        else:
            forecast_cache[cache_key] = None

    return forecast_cache[cache_key]


# ============================================================
# DISPLAY ALL GAMES
# ============================================================

for _, fixture in todays.iterrows():

    team1 = str(fixture["Team1"]).strip()
    team2 = str(fixture["Team2"]).strip()

    acr1 = str(
        fixture["Team1Acronym"]
    ).strip()

    acr2 = str(
        fixture["Team2Acronym"]
    ).strip()

    with st.container(border=True):

        st.subheader(
            f"{team1} vs {team2}"
        )

        st.caption(
            f"{str(fixture.get('Location', '')).strip()} "
            f"· Home: "
            f"{str(fixture.get('Home_Team', '')).strip()}"
        )

        try:
            first = predict_team(
                acr1,
                acr2,
                fixture["Home_Team"],
                fixture.get("Location", "")
            )

            second = predict_team(
                acr2,
                acr1,
                fixture["Home_Team"],
                fixture.get("Location", "")
            )

        except KeyError:
            st.warning(
                "Team acronym not found in All_Stats. "
                "Cannot forecast this matchup yet."
            )
            continue

        if first is None or second is None:
            st.warning(
                "Insufficient historical data "
                "for one or both teams/contexts."
            )
            continue

        ctx1 = make_context(acr1, acr2, fixture["Home_Team"], fixture.get("Location", ""), catalog)
        ctx2 = make_context(acr2, acr1, fixture["Home_Team"], fixture.get("Location", ""), catalog)

        # ----------------------------------------------------
        # HISTORICAL PREDICTION
        # ----------------------------------------------------

        m = model_matchup(first, second)
        historical = np.array([m["p1"], m["p2"]], dtype=float)
        ml_prediction = None
        if ml_engine is not None:
            try:
                ml_prediction = v3_predict_regressions(
                    ml_engine, catalog, acr1, acr2, ctx1, ctx2, fixture)
            except (KeyError, ValueError, TypeError) as exc:
                st.caption(f"V3 regression unavailable for matchup: {exc}")
        odds_pred, odds_details = v3_odds_estimate(current, fixture, ctx1, ctx2)
        tendencies = v3_tendency_adjustment(games, ctx1, ctx2, fixture)
        blended, active_shares = v3_blend_forecasts(
            historical, ml_prediction, odds_pred, tendencies)
        m["p1"], m["p2"] = map(float, blended)
        if not np.isfinite(blended).all():
            st.warning("Unable to produce a finite score estimate for this matchup.")
            continue
        simulations = v3_simulate(m["p1"], m["p2"], first, second,
                                  ml_engine, games, fixture.get("GameID", team1 + team2))
        range1 = np.quantile(simulations[:, 0], [0.025, 0.975])
        range2 = np.quantile(simulations[:, 1], [0.025, 0.975])
        favorite = key_text(fixture.get("Favorite", ""))
        spread = n(fixture.get("Spread", np.nan))
        total_line = n(fixture.get("OverUnder", np.nan))
        signed = np.nan
        if pd.notna(spread):
            if favorite in {team1.casefold(), acr1.casefold()}:
                signed = -abs(float(spread))
            elif favorite in {team2.casefold(), acr2.casefold()}:
                signed = abs(float(spread))
        probabilities = v3_event_probabilities(
            simulations, signed, float(total_line) if pd.notna(total_line) else np.nan)
        win1 = probabilities["ml1"]
        win2 = probabilities["ml2"]
        # Tied regulation-like scores were resolved by the joint OT simulator.

        # ----------------------------------------------------
        # PROJECTED SCORES
        # ----------------------------------------------------

        a, b = st.columns(2)

        with a:
            st.metric(
                f"{team1} projected points",
                f"{m['p1']:.1f}"
            )

            st.caption(
                "Approx. 95% simulated range: "
                f"{range1[0]:.0f}"
                "–"
                f"{range1[1]:.0f}"
            )

        with b:
            st.metric(
                f"{team2} projected points",
                f"{m['p2']:.1f}"
            )

            st.caption(
                "Approx. 95% simulated range: "
                f"{range2[0]:.0f}"
                "–"
                f"{range2[1]:.0f}"
            )

        margin = m["p1"] - m["p2"]
        total = m["p1"] + m["p2"]

        st.write(
            f"**Projected margin:** "
            f"{team1} {margin:+.1f}  |  "
            f"**Projected total:** {total:.1f}"
        )

        st.write(
            show_probability(
                f"Model win probability — {team1}",
                win1
            )
            + " · "
            + show_probability(
                team2,
                win2
            )
        )

        # ----------------------------------------------------
        # MARKET MONEYLINE: NORMALIZED TO 100% IF BOTH SIDES EXIST
        # ----------------------------------------------------
        st.markdown("#### Betting probabilities")
        raw1 = american_implied(fixture.get("ML1", np.nan))
        raw2 = american_implied(fixture.get("ML2", np.nan))
        if np.isfinite(raw1) and np.isfinite(raw2) and raw1 + raw2 > 0:
            implied1 = raw1 / (raw1 + raw2)
            implied2 = raw2 / (raw1 + raw2)
            st.caption("Moneyline probabilities normalized to remove sportsbook margin.")
        else:
            implied1, implied2 = raw1, raw2
            st.caption("A missing moneyline is shown as N/A; full-market normalization requires both sides.")

        st.markdown("**Moneyline**")
        betting_table([
            {"Selection": team1, "Odds": fixture.get("ML1", np.nan),
             "Implied Probability": implied1, "Model Probability": win1},
            {"Selection": team2, "Odds": fixture.get("ML2", np.nan),
             "Implied Probability": implied2, "Model Probability": win2},
        ], shade_model=True)

        # ----------------------------------------------------
        # SPREAD — derived from SAME joint score simulations as ML
        # ----------------------------------------------------
        st.markdown("**Spread**")
        if np.isfinite(signed):
            cover1 = probabilities["cover1"]
            cover2 = probabilities["cover2"]
            betting_table([
                {"Selection": f"{team1} {signed:+g}", "Odds": "—",
                 "Implied Probability": 0.5, "Model Probability": cover1},
                {"Selection": f"{team2} {-signed:+g}", "Odds": "—",
                 "Implied Probability": 0.5, "Model Probability": cover2},
            ])
            if probabilities.get("spread_push", 0) > 0:
                st.caption(f"Simulated spread push: {probabilities['spread_push']:.1%}")
        else:
            st.caption("No usable spread/favorite listed.")

        # ----------------------------------------------------
        # TOTAL — same joint score simulations, not independent bell curve
        # ----------------------------------------------------
        st.markdown("**Over / Under**")
        if pd.notna(total_line):
            betting_table([
                {"Selection": f"Over {float(total_line):g}", "Odds": "—",
                 "Implied Probability": 0.5, "Model Probability": probabilities["over"]},
                {"Selection": f"Under {float(total_line):g}", "Odds": "—",
                 "Implied Probability": 0.5, "Model Probability": probabilities["under"]},
            ])
            if probabilities.get("total_push", 0) > 0:
                st.caption(f"Simulated total push: {probabilities['total_push']:.1%}")
        else:
            st.caption("No total listed.")

        with st.expander("V3: objective regressions, ensemble weights and comparable odds"):
            st.caption("Active weights renormalize when data is insufficient. "
                       "Regression objectives are smooth optimization surrogates, "
                       "not demonstrated accuracy or profit guarantees.")
            st.dataframe(pd.DataFrame([
                {"Component": k, "Active weight": v} for k, v in active_shares.items()
            ]).style.format({"Active weight": "{:.1%}"}),
                hide_index=True, use_container_width=True)
            if ml_engine is not None and ml_engine.get("validation"):
                st.write("**Chronological holdout — ten-regression component only**")
                st.dataframe(pd.DataFrame([ml_engine["validation"]]), hide_index=True,
                             use_container_width=True)
                st.caption("Not full-model backtesting; these values are measured on later games "
                           "that the holdout regressions did not train on.")
            if ml_prediction is not None:
                st.write("**Ten training objectives and their score forecasts**")
                st.dataframe(ml_prediction["detail"], hide_index=True,
                             use_container_width=True)
            if not odds_details.empty:
                st.write("**Completed COMP_STAT games with comparable market lines (±5)**")
                st.dataframe(odds_details, hide_index=True, use_container_width=True)
            st.write("**Simulated probability diagnostics**")
            st.json({k: round(float(v), 4) for k, v in probabilities.items()})

        # ----------------------------------------------------
        # HISTORICAL MARKET PERFORMANCE
        # ----------------------------------------------------

        with st.expander(
            "Historical win / cover / over rates "
            "and diagnostic breakdown"
        ):
            rows = []

            for label, key in [
                ("Win rate", "WIN_RESULT"),
                ("Cover rate", "COVER_RESULT"),
                ("Over rate", "OVER_RESULT")
            ]:
                rows.append({
                    "Historical outcome": label,
                    team1: first["mean"].get(
                        key, np.nan
                    ),
                    team2: second["mean"].get(
                        key, np.nan
                    )
                })

            historical_df = pd.DataFrame(rows)

            st.dataframe(
                historical_df.style.format({
                    team1: "{:.1%}",
                    team2: "{:.1%}"
                }),
                hide_index=True,
                use_container_width=True
            )

            st.markdown("**When listed as favorite (historical)**")
            fav_rows = []
            for label, ctx in ((team1, ctx1), (team2, ctx2)):
                subset = games.loc[
                    (games["TEAM"] == ctx["TEAM"])
                    & (games["FAVORITE"] == ctx["TEAM"])
                ] if "FAVORITE" in games.columns else pd.DataFrame()
                # FAVORITE may contain acronyms; compare both forms.
                if "FAVORITE" in games.columns:
                    acronym = (acr1 if label == team1 else acr2).casefold()
                    subset = games.loc[
                        (games["TEAM"] == ctx["TEAM"])
                        & (games["FAVORITE"].astype(str).str.strip().str.casefold().isin([ctx["TEAM"], acronym]))
                    ]
                row = {"Team": label, "Games": len(subset), "Win as favorite": np.nan, "Cover as favorite": np.nan}
                if not subset.empty:
                    w = subset["_WEIGHT"].to_numpy(dtype=float)
                    for column, outname in (("WIN_RESULT", "Win as favorite"), ("COVER_RESULT", "Cover as favorite")):
                        x = subset[column].to_numpy(dtype=float)
                        mask = np.isfinite(x)
                        if mask.any():
                            row[outname] = float(np.average(x[mask], weights=w[mask]))
                fav_rows.append(row)
            st.dataframe(
                pd.DataFrame(fav_rows).style.format({
                    "Win as favorite": "{:.1%}", "Cover as favorite": "{:.1%}"
                }, na_rep="N/A"),
                hide_index=True, use_container_width=True
            )

            st.caption(
                "Historical rates use previous games "
                "and historical betting lines, "
                "not today's predicted scores."
            )

        # ----------------------------------------------------
        # HISTORICAL MARGIN THRESHOLDS (NEW SECTION)
        # ----------------------------------------------------
        with st.expander("Historical win / spread / total margins: ±5 and ±10"):
            st.caption(
                "Hit +5/+10 means the team exceeded the relevant threshold "
                "by at least that many points; Miss -5/-10 means it fell "
                "short by at least that many. Historical sportsbook lines "
                "are used for ATS and totals, and current-season games "
                "carry 3x weight. Effective games reflect concentration."
            )
            for heading, col in (
                ("Moneyline: final winning margin", "WIN_MARGIN"),
                ("Spread: margin relative to historical spread", "COVER_MARGIN_CALC"),
                ("Total: points relative to historical O/U", "OVER_MARGIN_CALC"),
            ):
                results = []
                for label, ctx in ((team1, ctx1), (team2, ctx2)):
                    history = matchup_margin_history(games, None, ctx, col)
                    rates = margin_thresholds(history)
                    if rates is None:
                        results.append({"Team": label, "Hit +5": np.nan,
                                        "Miss -5": np.nan, "Hit +10": np.nan,
                                        "Miss -10": np.nan, "Effective games": np.nan})
                    else:
                        results.append({"Team": label, **rates})
                st.markdown(f"**{heading}**")
                st.dataframe(
                    pd.DataFrame(results).style.format({
                        "Hit +5": "{:.1%}", "Miss -5": "{:.1%}",
                        "Hit +10": "{:.1%}", "Miss -10": "{:.1%}",
                        "Effective games": "{:.1f}"
                    }, na_rep="N/A"),
                    hide_index=True, use_container_width=True
                )
            st.caption(
                "Favorite win percentage and favorite cover percentage are "
                "different events. ATS historical margins are not the same "
                "thing as today's spread probability."
            )

        # V3 ten-objective diagnostics appear in the expanded panel above.

        # ----------------------------------------------------
        # PERIODS AND OVERTIME
        # ----------------------------------------------------

        with st.expander(
            "Period projections, overtime "
            "and model diagnostics"
        ):
            periods = pd.DataFrame({
                "Period": [
                    "First half",
                    "Second half",
                    "Full game"
                ],
                team1: [
                    m["h1"],
                    m["s1"],
                    m["p1"]
                ],
                team2: [
                    m["h2"],
                    m["s2"],
                    m["p2"]
                ],
            })

            st.dataframe(
                periods.style.format({
                    team1: "{:.1f}",
                    team2: "{:.1f}"
                }),
                hide_index=True,
                use_container_width=True
            )

            st.write(
                "**Estimated overtime occurrence:** "
                f"{m['ot_prob']:.1%}"
            )

            st.caption(
                "1H, 2H and full-game projections "
                "are currently estimated independently. "
                "Period totals may not exactly add up."
            )

            st.write(
                "**Observed historical team-game rows:** "
                f"{team1}: {first['team_games']} · "
                f"{team2}: {second['team_games']}"
            )

            c1, c2 = st.columns(2)

            with c1:
                st.write(
                    f"**{team1}: historical contexts**"
                )
                st.dataframe(
                    first["detail"],
                    hide_index=True,
                    use_container_width=True
                )

            with c2:
                st.write(
                    f"**{team2}: historical contexts**"
                )
                st.dataframe(
                    second["detail"],
                    hide_index=True,
                    use_container_width=True
                )

        # ----------------------------------------------------
        # CURRENT YEAR GAMES — DISPLAY ONLY
        # ----------------------------------------------------
        with st.expander("Current Year Games — results and betting history"):
            st.caption(
                "Completed 2026–27 games from COMP_STAT before the databook date. "
                "Final scores are shown team-first. ML and spread W/L refer to "
                "the listed team; Over W means the game went OVER the listed total. "
                "Green = win, red = loss, gray = push; N/A means no listed line."
            )
            show_current_year_results(current, team1, acr1)
            show_current_year_results(current, team2, acr2)

st.caption(
    "Score ranges come from a provisional joint simulation and may not achieve "
    "95% actual coverage. Probabilities and intervals have not been calibrated. "
    "When backtesting, All_Stats must reflect "
    "the selected historical date."
)
