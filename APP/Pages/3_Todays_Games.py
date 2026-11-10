
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
TEAM_COACH_CONTEXT_SHARE = 0.95
CONFERENCE_CONTEXT_SHARE = 0.05
PRIOR_GAMES = 8.0
MIN_SD = 6.0
TEAM_SCORE_SHARE = 0.75
OPP_DEFENSE_SHARE = 0.25
ML_CONTEXT_SHARE = 0.65
ML_CONFERENCE_SHARE = 0.05

ML_START_DATE = pd.Timestamp(2026, 11, 20)
ML_INFLUENCE = 0.30
ML_MIN_ROWS = 40

OUTCOMES = [
    "1H", "OPP_1H",
    "2H", "OPP_2H",
    "OT_IND",
    "PTS", "OPP_PTS",
]

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
    lower = min(lower, -fallback_sd)
    upper = max(upper, fallback_sd)
    return (max(0.0, predicted_mean + lower), predicted_mean + upper)


def model_matchup(first, second):
    t1 = context_score(first, second, "team_coach")
    t2 = context_score(second, first, "team_coach")
    c1 = context_score(first, second, "conference")
    c2 = context_score(second, first, "conference")

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
    if not np.isfinite(value) or abs(value) <= 3:
        return "background-color: #ffffff; color: #111111"
    strength = min(1.0, (abs(value) - 3.0) / 17.0)
    base = (255, 255, 255)
    target = (34, 153, 84) if value > 0 else (207, 52, 52)
    rgb = tuple(round(base[j] + strength * (target[j] - base[j])) for j in range(3))
    font = "#ffffff" if strength > 0.72 else "#111111"
    return f"background-color: rgb{rgb}; color: {font}; font-weight: bold"


def model_win_background(probability):
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
    baseline = normal_cdf((margin + signed) / sd_margin)
    hist1 = matchup_margin_history(games, first, ctx1, "WIN_MARGIN")
    hist2 = matchup_margin_history(games, second, ctx2, "WIN_MARGIN")
    if hist1 is None and hist2 is None:
        return baseline

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
    bandwidth = max(2.0, 0.25 * sd_margin)
    z = np.clip((margin + signed + v) / bandwidth, -30, 30)
    empirical = float(np.average(1.0 / (1.0 + np.exp(-z)), weights=w))
    samples = sum(h[2] for h in (hist1, hist2) if h is not None)
    trust = 0.65 * samples / (samples + 15.0)
    result = (1 - trust) * baseline + trust * empirical

    ats1, effective1 = historical_cover_rate_near_line(games, ctx1, signed)
    if np.isfinite(ats1):
        ats_share = 0.15 * effective1 / (effective1 + 12.0)
        result = (1 - ats_share) * result + ats_share * ats1
    return float(np.clip(result, 0.005, 0.995))


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
    "Historical/team-coach context + optional "
    "current-season ML (30% from Nov 20). "
    "Model weights and 95% intervals "
    "are provisional and not yet backtested."
)


# ============================================================
# TRAIN ML ONCE PER DATE / DATA UPDATE
# ============================================================

ml_engine = None

ml_note = (
    "ML inactive until November 20, 2026."
)

if today >= ML_START_DATE:
    if current.empty:
        ml_note = (
            "ML unavailable: COMP_STAT has no "
            "completed games before the selected date."
        )

    else:
        ml_engine, ml_problem = train_season_ml(
            (
                (DATA / FILES["season"]).stat().st_mtime_ns,
                today.isoformat()
            ),
            current
        )

        if ml_engine:
            ml_note = (
                f"ML active: {ml_engine['rows']} "
                "current-season team-game rows, "
                "3 regressors per score target, "
                "30% influence."
            )
        else:
            ml_note = (
                f"ML unavailable: {ml_problem}"
            )

st.caption(ml_note)


if todays.empty:
    st.info(
        "No games scheduled for the selected date."
    )
    st.stop()


# ============================================================
# FUTURE GAME ORDERING BY TEAM RATINGS
# ============================================================

# Once Rankings exposes a numeric Team Rating
# keyed by Acronym, sort matchups by:
# Team1_Rating + Team2_Rating
# Highest combined rating first.


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

        m = model_matchup(
            first,
            second
        )

        ml_prediction = None

        # ----------------------------------------------------
        # ADD ML WHEN AVAILABLE
        # ----------------------------------------------------

        if ml_engine is not None:
            try:
                ml_prediction = ml_predict_matchup(
                    ml_engine,
                    catalog,
                    acr1,
                    acr2
                )

                for idx in (1, 2):
                    team_est = m.get(f"team{idx}_context", np.nan)
                    conf_est = m.get(f"team{idx}_conference", np.nan)
                    if not np.isfinite(team_est):
                        team_est = m[f"p{idx}"]
                    if np.isfinite(conf_est):
                        history_score = (ML_CONTEXT_SHARE * team_est
                                         + ML_CONFERENCE_SHARE * conf_est)
                    else:
                        history_score = (1 - ML_INFLUENCE) * team_est
                    m[f"p{idx}"] = history_score + ML_INFLUENCE * ml_prediction[f"p{idx}"]

            except (
                KeyError,
                ValueError,
                TypeError
            ) as exc:
                st.caption(
                    f"ML unavailable for this matchup: {exc}"
                )

        range1 = score_interval(ctx1, ctx2, games, m["p1"], m["sd1"])
        range2 = score_interval(ctx2, ctx1, games, m["p2"], m["sd2"])

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
                "Uncalibrated 95% historical range: "
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
                "Uncalibrated 95% historical range: "
                f"{range2[0]:.0f}"
                "–"
                f"{range2[1]:.0f}"
            )

        margin = m["p1"] - m["p2"]
        total = m["p1"] + m["p2"]

        win1 = normal_cdf(
            margin / m["sd_margin"]
        )

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
                1 - win1
            )
        )

        # ----------------------------------------------------
        # MONEYLINE
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
             "Implied Probability": implied2, "Model Probability": 1 - win1},
        ], shade_model=True)

        # ----------------------------------------------------
        # SPREAD
        # ----------------------------------------------------

        favorite = str(fixture.get("Favorite", "")).strip().casefold()
        spread = n(fixture.get("Spread", np.nan))
        total_line = n(fixture.get("OverUnder", np.nan))

        st.markdown("**Spread**")
        if pd.notna(spread):
            magnitude = abs(float(spread))
            if favorite in {team1.casefold(), acr1.casefold()}:
                signed = -magnitude
            elif favorite in {team2.casefold(), acr2.casefold()}:
                signed = magnitude
            else:
                signed = np.nan

            if np.isfinite(signed):
                cover1 = empirical_cover_probability(
                    first, second, ctx1, ctx2, margin, signed, m["sd_margin"], games
                )

                cover1 = min(cover1, win1) if signed < 0 else max(cover1, win1) if signed > 0 else win1

                betting_table([
                    {"Selection": f"{team1} {signed:+g}", "Odds": "—",
                     "Implied Probability": 0.5, "Model Probability": cover1},
                    {"Selection": f"{team2} {-signed:+g}", "Odds": "—",
                     "Implied Probability": 0.5, "Model Probability": 1 - cover1},
                ])
            else:
                st.caption("Favorite does not match either team; spread omitted.")
        else:
            st.caption("No spread listed.")

        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        st.markdown("**Over / Under**")
        if pd.notna(total_line):
            over = normal_cdf((total - float(total_line)) / m["sd_total"])
            betting_table([
                {"Selection": f"Over {float(total_line):g}", "Odds": "—",
                 "Implied Probability": 0.5, "Model Probability": over},
                {"Selection": f"Under {float(total_line):g}", "Odds": "—",
                 "Implied Probability": 0.5, "Model Probability": 1 - over},
            ])
        else:
            st.caption("No total listed.")

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
        # HISTORICAL MARGIN THRESHOLDS
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

        # ----------------------------------------------------
        # MACHINE LEARNING DETAILS
        # ----------------------------------------------------

        if ml_prediction is not None:
            with st.expander(
                "Machine learning: three model "
                "projections (30% blend)"
            ):
                st.caption(
                    "Trained only on completed COMP_STAT "
                    "rows using pregame numerical statistics. "
                    "Team, coach, conference, and postgame "
                    "features are excluded."
                )

                model_rows = []

                for name, scores in (
                    ml_prediction["models"].items()
                ):
                    model_rows.append({
                        "Method": name,
                        team1: round(scores[0], 1),
                        team2: round(scores[1], 1)
                    })

                st.dataframe(
                    pd.DataFrame(model_rows),
                    hide_index=True,
                    use_container_width=True
                )

                st.write(
                    "**ML-only consensus:** "
                    f"{team1} {ml_prediction['p1']:.1f} – "
                    f"{team2} {ml_prediction['p2']:.1f}"
                )

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

        # ====================================================
        # CURRENT YEAR GAMES — RESULTS & BETTING HISTORY
        # ====================================================

        with st.expander(
            "Current Year Games — Results and Betting History"
        ):
            st.caption(
                "Completed current-season games only. "
                "Green = Win, Red = Loss, Gray = Push. "
                "O/U results represent an OVER bet."
            )

            def format_number(value, signed=False):
                try:
                    value = float(value)
                    if not np.isfinite(value):
                        return "N/A"
                    return (
                        f"{value:+g}" if signed
                        else f"{value:g}"
                    )
                except (ValueError, TypeError):
                    return "N/A"

            def result_color(value):
                if value == "W":
                    return (
                        "background-color: #c9efd2; "
                        "color: #14532d; font-weight: bold"
                    )
                if value == "L":
                    return (
                        "background-color: #f7c8c8; "
                        "color: #7f1d1d; font-weight: bold"
                    )
                if value == "Push":
                    return (
                        "background-color: #e5e7eb; "
                        "color: #374151; font-weight: bold"
                    )
                return ""

            for display_team, display_acr in (
                (team1, acr1),
                (team2, acr2)
            ):
                st.markdown(f"**{display_team}**")

                if current.empty:
                    st.caption(
                        "No completed current-season games."
                    )
                    continue

                team_mask = current["TEAM"].eq(
                    display_team.casefold().strip()
                )

                if "TEAM_ACR" in current.columns:
                    team_mask = team_mask | (
                        current["TEAM_ACR"]
                        .astype(str)
                        .str.strip()
                        .str.upper()
                        .eq(display_acr.upper())
                    )

                team_games = current.loc[
                    team_mask
                ].sort_values(
                    "_DATE",
                    ascending=False
                )

                rows = []

                for _, game in team_games.iterrows():
                    pts = n(game.get("PTS", np.nan))
                    opp_pts = n(
                        game.get("OPP_PTS", np.nan)
                    )

                    if pd.isna(pts) or pd.isna(opp_pts):
                        continue

                    margin = float(pts - opp_pts)
                    game_spread = n(
                        game.get("SPREAD", np.nan)
                    )
                    game_total = n(
                        game.get("TOTAL_LINE", np.nan)
                    )
                    game_ml = valid_odds(
                        game.get("ML", np.nan)
                    )

                    if np.isfinite(game_ml):
                        ml_result = (
                            "W" if margin > 0
                            else "L" if margin < 0
                            else "Push"
                        )
                    else:
                        ml_result = "N/A"

                    if pd.notna(game_spread):
                        ats_margin = (
                            margin + float(game_spread)
                        )
                        spread_result = (
                            "W" if ats_margin > 0
                            else "L" if ats_margin < 0
                            else "Push"
                        )
                    else:
                        spread_result = "N/A"

                    if pd.notna(game_total):
                        ou_margin = (
                            float(pts + opp_pts)
                            - float(game_total)
                        )
                        ou_result = (
                            "W" if ou_margin > 0
                            else "L" if ou_margin < 0
                            else "Push"
                        )
                    else:
                        ou_result = "N/A"

                    date_value = game["_DATE"]

                    rows.append({
                        "Date": date_value.strftime(
                            "%m/%d/%y"
                        ),
                        "Opponent": str(
                            game.get("OPP", "")
                        ).title(),
                        "HAN": str(
                            game.get("HAN", "")
                        ).upper(),
                        "Final": (
                            f"{pts:g}–{opp_pts:g}"
                        ),
                        "ML Odds": (
                            format_number(game_ml, True)
                        ),
                        "ML Result": ml_result,
                        "Spread": format_number(
                            game_spread, True
                        ),
                        "Spread Result": spread_result,
                        "O/U": format_number(game_total),
                        "Over Result": ou_result,
                    })

                if not rows:
                    st.caption(
                        "No completed games before "
                        "the databook date."
                    )
                    continue

                result_df = pd.DataFrame(rows)

                def shade_current_row(row):
                    styles = [
                        "" for _ in row.index
                    ]

                    pairs = [
                        ("ML Result", "ML Odds"),
                        ("Spread Result", "Spread"),
                        ("Over Result", "O/U"),
                    ]

                    for result_col, value_col in pairs:
                        color = result_color(
                            row[result_col]
                        )
                        styles[
                            row.index.get_loc(result_col)
                        ] = color
                        styles[
                            row.index.get_loc(value_col)
                        ] = color

                    return styles

                st.dataframe(
                    result_df.style.apply(
                        shade_current_row,
                        axis=1
                    ),
                    hide_index=True,
                    use_container_width=True
                )



st.caption(
    "Score ranges use weighted historical tails and may not achieve "
    "95% actual coverage. Probabilities and intervals have not been calibrated. "
    "When backtesting, All_Stats must reflect "
    "the selected historical date."
)
