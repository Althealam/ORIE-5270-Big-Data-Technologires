from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sklearn
## TODO: Feel free to import any useful modules from sklearn below
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_data(transactions_path: str, users_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load transactions and users CSVs and return (tx, users)."""
    tx = pd.read_csv(transactions_path)
    users = pd.read_csv(users_path)
    return tx, users


_SPEND_COLS = [
    "spend_rent",
    "spend_dining",
    "spend_groceries",
    "spend_travel",
    "spend_rideshare",
    "spend_other",
]

_CATEGORY_MAP = {
    "rideshare ": "rideshare",
    "ride_share": "rideshare",
    "uber/lyft": "rideshare",
    " dining": "dining",
    "restaurants": "dining",
    "groceries ": "groceries",
    "travel ": "travel",
    "flights": "travel",
    "other ": "other",
    "other": "other",
    "rent": "rent",
    "dining": "dining",
    "groceries": "groceries",
    "travel": "travel",
    "rideshare": "rideshare",
}


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the transactions DataFrame per the homework spec.

    Key requirements:
      1) Drop exact duplicate rows.
      2) Normalize primary_category: strip whitespace, lowercase, map synonyms.
      3) Missing values:
         - spend_* NaNs -> 0
         - monthly_spend NaN -> sum(spend_*)
         - payment_rate NaN -> median within age_group (computed from *input tx*)
         - apr NaN -> median within age_group (computed from *input tx*)
      4) Clip spend_* at 99.5th percentile (computed on *input tx*).
      5) payment_rate in [0,1]; apr in [0,0.40]
      6) user_id/month ints; age_group str
    """
    df = tx.copy()

    # (a) exact duplicates
    df = df.drop_duplicates()

    # Save per-age-group medians from the input tx (before filling)
    pay_median_by_age = tx.groupby("age_group", dropna=False)["payment_rate"].median()
    apr_median_by_age = tx.groupby("age_group", dropna=False)["apr"].median()

    # (b) normalize category labels
    if "primary_category" in df.columns:
        normalized = (
            df["primary_category"]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        df["primary_category"] = normalized.map(_CATEGORY_MAP).fillna(normalized)

    # (d) spend_* clipping thresholds from input tx
    clip_caps = {}
    for col in _SPEND_COLS:
        if col in tx.columns:
            clip_caps[col] = tx[col].quantile(0.995)

    # (c) missing in spend_*
    for col in _SPEND_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            cap = clip_caps.get(col, np.nan)
            if pd.notna(cap):
                df[col] = df[col].clip(upper=cap)

    # monthly_spend fill from (cleaned) spend sum if missing
    if "monthly_spend" in df.columns:
        spend_sum = df[[c for c in _SPEND_COLS if c in df.columns]].sum(axis=1)
        df["monthly_spend"] = pd.to_numeric(df["monthly_spend"], errors="coerce")
        df["monthly_spend"] = df["monthly_spend"].fillna(spend_sum)

    # payment_rate / apr age-group median imputation
    if "payment_rate" in df.columns:
        df["payment_rate"] = pd.to_numeric(df["payment_rate"], errors="coerce")
        df["payment_rate"] = df["payment_rate"].fillna(df["age_group"].map(pay_median_by_age))
        df["payment_rate"] = df["payment_rate"].fillna(df["payment_rate"].median())

    if "apr" in df.columns:
        df["apr"] = pd.to_numeric(df["apr"], errors="coerce")
        df["apr"] = df["apr"].fillna(df["age_group"].map(apr_median_by_age))
        df["apr"] = df["apr"].fillna(df["apr"].median())

    # (e) enforce valid ranges
    if "payment_rate" in df.columns:
        df["payment_rate"] = df["payment_rate"].clip(0.0, 1.0)
    if "apr" in df.columns:
        df["apr"] = df["apr"].clip(0.0, 0.40)

    # (f) dtypes
    if "user_id" in df.columns:
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").round().astype("Int64")
        df["user_id"] = df["user_id"].fillna(-1).astype(int)
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce").round().astype("Int64")
        df["month"] = df["month"].fillna(-1).astype(int)
    if "age_group" in df.columns:
        df["age_group"] = df["age_group"].astype(str)

    return df


def add_user_features(tx: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Merge user-level columns from users into tx on user_id WITHOUT changing number of rows in tx.
    Deduplicate users on user_id before merging.
    """
    before = len(tx)

    users_dedup = users.drop_duplicates(subset=["user_id"], keep="first")
    user_feature_cols = [c for c in users_dedup.columns if c != "user_id"]
    existing_user_cols = [c for c in user_feature_cols if c in tx.columns]

    tx_base = tx.drop(columns=existing_user_cols, errors="ignore")
    merged = tx_base.merge(users_dedup, on="user_id", how="left")
    
    after = len(merged)
    if after != before:
        raise ValueError(f"Row count changed after merge: {before} -> {after}")
    return merged


HARD_LEAKY_COLS = {
    "default_next",
    "profit_true",
    "chargeoff_loss",
    "pd_default",
    "balance_next",
    "interest_income",
    "interchange_rev",
    "reward_cost",
    "funding_cost",
}

LEAKY_SUBSTRINGS = [
    "default",
    "profit",
    "chargeoff",
    "interchange",
    "interest_income",
    "reward_cost",
    "funding_cost",
    "pd_",
    "label",
]


def _is_leaky(col: str) -> bool:
    c = col.lower()
    if col in HARD_LEAKY_COLS:
        return True
    return any(sub in c for sub in LEAKY_SUBSTRINGS)


def train_default_model(train_df: pd.DataFrame):
    """
    Train a Scikit-Learn Classifier to predict default_next.

    Returns
    -------
    model : fitted sklearn Pipeline
    feature_cols : list[str] raw feature columns used

    Constraints
    -----------
    - Do NOT use leaky columns (anything matching _is_leaky()).
    """
    df = train_df.copy()
    if "default_next" not in df.columns:
        raise ValueError("Missing target column default_next")

    feature_cols = [c for c in df.columns if c != "default_next" and not _is_leaky(c)]
    if len(feature_cols) == 0:
        raise ValueError("No valid feature columns found after leakage filtering")

    X = df[feature_cols].copy()
    y = pd.to_numeric(df["default_next"], errors="coerce").fillna(0).astype(int)

    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median"))]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X, y)

    return model, feature_cols


def evaluate_by_age_group(model, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Return per-age-group metrics: {age_group: {auc, brier, n}}."""
    if "age_group" not in df.columns:
        raise ValueError("df must contain age_group")
    if "default_next" not in df.columns:
        raise ValueError("df must contain default_next")

    p_hat = model.predict_proba(df[feature_cols])[:, 1]
    out: Dict[str, Dict[str, float]] = {}
    tmp = df.copy()
    tmp["_p_hat"] = p_hat

    for age, g in tmp.groupby("age_group", dropna=False):
        y_true = pd.to_numeric(g["default_next"], errors="coerce").fillna(0).astype(int).values
        y_prob = g["_p_hat"].values
        auc = float("nan") if np.unique(y_true).size < 2 else float(roc_auc_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))
        out[str(age)] = {"auc": auc, "brier": brier, "n": int(len(g))}

    return out

def compute_uplift_score(model, df: pd.DataFrame, feature_cols: List[str], top_frac: float = 0.10) -> float:
    """
    Compute the top-fraction uplift score for a fitted default model.

    Parameters
    ----------
    model : fitted sklearn-like classifier
        Must support ``predict_proba(df[feature_cols])[:, 1]``.
    df : pd.DataFrame
        Evaluation data containing ``default_next`` and the raw feature columns.
    feature_cols : list[str]
        Raw feature columns consumed by the model.
    top_frac : float, default 0.10
        Fraction of rows to target. Must lie in ``(0, 1]``.

    Returns
    -------
    float
        Extra number of defaults captured in the targeted slice relative to a
        random policy with the same targeting rate.
    """
    if "default_next" not in df.columns:
        raise ValueError("df must contain default_next")
    if not (0.0 < float(top_frac) <= 1.0):
        raise ValueError("top_frac must be in (0, 1]")

    n = len(df)
    if n == 0:
        return 0.0

    p_hat = model.predict_proba(df[feature_cols])[:, 1]
    k = int(np.ceil(float(top_frac) * n))
    k = max(1, min(k, n))

    y = pd.to_numeric(df["default_next"], errors="coerce").fillna(0).astype(float).values
    order = np.argsort(-p_hat)
    top_mean = float(np.mean(y[order[:k]]))
    base_rate = float(np.mean(y))

    if base_rate <= 0:
        return 0.0
    uplift_score = top_mean / base_rate

    return uplift_score


def policy_profit_impact(
    model,
    deploy_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float,
    offer_cost: float,
    delta_old: float,
    delta_genz: float,
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    """
    Stylized expected-profit simulation on deployment rows.

    Policy:
      - compute p_hat = P(default_next=1 | features)
      - offer incentive if p_hat <= threshold; cost offer_cost immediately
      - reduces default probability by delta (absolute), with cohort-specific values:
          delta_old for non-GenZ
          delta_genz for GenZ

    Returns:
      (profit_no_policy, profit_with_policy, breakdown)
    """
    df = deploy_df.copy()
    if "profit_true" not in df.columns:
        raise ValueError("deploy_df must contain profit_true")
    if "age_group" not in df.columns:
        raise ValueError("deploy_df must contain age_group")

    p_hat = model.predict_proba(df[feature_cols])[:, 1]
    offer_indicator = (p_hat <= threshold).astype(int)
    age_group_norm = df["age_group"].astype(str)
    delta = np.where(age_group_norm == "GenZ", delta_genz, delta_old)

    if "balance_new" in df.columns:
        balance = pd.to_numeric(df["balance_new"], errors="coerce").fillna(0.0)
    elif "statement_balance" in df.columns:
        balance = pd.to_numeric(df["statement_balance"], errors="coerce").fillna(0.0)
    else:
        raise ValueError("deploy_df must contain balance_new or statement_balance")

    profit_no = None
    profit_with = None

    breakdown: Dict[str, Dict[str, float]] = {}

    return profit_no, profit_with, breakdown


def stress_test_cohort_mix(
    df: pd.DataFrame,
    cohort: str,
    shares: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Stress test: reweight rows to simulate different cohort mixes (target cohort share varies).

    Returns DataFrame with columns: cohort, share, expected_profit
    """
    if shares is None:
        shares = [0.10, 0.20, 0.30, 0.45, 0.60]

    if "age_group" not in df.columns or "profit_true" not in df.columns:
        raise ValueError("df must contain age_group and profit_true")

    rows = []
    ## TODO: Implement it

    return pd.DataFrame(rows)
