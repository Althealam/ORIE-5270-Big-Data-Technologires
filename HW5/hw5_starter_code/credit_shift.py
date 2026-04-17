from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import sklearn
## TODO: Feel free to import any useful modules from sklearn below
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn import metrics


def load_data(transactions_path: str, users_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load transactions and users CSVs and return (tx, users)."""
    ## TODO: Implement it
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
    
    ## TODO: Implement it
    # 1. get the duplicate rows
    df = df.drop_duplicates(keep='first')

    # 2. normalize primary_category
    df['primary_category'] = df['primary_category'].str.strip()
    df['primary_category'] = df['primary_category'].str.lower()

    df["primary_category"] = df["primary_category"].replace(_CATEGORY_MAP)    # print(df['primary_category'].unique())

    # 3. missing values
    # - spend_* NaNs -> 0
    spend_cols = [col for col in df.columns if col.startswith("spend_")]
    df[spend_cols] = df[spend_cols].fillna(0)
    # - monthly_spend NaN -> sum(spend_*)
    spend_sum = df[spend_cols].sum(axis=1)
    df["monthly_spend"] = df['monthly_spend'].fillna(spend_sum)
    # - payment_rate NaN -> median within age_group (computed from *input tx*)
    payment_median = tx.groupby('age_group')['payment_rate'].median()
    df['payment_rate'] = df.apply(
        lambda row:
            payment_median[row["age_group"]]
            if pd.isna(row['payment_rate'])
            else row['payment_rate'],
            axis = 1
    )
    # - apr NaN -> median within age_group (computed from *input tx*)
    apr_median = tx.groupby('age_group')['apr'].median()
    df['apr'] = df.apply(
        lambda row:
            apr_median[row['age_group']]
            if pd.isna(row['apr'])
            else row['apr'],
            axis = 1
    )

    # 4. Clip spend_* at 99.5th percentile (computed on *input tx*)
    for col in spend_cols:
        upper = tx[col].quantile(0.995)
        df[col] = df[col].clip(upper=upper)
    
    # 5. payment_rate in [0,1]; apr in [0,0.40]
    df['payment_rate'] = df['payment_rate'].clip(0, 1)
    df['apr'] = df['apr'].clip(0, 0.40)

    # 6. user_id/month ints; age_group str
    df['user_id'] = df['user_id'].astype(int)
    df['month'] = df['month'].astype(int)
    df['age_group'] = df['age_group'].astype(str)
    return df



def add_user_features(tx: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """
    Merge user-level columns from users into tx on user_id WITHOUT changing number of rows in tx.
    Deduplicate users on user_id before merging.
    """
    before = len(tx)
    
    ## TODO: Implement it
    users = users.drop_duplicates(subset='user_id', keep='first')
    user_cols_to_add = [c for c in users.columns if c=='user_id' or c not in tx.columns]
    merged = tx.merge(users[user_cols_to_add], on='user_id', how='left')
    
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

    ## TODO: Implement it
    X, y = df[feature_cols], df['default_next']
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_cols),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    model = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ))
    ])
    model.fit(X, y)

    return model, feature_cols



def evaluate_by_age_group(model, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Return per-age-group metrics: {age_group: {auc, brier, n}}."""
    if "age_group" not in df.columns:
        raise ValueError("df must contain age_group")
    if "default_next" not in df.columns:
        raise ValueError("df must contain default_next")

    ## TODO: Implement it:
    p_hat = model.predict_proba(df[feature_cols])[:, 1]
    temp = df.copy()
    temp['p_hat'] = p_hat

    out = {}

    for group, sub in temp.groupby('age_group'):
        y_true = sub['default_next']
        y_prob = sub['p_hat']

        # if it only contains one class, then return nan
        if y_true.nunique()<2:
            auc = np.nan
        else:
            auc = metrics.roc_auc_score(y_true, y_prob)
        
        # brier_score = mean((y-p)**2)
        brier = metrics.brier_score_loss(y_true, y_prob)

        out[str(group)] = {
            'auc': float(auc) if pd.notna(auc) else np.nan,
            'brier': float(brier),
            'n': int(len(sub))
        }

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
    
    ## TODO: Implement it
    # 1. get the predicted_score
    p_hat = model.predict_proba(df[feature_cols])[:, 1]

    # 2. get the k
    n = len(df)
    k = math.ceil(top_frac*n)

    # 3. rank rows by predicted default probability from largest to smallest
    tmp = df.copy()
    tmp['_p_hat'] = p_hat
    top_k = tmp.sort_values(by="_p_hat", ascending=False).head(k)

    # 4. compute default rate in top k
    top_k_default_rate = top_k['default_next'].mean()

    # 5. compute overall default rate
    overall_default_rate = tmp['default_next'].mean()

    # 6. compute uplift score
    if overall_default_rate==0:
        return 0.0
    uplift_score = top_k_default_rate/overall_default_rate
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

    ## TODO: Implement it
    # step1: predict default probabilities
    p_hat = model.predict_proba(df[feature_cols])[:, 1]

    # step2: decide who receives the retention offer
    offer_indicator = [1 if x<=threshold else 0 for x in p_hat]
    df['offer_indicator'] = offer_indicator

    # step3: model the effect of the intervention
    df['delta'] = np.where(
        df['age_group'] == 'GenZ',
        delta_genz,
        delta_old
    )
    
    # step4: determine the relevant balance
    if 'balance_new' in df.columns:
        df['balance'] = df['balance_new']
    elif 'statement_balance' in df.columns:
        df['balance'] = df['statement_balance']
    else:
        raise ValueError("There are not 'balance_new' and 'statement_balance' in the columns")
    
    # step5: estimate avoided credit losses
    df['avoided_loss'] = df['offer_indicator']*df['delta']*0.90*df['balance']

    # step6: account for the cost of the incentive
    df['policy_cost'] = df['offer_indicator']*offer_cost
    
    # step7: profit without the policy
    profit_no = np.sum(df['profit_true'])

    # step8: profit with the policy
    df['profit_with_policy_row'] = df['profit_true']+df['avoided_loss']-df['policy_cost']
    profit_with = df['profit_with_policy_row'].sum()

    breakdown = {}
    for age, g in df.groupby('age_group'):
        breakdown[age] = {
            "n": len(g),
            "profit_no_policy": g['profit_true'].sum(),
            "profit_with_policy": g['profit_true'].sum(),
            "offer_rate": g["offer_indicator"].mean()
        }

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
    # define the target cohort
    target_mask = df['age_group']==cohort

    nt = target_mask.sum() # target group count
    no = (~target_mask).sum() # non-target group count

    results = []
    # iterate all the share and calculate expected profit
    for s in shares:
        if nt==0 or no==0:
            expected_profit = float(df['profit_true'].mean())
        else:
            weights = np.where(target_mask, s/nt, (1-s)/no)
            expected_profit = float(np.sum(weights*df['profit_true']))
        
        results.append({
            'cohort': cohort,
            'share': float(s),
            'expected_profit': expected_profit
        })

    return pd.DataFrame(results, columns=['cohort', 'share', 'expected_profit'])


# if __name__ == '__main__':

#     transactions_path = "/Users/althealam/Desktop/School/2026Spring/ORIE 5270-Big Data Technologies/ORIE-5270-Big-Data-Technologires/HW5/hw5_starter_code/transactions_train.csv"
#     users_path = "/Users/althealam/Desktop/School/2026Spring/ORIE 5270-Big Data Technologies/ORIE-5270-Big-Data-Technologires/HW5/hw5_starter_code/users.csv"

#     tx, users = load_data(transactions_path, users_path)
#     tx = clean_transactions(tx)
#     train_df = add_user_features(tx, users)
#     model, feature_cols = train_default_model(train_df)
#     probs = model.predict_proba(train_df[feature_cols])[:, 1]
#     score = compute_uplift_score(model, train_df, feature_cols, top_frac=0.1)
#     out = evaluate_by_age_group(model, train_df, feature_cols)
