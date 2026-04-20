from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .db import connect


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class TrainMetrics:
    accuracy: float
    roc_auc: Optional[float]
    n_samples: int
    n_positive: int


def _model_path(habit_id: int) -> Path:
    return ARTIFACTS_DIR / f"habit_{habit_id}_model.joblib"


def _meta_path(habit_id: int) -> Path:
    return ARTIFACTS_DIR / f"habit_{habit_id}_meta.json"


def _daterange(start: date, end: date) -> List[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def fetch_habit_logs_df(habit_id: int) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: date (datetime64), status (int).
    Missing days are filled as status=0 (missed) so that streak features are consistent.
    """
    with connect() as conn:
        rows = conn.execute(
            "SELECT log_date, status FROM habit_logs WHERE habit_id = ? ORDER BY log_date ASC",
            (habit_id,),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["date", "status"])

    df = pd.DataFrame([{"date": r["log_date"], "status": int(r["status"])} for r in rows])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    start = df["date"].min().date()
    end = df["date"].max().date()
    full = pd.DataFrame({"date": pd.to_datetime([d.isoformat() for d in _daterange(start, end)])})
    df = full.merge(df, on="date", how="left")
    df["status"] = df["status"].fillna(0).astype(int)
    return df


def compute_streak_and_consistency(df: pd.DataFrame, window_days: int = 30) -> Tuple[int, float]:
    """
    Streak: consecutive 1's ending at the latest date in df.
    Consistency score: mean(status) over last `window_days` days (or all if shorter).
    """
    if df.empty:
        return 0, 0.0

    statuses = df["status"].tolist()
    streak = 0
    for s in reversed(statuses):
        if int(s) == 1:
            streak += 1
        else:
            break

    tail = df.tail(window_days)
    consistency = float(tail["status"].mean()) if not tail.empty else 0.0
    return streak, consistency


def _features_for_index(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """
    Features based on day idx (inclusive) using history up to that day:
    - day_of_week
    - previous_day_status
    - current_streak_length (ending at idx)
    - missed_count_last_3_days (ending at idx)
    """
    d = df.loc[idx, "date"]
    day_of_week = float(pd.Timestamp(d).dayofweek)

    prev_status = float(df.loc[idx - 1, "status"]) if idx - 1 >= 0 else 0.0

    # Current streak ending at idx.
    streak = 0
    j = idx
    while j >= 0 and int(df.loc[j, "status"]) == 1:
        streak += 1
        j -= 1

    missed_last_3 = int(df.loc[max(0, idx - 2) : idx, "status"].eq(0).sum())
    return {
        "day_of_week": day_of_week,
        "prev_status": prev_status,
        "current_streak": float(streak),
        "missed_last_3": float(missed_last_3),
    }


def build_supervised_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build X, y where y is whether the NEXT day is missed (failure=1).
    For each day t (except last), features are computed from history up to t,
    and label is 1 if status(t+1)==0 else 0.
    """
    if df.empty or len(df) < 5:
        return pd.DataFrame(), pd.Series(dtype=int)

    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    for i in range(0, len(df) - 1):
        rows.append(_features_for_index(df, i))
        next_status = int(df.loc[i + 1, "status"])
        labels.append(1 if next_status == 0 else 0)

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="failure_next_day").astype(int)
    return X, y


def train_failure_model(habit_id: int) -> TrainMetrics:
    df = fetch_habit_logs_df(habit_id)
    X, y = build_supervised_dataset(df)
    if X.empty or y.nunique() < 2:
        raise ValueError("Not enough labeled history to train (need both success and failure examples).")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, preds))
    try:
        auc = float(roc_auc_score(y_test, probs))
    except Exception:
        auc = None

    joblib.dump(model, _model_path(habit_id))
    _meta_path(habit_id).write_text(
        json.dumps(
            {
                "trained_at": datetime.utcnow().isoformat() + "Z",
                "feature_columns": list(X.columns),
                "metrics": {"accuracy": acc, "roc_auc": auc},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return TrainMetrics(
        accuracy=acc,
        roc_auc=auc,
        n_samples=int(len(X)),
        n_positive=int(y.sum()),
    )


def load_failure_model(habit_id: int) -> LogisticRegression:
    path = _model_path(habit_id)
    if not path.exists():
        raise FileNotFoundError("Model not found. Train first via /train.")
    return joblib.load(path)


def predict_failure_probability(habit_id: int, for_date: date) -> float:
    """
    Predict P(failure) for `for_date` (meaning: probability the habit will be MISSED on that date).
    Features are computed from history up to the previous day.
    """
    model = load_failure_model(habit_id)
    df = fetch_habit_logs_df(habit_id)
    if df.empty:
        raise ValueError("No logs found for this habit.")

    # Ensure df includes the day before for_date (fill missing as 0).
    last_day = df["date"].max().date()
    if for_date <= df["date"].min().date():
        raise ValueError("for_date is before the first logged date; add more logs.")

    if for_date > last_day:
        # Extend df up to day before for_date with implied misses.
        start = last_day + timedelta(days=1)
        end = for_date - timedelta(days=1)
        if start <= end:
            extra = pd.DataFrame({"date": pd.to_datetime([d.isoformat() for d in _daterange(start, end)])})
            extra["status"] = 0
            df = pd.concat([df, extra], ignore_index=True).sort_values("date").reset_index(drop=True)

    prev_day = pd.to_datetime((for_date - timedelta(days=1)).isoformat())
    idx_list = df.index[df["date"] == prev_day].tolist()
    if not idx_list:
        raise ValueError("Unable to compute features for the requested date.")
    idx = int(idx_list[0])

    feats = _features_for_index(df, idx)
    X = pd.DataFrame([feats])
    prob = float(model.predict_proba(X)[:, 1][0])
    return prob


def risk_level(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Medium"
    return "High"

