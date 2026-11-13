"""Training pipeline for the loan approval classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "loan_dataset.csv"
TARGET = "approved"
CATEGORICAL = ["gender", "city", "employment_type", "bank"]
DROP = [TARGET, "applicant_name"]


@dataclass(frozen=True)
class TrainedModel:
    pipeline: Pipeline
    features: list[str]
    metrics: dict
    confusion: list[list[int]]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_pipeline(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=2000))])


def train(df: pd.DataFrame, seed: int = 42) -> TrainedModel:
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    y = df[TARGET]
    cat_cols = [c for c in CATEGORICAL if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    pipe = build_pipeline(num_cols, cat_cols).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
    }
    return TrainedModel(pipe, X.columns.tolist(), metrics, confusion_matrix(y_test, pred).tolist())


def approval_probability(model: TrainedModel, applicant: dict) -> float:
    row = pd.DataFrame([applicant])[model.features]
    return float(model.pipeline.predict_proba(row)[0, 1])
