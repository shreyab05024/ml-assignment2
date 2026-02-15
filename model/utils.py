# model/utils.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DataBundle:
    feature_names: list
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def load_and_split(scale: bool = True, test_size: float = 0.2, random_state: int = 42) -> DataBundle:
    """
    Loads Breast Cancer dataset, creates stratified train/test split,
    and (optionally) standard-scales features.
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    if scale:
        X_train = scaler.fit_transform(X_train_df)
        X_test = scaler.transform(X_test_df)
    else:
        # still fit scaler for consistency if you want to reuse later
        scaler.fit(X_train_df)
        X_train = X_train_df.to_numpy()
        X_test = X_test_df.to_numpy()

    return DataBundle(
        feature_names=list(X.columns),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
    )


def evaluate_classifier(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Returns the 6 required metrics.
    AUC is computed using predict_proba when available; otherwise decision_function if present.
    """
    y_pred = model.predict(X_test)

    # best-effort probabilities/scores for AUC
    y_score: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "AUC": float(roc_auc_score(y_test, y_score)) if y_score is not None else float("nan"),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
    }
    return metrics


def print_eval_details(model_name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)
    for k, v in evaluate_classifier(model, X_test, y_test).items():
        print(f"{k:>10}: {v:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", cr)


def save_artifact(filename: str, obj) -> Path:
    """
    Saves a python object to model/artifacts/<filename> using joblib.
    """
    path = ARTIFACT_DIR / filename
    joblib.dump(obj, path)
    return path