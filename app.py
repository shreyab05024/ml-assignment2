# app.py
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

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

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")
st.title("ML Assignment 2 — Classification Models Demo")
st.caption("Loads pre-trained .pkl models and scaler from model/artifacts/")

ARTIFACT_DIR = Path(__file__).resolve().parent / "model" / "artifacts"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

SCALER_FILE = "scaler.pkl"

# Expected feature list for Breast Cancer dataset (30 features)
# IMPORTANT: Your uploaded CSV must have these exact column names for safest results.
EXPECTED_FEATURES = [
    "Mean Radius","Mean Texture","Mean Perimeter","Mean Area","Mean Smoothness",
    "Mean Compactness","Mean Concavity","Mean Concave Points","Mean symmetry","Mean Fractal Dimension",
    "Radius Error","Texture Error","Perimeter Error","Area Error","Smoothness Error",
    "Compactness Error","Concavity Error","Concave Points Error","symmetry Error","Fractal Dimension Error",
    "Worst Radius","Worst Texture","Worst Perimeter","Worst Area","Worst Smoothness",
    "Worst Compactness","Worst Concavity","Worst Concave Points","Worst symmetry","Worst Fractal Dimension"
]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not ARTIFACT_DIR.exists():
        raise FileNotFoundError(
            f"Artifact folder not found: {ARTIFACT_DIR}\n"
            "Run the training scripts in model/ to generate .pkl files."
        )

    scaler_path = ARTIFACT_DIR / SCALER_FILE
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing scaler file: {scaler_path}\n"
            "Expected: model/artifacts/scaler.pkl"
        )
    scaler = joblib.load(scaler_path)

    models = {}
    missing = []
    for name, fn in MODEL_FILES.items():
        p = ARTIFACT_DIR / fn
        if p.exists():
            models[name] = joblib.load(p)
        else:
            missing.append(str(p))

    if missing:
        st.warning(
            "Some model files are missing. The dropdown will only show available models.\n\n"
            + "\n".join(missing)
        )

    if not models:
        raise FileNotFoundError(
            "No model .pkl files found in model/artifacts/. "
            "Please run training scripts to generate them."
        )

    return scaler, models


def compute_metrics(y_true, y_pred, y_score=None):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    out["AUC"] = roc_auc_score(y_true, y_score) if y_score is not None else np.nan
    return out


def get_auc_score_input(model, X):
    # Prefer predict_proba; fallback to decision_function
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def validate_and_prepare_features(df: pd.DataFrame):
    """
    Ensures uploaded CSV has expected feature columns.
    Extra columns are ignored. Missing columns stop execution.
    """
    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        return None, missing, []

    extra = [c for c in df.columns if c not in EXPECTED_FEATURES]
    X = df[EXPECTED_FEATURES].copy()
    return X, [], extra


# -------------------------------------------------------------------
# Load models/scaler
# -------------------------------------------------------------------
try:
    scaler, models = load_artifacts()
except Exception as e:
    st.Error(str(e))
    st.stop()

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.header("Controls")
model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
label_col = st.sidebar.text_input(
    "Optional: label column name in CSV (if present)",
    value="target",
    help="If your uploaded CSV has ground-truth labels, set the column name here (e.g., target/label/y).",
)
uploaded = st.sidebar.file_uploader("Upload CSV (test data)", type=["csv"])

# -------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.subheader("Loaded Artifacts")
    st.write(f"Artifacts directory: `{ARTIFACT_DIR}`")
    st.write("Available models:")
    st.write([f"- {k}" for k in models.keys()])

    st.markdown("---")
    st.subheader("Expected CSV Columns (30 features)")
    st.code("\n".join(EXPECTED_FEATURES), language="text")
    st.info(
        "Tip: Your uploaded CSV should include these exact column names. "
        "If you have labels, include a separate column (e.g., `target`)."
    )

with right:
    st.subheader(f"Predictions & Evaluation — {model_name}")
    model = models[model_name]

    if uploaded is None:
        st.warning("Upload a CSV from the sidebar to run predictions.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.Error(f"Could not read CSV: {e}")
        st.stop()

    y_true = None
    if label_col and label_col in df.columns:
        try:
            y_true = df[label_col].astype(int).to_numpy()
        except Exception:
            st.warning(f"Could not parse label column `{label_col}` as integers. Metrics will be skipped.")
        X_df = df.drop(columns=[label_col])
    else:
        X_df = df

    X, missing_cols, extra_cols = validate_and_prepare_features(X_df)

    if missing_cols:
        st.Error(
            "Uploaded CSV is missing required feature columns:\n\n"
            + ", ".join(missing_cols[:20])
            + (" ..." if len(missing_cols) > 20 else "")
        )
        st.stop()

    if extra_cols:
        st.info(
            "Extra columns detected (ignored):\n\n"
            + ", ".join(extra_cols[:20])
            + (" ..." if len(extra_cols) > 20 else "")
        )

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    y_pred = model.predict(X_scaled)

    out_df = df.copy()
    out_df["prediction"] = y_pred

    y_score = get_auc_score_input(model, X_scaled)
    if y_score is not None:
        out_df["score_or_prob_class_1"] = y_score

    st.write("**Preview of Predictions:**")
    st.dataframe(out_df.head(25), use_container_width=True)

    # Metrics + Confusion Matrix + Report (only if labels exist)
    if y_true is not None and len(y_true) == len(y_pred):
        st.markdown("---")
        st.subheader("Evaluation on Uploaded CSV (only if labels provided)")

        metrics = compute_metrics(y_true, y_pred, y_score if y_score is not None else None)
        st.write("**Metrics:**")
        st.dataframe(pd.DataFrame([metrics]).style.format("{:.4f}"), use_container_width=True)

        cm = confusion_matrix(y_true, y_pred)
        st.write("**Confusion Matrix:**")
        st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        st.write("**Classification Report:**")
        st.text(classification_report(y_true, y_pred, zero_division=0))
    else:
        st.info(
            "No label column found (or label parsing failed), so evaluation metrics/confusion matrix "
            "cannot be computed for the uploaded CSV."
        )

    # Download predictions
    st.markdown("---")
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )