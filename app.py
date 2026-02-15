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

# Use sklearn's official column naming (lowercase) but accept any case from uploads.
EXPECTED_FEATURES = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error","smoothness error",
    "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
    "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
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
            "Some model files are missing. Only available models will be shown.\n\n"
            + "\n".join(missing)
        )

    if not models:
        raise FileNotFoundError(
            "No model .pkl files found in model/artifacts/. "
            "Please generate them using the training scripts in model/."
        )

    return scaler, models


def compute_metrics(y_true, y_pred, y_score=None):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if y_score is not None else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return out


def get_auc_score_input(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def _normalize_cols(cols):
    return [str(c).strip().lower() for c in cols]


def validate_and_prepare_features(df: pd.DataFrame):
    """
    Accepts case-insensitive column matches.
    Extra columns are ignored.
    Missing columns are reported clearly.
    """
    df2 = df.copy()
    df2.columns = _normalize_cols(df2.columns)

    expected = [c.lower() for c in EXPECTED_FEATURES]
    missing = [c for c in expected if c not in df2.columns]

    extra = [c for c in df2.columns if c not in expected]

    if missing:
        return None, missing, extra

    X = df2[expected].copy()
    return X, [], extra


def example_csv_text():
    # Small sample only (header), to help evaluator upload correct format quickly
    return ",".join(EXPECTED_FEATURES) + "\n"


# -------------------------------------------------------------------
# Load artifacts
# -------------------------------------------------------------------
try:
    scaler, models = load_artifacts()
except Exception as e:
    st.error(str(e))   # FIXED: st.Error -> st.error
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

st.sidebar.download_button(
    "Download sample CSV header",
    data=example_csv_text().encode("utf-8"),
    file_name="sample_header.csv",
    mime="text/csv",
)

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
        "Your CSV must include these 30 feature columns (case-insensitive match supported). "
        "If you want evaluation metrics/confusion matrix, include a label column like `target`."
    )

with right:
    st.subheader(f"Predictions & Evaluation — {model_name}")
    st.caption("Metrics/Confusion Matrix/Report appear when a label column is provided.")

    # Always show the evaluation section headers (for grading visibility)
    metrics_placeholder = st.empty()
    cm_placeholder = st.empty()
    report_placeholder = st.empty()

    if uploaded is None:
        st.warning("Upload a CSV from the sidebar to run predictions.")
        st.stop()

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")   # FIXED: st.Error -> st.error
        st.stop()

    # Separate labels if present
    y_true = None
    df_cols_norm = _normalize_cols(df.columns)
    label_col_norm = label_col.strip().lower() if label_col else ""

    if label_col_norm and label_col_norm in df_cols_norm:
        # map original column name that matches (in case of different case)
        orig_label_col = df.columns[df_cols_norm.index(label_col_norm)]
        try:
            y_true = df[orig_label_col].astype(int).to_numpy()
        except Exception:
            st.warning(f"Could not parse label column `{orig_label_col}` as integers. Metrics will be skipped.")
            y_true = None
        X_df = df.drop(columns=[orig_label_col])
    else:
        X_df = df

    # Validate columns
    X, missing_cols, extra_cols = validate_and_prepare_features(X_df)

    if missing_cols:
        st.error(
            "Uploaded CSV is missing required feature columns (case-insensitive):\n\n"
            + ", ".join(missing_cols[:20])
            + (" ..." if len(missing_cols) > 20 else "")
        )
        st.info("Tip: Use the 'Download sample CSV header' button in the sidebar and format your CSV accordingly.")
        st.stop()

    if extra_cols:
        st.info(
            "Extra columns detected (ignored):\n\n"
            + ", ".join(extra_cols[:20])
            + (" ..." if len(extra_cols) > 20 else "")
        )

    # Scale + predict
    X_scaled = scaler.transform(X)
    model = models[model_name]
    y_pred = model.predict(X_scaled)

    out_df = df.copy()
    out_df["prediction"] = y_pred

    y_score = get_auc_score_input(model, X_scaled)
    if y_score is not None:
        out_df["score_or_prob_class_1"] = y_score

    st.write("**Preview of Predictions:**")
    st.dataframe(out_df.head(25), use_container_width=True)

    # Evaluation section (always visible; content depends on labels)
    st.markdown("---")
    st.subheader("Evaluation (Uploaded CSV)")

    if y_true is None:
        metrics_placeholder.info(
            "No label column detected, so evaluation metrics/confusion matrix/report cannot be computed.\n\n"
            "Add a label column (e.g., `target`) with values 0/1 to your CSV to enable evaluation."
        )
    else:
        metrics = compute_metrics(y_true, y_pred, y_score if y_score is not None else None)
        metrics_placeholder.write("**Metrics:**")
        metrics_placeholder.dataframe(pd.DataFrame([metrics]).style.format("{:.4f}"), use_container_width=True)

        cm = confusion_matrix(y_true, y_pred)
        cm_placeholder.write("**Confusion Matrix:**")
        cm_placeholder.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

        report_placeholder.write("**Classification Report:**")
        report_placeholder.text(classification_report(y_true, y_pred, zero_division=0))

    # Download predictions
    st.markdown("---")
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
    )