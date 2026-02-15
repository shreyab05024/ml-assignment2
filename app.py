# app.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

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

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False


st.set_page_config(page_title="ML Assignment 2 - Classifier Demo", layout="wide")

st.title("ML Assignment 2 — Classification Models Demo")
st.caption(
    "Dataset: Breast Cancer Wisconsin (UCI). Train/evaluate 6 models and test predictions via CSV upload."
)

# -----------------------------
# Helpers
# -----------------------------
def build_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }
    if _HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
        )
    return models


def compute_metrics(y_true, y_pred, y_prob=None):
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        out["AUC"] = roc_auc_score(y_true, y_prob)
    else:
        out["AUC"] = np.nan
    return out


@st.cache_resource
def train_and_evaluate():
    # Load dataset
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    X = df.drop(columns=["target"])
    y = df["target"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train models
    models = build_models()
    trained = {}
    results = []

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        trained[name] = model

        y_pred = model.predict(X_test_s)
        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_s)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics_row = {"Model": name, **metrics}
        results.append(metrics_row)

    results_df = pd.DataFrame(results).set_index("Model").sort_index()

    # Keep also raw test set for confusion matrix / report
    return {
        "feature_names": list(X.columns),
        "scaler": scaler,
        "models": trained,
        "X_test_scaled": X_test_s,
        "y_test": y_test.to_numpy(),
        "results_df": results_df,
    }


bundle = train_and_evaluate()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

model_names = list(bundle["models"].keys())
selected_model_name = st.sidebar.selectbox("Select a model", model_names, index=0)
selected_model = bundle["models"][selected_model_name]

label_col_hint = st.sidebar.text_input(
    "Optional: label column name in uploaded CSV (if present)",
    value="target",
    help="If your uploaded CSV includes ground-truth labels, enter the column name here (e.g., target/label/y).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("CSV Upload (Test Data)")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.subheader("Model Comparison (Held-out Test Split)")
    st.write("Metrics below are computed on an internal 80/20 train-test split.")

    # Display comparison table
    show_df = bundle["results_df"].copy()
    st.dataframe(show_df.style.format("{:.4f}"), use_container_width=True)

    st.subheader(f"Selected Model: {selected_model_name}")
    if selected_model_name in show_df.index:
        st.write("**Metrics (selected model):**")
        st.dataframe(show_df.loc[[selected_model_name]].style.format("{:.4f}"), use_container_width=True)

    st.markdown("---")
    st.subheader("Expected CSV Format")
    st.write(
        "Upload a CSV containing the same feature columns as the dataset. "
        "If you include labels too, add a label column (e.g., `target`) and specify its name in the sidebar."
    )
    st.code("\n".join(bundle["feature_names"]), language="text")

with right:
    st.subheader("Evaluation Outputs (Held-out Test Split)")
    X_test_s = bundle["X_test_scaled"]
    y_test = bundle["y_test"]

    # Evaluate selected model on held-out test split
    y_pred = selected_model.predict(X_test_s)
    y_prob = None
    if hasattr(selected_model, "predict_proba"):
        y_prob = selected_model.predict_proba(X_test_s)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    st.write("**Confusion Matrix (test split):**")
    st.write(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

    st.write("**Classification Report (test split):**")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    if uploaded is not None:
        st.markdown("---")
        st.subheader("Predictions on Uploaded CSV")

        try:
            up_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        # If label column exists, separate it
        y_true_upload = None
        if label_col_hint and label_col_hint in up_df.columns:
            y_true_upload = up_df[label_col_hint].copy()
            up_X = up_df.drop(columns=[label_col_hint])
        else:
            up_X = up_df.copy()

        # Validate columns (best-effort)
        expected = bundle["feature_names"]
        missing = [c for c in expected if c not in up_X.columns]
        extra = [c for c in up_X.columns if c not in expected]

        if missing:
            st.warning(
                "Your uploaded CSV is missing expected columns:\n"
                + ", ".join(missing[:20])
                + (" ..." if len(missing) > 20 else "")
            )
        if extra:
            st.info(
                "Your uploaded CSV has extra columns that will be ignored:\n"
                + ", ".join(extra[:20])
                + (" ..." if len(extra) > 20 else "")
            )

        # Reorder/select expected columns only
        up_X = up_X[[c for c in expected if c in up_X.columns]]

        # Handle if still empty
        if up_X.shape[1] == 0:
            st.error("No usable feature columns found after matching expected columns.")
            st.stop()

        # Scale & predict
        scaler = bundle["scaler"]
        up_X_s = scaler.transform(up_X)

        up_pred = selected_model.predict(up_X_s)
        out_df = up_df.copy()
        out_df["prediction"] = up_pred

        if hasattr(selected_model, "predict_proba"):
            out_df["probability_class_1"] = selected_model.predict_proba(up_X_s)[:, 1]

        st.write("**Preview:**")
        st.dataframe(out_df.head(20), use_container_width=True)

        # If labels exist, show metrics
        if y_true_upload is not None:
            try:
                y_true_upload = y_true_upload.astype(int).to_numpy()
                up_prob = out_df["probability_class_1"].to_numpy() if "probability_class_1" in out_df.columns else None
                upload_metrics = compute_metrics(y_true_upload, up_pred, up_prob)

                st.write("**Metrics (uploaded CSV):**")
                st.dataframe(
                    pd.DataFrame([upload_metrics], index=[selected_model_name]).style.format("{:.4f}"),
                    use_container_width=True,
                )

                cm_up = confusion_matrix(y_true_upload, up_pred)
                st.write("**Confusion Matrix (uploaded CSV):**")
                st.write(pd.DataFrame(cm_up, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

                st.write("**Classification Report (uploaded CSV):**")
                st.text(classification_report(y_true_upload, up_pred, zero_division=0))
            except Exception as e:
                st.warning(f"Could not compute uploaded-data metrics (label column issue): {e}")
        else:
            st.info(
                "No label column detected in uploaded CSV, so metrics/confusion matrix for uploaded data cannot be computed."
            )

        # Download predictions
        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download predictions as CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )

# Footer note for xgboost
if not _HAS_XGBOOST:
    st.warning(
        "XGBoost is not available in this environment. "
        "Install it (xgboost) and redeploy to include the 6th model."
    )