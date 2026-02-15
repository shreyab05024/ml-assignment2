# model/train_xgboost.py
from model.utils import load_and_split, print_eval_details, save_artifact

def main():
    # Import inside main so the script fails with a clear message if xgboost isn't installed
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        raise SystemExit(
            "XGBoost is not installed. Run: pip install xgboost\n"
            f"Original error: {e}"
        )

    data = load_and_split(scale=True)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(data.X_train, data.y_train)

    print_eval_details("XGBoost", model, data.X_test, data.y_test)

    save_artifact("xgboost.pkl", model)
    save_artifact("scaler.pkl", data.scaler)
    print("\nSaved: artifacts/xgboost.pkl and artifacts/scaler.pkl")

if __name__ == "__main__":
    main()