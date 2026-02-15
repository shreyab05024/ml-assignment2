# model/train_xgboost.py

from model.utils import load_and_split, print_eval_details, save_artifact

def main():
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise SystemExit(
            "XGBoost is not installed.\n"
            "Run: pip install xgboost"
        )

    # Load dataset
    data = load_and_split(scale=True)

    # Define model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )

    # Train
    model.fit(data.X_train, data.y_train)

    # Evaluate
    print_eval_details("XGBoost", model, data.X_test, data.y_test)

    # Save model
    save_artifact("xgboost.pkl", model)
    save_artifact("scaler.pkl", data.scaler)

    print("\nModel saved to model/artifacts/xgboost.pkl")


if __name__ == "__main__":
    main()