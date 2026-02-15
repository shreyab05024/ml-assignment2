# model/train_random_forest.py
from sklearn.ensemble import RandomForestClassifier
from model.utils import load_and_split, print_eval_details, save_artifact

def main():
    data = load_and_split(scale=True)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    model.fit(data.X_train, data.y_train)

    print_eval_details("Random Forest", model, data.X_test, data.y_test)

    save_artifact("random_forest.pkl", model)
    save_artifact("scaler.pkl", data.scaler)
    print("\nSaved: artifacts/random_forest.pkl and artifacts/scaler.pkl")

if __name__ == "__main__":
    main()