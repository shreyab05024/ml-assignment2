# model/train_logistic_regression.py
from sklearn.linear_model import LogisticRegression
from model.utils import load_and_split, print_eval_details, save_artifact

def main():
    data = load_and_split(scale=True)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(data.X_train, data.y_train)

    print_eval_details("Logistic Regression", model, data.X_test, data.y_test)

    save_artifact("logistic_regression.pkl", model)
    save_artifact("scaler.pkl", data.scaler)  # save once (overwrites same file safely)
    print("\nSaved: artifacts/logistic_regression.pkl and artifacts/scaler.pkl")

if __name__ == "__main__":
    main()