# model/train_naive_bayes.py
from sklearn.naive_bayes import GaussianNB
from model.utils import load_and_split, print_eval_details, save_artifact

def main():
    data = load_and_split(scale=True)

    model = GaussianNB()
    model.fit(data.X_train, data.y_train)

    print_eval_details("Naive Bayes (Gaussian)", model, data.X_test, data.y_test)

    save_artifact("naive_bayes.pkl", model)
    save_artifact("scaler.pkl", data.scaler)
    print("\nSaved: artifacts/naive_bayes.pkl and artifacts/scaler.pkl")

if __name__ == "__main__":
    main()