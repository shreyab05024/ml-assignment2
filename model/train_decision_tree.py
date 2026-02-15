# model/train_decision_tree.py
from sklearn.tree import DecisionTreeClassifier
from model.utils import load_and_split, print_eval_details, save_artifact

def main():
    data = load_and_split(scale=True)

    model = DecisionTreeClassifier(random_state=42, max_depth=None)
    model.fit(data.X_train, data.y_train)

    print_eval_details("Decision Tree", model, data.X_test, data.y_test)

    save_artifact("decision_tree.pkl", model)
    save_artifact("scaler.pkl", data.scaler)
    print("\nSaved: artifacts/decision_tree.pkl and artifacts/scaler.pkl")

if __name__ == "__main__":
    main()