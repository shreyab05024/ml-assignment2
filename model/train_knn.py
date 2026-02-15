# model/train_knn.py
from sklearn.neighbors import KNeighborsClassifier
from model.utils import load_and_split, print_eval_details, save_artifact

def main():
    data = load_and_split(scale=True)

    model = KNeighborsClassifier(n_neighbors=5, metric="minkowski")
    model.fit(data.X_train, data.y_train)

    print_eval_details("KNN", model, data.X_test, data.y_test)

    save_artifact("knn.pkl", model)
    save_artifact("scaler.pkl", data.scaler)
    print("\nSaved: artifacts/knn.pkl and artifacts/scaler.pkl")

if __name__ == "__main__":
    main()