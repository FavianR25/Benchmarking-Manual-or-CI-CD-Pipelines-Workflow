import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from metrics_logger import log_metrics

def train_svm():
    df = pd.read_csv("dataset/processed.csv")

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = SVC()
    model.fit(X_train, y_train)

    log_metrics("svm_manual")

if __name__ == "__main__":
    train_svm()
