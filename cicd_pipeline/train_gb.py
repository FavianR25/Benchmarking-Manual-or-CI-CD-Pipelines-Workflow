import time
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import log_cicd

def train_gb():
    print("[CI/CD] Training Gradient Boosting...")

    df = pd.read_csv("cicd_pipeline/processed_cicd.csv")
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    start_time = time.time()

    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    preds = model.predict(X_test)
    error_rate = 1 - accuracy_score(y_test, preds)

    log_cicd({
        "workflow": "cicd",
        "model": "gradient_boosting",
        "deployment_time": train_time,
        "error_rate": error_rate,
        "reproducibility": 1,
        "setup_time": 0,
        "cpu_usage": 0,
        "memory_usage_mb": 0
    })

    print("[CI/CD] Gradient Boosting training completed.")

if __name__ == "__main__":
    train_gb()

