import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_cicd():
    print("[CI/CD] Preprocessing...")

    df = pd.read_csv("train.csv")

    # Drop columns identical to manual pipeline
    df = df.drop(["Cabin", "Ticket", "Name"], axis=1)

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # One-hot encoding
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    # Scale numerical data (same as manual pipeline)
    scaler = StandardScaler()
    numeric_cols = ["Age", "Fare"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    output_path = "cicd_pipeline/processed_cicd.csv"
    df.to_csv(output_path, index=False)

    print(f"[CI/CD] Preprocessing completed → {output_path}")

if __name__ == "__main__":
    preprocess_cicd()
