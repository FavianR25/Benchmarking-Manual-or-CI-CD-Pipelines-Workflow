# cicd_pipeline/preprocess.py

import pandas as pd

def preprocess():
    print("[CI/CD] Preprocessing started...")

    df = pd.read_csv("train.csv")

    # Drop non-numerical & high-missing columns
    df = df.drop(columns=["Name", "Ticket", "Cabin"])

    # Fill missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Encode Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # One-hot Embarked
    df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    df.to_csv("cicd_pipeline/processed_cicd.csv", index=False)

    print("[CI/CD] Preprocessing complete → cicd_pipeline/processed_cicd.csv")


if __name__ == "__main__":
    preprocess()
