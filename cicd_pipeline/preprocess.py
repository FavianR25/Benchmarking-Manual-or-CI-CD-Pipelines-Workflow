import pandas as pd
import os

INPUT_PATH = "dataset/train.csv"
OUTPUT_PATH = "cicd_pipeline/processed_cicd.csv"

def preprocess():
    print("CI/CD Preprocessing started...")

    df = pd.read_csv(INPUT_PATH)

    df.fillna({
        "Age": df["Age"].median(),
        "Fare": df["Fare"].median()
    }, inplace=True)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["Embarked"].fillna(0, inplace=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessing completed. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()

