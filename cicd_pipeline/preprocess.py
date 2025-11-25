import pandas as pd

INPUT_PATH = "train.csv"
OUTPUT_PATH = "cicd_pipeline/processed_cicd.csv"

def preprocess():
    print("[CI/CD] Preprocessing started...")

    df = pd.read_csv(INPUT_PATH)

    # === 1. Drop columns that are not numeric or unnecessary ===
    df = df.drop(columns=["Name", "Ticket", "Cabin"])

    # === 2. Fill numeric missing values ===
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    # === 3. Encode categorical columns ===
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["Embarked"].fillna(0, inplace=True)

    # === 4. Save cleaned dataset ===
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[CI/CD] Preprocessing completed. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    preprocess()
