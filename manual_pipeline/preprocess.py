import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_manual():
    # Load dataset
    df = pd.read_csv("dataset/train.csv")

    # Drop columns not needed
    df = df.drop(["Cabin", "Ticket", "Name"], axis=1)

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # One-hot encoding
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    # Scale certain numeric columns
    scaler = StandardScaler()
    numeric_cols = ["Age", "Fare"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save processed dataset
    output_path = "manual_pipeline/processed_manual.csv"
    df.to_csv(output_path, index=False)
    print(f"Preprocessing completed. Saved to {output_path}")

if __name__ == "__main__":
    preprocess_manual()
  
