import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def preprocess_data():
    df = pd.read_csv("dataset/train.csv")

    df = df.drop(["Cabin", "Ticket", "Name"], axis=1)

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

    df = df.dropna()

    scaler = StandardScaler()
    numeric_cols = ["Age", "Fare"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df.to_csv("dataset/processed.csv", index=False)
    print("Preprocessing Selesai. File disimpan di dataset/processed.csv")

if __name__ == "__main__":
    preprocess_data()
