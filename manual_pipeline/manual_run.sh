#!/bin/bash

echo "========================="
echo "MANUAL PIPELINE RUN START"
echo "========================="

echo "[1/3] Preprocessing..."
python3 manual_pipeline/preprocess.py

echo "[2/3] Training SVM..."
python3 manual_pipeline/train_svm.py

echo "[3/3] Training Gradient Boosting..."
python3 manual_pipeline/train_gb.py

echo "========================="
echo "MANUAL PIPELINE COMPLETED"
echo "Logs saved in manual_pipeline/log_manual.csv"
echo "========================="

