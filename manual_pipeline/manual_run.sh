#!/bin/bash

PYTHON="/c/Python314/python"

echo "========================="
echo "MANUAL PIPELINE RUN START"
echo "========================="

echo "[1/3] Preprocessing..."
$PYTHON manual_pipeline/preprocess.py

echo "[2/3] Training SVM..."
$PYTHON manual_pipeline/train_svm.py

echo "[3/3] Training Gradient Boosting..."
$PYTHON manual_pipeline/train_gb.py

echo "========================="
echo "MANUAL PIPELINE COMPLETED"
echo "Logs saved in manual_pipeline/log_manual.csv"
echo "========================="
