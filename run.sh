#!/bin/bash
# Quick run script for Road Anomaly Detection System

cd "$(dirname "$0")"
source ~/venvs/road_model/bin/activate

echo "============================================================"
echo "Road Anomaly Detection System"
echo "============================================================"
echo ""
echo "Starting system..."
echo "Press Ctrl+C to stop"
echo ""

python main.py

