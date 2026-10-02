"""
RAIN-X Master Training & Verification Script
Trains all 6 post-processing models on 2018-2021 data, validates on 2022,
and evaluates on held-out 2023-2024 test data.
Prints the official SIH 26080 Verification Scorecard:
  RMSE, MAE, Bias, POD, FAR, CSI, ETS, FSS
"""

import os
import sys
import json
import numpy as np

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data.dataset_generator import build_full_climatological_dataset
from src.pipeline import RainXEngine


def main():
    print("=" * 75)
    print(" RAIN-X: Regime-Aware Neural Post-Processing Engine (SIH 26080)")
    print(" National Centre for Medium Range Weather Forecasting (NCMRWF) Problem")
    print("=" * 75)

    data_dir = os.path.abspath("data")
    train_file = f"{data_dir}/processed/train_dataset.npz"

    # Step 1: Data Verification / Generation
    if not os.path.exists(train_file):
        print("\n[Step 1/3] Generating Synoptic Climatological Datasets (2018-2024)...")
        build_full_climatological_dataset(data_dir)
    else:
        print("\n[Step 1/3] Climatological Datasets found in data/processed/.")

    # Step 2: Initialize Engine and Train Models
    print("\n[Step 2/3] Initializing RAIN-X Engine & Training Model Comparison Layer...")
    engine = RainXEngine(data_dir=data_dir)
    engine.load_or_train_models()

    # Step 3: Run Official Verification on Held-Out Test Set (2023-2024)
    print("\n[Step 3/3] Running Rigorous Held-Out Evaluation (2023-2024 Test Set)...")
    verif = engine.run_held_out_verification()

    overall = verif["stratified"]["Overall Domain"]

    print("\n" + "=" * 75)
    print(" OFFICIAL SIH 26080 VERIFICATION BENCHMARK SCORECARD (>64.5 mm Heavy Rain)")
    print("=" * 75)
    header = f"{'MODEL':<24} | {'RMSE (mm)':<9} | {'CSI':<6} | {'ETS':<6} | {'POD':<6} | {'FAR':<6} | {'FSS':<6}"
    print(header)
    print("-" * 75)

    model_display_names = {
        "raw_nwp": "0. Raw NWP Forecast",
        "linear_bc": "1. Linear Bias Corr.",
        "quantile_mapping": "2. Quantile Mapping",
        "random_forest": "3. Random Forest",
        "lightgbm": "4. LightGBM Post-Proc.",
        "rain_x_moe": "5. RAIN-X MoE (Regime)"
    }

    for m_key, m_disp in model_display_names.items():
        m_eval = overall[m_key]
        line = (
            f"{m_disp:<24} | "
            f"{m_eval['rmse']:<9.1f} | "
            f"{m_eval['csi']:<6.3f} | "
            f"{m_eval['ets']:<6.3f} | "
            f"{m_eval['pod']:<6.3f} | "
            f"{m_eval['far']:<6.3f} | "
            f"{m_eval['fss']:<6.3f}"
        )
        print(line)

    print("=" * 75)
    
    # Save scorecard to json for API and Web Console
    metrics_path = f"{data_dir}/processed/scorecard.json"
    with open(metrics_path, "w") as f:
        json.dump(verif, f, indent=2)
    print(f"\n[Scorecard] Saved complete verification metrics to {metrics_path}")

    # Test single point prediction
    print("\n--- Example Operational Point Prediction (Anakapalli, Visakhapatnam) ---")
    pred = engine.predict_district("Andhra Pradesh", "Visakhapatnam", "Anakapalli", lead_time_hrs=24)
    print(f"Station: {pred['location']}")
    print(f"Detected Regime: {pred['regime_info']['dominant_regime']} (Confidence: {pred['regime_info']['confidence']}%)")
    print(f"Raw NWP Forecast: {pred['raw_nwp_mm']} mm")
    print(f"RAIN-X Corrected: {pred['corrected_mm']} mm (Interval: {pred['interval_q10_q50_q90'][0]} - {pred['interval_q10_q50_q90'][2]} mm)")
    print(f"Confidence Score: {pred['confidence_score']}%")
    print(f"IMD Alert: {pred['alert']['code']} - {pred['alert']['name']}")
    print(f"Meteorological Reason: {pred['explanation']['meteorological_reason']}")
    print(f"Analyst Agent Action: {pred['agent_briefing']['actionable_recommendation']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
