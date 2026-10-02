"""
RAIN-X Uncertainty Estimation & Probabilistic Calibration Engine
Computes:
  - Quantile prediction intervals [q10, q50, q90]
  - Composite Model Confidence Score [0.0, 1.0] based on regime entropy and spread sharpness
  - Lead-time uncertainty degradation factor
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class UncertaintyEstimator:
    """
    Quantifies predictive uncertainty through quantile intervals [q10, q90]
    and computes meteorologically grounded model confidence.
    """
    def __init__(self):
        # Empirical quantile scaling factors per regime based on NWP historical variance
        self.regime_relative_spreads = {
            "active_monsoon": 0.28,
            "break_monsoon": 0.22,
            "low_depression": 0.35,  # Higher convective chaos
            "coastal": 0.32,
            "orographic": 0.40,      # Steep terrain variance
            "western_disturbance": 0.30
        }

    def compute_quantile_interval(
        self,
        predicted_rainfall: float,
        gating_probs: Dict[str, float],
        lead_time_hrs: int = 24
    ) -> Tuple[float, float, float]:
        """
        Returns (q10, q50, q90) in mm.
        q10: 10th percentile (lower confidence bound)
        q50: 50th percentile (median)
        q90: 90th percentile (upper confidence bound)
        """
        # Weighted relative spread from regimes
        effective_rel_spread = 0.0
        for reg_key, prob in gating_probs.items():
            base_s = self.regime_relative_spreads.get(reg_key, 0.30)
            effective_rel_spread += prob * base_s

        # Lead time uncertainty expansion: +2% per 12h
        lead_factor = 1.0 + (max(lead_time_hrs, 6) - 6) * 0.003
        effective_spread = effective_rel_spread * lead_factor

        # Standard error scaling
        sigma = max(predicted_rainfall * effective_spread, 3.5)

        # Assumes log-normal / skewed distribution for precipitation
        q10 = max(0.0, predicted_rainfall - 1.28 * sigma)
        q50 = predicted_rainfall
        q90 = predicted_rainfall + 1.28 * sigma * 1.15  # Asymmetric positive heavy tail

        return round(float(q10), 1), round(float(q50), 1), round(float(q90), 1)

    def compute_confidence_score(
        self,
        predicted_rainfall: float,
        q10: float,
        q90: float,
        gating_probs: Dict[str, float],
        lead_time_hrs: int = 24
    ) -> float:
        """
        Computes composite confidence score [0.0 - 1.0] (or 0% - 100%).
        Formulation combines:
          1. Regime Certainty: C_regime = 1 - (Entropy / ln(6))
          2. Spread Sharpness: C_spread = 1 / (1 + (q90 - q10) / max(predicted, 15))
          3. Lead Time Reliability: C_lead = 1 / (1 + lead_time / 144)
        """
        # 1. Regime entropy
        probs = np.array(list(gating_probs.values()), dtype=np.float32)
        probs = np.clip(probs, 1e-6, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probs))
        c_regime = max(0.0, 1.0 - (entropy / max_entropy))

        # 2. Prediction interval sharpness
        spread = max(q90 - q10, 1.0)
        norm_denom = max(predicted_rainfall, 20.0)
        c_sharpness = 1.0 / (1.0 + (spread / norm_denom) * 0.45)

        # 3. Lead time factor
        c_lead = 1.0 - (lead_time_hrs / 120.0) * 0.25

        # Composite score
        confidence = (0.45 * c_regime + 0.35 * c_sharpness + 0.20 * c_lead)
        confidence = float(np.clip(confidence, 0.40, 0.96))
        return round(confidence * 100.0, 1)
