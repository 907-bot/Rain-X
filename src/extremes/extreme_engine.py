"""
RAIN-X Extreme Rainfall Engine
Computes calibrated exceedance probabilities for official IMD operational thresholds:
  - Moderate Rain: > 15.6 mm
  - Heavy Rain: > 64.5 mm
  - Very Heavy Rain: > 115.5 mm
  - Extremely Heavy Rain: > 204.4 mm
Generates official IMD Alert Color Codes (Green, Yellow, Orange, Red) and disaster warnings.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# Official IMD Meteorological Thresholds (mm/24h)
THRESHOLDS = {
    "moderate": 15.6,
    "heavy": 64.5,
    "very_heavy": 115.5,
    "extremely_heavy": 204.4
}

ALERT_LEVELS = {
    "GREEN": {"name": "No Warning", "color": "#10B981", "desc": "No severe weather expected. Normal activities may continue."},
    "YELLOW": {"name": "Watch (Be Updated)", "color": "#F59E0B", "desc": "Moderate rain with localized waterlogging. Keep track of weather updates."},
    "ORANGE": {"name": "Alert (Be Prepared)", "color": "#F97316", "desc": "Heavy to very heavy rainfall expected. Risk of localized flooding and disruption."},
    "RED": {"name": "Warning (Take Action)", "color": "#EF4444", "desc": "Extremely heavy rainfall imminent. High risk of severe flash floods, landslides, and transport breakdown."}
}


class ExtremeRainfallEngine:
    """
    Calibrates extreme precipitation exceedance probabilities using
    Generalized Extreme Value (GEV) and logistic-sigmoid transfer functions
    conditioned on regime uncertainty and atmospheric moisture.
    """
    def __init__(self):
        pass

    def compute_exceedance_probabilities(
        self,
        predicted_rainfall: float,
        uncertainty_spread: float = 15.0,
        regime_heavy_boost: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculates calibrated probabilities P(R > threshold) for the 4 IMD tiers.
        """
        probs = {}
        # Effective scale parameter
        scale = max(uncertainty_spread * 0.7, 4.0)

        for key, tau in THRESHOLDS.items():
            # Distance from threshold normalized by spread
            z = (predicted_rainfall - tau) / scale
            # Logistic sigmoid with tail-fattening factor for heavy convective regimes
            base_p = 1.0 / (1.0 + np.exp(-z))
            
            # Apply regime heavy tail adjustment
            adjusted_p = base_p ** (1.0 / regime_heavy_boost)
            adjusted_p = float(np.clip(adjusted_p, 0.01, 0.99))
            probs[key] = round(adjusted_p * 100.0, 1)

        return probs

    def determine_alert_level(
        self,
        predicted_rainfall: float,
        probs: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Maps rainfall and probabilities to the official IMD Color Alert.
        """
        p_heavy = probs.get("heavy", 0.0)
        p_very_heavy = probs.get("very_heavy", 0.0)
        p_extremely_heavy = probs.get("extremely_heavy", 0.0)

        if predicted_rainfall >= THRESHOLDS["extremely_heavy"] or p_extremely_heavy >= 25.0 or p_very_heavy >= 60.0:
            level = "RED"
        elif predicted_rainfall >= THRESHOLDS["heavy"] or p_heavy >= 50.0 or p_very_heavy >= 30.0:
            level = "ORANGE"
        elif predicted_rainfall >= THRESHOLDS["moderate"] or p_heavy >= 25.0:
            level = "YELLOW"
        else:
            level = "GREEN"

        alert_info = ALERT_LEVELS[level].copy()
        alert_info["code"] = level
        alert_info["action_plan"] = self._get_action_plan(level)
        return alert_info

    def _get_action_plan(self, level: str) -> List[str]:
        if level == "RED":
            return [
                "Issue red alert to State Disaster Management Authority (SDMA) & NDRF.",
                "Evacuate vulnerable low-lying riverine and landslide-prone foothill zones.",
                "Halt non-essential maritime, fishing, and hill highway transit.",
                "Open emergency sluice gates on reservoirs reaching FSL capacity."
            ]
        elif level == "ORANGE":
            return [
                "Deploy municipal desiltation and dewatering pumps in urban centers.",
                "Place disaster response teams and quick response medical units on standby.",
                "Advise fishing vessels to return to coastal harbors."
            ]
        elif level == "YELLOW":
            return [
                "Instruct civic agencies to monitor localized drainage bottlenecks.",
                "Issue advisory to farmers regarding standing crop runoff."
            ]
        else:
            return ["Routine meteorological monitoring. No civil restriction required."]
