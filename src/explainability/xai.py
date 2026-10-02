"""
RAIN-X Meteorological Explainability Engine (XAI)
Generates physical attribution, atmospheric contributor diagnostics,
regime historical bias analysis, and auditable plain-language explanations.
"""

from typing import Dict, List, Any
import numpy as np


class MeteorologicalXAI:
    """
    Translates neural activations, regime gating weights, and atmospheric state gradients
    into domain-specific meteorological explanations.
    """
    def __init__(self):
        # Known historical NWP systematic error profiles per regime
        self.regime_historical_profiles = {
            "active_monsoon": {
                "typical_bias": "+15% to +25% overprediction in light convective rain, underprediction of localized heavy spells",
                "physical_cause": "Convective parameterization schemes (e.g. Tiedtke / Kain-Fritsch) trigger too early and produce widespread light precipitation while dispersing concentrated core updrafts."
            },
            "break_monsoon": {
                "typical_bias": "Spurious central India convection (+10 mm error), underprediction of Himalayan foothills orographic bursts",
                "physical_cause": "Model struggles with dry air intrusion over central India and fails to confine moisture along the sub-Himalayan shear line."
            },
            "low_depression": {
                "typical_bias": "Underprediction of core eyewall/shear rainfall (-20% to -35%), slight track displacement",
                "physical_cause": "Coarse 0.25° NWP grid underestimates localized boundary layer vortex convergence and vortex moisture pump."
            },
            "coastal": {
                "typical_bias": "Underprediction of night-to-early-morning coastal convergence peaks (-25%)",
                "physical_cause": "Unresolved land-sea thermal friction contrast and sea breeze convergence lines along the coastal boundary."
            },
            "orographic": {
                "typical_bias": "Severe windward underprediction (-40% to -60%), windward/leeward spatial blurring",
                "physical_cause": "Smoothed model orography fails to capture steep vertical uplifting along the Western Ghats / Meghalaya escarpments."
            },
            "western_disturbance": {
                "typical_bias": "Misplacement of precipitation bands and snow/rain line elevation errors",
                "physical_cause": "Baroclinic mid-latitude wave interaction with high complex Himalayan terrain."
            }
        }

    def generate_explanation(
        self,
        raw_nwp: float,
        corrected_rain: float,
        regime_info: Dict[str, Any],
        phys_features: Dict[str, float],
        location_name: str = "Target Station"
    ) -> Dict[str, Any]:
        """
        Synthesizes a full meteorological XAI audit report.
        """
        dom_key = regime_info.get("dominant_key", "active_monsoon")
        dom_name = regime_info.get("dominant_regime", "Active Monsoon")
        confidence = regime_info.get("confidence", 80.0)
        profile = self.regime_historical_profiles.get(dom_key, self.regime_historical_profiles["active_monsoon"])

        delta_mm = corrected_rain - raw_nwp
        pct_change = (delta_mm / (raw_nwp + 1e-4)) * 100.0

        # Determine directional reason
        if delta_mm > 1.0:
            direction = "INCREASED"
            adjustment_summary = f"RAIN-X augmented forecast by +{delta_mm:.1f} mm (+{pct_change:.1f}%)"
            reason = f"Historical NWP runs under {dom_name} systematically underpredict precipitation due to: {profile['physical_cause']}"
        elif delta_mm < -1.0:
            direction = "DECREASED"
            adjustment_summary = f"RAIN-X reduced forecast by {abs(delta_mm):.1f} mm ({pct_change:.1f}%)"
            reason = f"Historical NWP runs under {dom_name} suffer from convective over-triggering: {profile['typical_bias']}"
        else:
            direction = "CONSISTENT"
            adjustment_summary = "RAIN-X confirmed raw NWP within physical uncertainty bounds"
            reason = f"NWP state aligns well with historical {dom_name} balance."

        # Meteorological Driver Diagnostics
        contributors = []
        mfc = phys_features.get("mfc", 0.0)
        if mfc > 0.8:
            contributors.append({"factor": "Moisture Flux Convergence (-∇·qV)", "effect": "POSITIVE", "weight": "+++", "desc": "Strong influx of precipitable water into boundary layer"})
        elif mfc < -0.5:
            contributors.append({"factor": "Moisture Flux Divergence", "effect": "NEGATIVE", "weight": "--", "desc": "Drying trend suppressing convective organization"})

        vort = phys_features.get("vorticity_850", 0.0)
        if vort > 1.8:
            contributors.append({"factor": "850 hPa Cyclonic Vorticity (ζ)", "effect": "POSITIVE", "weight": "+++", "desc": "Organized low-level cyclonic shear sustaining updrafts"})

        slp_a = phys_features.get("slp_anomaly", 0.0)
        if slp_a < -2.0:
            contributors.append({"factor": "Surface Pressure Deficit (ΔSLP)", "effect": "POSITIVE", "weight": "++", "desc": f"Deep depression depression anomaly ({slp_a:.1f} hPa)"})

        ofi = phys_features.get("ofi", 0.0)
        if ofi > 1.2:
            contributors.append({"factor": "Orographic Windward Uplift (V·∇h)", "effect": "POSITIVE", "weight": "+++", "desc": "Strong orthogonal onshore flow impinging on mountain slope"})

        llj = phys_features.get("llj_speed", 0.0)
        if llj > 12.0:
            contributors.append({"factor": "Monsoon Low-Level Jet (LLJ)", "effect": "POSITIVE", "weight": "++", "desc": f"Robust {llj:.1f} m/s westerly momentum transport"})

        if not contributors:
            contributors.append({"factor": "Equilibrium Synoptic Forcing", "effect": "NEUTRAL", "weight": "+", "desc": "Moderate ambient moisture and weak vertical shear"})

        return {
            "location": location_name,
            "regime": dom_name,
            "regime_key": dom_key,
            "regime_confidence": confidence,
            "raw_nwp_mm": round(raw_nwp, 1),
            "corrected_mm": round(corrected_rain, 1),
            "delta_mm": round(delta_mm, 1),
            "pct_change": round(pct_change, 1),
            "direction": direction,
            "adjustment_summary": adjustment_summary,
            "meteorological_reason": reason,
            "historical_nwp_bias": profile["typical_bias"],
            "contributors": contributors
        }
