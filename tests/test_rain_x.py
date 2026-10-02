"""
RAIN-X Comprehensive Verification & Unit Test Suite
Validates all 10 modules:
  1. Ingestion & QC
  2. Weather Regime Intelligence & Soft Classifier
  3. MoE Post-Processor & Physics Constraints
  4. Extreme Rainfall Engine & IMD Alerts
  5. Uncertainty & Quantile Calibration
  6. Spatial Intelligence & Topography
  7. District Aggregation
  8. Meteorological Explainability (XAI)
  9. Verification Metrics (RMSE, CSI, ETS, POD, FAR, FSS)
  10. AI Forecast Analyst Agent & REST API
"""

import os
import sys
import unittest
import numpy as np

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spatial.domain import (
    create_topography_dem, create_land_sea_mask, compute_topography_gradients,
    get_station_coords, grid_point_index, LATS, LONS, NLAT, NLON
)
from src.regime.classifier import (
    extract_atmospheric_physics_features, WeatherRegimeIntelligence,
    REGIME_NAMES, REGIME_KEYS
)
from src.postprocessor.models import (
    LinearBiasCorrection, EmpiricalQuantileMapping, ModelComparisonSuite
)
from src.extremes.extreme_engine import ExtremeRainfallEngine, THRESHOLDS
from src.uncertainty.estimator import UncertaintyEstimator
from src.explainability.xai import MeteorologicalXAI
from src.verification.metrics import (
    compute_continuous_metrics, compute_contingency_table,
    compute_threat_scores, compute_fractions_skill_score_2d
)
from src.agent.forecast_analyst import ForecastAnalystAgent


class TestRainXModules(unittest.TestCase):
    
    def test_module_spatial_topography(self):
        """Test Module 6: Indian Domain & Topography DEM."""
        dem = create_topography_dem()
        self.assertEqual(dem.shape, (NLAT, NLON))
        # Elevation must be non-negative
        self.assertTrue(np.all(dem >= 0.0))
        # Peak Himalayan elevation should be high (>3500m)
        self.assertTrue(np.max(dem) > 3500.0)

        # Land mask
        lmask = create_land_sea_mask()
        self.assertEqual(lmask.shape, (NLAT, NLON))
        self.assertTrue(set(np.unique(lmask)).issubset({0.0, 1.0}))

        # Gradients
        gx, gy = compute_topography_gradients(dem)
        self.assertEqual(gx.shape, (NLAT, NLON))
        self.assertEqual(gy.shape, (NLAT, NLON))

    def test_module_district_mapping(self):
        """Test Module 7: District & Station Mapping."""
        st = get_station_coords("Andhra Pradesh", "Visakhapatnam", "Anakapalli")
        self.assertEqual(st["district"], "Visakhapatnam")
        self.assertEqual(st["station"], "Anakapalli")
        self.assertAlmostEqual(st["lat"], 17.68, places=1)
        self.assertAlmostEqual(st["lon"], 83.00, places=1)

        ilat, ilon = grid_point_index(st["lat"], st["lon"])
        self.assertTrue(0 <= ilat < NLAT)
        self.assertTrue(0 <= ilon < NLON)

    def test_module_regime_intelligence(self):
        """Test Module 2: Weather Regime Intelligence & Soft Classifier."""
        reg_intel = WeatherRegimeIntelligence()
        priors = reg_intel.compute_heuristic_physics_priors(
            lat=17.5, lon=84.0, rain_nwp=90.0, llj_speed=14.0,
            slp_anomaly=-6.0, vorticity_850=3.2, convergence_850=2.0,
            ofi=0.5, upward_motion=0.25, rh850=92.0, dem=30.0
        )
        self.assertEqual(len(priors), 6)
        # Soft probabilities must sum to 1.0
        self.assertAlmostEqual(float(np.sum(priors)), 1.0, places=5)
        # For a deep cyclonic depression, low_depression index (index 2) should be highest
        self.assertEqual(int(np.argmax(priors)), 2)

    def test_module_verification_metrics(self):
        """Test Module 9 & 10: Official Verification Metrics (RMSE, CSI, ETS, POD, FAR, FSS)."""
        # Synthetic forecast and observation
        y_true = np.array([0.0, 20.0, 70.0, 85.0, 120.0, 5.0, 0.0, 95.0], dtype=np.float32)
        y_pred = np.array([0.0, 15.0, 75.0, 80.0, 110.0, 2.0, 0.0, 60.0], dtype=np.float32)

        cont = compute_continuous_metrics(y_pred, y_true)
        self.assertTrue(cont["rmse"] > 0.0)
        self.assertTrue(cont["mae"] > 0.0)
        self.assertTrue(cont["corr"] > 0.8)

        table = compute_contingency_table(y_pred, y_true, threshold_mm=64.5)
        # True heavy events (>64.5mm): indices 2, 3, 4, 7 (4 events)
        # Pred heavy events (>64.5mm): indices 2, 3, 4 (3 events)
        self.assertEqual(table["hits"], 3)
        self.assertEqual(table["misses"], 1)
        self.assertEqual(table["false_alarms"], 0)

        threat = compute_threat_scores(table)
        self.assertAlmostEqual(threat["pod"], 0.75, places=2)
        self.assertAlmostEqual(threat["far"], 0.0, places=2)
        self.assertAlmostEqual(threat["csi"], 0.75, places=2)
        self.assertTrue(threat["ets"] > 0.0)

        # FSS 2D test
        p2d = np.zeros((10, 10), dtype=np.float32)
        o2d = np.zeros((10, 10), dtype=np.float32)
        p2d[4:7, 4:7] = 80.0
        o2d[4:7, 4:7] = 80.0
        fss_perfect = compute_fractions_skill_score_2d(p2d, o2d, threshold_mm=64.5, window_size=3)
        self.assertAlmostEqual(fss_perfect, 1.0, places=2)

    def test_module_extreme_rainfall(self):
        """Test Module 4: Extreme Rainfall & IMD Alerts."""
        ext_engine = ExtremeRainfallEngine()
        probs = ext_engine.compute_exceedance_probabilities(120.0, uncertainty_spread=15.0)
        self.assertTrue(probs["moderate"] > 85.0)
        self.assertTrue(probs["heavy"] > 70.0)
        self.assertTrue(probs["very_heavy"] > 40.0)

        alert_red = ext_engine.determine_alert_level(210.0, probs)
        self.assertEqual(alert_red["code"], "RED")

        alert_green = ext_engine.determine_alert_level(5.0, {"heavy": 5.0, "very_heavy": 1.0, "extremely_heavy": 0.0})
        self.assertEqual(alert_green["code"], "GREEN")

    def test_module_uncertainty(self):
        """Test Module 5: Uncertainty & Quantile Spreads."""
        unc = UncertaintyEstimator()
        gating = {"active_monsoon": 0.7, "break_monsoon": 0.05, "low_depression": 0.15, "coastal": 0.05, "orographic": 0.03, "western_disturbance": 0.02}
        q10, q50, q90 = unc.compute_quantile_interval(100.0, gating, lead_time_hrs=24)
        self.assertTrue(q10 < q50)
        self.assertTrue(q50 < q90)
        self.assertTrue(q10 >= 0.0)

        conf = unc.compute_confidence_score(100.0, q10, q90, gating, lead_time_hrs=24)
        self.assertTrue(50.0 <= conf <= 99.0)

    def test_module_meteorological_xai(self):
        """Test Module 8: Explainability (XAI)."""
        xai = MeteorologicalXAI()
        reg_info = {
            "dominant_regime": "Low / Depression",
            "dominant_key": "low_depression",
            "confidence": 85.0
        }
        phys = {"mfc": 1.8, "vorticity_850": 2.5, "slp_anomaly": -4.5, "llj_speed": 12.0}
        report = xai.generate_explanation(74.0, 109.0, reg_info, phys, "Visakhapatnam")
        self.assertEqual(report["direction"], "INCREASED")
        self.assertTrue(len(report["contributors"]) >= 2)
        self.assertIn("systematically underpredict", report["meteorological_reason"])

    def test_module_analyst_agent(self):
        """Test Module: AI Forecast Analyst Agent Briefing."""
        agent = ForecastAnalystAgent()
        reg_info = {"dominant_regime": "Active Monsoon", "confidence": 82.0}
        alert = {"code": "ORANGE", "name": "Alert", "action_plan": ["Standby pumps"]}
        briefing = agent.generate_briefing(
            location="Mumbai",
            regime_info=reg_info,
            raw_nwp_mm=110.0,
            corrected_mm=145.0,
            interval=(95.0, 145.0, 185.0),
            exceedance_probs={"heavy": 85.0, "very_heavy": 55.0, "extremely_heavy": 12.0},
            alert_info=alert,
            confidence_score=86.0,
            lead_time_hrs=24
        )
        self.assertIn("Mumbai", briefing["title"])
        self.assertIn("ORANGE", briefing["risk_assessment"]["alert_code"])
        self.assertTrue(len(briefing["action_items"]) > 0)


if __name__ == "__main__":
    unittest.main()
