"""
RAIN-X Dataset Generator & Historical Benchmark Archive
Generates scientifically grounded atmospheric synoptic fields and rainfall observations
across chronological partitions:
  - 2018-2021: Training set (monsoon, depression, WD, breaks, coastal)
  - 2022: Validation set
  - 2023-2024: Held-out Test set + Famous SIH Benchmark Extreme Events:
      1. Cyclone Michaung (Dec 2023 - Coastal/Depression)
      2. Wayanad Extreme Orographic Event (July 2024 - Orographic Cloudburst)
      3. July 2023 North India Western Disturbance (WD + Monsoon)
      4. August 2023 Core Monsoon Break Spell (Break Monsoon)
      5. Bay of Bengal Deep Depression (Sept 2023)
      6. Active Monsoon Peak Surge (July 2022)
"""

import os
import sys
import json
import numpy as np
from typing import Dict, List, Tuple, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spatial.domain import (
    LATS, LONS, NLAT, NLON, create_topography_dem,
    create_land_sea_mask, compute_topography_gradients, DISTRICT_CATALOG
)
from src.regime.classifier import (
    extract_atmospheric_physics_features, WeatherRegimeIntelligence,
    REGIME_NAMES, REGIME_KEYS
)


def generate_synoptic_field(
    regime_type: str,
    date_str: str,
    lead_time_hrs: int = 24,
    dem: np.ndarray = None,
    grad_x: np.ndarray = None,
    grad_y: np.ndarray = None,
    land_mask: np.ndarray = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Synthesizes physically consistent 2D gridded atmospheric state and matching
    NWP forecast + observed IMD precipitation fields based on meteorological equations.
    """
    rng = np.random.RandomState(seed)
    lon_grid, lat_grid = np.meshgrid(LONS, LATS)

    # Base atmospheric thermodynamic state
    t2m = 273.15 + 28.0 - 0.4 * (lat_grid - 15.0) - (dem / 1000.0) * 6.5
    mslp = 1008.0 - 0.1 * (lat_grid - 20.0)
    rh850 = 70.0 + rng.uniform(-5.0, 5.0, size=(NLAT, NLON))
    tcwv = 45.0 + rng.uniform(-3.0, 3.0, size=(NLAT, NLON))
    z500 = 5840.0 + (lat_grid - 20.0) * (-2.0)
    omega500 = rng.uniform(-0.05, 0.05, size=(NLAT, NLON))
    u850 = np.full((NLAT, NLON), 6.0, dtype=np.float32)
    v850 = np.full((NLAT, NLON), 1.0, dtype=np.float32)
    u200 = np.full((NLAT, NLON), -15.0, dtype=np.float32) # Tropical Easterly Jet
    v200 = np.zeros((NLAT, NLON), dtype=np.float32)

    rain_nwp = np.zeros((NLAT, NLON), dtype=np.float32)
    rain_obs = np.zeros((NLAT, NLON), dtype=np.float32)

    # -------------------------------------------------------------
    # 1. ACTIVE MONSOON REGIME
    # -------------------------------------------------------------
    if regime_type == "active_monsoon":
        # Strong low-level jet across peninsular India (8N - 18N)
        peninsula_mask = (lat_grid >= 10.0) & (lat_grid <= 18.0)
        u850 += 12.0 * np.exp(-((lat_grid - 14.5) ** 2) / 10.0)
        # Trough line at ~21N with negative SLP
        mslp -= 6.0 * np.exp(-((lat_grid - 21.0) ** 2) / 8.0)
        rh850 = np.clip(rh850 + 15.0, 0.0, 95.0)
        omega500 -= 0.18 * np.exp(-((lat_grid - 21.0) ** 2) / 12.0)

        # Widespread rainfall over Central India & Western Ghats
        core_rain = 75.0 * np.exp(-((lat_grid - 21.5) ** 2) / 10.0 - ((lon_grid - 80.0) ** 2) / 60.0)
        ghats_rain = 90.0 * (dem / 1200.0) * (lon_grid <= 75.5) * (lat_grid <= 20.0)
        rain_obs = core_rain + ghats_rain + rng.exponential(6.0, size=(NLAT, NLON)) * land_mask
        
        # Raw NWP underpredicts extreme core, overpredicts surrounding light drizzle
        rain_nwp = 0.72 * core_rain + 0.55 * ghats_rain + rng.exponential(12.0, size=(NLAT, NLON)) * land_mask

    # -------------------------------------------------------------
    # 2. BREAK MONSOON REGIME
    # -------------------------------------------------------------
    elif regime_type == "break_monsoon":
        # Trough shifted north to Himalayan foothills (~27N - 30N)
        mslp += 4.5 * np.exp(-((lat_grid - 20.0) ** 2) / 15.0)  # Positive SLP in central India (dry)
        u850 = 3.0 + rng.uniform(-2.0, 2.0, size=(NLAT, NLON))
        
        # High rainfall concentrated in foothills & Northeast (Meghalaya/Assam)
        foothill_mask = (lat_grid >= 26.0) & (lat_grid <= 30.0) & (lon_grid >= 82.0) & (lon_grid <= 95.0)
        foothill_rain = 130.0 * np.exp(-((lat_grid - 27.5) ** 2) / 2.0) * foothill_mask
        ne_rain = 150.0 * (dem / 1400.0) * (lon_grid >= 90.0) * (lat_grid >= 24.5) * (lat_grid <= 27.0)
        rain_obs = foothill_rain + ne_rain + rng.exponential(2.0, size=(NLAT, NLON)) * land_mask

        # Raw NWP falsely triggers central India convection (15-25 mm) and underestimates foothill bursts
        spurious_ci = 22.0 * np.exp(-((lat_grid - 21.0) ** 2) / 8.0 - ((lon_grid - 78.0) ** 2) / 30.0)
        rain_nwp = 0.60 * (foothill_rain + ne_rain) + spurious_ci + rng.exponential(4.0, size=(NLAT, NLON)) * land_mask

    # -------------------------------------------------------------
    # 3. LOW / DEPRESSION / CYCLONE REGIME
    # -------------------------------------------------------------
    elif regime_type == "low_depression":
        # Deep cyclonic vortex center (e.g. Bay of Bengal moving onto coast ~17.5N, 84.5E)
        c_lat, c_lon = 17.5, 84.5
        dist_sq = (lat_grid - c_lat) ** 2 + (lon_grid - c_lon) ** 2
        
        # Deep pressure deficit (up to -16 hPa at center)
        mslp -= 16.0 * np.exp(-dist_sq / 12.0)
        # Tangential cyclonic winds
        r = np.sqrt(dist_sq) + 1e-4
        tangential = 26.0 * (r / 2.0) * np.exp(-r / 2.5)
        u850 += -tangential * (lat_grid - c_lat) / r
        v850 += tangential * (lon_grid - c_lon) / r
        rh850 = np.clip(rh850 + 25.0 * np.exp(-dist_sq / 16.0), 0.0, 99.0)
        omega500 -= 0.45 * np.exp(-dist_sq / 10.0)

        # Intense spiral rainbands and eyewall
        dep_rain = 160.0 * np.exp(-dist_sq / 7.0) + 70.0 * np.exp(-((dist_sq - 9.0) ** 2) / 16.0)
        rain_obs = dep_rain + rng.exponential(8.0, size=(NLAT, NLON))

        # Raw NWP underestimates peak eye/coastal convergence (-30% to -40%)
        rain_nwp = 0.65 * dep_rain + rng.exponential(10.0, size=(NLAT, NLON))

    # -------------------------------------------------------------
    # 4. COASTAL RAINFALL REGIME
    # -------------------------------------------------------------
    elif regime_type == "coastal":
        # Moisture advection onshore along West & East coast
        u850 = 9.0 + rng.uniform(-1.0, 1.0, size=(NLAT, NLON))
        v850 = 3.0 + rng.uniform(-1.0, 1.0, size=(NLAT, NLON))
        rh850 = np.clip(rh850 + 18.0, 0.0, 95.0)

        # Coastal strips: West coast (lon 72.8 - 75.0, lat 10-19) and East coast (lon 80-86, lat 14-21)
        w_coast = (lon_grid >= 72.8) & (lon_grid <= 74.8) & (lat_grid >= 9.0) & (lat_grid <= 19.5)
        e_coast = (lon_grid >= 80.0) & (lon_grid <= 85.5) & (lat_grid >= 14.0) & (lat_grid <= 21.0)
        
        coast_rain = 85.0 * w_coast * np.exp(-((lon_grid - 73.5) ** 2) / 0.5) + \
                     70.0 * e_coast * np.exp(-((lon_grid - 82.5) ** 2) / 0.8)
        rain_obs = coast_rain + rng.exponential(5.0, size=(NLAT, NLON)) * land_mask

        # Raw NWP misses fine-scale coastal thermal boundary convergence
        rain_nwp = 0.68 * coast_rain + rng.exponential(8.0, size=(NLAT, NLON)) * land_mask

    # -------------------------------------------------------------
    # 5. OROGRAPHIC RAINFALL REGIME
    # -------------------------------------------------------------
    elif regime_type == "orographic":
        # Strong low level westerlies perpendicular to Western Ghats
        u850 = 16.0 + rng.uniform(-2.0, 2.0, size=(NLAT, NLON))
        v850 = 4.0 + rng.uniform(-1.0, 1.0, size=(NLAT, NLON))
        rh850 = np.clip(rh850 + 22.0, 0.0, 98.0)

        # Physical uplifting: OFI = u * dh/dx + v * dh/dy
        ofi = np.maximum(u850 * grad_x + v850 * grad_y, 0.0)
        oro_rain = 180.0 * np.clip(ofi / 3.5, 0.0, 1.5) * (dem > 400.0)
        
        # Wayanad & Mahabaleshwar localized cloudburst peak
        wayanad_peak = 140.0 * np.exp(-((lat_grid - 11.55) ** 2 + (lon_grid - 76.12) ** 2) / 0.2)
        maha_peak = 110.0 * np.exp(-((lat_grid - 17.92) ** 2 + (lon_grid - 73.65) ** 2) / 0.25)
        rain_obs = oro_rain + wayanad_peak + maha_peak + rng.exponential(6.0, size=(NLAT, NLON)) * land_mask

        # Coarse NWP severely blurs steep mountain slopes: underestimates peak windward by ~50%
        rain_nwp = 0.48 * (oro_rain + wayanad_peak + maha_peak) + rng.exponential(8.0, size=(NLAT, NLON)) * land_mask

    # -------------------------------------------------------------
    # 6. WESTERN DISTURBANCE REGIME
    # -------------------------------------------------------------
    elif regime_type == "western_disturbance":
        # Upper level westerly jet (>35 m/s) and mid-tropospheric trough over NW India (28N-35N, 72E-80E)
        u200 = 38.0 + rng.uniform(-3.0, 3.0, size=(NLAT, NLON))
        trough_mask = (lat_grid >= 28.0) & (lat_grid <= 35.5) & (lon_grid >= 71.0) & (lon_grid <= 81.0)
        z500 -= 60.0 * np.exp(-((lon_grid - 75.5) ** 2) / 12.0) * trough_mask
        mslp -= 4.0 * trough_mask
        omega500 -= 0.22 * trough_mask

        # Precipitation over Western Himalayas (Himachal, Uttarakhand, J&K, Punjab)
        wd_rain = 125.0 * np.exp(-((lat_grid - 31.5) ** 2) / 4.0 - ((lon_grid - 77.0) ** 2) / 8.0) * trough_mask
        rain_obs = wd_rain + rng.exponential(5.0, size=(NLAT, NLON)) * land_mask

        # NWP displacement and cold-front timing error
        rain_nwp = 0.62 * wd_rain + rng.exponential(8.0, size=(NLAT, NLON)) * land_mask

    # Zero out ocean rain for observation if land-only mask is applied
    rain_obs = np.maximum(rain_obs * land_mask, 0.0).astype(np.float32)
    rain_nwp = np.maximum(rain_nwp * land_mask, 0.0).astype(np.float32)

    return {
        "regime": regime_type,
        "date": date_str,
        "lead_time_hrs": lead_time_hrs,
        "rain_nwp": rain_nwp,
        "rain_obs": rain_obs,
        "u850": u850.astype(np.float32),
        "v850": v850.astype(np.float32),
        "u200": u200.astype(np.float32),
        "v200": v200.astype(np.float32),
        "mslp": mslp.astype(np.float32),
        "rh850": rh850.astype(np.float32),
        "z500": z500.astype(np.float32),
        "omega500": omega500.astype(np.float32),
        "t2m": t2m.astype(np.float32),
        "tcwv": tcwv.astype(np.float32),
        "dem": dem,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "land_mask": land_mask
    }


def build_full_climatological_dataset(
    output_dir: str = "/Users/abhishekadari/projects/Rain-X/data"
) -> Dict[str, Any]:
    """
    Builds the complete RAIN-X dataset:
      - Training partition (2018-2021)
      - Validation partition (2022)
      - Test partition (2023-2024)
      - Benchmark Historical Events catalog
    """
    os.makedirs(f"{output_dir}/processed", exist_ok=True)
    os.makedirs(f"{output_dir}/geo", exist_ok=True)
    os.makedirs(f"{output_dir}/benchmarks", exist_ok=True)

    print("[DataGenerator] Synthesizing Indian Topography DEM and Land-Sea Mask...")
    dem = create_topography_dem()
    land_mask = create_land_sea_mask()
    grad_x, grad_y = compute_topography_gradients(dem)

    np.savez_compressed(
        f"{output_dir}/geo/india_geo_grid.npz",
        lats=LATS,
        lons=LONS,
        dem=dem,
        land_mask=land_mask,
        grad_x=grad_x,
        grad_y=grad_y
    )

    regimes_pool = [
        "active_monsoon",
        "break_monsoon",
        "low_depression",
        "coastal",
        "orographic",
        "western_disturbance"
    ]

    print("[DataGenerator] Generating Chronological Datasets (2018 - 2024)...")
    
    # 1. Benchmark Famous Events
    benchmarks = [
        {"id": "cyclone_michaung_2023", "regime": "low_depression", "date": "2023-12-04", "title": "Cyclone Michaung (Dec 2023)", "desc": "Severe cyclonic storm producing catastrophic coastal and eyewall deluge (>200mm) over Andhra Pradesh and Chennai."},
        {"id": "wayanad_cloudburst_2024", "regime": "orographic", "date": "2024-07-30", "title": "Wayanad Orographic Cloudburst (July 2024)", "desc": "Extreme orographic windward rainfall (>260mm) along Western Ghats escarpment triggering devastating debris flows."},
        {"id": "north_india_wd_2023", "regime": "western_disturbance", "date": "2023-07-09", "title": "North India WD Extreme (July 2023)", "desc": "Upper-tropospheric westerly trough interacting with monsoon flow causing catastrophic Beas & Yamuna floods."},
        {"id": "monsoon_break_spell_2023", "regime": "break_monsoon", "date": "2023-08-18", "title": "Monsoon Break Spell (Aug 2023)", "desc": "Monsoon trough shifted north to foothills; dry central India vs intense sub-Himalayan / Meghalaya downpours."},
        {"id": "bay_of_bengal_depression_2023", "regime": "low_depression", "date": "2023-09-13", "title": "Bay of Bengal Deep Depression (Sept 2023)", "desc": "Monsoon depression tracking west-northwest across Odisha and Central India with heavy spiral rainbands."},
        {"id": "active_monsoon_peak_2022", "regime": "active_monsoon", "date": "2022-07-16", "title": "Active Monsoon Peak Surge (July 2022)", "desc": "Intense southwesterly low-level jet driving widespread heavy precipitation across peninsular & central India."}
    ]

    benchmark_records = {}
    for bm in benchmarks:
        print(f"  Generating Benchmark Case: {bm['title']}...")
        field = generate_synoptic_field(
            regime_type=bm["regime"],
            date_str=bm["date"],
            lead_time_hrs=24,
            dem=dem,
            grad_x=grad_x,
            grad_y=grad_y,
            land_mask=land_mask,
            seed=hash(bm["id"]) % 100000
        )
        save_path = f"{output_dir}/benchmarks/{bm['id']}.npz"
        np.savez_compressed(save_path, **field)
        
        benchmark_records[bm["id"]] = {
            "id": bm["id"],
            "title": bm["title"],
            "regime": bm["regime"],
            "date": bm["date"],
            "description": bm["desc"],
            "file": save_path
        }

    with open(f"{output_dir}/benchmarks/manifest.json", "w") as f:
        json.dump(benchmark_records, f, indent=2)

    # 2. Extract Training, Validation, and Test Samples across Key District/Grid Points
    print("[DataGenerator] Sampling Synoptic Samples for Training (2018-2021), Validation (2022), Test (2023-2024)...")
    
    # We will generate multi-regime synoptic cases and extract tabular feature samples
    partitions = {
        "train": {"years": [2018, 2019, 2020, 2021], "n_cases": 48},
        "val": {"years": [2022], "n_cases": 18},
        "test": {"years": [2023, 2024], "n_cases": 24}
    }

    feature_manifest = {}
    regime_intel = WeatherRegimeIntelligence()

    for part_name, conf in partitions.items():
        X_list = []
        y_nwp_list = []
        y_obs_list = []
        gating_list = []
        regime_label_list = []
        case_idx = 0

        for yr in conf["years"]:
            for r_idx, reg_key in enumerate(regimes_pool):
                case_idx += 1
                lead = int(np.random.choice([6, 12, 24, 48, 72]))
                case_field = generate_synoptic_field(
                    regime_type=reg_key,
                    date_str=f"{yr}-07-15",
                    lead_time_hrs=lead,
                    dem=dem,
                    grad_x=grad_x,
                    grad_y=grad_y,
                    land_mask=land_mask,
                    seed=(yr * 100 + r_idx * 10 + case_idx)
                )

                # Extract physics features
                phys = extract_atmospheric_physics_features(
                    rain_nwp=case_field["rain_nwp"],
                    u850=case_field["u850"],
                    v850=case_field["v850"],
                    u200=case_field["u200"],
                    v200=case_field["v200"],
                    mslp=case_field["mslp"],
                    rh850=case_field["rh850"],
                    z500=case_field["z500"],
                    omega500=case_field["omega500"],
                    t2m=case_field["t2m"],
                    tcwv=case_field["tcwv"],
                    dem=dem,
                    grad_x=grad_x,
                    grad_y=grad_y,
                    lats=LATS,
                    lons=LONS
                )

                # Sample grid points over Indian land domain (districts and grid centers)
                # Sample 40 land points per case
                land_indices = np.argwhere(land_mask > 0.5)
                chosen_pts = land_indices[np.random.choice(len(land_indices), size=40, replace=False)]

                for ilat, ilon in chosen_pts:
                    lat_val = float(LATS[ilat])
                    lon_val = float(LONS[ilon])
                    rnwp = float(case_field["rain_nwp"][ilat, ilon])
                    robs = float(case_field["rain_obs"][ilat, ilon])

                    vec = regime_intel.build_feature_vector(
                        rain_nwp=rnwp,
                        u850=float(case_field["u850"][ilat, ilon]),
                        v850=float(case_field["v850"][ilat, ilon]),
                        mslp=float(case_field["mslp"][ilat, ilon]),
                        rh850=float(case_field["rh850"][ilat, ilon]),
                        z500=float(case_field["z500"][ilat, ilon]),
                        omega500=float(case_field["omega500"][ilat, ilon]),
                        t2m=float(case_field["t2m"][ilat, ilon]),
                        tcwv=float(case_field["tcwv"][ilat, ilon]),
                        vorticity_850=float(phys["vorticity_850"][ilat, ilon]),
                        convergence_850=float(phys["convergence_850"][ilat, ilon]),
                        mfc=float(phys["mfc"][ilat, ilon]),
                        ofi=float(phys["ofi"][ilat, ilon]),
                        dem=float(dem[ilat, ilon]),
                        lat=lat_val,
                        lon=lon_val
                    )

                    # Compute soft gating prior
                    prior = regime_intel.compute_heuristic_physics_priors(
                        lat=lat_val,
                        lon=lon_val,
                        rain_nwp=rnwp,
                        llj_speed=float(phys["llj_speed"][ilat, ilon]),
                        slp_anomaly=float(phys["slp_anomaly"][ilat, ilon]),
                        vorticity_850=float(phys["vorticity_850"][ilat, ilon]),
                        convergence_850=float(phys["convergence_850"][ilat, ilon]),
                        ofi=float(phys["ofi"][ilat, ilon]),
                        upward_motion=float(phys["upward_motion"][ilat, ilon]),
                        rh850=float(case_field["rh850"][ilat, ilon]),
                        dem=float(dem[ilat, ilon])
                    )

                    X_list.append(vec)
                    y_nwp_list.append(rnwp)
                    y_obs_list.append(robs)
                    gating_list.append(prior)
                    regime_label_list.append(r_idx)

        X_arr = np.array(X_list, dtype=np.float32)
        y_nwp_arr = np.array(y_nwp_list, dtype=np.float32)
        y_obs_arr = np.array(y_obs_list, dtype=np.float32)
        gating_arr = np.array(gating_list, dtype=np.float32)
        regime_labels_arr = np.array(regime_label_list, dtype=np.int64)

        part_file = f"{output_dir}/processed/{part_name}_dataset.npz"
        np.savez_compressed(
            part_file,
            X=X_arr,
            y_nwp=y_nwp_arr,
            y_obs=y_obs_arr,
            gating_probs=gating_arr,
            regime_labels=regime_labels_arr
        )
        print(f"  Saved {part_name} partition: {len(X_arr)} samples -> {part_file}")
        feature_manifest[part_name] = {
            "samples": len(X_arr),
            "file": part_file
        }

    with open(f"{output_dir}/processed/dataset_manifest.json", "w") as f:
        json.dump(feature_manifest, f, indent=2)

    print("[DataGenerator] Complete dataset successfully generated.")
    return feature_manifest


if __name__ == "__main__":
    build_full_climatological_dataset()
