"""
RAIN-X NWP Ingestion Engine
Handles ingestion, quality control (QC), spatial alignment,
and regridding of NWP model forecasts (NCMRWF NCUM, GFS, ECMWF)
and IMD 0.25° gridded rainfall observations.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.spatial.domain import LATS, LONS, NLAT, NLON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RESOLUTION


class NWPIngestionEngine:
    """
    Ingests atmospheric model forecast fields and observation grids.
    Supports NetCDF, GRIB-2 (via cfgrib/xarray), and structured NumPy arrays.
    """
    def __init__(self, target_lats: np.ndarray = LATS, target_lons: np.ndarray = LONS):
        self.target_lats = target_lats
        self.target_lons = target_lons
        self.n_lat = len(target_lats)
        self.n_lon = len(target_lons)

    def quality_control(self, field: np.ndarray, var_name: str) -> np.ndarray:
        """
        Applies meteorological bounds and quality control to prevent non-physical artifacts.
        """
        qc_field = field.copy()
        
        # Check NaNs and Infs
        nan_mask = np.isnan(qc_field) | np.isinf(qc_field)
        if np.any(nan_mask):
            qc_field[nan_mask] = 0.0

        # Physical range assertions
        if "rain" in var_name or "precip" in var_name:
            qc_field = np.clip(qc_field, 0.0, 1200.0)  # max recorded daily rain ~ 1168 mm (Cherrapunji)
        elif "mslp" in var_name or "pressure" in var_name:
            qc_field = np.clip(qc_field, 880.0, 1080.0) # hPa
        elif "rh" in var_name:
            qc_field = np.clip(qc_field, 0.0, 100.0)    # %
        elif "wind" in var_name or var_name in ["u850", "v850", "u200", "v200"]:
            qc_field = np.clip(qc_field, -150.0, 150.0) # m/s
        elif "t2m" in var_name:
            qc_field = np.clip(qc_field, 220.0, 335.0)  # Kelvin

        return qc_field

    def ingest_npz_case(self, file_path: str) -> Dict[str, Any]:
        """
        Ingests a preprocessed synoptic case file containing NWP forecast fields
        and matching IMD gridded observation.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Case file not found: {file_path}")

        data = np.load(file_path, allow_pickle=True)
        res = {}
        for k in data.files:
            val = data[k]
            if isinstance(val, np.ndarray) and val.dtype in [np.float32, np.float64]:
                res[k] = self.quality_control(val, k)
            else:
                res[k] = val

        return res

    def regrid_to_target(
        self,
        source_field: np.ndarray,
        src_lats: np.ndarray,
        src_lons: np.ndarray
    ) -> np.ndarray:
        """
        Bilinear spatial interpolation to the canonical IMD 0.25° x 0.25° grid.
        """
        if len(src_lats) == self.n_lat and len(src_lons) == self.n_lon:
            if np.allclose(src_lats, self.target_lats) and np.allclose(src_lons, self.target_lons):
                return source_field.astype(np.float32)

        # Bilinear interpolation
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((src_lats, src_lons), source_field, bounds_error=False, fill_value=0.0)
        
        lon_grid, lat_grid = np.meshgrid(self.target_lons, self.target_lats)
        pts = np.stack([lat_grid.flatten(), lon_grid.flatten()], axis=-1)
        regridded = interp(pts).reshape((self.n_lat, self.n_lon))
        return regridded.astype(np.float32)
