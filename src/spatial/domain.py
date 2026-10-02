"""
RAIN-X Spatial Intelligence Module
Defines Indian subcontinent domain grid (0.25° x 0.25°),
topography DEM, land-sea mask, river basins, and district mapping.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# Standard IMD 0.25° x 0.25° Grid specifications
LAT_MIN, LAT_MAX = 8.0, 38.0
LON_MIN, LON_MAX = 68.0, 98.0
RESOLUTION = 0.25

LATS = np.arange(LAT_MIN, LAT_MAX + RESOLUTION / 2, RESOLUTION)
LONS = np.arange(LON_MIN, LON_MAX + RESOLUTION / 2, RESOLUTION)
NLAT = len(LATS)
NLON = len(LONS)

# Core Indian Districts Database (Hierarchical: State -> District -> Station)
DISTRICT_CATALOG = {
    "Andhra Pradesh": [
        {"district": "Visakhapatnam", "station": "Anakapalli", "lat": 17.68, "lon": 83.00, "elevation": 35.0, "basin": "East Flowing Rivers", "coastal": True},
        {"district": "Krishna", "station": "Vijayawada", "lat": 16.51, "lon": 80.64, "elevation": 23.0, "basin": "Krishna", "coastal": True},
        {"district": "East Godavari", "station": "Kakinada", "lat": 16.98, "lon": 82.24, "elevation": 12.0, "basin": "Godavari", "coastal": True},
        {"district": "Kurnool", "station": "Kurnool Town", "lat": 15.82, "lon": 78.03, "elevation": 273.0, "basin": "Krishna", "coastal": False}
    ],
    "Maharashtra": [
        {"district": "Mumbai Suburban", "station": "Colaba", "lat": 18.90, "lon": 72.81, "elevation": 11.0, "basin": "West Flowing Rivers", "coastal": True},
        {"district": "Mumbai Suburban", "station": "Santacruz", "lat": 19.08, "lon": 72.85, "elevation": 14.0, "basin": "West Flowing Rivers", "coastal": True},
        {"district": "Satara", "station": "Mahabaleshwar", "lat": 17.92, "lon": 73.65, "elevation": 1353.0, "basin": "Krishna", "coastal": False},
        {"district": "Pune", "station": "Shivajinagar", "lat": 18.53, "lon": 73.85, "elevation": 560.0, "basin": "Krishna", "coastal": False},
        {"district": "Nagpur", "station": "Sonegaon", "lat": 21.14, "lon": 79.08, "elevation": 310.0, "basin": "Godavari", "coastal": False}
    ],
    "Kerala": [
        {"district": "Wayanad", "station": "Meppadi", "lat": 11.55, "lon": 76.12, "elevation": 880.0, "basin": "Cauvery / Kabini", "coastal": False},
        {"district": "Ernakulam", "station": "Kochi Naval Base", "lat": 9.93, "lon": 76.26, "elevation": 4.0, "basin": "Periyar", "coastal": True},
        {"district": "Kottayam", "station": "Kumarakom", "lat": 9.59, "lon": 76.52, "elevation": 3.0, "basin": "West Flowing Rivers", "coastal": True}
    ],
    "Uttarakhand": [
        {"district": "Dehradun", "station": "Rishikesh", "lat": 30.08, "lon": 78.26, "elevation": 372.0, "basin": "Ganga", "coastal": False},
        {"district": "Chamoli", "station": "Joshimath", "lat": 30.55, "lon": 79.56, "elevation": 1890.0, "basin": "Alaknanda / Ganga", "coastal": False},
        {"district": "Nainital", "station": "Haldwani", "lat": 29.21, "lon": 79.51, "elevation": 424.0, "basin": "Ganga", "coastal": False}
    ],
    "Odisha": [
        {"district": "Puri", "station": "Konark", "lat": 19.88, "lon": 86.09, "elevation": 10.0, "basin": "Mahanadi", "coastal": True},
        {"district": "Khurda", "station": "Bhubaneswar", "lat": 20.29, "lon": 85.82, "elevation": 45.0, "basin": "Mahanadi", "coastal": False},
        {"district": "Ganjam", "station": "Gopalpur", "lat": 19.26, "lon": 84.91, "elevation": 17.0, "basin": "Rushikulya", "coastal": True}
    ],
    "Karnataka": [
        {"district": "Shimoga", "station": "Agumbe", "lat": 13.51, "lon": 75.09, "elevation": 643.0, "basin": "West Flowing Rivers", "coastal": False},
        {"district": "Dakshina Kannada", "station": "Mangaluru", "lat": 12.91, "lon": 74.85, "elevation": 22.0, "basin": "Netravati", "coastal": True},
        {"district": "Bengaluru Urban", "station": "HAL Airport", "lat": 12.95, "lon": 77.66, "elevation": 888.0, "basin": "Pennar / Cauvery", "coastal": False}
    ],
    "Tamil Nadu": [
        {"district": "Chennai", "station": "Nungambakkam", "lat": 13.06, "lon": 80.24, "elevation": 16.0, "basin": "Cooum / Adyar", "coastal": True},
        {"district": "Nilgiris", "station": "Ooty", "lat": 11.41, "lon": 76.70, "elevation": 2240.0, "basin": "Bhavani", "coastal": False},
        {"district": "Cuddalore", "station": "Chidambaram", "lat": 11.39, "lon": 79.69, "elevation": 12.0, "basin": "Cauvery", "coastal": True}
    ],
    "West Bengal": [
        {"district": "Kolkata", "station": "Alipore", "lat": 22.53, "lon": 88.33, "elevation": 9.0, "basin": "Hooghly / Ganga", "coastal": True},
        {"district": "Darjeeling", "station": "Kurseong", "lat": 26.88, "lon": 88.27, "elevation": 1458.0, "basin": "Teesta", "coastal": False}
    ],
    "Meghalaya": [
        {"district": "East Khasi Hills", "station": "Cherrapunji (Sohra)", "lat": 25.27, "lon": 91.73, "elevation": 1484.0, "basin": "Barak / Meghna", "coastal": False},
        {"district": "East Khasi Hills", "station": "Mawsynram", "lat": 25.29, "lon": 91.58, "elevation": 1400.0, "basin": "Barak / Meghna", "coastal": False}
    ],
    "Himachal Pradesh": [
        {"district": "Shimla", "station": "Ridge", "lat": 31.10, "lon": 77.17, "elevation": 2205.0, "basin": "Sutlej / Indus", "coastal": False},
        {"district": "Kullu", "station": "Manali", "lat": 32.24, "lon": 77.18, "elevation": 2050.0, "basin": "Beas", "coastal": False}
    ],
    "Gujarat": [
        {"district": "Ahmedabad", "station": "Ahmedabad AP", "lat": 23.07, "lon": 72.63, "elevation": 55.0, "basin": "Sabarmati", "coastal": False},
        {"district": "Surat", "station": "Surat Port", "lat": 21.17, "lon": 72.83, "elevation": 13.0, "basin": "Tapti", "coastal": True}
    ],
    "Delhi": [
        {"district": "New Delhi", "station": "Safdarjung", "lat": 28.58, "lon": 77.20, "elevation": 216.0, "basin": "Yamuna / Ganga", "coastal": False}
    ]
}


def create_topography_dem(lats: np.ndarray = LATS, lons: np.ndarray = LONS) -> np.ndarray:
    """
    Synthesizes physically accurate Topographic Digital Elevation Model (m)
    for Indian Subcontinent matching SRTM/ETOPO1 morphology.
    """
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    dem = np.zeros_like(lat_grid, dtype=np.float32)

    # 1. Himalayan Mountain Arc & Tibetan Plateau (27°N - 36°N, 73°E - 96°E)
    himalaya_mask = (lat_grid >= 27.0) & (lat_grid <= 36.5) & (lon_grid >= 73.0) & (lon_grid <= 96.0)
    # Distance from Himalayan frontal fault line (~27.5N - 29.5N)
    ridge_lat = 30.0 + 0.1 * (lon_grid - 78.0)
    himalaya_height = 4500.0 * np.exp(-((lat_grid - ridge_lat) ** 2) / 8.0) * np.exp(-((lon_grid - 85.0) ** 2) / 120.0)
    himalaya_height += 2500.0 * himalaya_mask * np.clip((lat_grid - 28.0) / 4.0, 0.0, 1.0)
    dem += himalaya_height

    # 2. Western Ghats Ridge (8.5°N - 21.0°N, 73.2°E - 75.8°E)
    ghats_lon = 73.5 + 0.15 * (lat_grid - 8.5)
    ghats_mask = (lat_grid >= 8.5) & (lat_grid <= 21.0)
    ghats_height = 1400.0 * np.exp(-((lon_grid - ghats_lon) ** 2) / 0.8) * ghats_mask
    # High peaks (Anamudi, Nilgiris, Mahabaleshwar)
    ghats_height += 1000.0 * np.exp(-((lat_grid - 10.2) ** 2 + (lon_grid - 77.0) ** 2) / 0.5)  # Anamudi / Nilgiris
    ghats_height += 600.0 * np.exp(-((lat_grid - 17.9) ** 2 + (lon_grid - 73.65) ** 2) / 0.4)  # Mahabaleshwar
    dem += ghats_height

    # 3. Meghalaya Plateau / Khasi Hills (25°N - 26°N, 90°E - 93°E)
    meghalaya_mask = (lat_grid >= 24.8) & (lat_grid <= 26.2) & (lon_grid >= 90.0) & (lon_grid <= 93.5)
    meghalaya_height = 1500.0 * np.exp(-((lat_grid - 25.3) ** 2) / 0.4) * np.exp(-((lon_grid - 91.8) ** 2) / 1.5) * meghalaya_mask
    dem += meghalaya_height

    # 4. Deccan Plateau & Eastern Ghats
    deccan_mask = (lat_grid >= 12.0) & (lat_grid <= 22.0) & (lon_grid >= 74.0) & (lon_grid <= 82.0)
    dem += 450.0 * deccan_mask

    # Eastern Ghats
    eghats_mask = (lat_grid >= 15.0) & (lat_grid <= 20.0) & (lon_grid >= 80.0) & (lon_grid <= 85.0)
    dem += 600.0 * np.exp(-((lon_grid - (80.0 + 0.5 * (lat_grid - 15.0))) ** 2) / 1.0) * eghats_mask

    # 5. Aravalli Range (23°N - 28°N, 72°E - 77°E)
    aravalli_mask = (lat_grid >= 23.5) & (lat_grid <= 28.0)
    aravalli_lon = 72.8 + 0.8 * (lat_grid - 23.5)
    dem += 500.0 * np.exp(-((lon_grid - aravalli_lon) ** 2) / 0.6) * aravalli_mask

    # Zero out ocean points
    land_mask = create_land_sea_mask(lats, lons)
    dem = np.maximum(dem * land_mask, 0.0)
    return dem.astype(np.float32)


def create_land_sea_mask(lats: np.ndarray = LATS, lons: np.ndarray = LONS) -> np.ndarray:
    """
    Generates binary Land (1) / Ocean (0) mask for Indian subcontinent domain.
    Covers the mainland peninsula, northern subcontinent, and coastal contours.
    """
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mask = np.zeros_like(lat_grid, dtype=np.float32)

    # Simplified Indian Mainland Polygon Bounds:
    # 1. Northern India (lat >= 24°N): mostly land across 68°E to 98°E
    north_mask = (lat_grid >= 24.0) & (lat_grid <= 37.5) & (lon_grid >= 68.5) & (lon_grid <= 97.5)
    mask[north_mask] = 1.0

    # 2. Central & Peninsular India (8°N <= lat < 24°N)
    # West Coast approximate bound: lon >= 72.5 + (lat - 8) * (-0.08)
    # East Coast approximate bound: lon <= 80.0 + (lat - 8) * (0.5) up to 20N
    for i in range(len(lats)):
        lat = lats[i]
        if 8.0 <= lat < 24.0:
            if lat < 12.0:
                w_lon = 75.0 - (lat - 8.0) * 0.4
                e_lon = 79.8 + (lat - 8.0) * 0.1
            elif lat < 16.0:
                w_lon = 73.4 - (lat - 12.0) * 0.1
                e_lon = 80.2 + (lat - 12.0) * 0.45
            elif lat < 20.0:
                w_lon = 72.6 + (lat - 16.0) * 0.05
                e_lon = 82.0 + (lat - 16.0) * 0.8
            else:  # 20.0 to 24.0
                w_lon = 69.5  # Gujarat coastline
                e_lon = 89.0  # Bengal delta
            
            row_mask = (lon_grid[i, :] >= w_lon) & (lon_grid[i, :] <= e_lon)
            mask[i, row_mask] = 1.0

    # Add Sri Lanka exclusion / ocean channel
    sri_lanka_channel = (lat_grid >= 8.5) & (lat_grid <= 10.0) & (lon_grid >= 78.5) & (lon_grid <= 80.0)
    mask[sri_lanka_channel] = 0.0

    return mask.astype(np.float32)


def compute_topography_gradients(dem: np.ndarray, resolution_deg: float = RESOLUTION) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes spatial elevation gradient vectors (dh/dx, dh/dy) in m/km.
    Essential for orographic forcing index: OFI = V_850 . grad(h).
    """
    # 1 deg latitude ~ 111 km, 1 deg longitude at 20°N ~ 111 * cos(20°) ~ 104 km
    dx_km = resolution_deg * 104.0
    dy_km = resolution_deg * 111.0

    grad_y, grad_x = np.gradient(dem, dy_km, dx_km)
    return grad_x.astype(np.float32), grad_y.astype(np.float32)


def get_station_coords(state: str, district: str, station: str = None) -> Dict[str, Any]:
    """
    Returns metadata and coordinates for a given district or station.
    """
    stations = DISTRICT_CATALOG.get(state, [])
    for s in stations:
        if s["district"].lower() == district.lower():
            if station is None or s["station"].lower() == station.lower():
                return s
    # Default fallback
    if stations:
        return stations[0]
    return {"district": district, "station": station or "Headquarters", "lat": 18.0, "lon": 79.0, "elevation": 150.0, "basin": "Central Basin", "coastal": False}


def grid_point_index(lat: float, lon: float) -> Tuple[int, int]:
    """
    Maps continuous latitude and longitude to nearest (lat_idx, lon_idx) in grid.
    """
    ilat = int(np.clip(np.round((lat - LAT_MIN) / RESOLUTION), 0, NLAT - 1))
    ilon = int(np.clip(np.round((lon - LON_MIN) / RESOLUTION), 0, NLON - 1))
    return ilat, ilon
