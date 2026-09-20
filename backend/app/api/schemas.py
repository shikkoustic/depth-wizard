"""
DepthWizard - Pydantic Schemas for API requests and responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ElevationStats(BaseModel):
    min_elevation: float
    max_elevation: float
    mean_elevation: float
    std_elevation: float
    max_object_height: float
    mean_object_height: float


class GCPStats(BaseModel):
    gcp_count: int
    model_type: str
    offset_m: float
    residual_rmse_before: float
    residual_rmse_after: float


class ProcessResponse(BaseModel):
    success: bool
    task_id: str
    is_georeferenced: bool
    crs: Optional[str] = None
    bounds: Optional[List[float]] = None
    width: int
    height: int
    elevation_type: str  # "Absolute DSM (Metric)" or "Relative DSM (rDSM)"
    dem_source: Optional[str] = None
    stats: ElevationStats
    gcp_stats: Optional[GCPStats] = None
    grid_rows: int
    grid_cols: int
    height_grid: List[List[float]]  # Downsampled for Three.js terrain mesh
    reference_grid: Optional[List[List[float]]] = None  # Downsampled LiDAR reference for comparison
    pixel_size: Optional[List[float]] = None  # [dx, dy] in meters
    ground_width_m: Optional[float] = None
    ground_height_m: Optional[float] = None
    texture_url: str
    dsm_preview_url: str
    confidence_map_url: str
    geotiff_download_url: Optional[str] = None



class BaselineItem(BaseModel):
    method: str
    mae: float
    rmse: float
    correlation: float
    status: str


class EvaluationResponse(BaseModel):
    success: bool
    mae: float
    rmse: float
    correlation: float
    valid_pixel_count: int
    datum_offset_adjusted: Optional[float] = None
    breakdown: Dict[str, Any]
    baselines: List[BaselineItem]
    height_bands: Optional[Dict[str, Any]] = None
