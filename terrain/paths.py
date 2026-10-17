"""Where the terrain rasters live.

The rasters (ALOS AW3D30 elevation reprojected to UTM 47N, and the slope derived from it) are too large
for the repository. By default they are read from processing/data/; set RESERVE_RASTER_DIR to use
another folder.
"""
import os

RASTER_DIR = os.environ.get("RESERVE_RASTER_DIR", os.path.join("processing", "data"))
DEM_PATH = os.path.join(RASTER_DIR, "reserve_UTM47N.tif")      # elevation, UTM 47N metres
SLOPE_PATH = os.path.join(RASTER_DIR, "reserve_slope.tif")     # slope in degrees, same grid
