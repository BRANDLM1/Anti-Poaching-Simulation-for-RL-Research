"""Merge the cleaned ALOS AW3D30 tiles into one elevation raster.

Run from the project root: python processing/merge_tiles.py
Reads processing/data/cleaned/*_cleaned.tif, writes processing/data/reserve_merged.tif.
Needs GDAL's gdal_merge.py: on PATH, or set GDAL_MERGE to its full path inside OSGeo4W

The merged raster was then reprojected to UTM 47N and turned into a slope raster with GDAL
(gdalwarp, gdaldem slope) to produce reserve_UTM47N.tif and reserve_slope.tif.
"""
import glob
import os
import shutil
import subprocess
import sys

CLEANED = os.path.join("processing", "data", "cleaned")
OUTPUT = os.path.join("processing", "data", "reserve_merged.tif")
GDAL_MERGE = os.environ.get("GDAL_MERGE") or shutil.which("gdal_merge.py") or "gdal_merge.py"

files = sorted(glob.glob(os.path.join(CLEANED, "*_cleaned.tif")))
if not files:
    sys.exit(f"No *_cleaned.tif files found in {CLEANED}")

print(f"Merging {len(files)} tiles into {OUTPUT}")
subprocess.run(["python", GDAL_MERGE, "-o", OUTPUT, "-n", "-9999", "-a_nodata", "-9999", *files], check=True)
print("Merge complete.")
