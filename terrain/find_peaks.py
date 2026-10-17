import rasterio
import numpy as np
from scipy.ndimage import maximum_filter
import pyproj

def find_optimal_observer_points():
    """
    Analyzes a DTM to find the best observation points based on elevation
    and local topography, using the logic from your successful script.
    """
    from terrain.paths import DEM_PATH as dsm_path, SLOPE_PATH as slope_path

    try:
        with rasterio.open(dsm_path) as dsm_src, rasterio.open(slope_path) as slope_src:
            dsm_array = dsm_src.read(1)
            slope_array = slope_src.read(1)

            nodata = dsm_src.nodata

            local_max = maximum_filter(dsm_array, size=200)

            # Create masks
            local_max_mask = (dsm_array == local_max)
            elevation_mask = (dsm_array >= 1800)  # Find high-elevation peaks
            slope_mask = (slope_array >= 10)
            valid_data_mask = (dsm_array != nodata)
            
            # Combine masks to find the final peak locations
            peak_mask = (local_max_mask & elevation_mask & valid_data_mask & slope_mask)

            # Get the pixel coordinates of the peaks
            rows, cols = np.where(peak_mask)

            if len(rows) == 0:
                print("No peaks found with the current criteria. Try lowering Min_Elevation.")
                return

            # Convert pixel coordinates to UTM
            utm_x, utm_y = dsm_src.xy(rows, cols)
            
            target_crs = pyproj.CRS("EPSG:4326") # WGS 84 (Latitude/Longitude)
            source_crs = pyproj.CRS("EPSG:32647") # WGS 84 / UTM zone 47N
            transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

            lon, lat = transformer.transform(utm_x, utm_y)

            # Get the elevation of each peak
            peak_elevations = dsm_array[rows, cols]

            print("--- Optimal Observation Points Found ---")
            print("These points are true local maxima and should provide good visibility.\n")
            for i in range(len(utm_x)):
                print(f"Peak #{i+1}:")
                print(f"  Longitude Latitude: ({lon[i]:.6f}, {lat[i]:.6f})")
                print(f"  UTM Coordinates (X, Y): ({utm_x[i]:.6f}, {utm_y[i]:.6f})")
                print(f"  Elevation: {peak_elevations[i]}m\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    find_optimal_observer_points()