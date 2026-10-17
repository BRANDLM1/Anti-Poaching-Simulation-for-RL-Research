"""Plot the reserve's terrain with Reserve.csv nodes and candidate peaks on top.

Run from the project root:  python -m terrain.plot_map
Saves data/reserve_map.png.
"""
import numpy as np
import pandas as pd
import pyproj
import rasterio
import rasterio.transform
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from scipy.ndimage import maximum_filter

from terrain.paths import DEM_PATH, SLOPE_PATH
DOWNSAMPLE = 4            # plot at ~100 m pixels instead of 25 m, for speed

# Same criteria as utils/Significant_Peaks.py, scaled to the downsampled grid
PEAK_WINDOW_PX = 200 // DOWNSAMPLE
MIN_ELEVATION = 1800
MIN_SLOPE = 10

TYPE_STYLE = {   # marker, colour, label
    "rangerh":      ("s", "blue",    "Ranger station"),
    "poacherh":     ("v", "black",   "Entry / border"),
    "civilization": ("o", "orange",  "Village"),
    "mountain":     ("^", "red",     "Peak (Reserve.csv)"),
    "camera":       ("D", "magenta", "Camera"),
    "water":        ("P", "cyan",    "Water"),
}


def read_downsampled(path):
    with rasterio.open(path) as src:
        shape = (src.height // DOWNSAMPLE, src.width // DOWNSAMPLE)
        data = src.read(1, out_shape=shape, resampling=Resampling.average).astype(float)
        data[data == src.nodata] = np.nan
        transform = src.transform * src.transform.scale(src.width / shape[1], src.height / shape[0])
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
    return data, transform, extent


def main():
    dem, transform, extent = read_downsampled(DEM_PATH)
    slope, _, _ = read_downsampled(SLOPE_PATH)

    # Candidate peaks: local maxima that are high and steep
    filled = np.nan_to_num(dem, nan=-1)
    is_peak = (filled == maximum_filter(filled, size=PEAK_WINDOW_PX)) & (dem >= MIN_ELEVATION) & (slope >= MIN_SLOPE)
    rows, cols = np.where(is_peak)
    peak_x, peak_y = rasterio.transform.xy(transform, rows, cols)

    nodes: pd.DataFrame = pd.read_csv("data/Reserve.csv")
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32647", always_xy=True)
    nodes["x"], nodes["y"] = to_utm.transform(nodes["Longitude"].values, nodes["Latitude"].values)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(dem, cmap="terrain", extent=extent, vmin=1000, vmax=2600)
    fig.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.6)

    ax.scatter(peak_x, peak_y, marker="x", color="darkred", s=40, label="Peak finder candidates")

    # Reserve perimeter: border1, border2, ... joined in order and closed back to the start
    names = nodes["Node"].astype(str)
    is_border = names.str.fullmatch(r"border\d+")
    borders = nodes.loc[is_border].copy()
    borders["n"] = names.loc[is_border].str.extract(r"(\d+)", expand=False).astype(int)
    borders = borders.sort_values(by="n")
    ax.plot(list(borders["x"]) + [borders["x"].iloc[0]], list(borders["y"]) + [borders["y"].iloc[0]],
            color="black", linewidth=1.2, linestyle="--", label="Reserve perimeter")
    for node_type, (marker, colour, label) in TYPE_STYLE.items():
        subset = nodes[nodes["Type"] == node_type]
        ax.scatter(subset["x"], subset["y"], marker=marker, color=colour, edgecolor="white",
                   s=80, label=label, zorder=3)
    for _, row in nodes.iterrows():
        ax.annotate(str(row["Node"]), (float(row["x"]), float(row["y"])), fontsize=7, xytext=(3, 3), textcoords="offset points")

    ax.set_title("Reserve terrain with Reserve.csv nodes and peak-finder candidates")
    ax.set_xlabel("Easting (m, UTM 47N)")
    ax.set_ylabel("Northing (m, UTM 47N)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig("data/reserve_map.png", dpi=110)
    print(f"Saved data/reserve_map.png ({len(peak_x)} peak candidates)")


if __name__ == "__main__":
    main()
