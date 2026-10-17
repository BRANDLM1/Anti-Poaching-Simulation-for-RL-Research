from typing import List, Tuple, Dict, Optional
from math import sqrt
from functools import lru_cache
import numpy as np
import rasterio
import pyproj
import heapq
import time

from terrain.paths import DEM_PATH as elevation_path, SLOPE_PATH as slope_path

dsm_src = rasterio.open(elevation_path)


def euclidean_distance(u, v, G):
    """Straight-line distance in metres between two graph nodes (UTM positions)."""
    pos_u = G.nodes[u]['pos']
    pos_v = G.nodes[v]['pos']
    return ((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2) ** 0.5


def custom_dist(u, v, G):
    """Terrain distance in metres between two graph nodes: A* over the slope raster."""
    start = (G.nodes[u]['Latitude'], G.nodes[u]['Longitude'])
    goal = (G.nodes[v]['Latitude'], G.nodes[v]['Longitude'])
    return main(start, goal, downsample_factor=5)


@lru_cache(maxsize=None)
def load_grids(downsample_factor):
    with rasterio.open(slope_path) as slope_src:
        slope_grid = slope_src.read(1)
    grid = (slope_grid < 0).astype(np.uint8)
    return grid[::downsample_factor, ::downsample_factor], slope_grid[::downsample_factor, ::downsample_factor]

def gcs_to_array(lat: float, lon: float)-> Tuple[int, int]:
        
    source_crs = pyproj.CRS("EPSG:4326") # WGS 84 (Latitude/Longitude)
    target_crs = pyproj.CRS("EPSG:32647") # WGS 84 / UTM zone 47N
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

    utm_x, utm_y = transformer.transform(lon, lat)
    row, col = dsm_src.index(utm_x, utm_y)
    
    print(f'Function converted {lat,lon} to {row,col}')

    return row, col

def calculate_path_length_meters(path: List[Tuple[int, int]]) -> float:
    """
    Calculate the total length of path A-->B in meters.
    Args:
        path: List of array positions (row, col) - already at full resolution
    Returns:
        Total path length in meters
    """

    if len(path) < 2.0:
        return 0.0
    
    total_distance = 0.0
    
    for i in range(len(path) - 1):
        # Current and next positions
        current_row, current_col = path[i]
        next_row, next_col = path[i + 1]
        
        # Convert array indices to UTM coordinates (in meters)
        current_x, current_y = dsm_src.xy(current_row, current_col)
        next_x, next_y = dsm_src.xy(next_row, next_col)
        
        # Calculate distance between consecutive points
        distance = sqrt((next_x - current_x)**2 + (next_y - current_y)**2)
        total_distance += distance

    print(f'Function calculated path distance is {total_distance} meters')
    return total_distance

def main(start_gcs, goal_gcs, downsample_factor):

    # Retrieve array postions
    start_pos = gcs_to_array(*start_gcs)
    goal_pos = gcs_to_array(*goal_gcs)
    obstacles, slope = load_grids(downsample_factor)

    rough_start = (start_pos[0] // downsample_factor, start_pos[1] // downsample_factor)
    rough_goal = (goal_pos[0] // downsample_factor, goal_pos[1] // downsample_factor)

    path = find_path(obstacles, slope, rough_start, rough_goal)
    if not path:
        print("No path found")
        return float('inf')
    return calculate_path_length_meters([(r*downsample_factor, c*downsample_factor) for r, c in path])
    
def reconstruct_path(closed_list: Dict, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Reconstructs the path from the 'closed_list' dictionary.
    """
    path = []
    current = goal
    
    if goal not in closed_list:
        # Add the start node itself to indicate where the search began.
        if start == goal: return [start] # Special case: start is the goal
        return [] # Else goal was not reached

    while current != start:
        path.append(current)
        current = closed_list.get(current)
        if current is None:
             # This would indicate a broken path chain, should not happen with correct A*
             print("Error: Path reconstruction failed, broken chain.")
             return []

    path.append(start)  # Add the start node to the path
    return path[::-1]

def calculate_heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Estimate distance between two points
    """
    x1, y1 = pos1
    x2, y2 = pos2

    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_valid_neighbors(grid: np.ndarray, position: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get valid neighbors in grid
    
    Args:
        grid: 2D numpy array ( 0 = walkable ,  1 = obstacle )
        position: Current pos (x, y)
    
    Returns:
        List of valid neighboring positions
    """
    x, y = position
    rows, cols = grid.shape
    
    # All possible moves (including diagonals)
    possible_moves = [
        (x+1, y), (x-1, y),    # Right, Left
        (x, y+1), (x, y-1),    # Up, Down
        (x+1, y+1), (x-1, y-1),  # Diagonal moves
        (x+1, y-1), (x-1, y+1)
    ]
    
    return [
        (nx, ny) for nx, ny in possible_moves
        if 0 <= nx < rows and 0 <= ny < cols  # Within grid bounds
        and grid[nx, ny] == 0                # Not an obstacle
    ]

def create_node(position: Tuple[int, int], g: float = float('inf'), 
                h: float = 0.0, parent: Optional[Dict] = None) -> Dict:
    """
    Create a node for the A* algorithm.
    Args:
        position: (x, y) coordinates of the node
        g: Cost from start to this node (default: infinity)
        h: Estimated cost from this node to goal (default: 0)
        parent: Parent node (default: None)
    
    Returns:
        Dictionary containing node information
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

def find_path(grid: np.ndarray, slope_grid: np.ndarray, start: Tuple[int, int], 
              goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find the optimal path using A* algorithm.
    Must take in start and end array positions
    Args:
        grid: 2D numpy array (0 = free space, 1 = obstacle)
        start: Starting position (x, y)
        goal: Goal position (x, y)
    
    Returns:
        List of positions representing the optimal path
    """
    rows, cols = grid.shape
    for name, p in (("Start", start), ("Goal", goal)):
        if not (0 <= p[0] < rows and 0 <= p[1] < cols):
            print(f"Error: {name} position is out of grid bounds.")
            return []
        if grid[p] == 1:
            print(f"Error: {name} position is in no data.")
            return []

    # The priority queue stores tuples of (f_score, position)
    open_list = [(calculate_heuristic(start, goal), start)] 

    # For path reconstruction
    closed_list = {}

    # Tracks current exact cost from start to node n
    g_score = np.full(grid.shape, float('inf'))
    # Initialize starting node to cost 0
    g_score[start] = 0

    while open_list:
        
        # Get node with lowest f value
        _, current_pos = heapq.heappop(open_list)
        
        # Check if we've reached the goal
        if current_pos == goal:
            # Return final reconstructed path
            print("Path Found: Reconstructing path...")
            return reconstruct_path(closed_list, start, goal)
        
        # Explore neighbors
        for neighbor_pos in get_valid_neighbors(grid, current_pos):

            current_slope = slope_grid[current_pos]
            neighbor_slope = slope_grid[neighbor_pos]

            if neighbor_slope < 0:
                continue

            # Calculate Eucilidean Distance from node ni to ni+1
            distance = sqrt((neighbor_pos[0] - current_pos[0])**2 + (neighbor_pos[1] - current_pos[1])**2)

            # Final g(n) cost to move(edge weight) from ni to ni+1 using cost surface and pixel navigation
            move_cost = (1 + (current_slope + neighbor_slope) / 2) * distance
            
            # Calculate new path cost
            tentative_g = g_score[current_pos] + move_cost
            
            
            # If this path is better, update and add to the open list
            if tentative_g < g_score[neighbor_pos]:
                closed_list[neighbor_pos] = current_pos
                g_score[neighbor_pos] = tentative_g
                f_score = tentative_g + calculate_heuristic(neighbor_pos, goal)
                heapq.heappush(open_list, (f_score, neighbor_pos))
                
    return [] # No path found
