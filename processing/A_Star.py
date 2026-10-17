from typing import List, Tuple, Dict, Optional
import numpy as np
import heapq
from math import sqrt
import plotly.graph_objects as go

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
    
    Args:
        grid: 2D numpy array (0 = free space, 1 = obstacle)
        start: Starting position (x, y)
        goal: Goal position (x, y)
    
    Returns:
        List of positions representing the optimal path
    """
    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
        print("Error: Start position is out of grid bounds.")
        return []
    
    print("Starting path search...")

     # The priority queue stores tuples of (f_score, position)
    open_list = [(calculate_heuristic(start, goal), start)] 

    # For path reconstruction
    closed_list = {}

    # Tracks current exact cost from start to node n
    g_score = {pos: float('inf') for pos in np.ndindex(grid.shape)}
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
    
def visualize_path(
    elevation_grid_cleaned: np.ndarray, 
    path: List[Tuple[int, int]],
    downsample_factor: int,
    vertical_exaggeration: float,
    ):
    
    if not path:
        print("No path provided to visualize.")
        return
        
    elevation_data = elevation_grid_cleaned.astype(np.float32)

    rows_orig, cols_orig = elevation_data.shape

    x_coords_orig = np.arange(cols_orig)
    y_coords_orig = np.arange(rows_orig)
    
    z_data_downsampled = elevation_data[::downsample_factor, ::downsample_factor]
    # coords should match the size of z_data_downsampled
    x_coords_surface = x_coords_orig[::downsample_factor]
    y_coords_surface = y_coords_orig[::downsample_factor]
    # Bounding box for performance
    path_rows_orig, path_cols_orig = zip(*path)
    
    # Get the elevation z val for each point on path edge with cleaned data
    # Offset z slightly accounting for person height for visibility
    path_z = []
    for r, c in path:
        # Ensure indices are within bounds
        if 0 <= r < elevation_data.shape[0] and 0 <= c < elevation_data.shape[1]:
            elev = elevation_data[r, c]
            path_z.append(elev + 30 if not np.isnan(elev) else 0) # Assign 0 or a low value if it's NaN
        else:
            path_z.append(0) # Default to 0 if out of bounds

    start_node = path[0]
    end_node = path[-1]

    
    start_z_marker = elevation_data[start_node] + 50 if not np.isnan(elevation_data[start_node]) else 0
    end_z_marker = elevation_data[end_node] + 50 if not np.isnan(elevation_data[end_node]) else 0

    # Create the 3D terrain surface
    surface = go.Surface(
        x=x_coords_surface,
        y=y_coords_surface,
        z=z_data_downsampled,
        colorscale='earth',
        cmin=np.nanmin(z_data_downsampled), # Safely ignores nan values
        cmax=np.nanmax(z_data_downsampled), # Safely ignores nan values
        showscale= False 
    )

    # Create the 3D line for the path
    path_line = go.Scatter3d(
        x=path_cols_orig, 
        y=path_rows_orig, 
        z=path_z,
        mode='lines',
        line=dict(color='red', width=8),
        name='A* Path'
    )
    
    # Create markers for the start and end points
    start_marker = go.Scatter3d(
        x=[start_node[1]], y=[start_node[0]], z=[start_z_marker],
        mode='markers', marker=dict(color='lime', size=10, symbol='circle'), name='Start'
    )
    end_marker = go.Scatter3d(
        x=[end_node[1]], y=[end_node[0]], z=[end_z_marker],
        mode='markers', marker=dict(color='magenta', size=10, symbol='x'), name='End'
    )
    
    # Show figure
    fig = go.Figure(data=[surface, path_line, start_marker, end_marker])

    fig.update_layout(
        title='Interactive 3D Visualization of A* Path',
        scene=dict(
            xaxis_title='Pixel Column (East-West)',
            yaxis_title='Pixel Row (North-South)',
            zaxis_title='Elevation (meters)',
            # Vertical exaggeration of path, control variable
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=vertical_exaggeration) 
        ),
        margin=dict(l=0, r=0, b=0, t=40) # Reduce margins
    )

    fig.show()


    