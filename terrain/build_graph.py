import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import time
import pyproj
import rasterio
import subprocess
import os
import random
from terrain.paths import DEM_PATH
from terrain.terrain_distance import custom_dist, euclidean_distance

def calculate_viewshed_scores(G, dtm_path, observer_nodes, max_distance=25000, observer_height=10):
    """
    Calculates and adds a 'Viewshed' score to each node in the graph using GDAL.

    Viewsshed score (0-1.0) = proportion of potentially observable points actually visible at a given node

    Args:
        G (nx.Graph): The graph. Nodes must have a 'pos' attribute with UTM coordinates.
        dtm_path (str): The file path to your DTM GeoTIFF in the same UTM zone.
        observer_nodes (dict): A dictionary where keys are observer IDs and values are
                            their (utm_x, utm_y) pos tuples.
        max_distance (int): The maximum distance in meters for the viewshed calculation.
        observer_height (float): The height in meters of the observer above the ground.
    """
    print(f"\nStarting viewshed analysis for {len(observer_nodes)} optimal observation points...")
    
    # Initialize a temporary counter for visibility on each node
    visibility_counts = {node_id: 0 for node_id in G.nodes()}
    observer_node_ids = list(observer_nodes.keys())

    # Add observer points to the graph if they don't exist
    for obs_id, obs_pos in observer_nodes.items():
        if obs_id not in G:
            G.add_node(obs_id, Type='optimal_peak', pos=obs_pos)

    for obs_id, obs_pos in observer_nodes.items():
        output_tif = f"temp_viewshed_{obs_id}.tif"
        
        gdal_command = [
            'gdal_viewshed',
            '-md', str(max_distance),     # Max distance
            '-ox', str(obs_pos[0]),       # Observer UTM X
            '-oy', str(obs_pos[1]),       # Observer UTM Y
            '-oz', str(observer_height),  # Observer height
            '-a_nodata', '0',             # NoData value for non-visible areas
            '-vv', '255',                 # Value for visible areas
            dtm_path,                     # Input DTM
            output_tif                    # Output viewshed raster
        ]
        
        try:
            # Run the command, checking for errors and capturing output
            subprocess.run(gdal_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: gdal_viewshed failed for observer '{obs_id}'.")
            print(f"    GDAL Error: {e.stderr}")
            continue # Skip to the next observer
        except FileNotFoundError:
            print("    FATAL ERROR: 'gdal_viewshed' command not found. Ensure GDAL is installed and in your system's PATH.")
            return 

        #  Sample output raster for all nodes
        try:
            with rasterio.open(output_tif) as viewshed_src:
                # Ensure all nodes, including new observers are checked
                all_node_ids = list(G.nodes())
                all_node_coords = [G.nodes[nid]['pos'] for nid in all_node_ids]
                
                visibility_values = [val[0] for val in viewshed_src.sample(all_node_coords)]
                
                for i, node_id in enumerate(all_node_ids):
                    if visibility_values[i] > 0:
                        # Lazily initialize counter for newly added observer nodes
                        if node_id not in visibility_counts:
                            visibility_counts[node_id] = 0
                        visibility_counts[node_id] += 1
        finally:
            # Ensure the temporary file is always removed
            if os.path.exists(output_tif):
                os.remove(output_tif)

    # Normalize counts and add the final 'Viewshed' attribute to the graph
    num_observers = len(observer_node_ids)
    final_viewshed_scores = {node_id: 0.0 for node_id in G.nodes()} # Default to 0
    if num_observers > 0:
        for node_id, count in visibility_counts.items():
            final_viewshed_scores[node_id] = count / num_observers
    
    nx.set_node_attributes(G, final_viewshed_scores, 'Viewshed')
    print("  Viewshed analysis complete.")

def main():

    print("Initializing graph and transforming coordinates...")
    nodes_df = pd.read_csv('data/Reserve.csv')
    nodes_df.set_index('Node', inplace=True)
    G = nx.Graph()

    # Define the transformer from Lat/Lon (EPSG:4326) to UTM Zone 47N (EPSG:32647)
    transformer = pyproj.Transformer.from_crs("epsG:4326", "epsg:32647", always_xy=True)

    for node_id, attributes in nodes_df.to_dict('index').items():
        # Convert coordinates to UTM
        utm_x, utm_y = transformer.transform(attributes['Longitude'], attributes['Latitude'])
        attributes['pos'] = (utm_x, utm_y)
        # Add to graph
        G.add_node(node_id, **attributes)

    print(f"Graph created with {G.number_of_nodes()} initial nodes (UTM coordinates).")

    # KDTree sparse graph set to 10 KNN
    # Avoiding complete graph for realism and compute time
    start_time = time.time()
    nodes = list(G.nodes())
    positions = np.array([G.nodes[n]['pos'] for n in nodes])
    kdtree = KDTree(positions)
    KNN = 10

    for i, node_u in enumerate(nodes):
        distances, indices = kdtree.query(positions[i], k=KNN + 1)
        for neighbor_index in np.asarray(indices)[1:]:
            node_v = nodes[neighbor_index]
            if not G.has_edge(node_u, node_v):
                distance_cost = custom_dist(node_u, node_v, G)
                if distance_cost != float('inf'):
                    G.add_edge(node_u, node_v, terrain_dist=distance_cost)

    minutes, seconds = divmod(time.time() - start_time, 60)
    print(f"Computed sparse graph with {G.number_of_edges()} edges in {int(minutes)}m {seconds:.2f}s.")

    print("Calculating viewshed, centrality, and other weights...")

    mountains = [n for n, d in G.nodes(data=True) if d.get('Type') == 'mountain']
    observers = {n: G.nodes[n]['pos'] for n in mountains}
    calculate_viewshed_scores(G, DEM_PATH, observers)

    # Centrality and weight calcs
    centrality = nx.betweenness_centrality(G, weight='terrain_dist')
    nx.set_node_attributes(G, centrality, 'centrality')

    water_nodes = [n for n, d in G.nodes(data=True) if d.get('Type') == 'water']
    civ_nodes = [n for n, d in G.nodes(data=True) if d.get('Type') == 'civilization']
    sighting_nodes = [n for n, d in G.nodes(data=True) if d.get('Type') == 'sighting']
    station_nodes = [n for n, d in G.nodes(data=True) if d.get('Type') == 'rangerh']

    for node_id in G.nodes():
        dist_to_water = min([euclidean_distance(node_id, w, G) for w in water_nodes]) if water_nodes else float('inf')
        dist_to_civ = min([euclidean_distance(node_id, c, G) for c in civ_nodes]) if civ_nodes else float('inf')
        dist_to_sighting = min([euclidean_distance(node_id, s, G) for s in sighting_nodes]) if sighting_nodes else float('inf')
        dist_to_station = min([euclidean_distance(node_id, r, G) for r in station_nodes]) if station_nodes else float('inf')
        
        G.nodes[node_id]['dist_to_water'] = dist_to_water
        G.nodes[node_id]['dist_to_civ'] = dist_to_civ
        G.nodes[node_id]['dist_to_sighting'] = dist_to_sighting
        G.nodes[node_id]['dist_to_station'] = dist_to_station

        node_centrality = G.nodes[node_id].get('centrality', 0)
        viewshed_score = G.nodes[node_id].get('Viewshed', 0)

        G.nodes[node_id]['Ranger_Base_Weight'] = (viewshed_score / 10) - (1 / (dist_to_civ + 1)) - (1 / (dist_to_station + 1))
        G.nodes[node_id]['Ranger_Significance'] = (node_centrality * 1000) + (10 / (dist_to_water + 1))
        G.nodes[node_id]['Poacher_Base_Weight'] = (viewshed_score * 10) + (5 / (dist_to_sighting + 1))
        G.nodes[node_id]['Poacher_Significance'] = (node_centrality * 1000) + (10 / (dist_to_water + 1)) + dist_to_sighting


    max_civ_centrality_score = max([G.nodes[n].get('centrality', 0) for n, d in G.nodes(data=True) if d.get('Type') == 'civilization'], default=0)
    for node_id, node_data in G.nodes(data=True):
        if node_data.get('Type') == 'civilization':
            activity = (G.nodes[node_id].get('centrality', 0) / max_civ_centrality_score)
            G.nodes[node_id]['herder_activity'] = max(0.10, activity)
            
    print("Map Complete.")

    # Save results
    node_data = [{'Node': node, **G.nodes[node]} for node in G.nodes()]
    df_final = pd.DataFrame(node_data)
    df_final.to_csv("data/sparse_weighted_graph.csv", index=False)
    print("Final map saved to 'data/sparse_weighted_graph.csv'")

    nx.write_gml(G, "data/sparse_weighted_graph.gml")
    print("Final map saved to 'data/sparse_weighted_graph.gml'")

if __name__ == "__main__":
    main()