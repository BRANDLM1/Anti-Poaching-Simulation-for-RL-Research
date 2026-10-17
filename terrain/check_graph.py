"""Quick check after rebuilding the graph: one connected component, no isolated nodes.

Run from the project root:  python -m terrain.check_graph
"""
import networkx as nx

from simulation.config import SimConfig

G = nx.read_gml(SimConfig().graph_file)
print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print("components:", nx.number_connected_components(G), "| isolated:", list(nx.isolates(G)))
