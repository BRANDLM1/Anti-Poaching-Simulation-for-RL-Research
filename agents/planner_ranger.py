import networkx as nx
import math
import numpy as np
from simulation.weather import RAIN, SNOW
from agents.base import ControlPanel

def get_travel_time(u, v, reserve_state, cfg):
    """Calculates the time (in ticks) to travel between two nodes."""
  
    G = reserve_state.graph
    
    # Check for obstacles
    if tuple(sorted((u, v))) in reserve_state.blocked_edges:
        return float('inf') # Impassable

    base_speed_modifier = 1.0 # Default speed

    node_u_type = G.nodes[u]['Type']
    node_v_type = G.nodes[v]['Type']

    if node_u_type == 'civilization' or node_v_type == 'civilization':
        base_speed_modifier *= 2.0
    
    if node_u_type == 'water' or node_v_type == 'water':
        base_speed_modifier *= 1.5

    if reserve_state.weather == RAIN and (node_u_type == 'water' or node_v_type == 'water'):
        base_speed_modifier *= 2.0
    
    if reserve_state.weather == SNOW:
        base_speed_modifier *= 2.5

    # Use .get_edge_data() to be safe against missing edges, though the self-loop is the main issue
    edge_data = G.get_edge_data(u, v, default={'terrain_dist': 0})
    terrain_dist = edge_data.get('terrain_dist', 0)
    
    travel_time = ((terrain_dist * base_speed_modifier) / (cfg.bike_m_per_tick)) + (cfg.visit_minutes.get(node_v_type, 0) / cfg.tick_minutes)
    return travel_time

def rank_patrol(path, reserve_state, patrol_time_limit, cfg):
    """
    SA energy for produced route, higher val is better
    """
    if len(path) < 2:
        if path:
            node_id = path[0]
            return reserve_state.graph.nodes[node_id]["Ranger_Significance"]
        return 0 #empty path
    
    # Can't double count visit, identify unique node traversals
    # Sum up significance scores
    total_score = sum(reserve_state.graph.nodes[node].get('Ranger_Significance') for node in dict.fromkeys(path))   # unique, in a fixed order
    
    patrol_time = 0

    # Calculate the total travel time for the path
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]

        if u == v:
            continue

        patrol_time += get_travel_time(u, v, reserve_state, cfg)

    # Reject patrols exceeding time limit
    if patrol_time > patrol_time_limit:
        return total_score - 10000 * (patrol_time - patrol_time_limit)
    
    return total_score
def run_patrol_sim(G, reserve_state, start_node, patrol_time_limit, initial_temp, cooling_rate, max_iterations, cfg, rng):
    """
    Simulated annealing to find a near-optimal patrol path for the ranger.
    """

    travel = lambda u, v, d: get_travel_time(u, v, reserve_state, cfg)
    return_time = nx.single_source_dijkstra_path_length(G, start_node, weight=travel)

    current_path, current_time= [start_node], 0
    
    while True:
        here = current_path[-1]
        valid_neighbors = [n for n in G.neighbors(here)  if current_time + get_travel_time(here, n, reserve_state, cfg) + return_time[n] <= patrol_time_limit]

        if not valid_neighbors:
            break

        nxt = rng.choice(valid_neighbors)
        current_time += get_travel_time(here, nxt, reserve_state, cfg)
        current_path.append(nxt)
    
    back = nx.shortest_path(G, current_path[-1], start_node, weight=travel)
    current_path += back[1:]

    best_path = list(current_path)
    best_score = rank_patrol(best_path, reserve_state, patrol_time_limit, cfg)
    
    temp = initial_temp

    for i in range(max_iterations):
        if temp <= 0.1:
            break

        candidate_path = list(best_path)
        if len(candidate_path) < 3:
             continue

        idx = rng.randint(1, len(candidate_path) - 2)
        node_before = candidate_path[idx-1]
        node_after = candidate_path[idx+1]
        
        common_neighbors = set(G.neighbors(node_before)) & set(G.neighbors(node_after))
        if not common_neighbors:
            continue
            
        inserted_node = rng.choice(sorted(common_neighbors))   # sorted: set order changes per process
        candidate_path[idx] = inserted_node

        candidate_score = rank_patrol(candidate_path, reserve_state, patrol_time_limit, cfg)

        if candidate_score > best_score or rng.random() < np.exp((candidate_score - best_score) / temp):
            best_path = candidate_path
            best_score = candidate_score

        temp *= cooling_rate
        
    return best_path


class PlannerRanger(ControlPanel):
    """R0 ranger: plans and executes a patrol by simulated annealing at the start of each shift.
    Capable of reading only public conditions (weather, blocked paths) from the world when planning.
    """

    def __init__(self, graph, rng, sim, cfg, initial_temp=100, cooling_rate=0.995, max_iterations=2000):
        super().__init__(graph, rng)
        self.sim = sim
        self.cfg = cfg
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.route = []
        self.route_index = 0

    def start_episode(self, node, goal, situation=None):
        """A shift begins at `node`; situation["budget_ticks"] is how long the shift lasts."""
        if situation is None:
            raise ValueError("PlannerRanger needs situation['budget_ticks'] at the start of a shift")
        self.route = run_patrol_sim(self.graph, self.sim, node, situation["budget_ticks"],
                                    self.initial_temp, self.cooling_rate, self.max_iterations,
                                    self.cfg, self.rng)
        self.route_index = 1

    def next_move(self, node, goal, situation):
        """The next stop on the planned route, or None once the route is finished."""
        if self.route_index < len(self.route):
            nxt = self.route[self.route_index]
            self.route_index += 1
            return nxt
        return None
