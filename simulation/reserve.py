# reserve.py
import numpy as np

from simulation.weather import CLEAR, RAIN, SNOW, DETECTION_RADIUS_M, DISPLAY, next_weather

def bounds_from_graph(G, margin=2000):
    """(min_x, min_y, max_x, max_y) around every node, plus a 2 km margin. Shared by the world and the brains."""
    xs = [d["pos"][0] for _, d in G.nodes(data=True)]
    ys = [d["pos"][1] for _, d in G.nodes(data=True)]
    return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)

class Simulation:
    def __init__(self, G, weather_rng, obstacle_rng):
        self.graph = G
        self.current_day = 0
        self.weather = CLEAR
        self.blocked_edges = set() # blocked edges {(u,v), (x,y)}
        self.map_bounds = bounds_from_graph(G)
        self.weather_rng = weather_rng      # own stream: weather never depends on anything else
        self.obstacle_rng = obstacle_rng    # own stream: switching obstacles off can't change the weather

    def check_for_capture(self, poacher, ranger, radius_factor=1.0):
        G = self.graph
        # Condition 1: Same node check
        if poacher.current_node == ranger.current_node:
            return True, "Caught at same node"
        # Cond 2 deleted random static chance of capture
        # Condition 3: Line of sight at step check
        gap = np.linalg.norm(np.array(poacher.get_current_pos()) - np.array(ranger.get_current_pos()))
        if gap <= DETECTION_RADIUS_M[self.weather] * radius_factor:
            return True, f"Visual detection ({DISPLAY[self.weather]})"

        return False, ""

    def update_environment(self):
        self.current_day += 1
        print(f"--- Day {self.current_day}: Updating Reserve Environment (New Day New Me!) ---")
        self._update_weather()
        self._update_obstacles()
        ## Add other updates

    def _update_weather(self):
        # Stochastic weather
        # Import annual weather stats for more accuracy (AFTER SUBMISSION)
        self.weather = next_weather(self.weather, self.weather_rng)
        print(f"Weather Update: {DISPLAY[self.weather]}")

    def _update_obstacles(self):
        # Dependent on snow weather
        self.blocked_edges.clear()
        if self.weather == SNOW:
            # There's a 80% chance of 1-3 random paths becoming blocked
            if self.obstacle_rng.random() < 0.8:
                for i in range(3):
                    # Pick a random edge from the graph to block
                    u, v = self.obstacle_rng.choice(list(self.graph.edges()))
                    self.blocked_edges.add(tuple(sorted((u, v))))
                    # Canonical order
                    print(f"Obstacle: Path between {u} and {v} is now snowed in. (-_-)")
        if self.weather == RAIN:
            # There's a 20% chance of a random path becoming blocked
            if self.obstacle_rng.random() < 0.2:
                # Pick a random edge from the graph to block
                u, v = self.obstacle_rng.choice(list(self.graph.edges()))
                self.blocked_edges.add(tuple(sorted((u, v)))) 
                # Canonical order
                print(f"Obstacle: Path between {u} and {v} is now flooded. (O-O)")