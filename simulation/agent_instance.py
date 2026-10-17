import math

import numpy as np 

class AgentInstance:
    def __init__(self, start_node, graph, speed_m_per_tick: float = 50.0):
        self.graph = graph
        self.start_node = start_node
        self.current_node = start_node
        self.destination_node = None
        self.progress_on_edge = 0.0  # Progress from 0.0 to 1.0
        self.path_taken = [start_node]
        self.just_arrived = False # Flag for event-based checks
        self.returning = False      # driving back to current_node after finding the edge blocked
        self.just_blocked = False   # one-tick flag: found the block this tick
        self.just_returned = False  # one-tick flag: back where the blocked move started
        self.speed = speed_m_per_tick # Meters per tick
        self.busy_until = 0
        self.speed_factor = 1.0
        self.slow_until = 0
        self.forced_next: str | None = None
        self.outcome: tuple[str, str] | None = None

    def is_busy(self, now):
        return now < self.busy_until

    def get_current_pos(self):
        """Calculates the agent's precise (x, y) coordinates, interpolating along edges."""
        if self.destination_node is None:
            # Agent is stationary at a node
            return self.graph.nodes[self.current_node]['pos']
        else:
            # Agent is moving between nodes, interpolate position
            pos_u = np.array(self.graph.nodes[self.current_node]['pos'])
            pos_v = np.array(self.graph.nodes[self.destination_node]['pos'])
            return pos_u + self.progress_on_edge * (pos_v - pos_u)

    def take_forced_next(self):
        if self.forced_next is not None:
            self.destination_node = self.forced_next
            self.forced_next = None
        return self.destination_node

    def update_movement(self, now, cfg, blocked_edges=()):
        """Updates the agent's progress along its current edge for one tick.

        blocked_edges holds sorted (u, v) pairs. A blocked edge is found cfg.block_found_at of the way
        along; the agent then drives back (progress runs down to 0) and is idle at current_node again.
        """

        self.just_arrived = False
        self.just_blocked = False
        self.just_returned = False
        if self.is_busy(now):
            return False

        if now >= self.slow_until:
            self.speed_factor = 1.0

        if self.destination_node is None:
            return False #Not moving

        edge_length = self.graph.edges[self.current_node, self.destination_node]['terrain_dist']

        # Avoid division by zero on very short edges
        progress_made = 1.0 if edge_length <= 0 else self.speed * self.speed_factor / edge_length

        if self.returning:
            self.progress_on_edge -= progress_made
            if self.progress_on_edge <= 0:
                # Back at the start of the blocked edge: no visit time, not a new arrival
                self.destination_node = None
                self.progress_on_edge = 0.0
                self.returning = False
                self.just_returned = True
            return False

        self.progress_on_edge += progress_made

        # Checked before arrival, so a short edge crossed in one tick still finds its block
        edge = tuple(sorted((self.current_node, self.destination_node)))
        if self.progress_on_edge >= cfg.block_found_at and edge in blocked_edges:
            self.progress_on_edge = cfg.block_found_at
            self.returning = True
            self.just_blocked = True
            return False

        # Check if the agent has arrived at its destination
        if self.progress_on_edge >= 1.0:
            self.current_node = self.destination_node
            self.destination_node = None
            self.progress_on_edge = 0.0
            self.path_taken.append(self.current_node)
            self.just_arrived = True # Set flag for one tick
            minutes = cfg.visit_minutes.get(self.graph.nodes[self.current_node]["Type"], 0)
            self.busy_until = max(self.busy_until, now + math.ceil(minutes / cfg.tick_minutes)) 
            return True
        else:
            return False
