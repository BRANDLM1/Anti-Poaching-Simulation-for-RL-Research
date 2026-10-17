# shortest_path_poacher.py
"""Baseline fastest route Dijkstra over terrain_dist to the current goal.
No learning/memory beyond today's blocked edges.
"""
import networkx as nx

from agents.base import ControlPanel


class ShortestPathPoacher(ControlPanel):

    def __init__(self, graph, rng):
        super().__init__(graph, rng)
         # Set blocked edges, sorted (u, v), avoid until midnight
        self.blocked_today = set()

    def next_move(self, node, goal, situation):
        # Route around known blocks, if cut off still try the normal route
        open_graph = nx.restricted_view(self.graph, [], self.blocked_today)
        try:
            path = nx.dijkstra_path(open_graph, node, goal, weight="terrain_dist")
        except nx.NetworkXNoPath:
            path = nx.dijkstra_path(self.graph, node, goal, weight="terrain_dist")
        return path[1] if len(path) > 1 else None

    def observe(self, event):
        if event["kind"] == "blocked":
            u, v = sorted(event["edge"])
            self.blocked_today.add((u, v))

    def end_day(self):
        self.blocked_today.clear()
