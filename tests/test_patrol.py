import networkx as nx
from simulation.config import SimConfig
from simulation.reserve import Simulation
from agents.planner_ranger import run_patrol_sim, get_travel_time

cfg = SimConfig()
G = nx.read_gml(cfg.graph_file)
sim = Simulation(G, cfg.rng("weather"), cfg.rng("obstacles"))

budget = cfg.shifts[0].duration_minutes / cfg.tick_minutes          # 240 ticks
path = run_patrol_sim(G, sim, "station", budget, initial_temp=100, cooling_rate=0.995,
                      max_iterations=2000, cfg=cfg, rng=cfg.rng("ranger_plan"))

total = sum(get_travel_time(u, v, sim, cfg) for u, v in zip(path, path[1:]))
print(path)
print(f"starts/ends at station: {path[0] == path[-1] == 'station'} | ticks: {total:.0f} of {budget:.0f} | places: {len(set(path))}")