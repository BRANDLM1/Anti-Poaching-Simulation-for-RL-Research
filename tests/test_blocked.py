# test_blocked.py: blocked edges are found partway and driven back from; forced moves reach the brain.
import networkx as nx

from agents.dqn_poacher import DQNAgent
from agents.planner_ranger import PlannerRanger
from simulation.agent_instance import AgentInstance
from simulation.config import SimConfig
from simulation.world import World

cfg = SimConfig(days=2, trip_start_chance_per_day=1.0)
G = nx.read_gml(cfg.graph_file)

# 1. Body: even the shortest edge (crossed in one tick) finds its block, then returns to where it started
u, v = min(G.edges, key=lambda e: G.edges[e]["terrain_dist"])
body = AgentInstance(u, G, cfg.bike_m_per_tick)
body.destination_node = v
blocked = {tuple(sorted((u, v)))}
flags = []
for now in range(5):
    body.update_movement(now, cfg, blocked)
    flags.append((body.just_blocked, body.just_returned, round(body.progress_on_edge, 2)))
print(f"1. edge {u}-{v} ({G.edges[u, v]['terrain_dist']:.0f} m): {flags[:2]}")
print(f"   back at {body.current_node}, idle: {body.current_node == u and body.destination_node is None}")

# 2. World: block every edge out of every entry point, so poachers can never leave their entry
def make_poacher(world, i):
    return DQNAgent(world.G, world.cfg.rng(f"poacher{i + 1}_brain"))

def make_ranger(world, i):
    return PlannerRanger(world.G, world.cfg.rng(f"ranger{i + 1}_brain"), world.sim, world.cfg)

world = World(cfg, make_poacher, make_ranger)
entry_edges = {tuple(sorted(e)) for n, d in G.nodes(data=True) if d["Type"] == "poacherh" for e in G.edges(n)}
original = world.sim._update_obstacles


def block_entries():
    original()
    world.sim.blocked_edges.update(entry_edges)


world.sim._update_obstacles = block_entries
for _ in range(cfg.ticks_per_day):
    world.step()
kinds = [e["kind"] for e in world.log.events if e["who"] == "poacher1"]
print(f"2. poacher1 hit_block {kinds.count('hit_block')}, turned_back {kinds.count('turned_back')}, "
      f"arrived {kinds.count('arrived')} (expect equal-ish counts, 0 arrivals)")
poacher_brain = world.parties[0].brain
assert isinstance(poacher_brain, DQNAgent)
print(f"   brain remembers {len(poacher_brain.blocked_today)} blocked edges today")

# 3. Forced move: the brain is told, not asked
party = world.parties[0]
if party.active and party.body is not None:
    body = party.body
    while body.destination_node is not None or body.is_busy(world.now):
        world.step()
    body.forced_next = next(iter(G.neighbors(body.current_node)))
    first = len(world.log.events)
    world.step()
    print(f"3. events: {[e['kind'] for e in world.log.events[first:] if e['who'] == 'poacher1']}")
