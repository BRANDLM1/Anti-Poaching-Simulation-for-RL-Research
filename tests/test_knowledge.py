import random

import networkx as nx

from agents.dqn_poacher import DQNAgent

from simulation.config import SimConfig

G = nx.read_gml(SimConfig().graph_file)

agent = DQNAgent(G, random.Random(0))

print(f'{agent.known["civ1"]}, .... {agent.known["peak7"]}')
print(f'{G.nodes["civ1"]["Type"]}')

v1, v2 = agent.encode_node("civ1"), agent.encode_node("peak7")

print(len(v1), len(v2))     # 17 17
print(v1)                   # type flag 0, all six type slots 0, hidden distances 0,0 pairs
print(v2)                   # type flag 1, a 1 in the mountain slot
agent.observe({"kind": "arrived", "node": "civ1"})
print(agent.encode_node("civ1"))   # now the type flag is 1, with a 1 in the civilization slot