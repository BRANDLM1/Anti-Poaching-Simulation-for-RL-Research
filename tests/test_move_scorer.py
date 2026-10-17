# test_move_scorer.py: encode_move gives one row of numbers per neighbor
import random

import networkx as nx
import torch

from agents.dqn_poacher import DQNAgent
from simulation.config import SimConfig

G = nx.read_gml(SimConfig().graph_file)
a = DQNAgent(G, random.Random(0), use_road_distance=True)

nbrs = list(G.neighbors("border3"))
rows = torch.tensor([a.encode_move("border3", n, "peak1") for n in nbrs])

 # expect (degree of border3, 27): 17 node + 4 move + 6 lesson inputs
print(rows.shape)

for n, row in zip(nbrs, rows):
    # print positions 17-20 following the 17 encode_node numbers: edge length, progress, threat, road progress
    print(f"{n:10s} edge {row[17]:.3f}  progress {row[18]:+.3f}  threat {row[19]:.2f}  road {row[20]:+.3f}")
