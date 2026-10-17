# test_learning.py: can learn() fit a small case?
import random
from types import SimpleNamespace

import networkx as nx

from simulation.config import SimConfig
import torch

from agents.dqn_poacher import DQNAgent

G = nx.read_gml(SimConfig().graph_file)
a = DQNAgent(G, random.Random(0), batch_size=8)
view = a.last_view = SimpleNamespace(weather="Clear", night=False, map_bounds=a.map_bounds)

neighbors, rows = a.candidate_moves("border3", "peak1", view)
print(f"input size {a.input_size}, rows {tuple(rows.shape)}")

# 1. A move that ends the trip with reward +10: its score should approach 10
a.replay_memory.clear()
for _ in range(8):
    a.remember(rows[0], 10.0, None, True)
for _ in range(300):
    a.learn(8)
with torch.no_grad():
    print(f"1. done move scores {a.main_network(rows[:1]).item():.2f} (expect ~10)")

# 2. A non-final move (reward 0) whose options include the +10 one: target = 0.99 * max(next scores)
a.update_target_network()
_, next_rows = a.candidate_moves("border3", "peak1", view)
a.replay_memory.clear()
for _ in range(4):
    a.remember(rows[1], 0.0, next_rows, False)    # next options padded to max_degree, masked
    a.remember(rows[0], 10.0, None, True)
for _ in range(300):
    a.learn(8)
with torch.no_grad():
    best_next = a.target_network(next_rows).max().item()
    print(f"2. earlier move scores {a.main_network(rows[1:2]).item():.2f} "
          f"(expect ~0.99 x {best_next:.2f} = {0.99 * best_next:.2f}; no nan)")
