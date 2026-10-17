# test_reasoning_bank.py: the bank's plumbing with a fake LLM (free, no API key needed).
import random

import networkx as nx

from agents.dqn_poacher import LESSON_CONCERNS
from agents.dqnrb_poacher import DQNRBAgent
from run_ablation import SETTINGS, play
from simulation.config import SimConfig


class FakeWriter:
    """Pretends to be the LLM: after a capture, weigh villages and capture sites heavily."""
    def __init__(self):
        self.debriefs = []

    def __call__(self, debrief):
        self.debriefs.append(debrief)
        caught = "Trip outcome: caught" in debrief
        weights = {name: 0.2 for name in LESSON_CONCERNS}
        if caught:
            weights.update(avoid_villages=0.9, avoid_capture_sites=0.8)
        return weights, "fake lesson after " + ("a capture" if caught else "a trip")


G = nx.read_gml(SimConfig().graph_file)
runs = []
for trigger, use in [("every_trip", "inputs"), ("outcomes", "inputs"), ("every_trip", "bonus")]:
    writer = FakeWriter()
    brain = DQNRBAgent(G, random.Random(1), lesson_trigger=trigger, lesson_use=use, writer=writer,
                       use_knowledge=False, use_threat=False,
                       judge_below_epsilon=0.9 if use == "bonus" else 1.0)
    world = play(SimConfig(days=30, trip_start_chance_per_day=1.0, seed=1), brain, SETTINGS, keep_tracks=False)

    events = [row["event"] for row in brain.bank.log]
    print(f"{trigger}/{use}: {len(world.trips)} trips, {len(writer.debriefs)} lessons asked for, "
          f"written {events.count('written')}, dropped {events.count('dropped')}, retired {events.count('retired')}, "
          f"active now {len(brain.bank.lessons)}")
    print("   weights now:", {k: round(v, 2) for k, v in brain.lesson_weights.items()})
    runs.append((brain, writer))

# v2 (bonus mode, the last run above): lessons lower move scores directly; network inputs stay zero
brain, writer = runs[-1]
view = brain.last_view
neighbors, rows = brain.candidate_moves("border3", "peak1", view)
bonus = brain.move_bonus("border3", neighbors)
assert bonus is not None
print("bonus mode: lesson inputs all zero:", bool((rows[:, -6:] == 0).all()),
      "| score bonus per move (min, max):", [round(float(bonus.min()), 2), round(float(bonus.max()), 2)])
print("debriefs name who caught the poacher:", any(", by " in d for d in writer.debriefs))

# The network really sees the lessons (inputs mode): some moves get non-zero lesson inputs
brain = DQNRBAgent(G, random.Random(1), writer=FakeWriter(), use_knowledge=False, use_threat=False)
brain.lesson_weights = {name: 0.5 for name in LESSON_CONCERNS}
neighbors, rows = brain.candidate_moves("border3", "peak1", view)
print("lesson inputs for border3's moves (max per concern):", rows[:, -6:].max(dim=0).values.round(decimals=2).tolist())

# The debrief must hold only what the poacher knows
print("\nOne debrief:\n" + writer.debriefs[-1])
leaks = [word for word in ["Poacher_Significance", "Ranger_Significance", "herder_activity", "Base_Weight"]
         if word in "".join(writer.debriefs)]
print("\nhidden facts in debriefs:", leaks or "none")
