# dqnrb_poacher.py

import json

import torch

from agents.dqn_poacher import DQNAgent, LESSON_CONCERNS, NODE_TYPES
from agents.reasoning_bank import ReasoningBank, LLMLessonWriter

STATION_NEAR_M = 10_000     # "near the station" fades to 0 at this distance


def caught_by(reason):
    """Who caught the poacher, in its own words (it would know a herder from a ranger)."""
    if "herder" in reason:
        return "a local herder"
    if "station" in reason:
        return "rangers at the ranger station"
    if reason:
        return "a ranger on patrol"
    return "someone (unknown)"

TYPE_NAMES = {"poacherh": "entry point", "rangerh": "ranger station", "civilization": "village",
              "mountain": "mountain", "camera": "camera", "water": "water", None: "unknown"}


class DQNRBAgent(DQNAgent):
    """DQN poacher with a reasoning bank. LLM writes lessons from trip debriefs for network to read.

    lesson_trigger: "every_trip" (a lesson after every training trip) or "outcomes" (only after a capture
    or a success). Lessons are written and judged only while training.
    lesson_use: "inputs" or "bonus"
    (v1 inputs: weight x hit as 6 extra network inputs. Network must derive meaning)
    (v2 bonus: same numbers lower a move's score directly when choosing)
    bonus_strength: how many reward units a fully-matched, fully-weighted concern subtracts.
    judge_below_epsilon: lessons collect strikes only once exploration is below this
    `writer` is anything callable debrief -> (weights, text) or None; defaults to Claude (LLMLessonWriter).
    """

    def __init__(self, graph, rng, lesson_trigger="every_trip", lesson_use="inputs", bonus_strength=2.0,
                 judge_below_epsilon=1.0, writer=None, **kwargs):
        if lesson_trigger not in ("every_trip", "outcomes"):
            raise ValueError(f"Unknown lesson_trigger: {lesson_trigger!r}")
        if lesson_use not in ("inputs", "bonus"):
            raise ValueError(f"Unknown lesson_use: {lesson_use!r}")
        self.lesson_trigger = lesson_trigger
        self.lesson_use = lesson_use
        self.bonus_strength = bonus_strength
        self.judge_below_epsilon = judge_below_epsilon
        # The bank must exist before DQNAgent.__init__, which sizes the network by calling encode_move
        self.bank = ReasoningBank(writer if writer is not None else LLMLessonWriter())
        self.lesson_weights = self.bank.weights()
        super().__init__(graph, rng, **kwargs)
        self.trips_done = 0
        self.trip_route = []

    # what the network sees
    def lesson_features(self, node, neighbor):
        """Network inputs weight x hit per concern in v1 inputs. Zeros in bonus mode."""
        if self.lesson_use == "bonus":
            return [0.0] * len(LESSON_CONCERNS)
        return self._weighted_hits(neighbor)

    def move_bonus(self, node, neighbors):
        """Each move's score drops by bonus_strength x average weighted hit over the concerns."""
        if self.lesson_use != "bonus":
            return None
        penalties = [sum(self._weighted_hits(n)) / len(LESSON_CONCERNS) for n in neighbors]
        return -self.bonus_strength * torch.tensor(penalties)

    def _weighted_hits(self, neighbor):
        """Weight of each concern x how much moving to `neighbor` hits it (poacher's knowledge only)."""
        facts = self.known[neighbor]
        node_type = facts["Type"]
        station_d = facts["dist_to_station"]
        row, col = self._reserve_to_grid(self.graph.nodes[neighbor]["pos"], self.map_bounds)
        hits = {
            "avoid_villages": 1.0 if node_type == "civilization" else 0.0,
            "avoid_station_area": 1.0 if node_type == "rangerh"
                                  else (max(0.0, 1 - station_d / STATION_NEAR_M) if station_d is not None else 0.0),
            "avoid_high_centrality": facts["centrality"] / self.max_centrality,
            "prefer_low_viewshed": facts["Viewshed"],
            "avoid_capture_sites": min(facts["times_caught"], 5) / 5,
            "avoid_threat_cells": float(self.threat_grid[row, col]),
        }
        return [self.lesson_weights[name] * hits[name] for name in LESSON_CONCERNS]

    # trip data for debrief

    def start_episode(self, node, goal, situation=None):
        super().start_episode(node, goal, situation)
        self.trip_route = [node]
        self.trip_weather = []
        self.trip_night_moves = 0
        self.trip_caught_near = None
        self.trip_caught_by = None

    def next_move(self, node, goal, situation):
        self.trip_weather.append(situation["weather"])
        self.trip_night_moves += bool(situation["night"])
        return super().next_move(node, goal, situation)

    def observe(self, event, reserve=None):
        super().observe(event, reserve)
        if event["kind"] == "arrived":
            self.trip_route.append(event["node"])
        elif event["kind"] == "captured":
            self.trip_caught_near = event["node"]
            self.trip_caught_by = caught_by(event.get("reason", ""))

    def end_episode(self, outcome):
        super().end_episode(outcome) # epsilon, target network, last_trip_stats
        if not self.training:
            return
        self.trips_done += 1
        if self.epsilon < self.judge_below_epsilon: # random exploration says nothing about the lessons
            self.bank.judge(outcome, self.trips_done)
        wants_lesson = self.lesson_trigger == "every_trip" or outcome.startswith("caught") \
            or outcome == "escaped_with_leopard"
        if wants_lesson:
            self.bank.write(self.debrief(outcome), self.trips_done)
        self.lesson_weights = self.bank.weights()
        self.last_trip_stats.update({"lessons_active": len(self.bank.lessons), **{f"w_{name}": round(w, 3) for name, w in self.lesson_weights.items()}})

    def debrief(self, outcome):
        """The trip as the poacher experienced it with its route, outcome and knowledge"""
        def place(n):
            return f"{n} ({TYPE_NAMES[self.known[n]['Type']]})"

        moves = len(self.trip_weather)
        weather = max(set(self.trip_weather), key=self.trip_weather.count) if self.trip_weather else "unknown"
        caught_places = [f"{n} x{f['times_caught']}" for n, f in self.known.items() if f["times_caught"]]
        known_counts = {TYPE_NAMES[t]: sum(f["Type"] == t for f in self.known.values()) for t in NODE_TYPES}
        lines = [
            f"Trip outcome: {outcome.replace('_', ' ')}.",
            f"Route ({moves} moves): " + " -> ".join(place(n) for n in self.trip_route),
            f"Caught near: {place(self.trip_caught_near)}, by {self.trip_caught_by}." if self.trip_caught_near
            else "Not caught.",
            f"Conditions: mostly {weather}; {self.trip_night_moves} of {moves} moves at night.",
            "Places known so far: " + ", ".join(f"{k} {v}" for k, v in known_counts.items() if v) + ".",
            "Places where I have been caught before: " + (", ".join(caught_places) or "none") + ".",
            "Current weights: " + json.dumps({k: round(v, 2) for k, v in self.lesson_weights.items()}),
            "Return updated weights for every concern and a one-sentence lesson.",
        ]
        return "\n".join(lines)

