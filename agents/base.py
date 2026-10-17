from abc import ABC, abstractmethod


class ControlPanel(ABC):
    """Agent brain responding to sim world instructions, chooses moves and learns from what it's told.

    sim world instructions as in when episodes (trips, shifts) start, where agents are placed,
    rewards, captures, etc..
    
    Only next_move is required; every other hook does nothing unless a brain needs it.
    """

    def __init__(self, graph, rng):
        self.graph = graph      # to look up neighbors and node facts
        self.rng = rng          # random stream for reproducibilty

    def start_episode(self, node, goal, situation=None):
        """The world placed this agent at `node` (a trip or shift began). `goal` may be None."""

    @abstractmethod
    def next_move(self, node, goal, situation):
        """The agent is idle at `node`. Return a neighboring node to move to, or None to stay put this tick."""

    def observe(self, event):
        """Something happened to this agent: {"kind": ..., "node": ..., "pos": ...} plus extras by kind.

        Examples:
        arrived      reached `node`
        captured     caught near `node`
        blocked      found `edge` (node, other) blocked partway along; now driving back to `node`
        returned     back at `node` after a blocked edge (idle again, not a new arrival)
        forced_move  a random event sent the agent from `node` toward `to` without asking the brain
        """

    def feedback(self, reward, done):
        """The world's score for this tick. `done` is True on the tick the episode ended."""

    def end_episode(self, outcome):
        """The trip or shift is over and report outcome."""

    def end_day(self):
        """On midnight fade memories, etc.."""


class Wanderer(ControlPanel):
    """Baseline agent moves to a random neighbor."""

    def next_move(self, node, goal, situation):
        return self.rng.choice(list(self.graph.neighbors(node)))
