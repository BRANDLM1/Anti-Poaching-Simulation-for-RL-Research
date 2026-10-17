from enum import Enum
from dataclasses import dataclass

import networkx as nx

class Phase(Enum):
    PLANNED = "planned"         # decided at midnight, starts at start_tick (random time of day)
    INBOUND = "inbound"
    HUNTING = "hunting"
    OUTBOUND = "outbound"
    ESCAPED = "escaped"
    CAUGHT = "caught"
    TIMED_OUT = "timed_out"     # ran past cfg.max_trip_hours: a failed trip, not a capture

@dataclass
class Trip:
    entry: str
    target: str
    start_tick: int
    phase: Phase = Phase.PLANNED
    exit: str | None = None
    leopard_found: bool = False
    hunting_since: int | None = None
    outcome: str | None = None      # set once, when the trip ends

def entry_points(G):
    """Every node a poacher can enter or leave the reserve through."""
    return [n for n, d in G.nodes(data=True) if d["Type"] == "poacherh"]

def nearest_exit(G, node):
    """The entry point with the shortest road distance from `node`."""
    return min(entry_points(G), key=lambda e: nx.shortest_path_length(G, node, e, weight="terrain_dist"))

# Starts trip at midnight for now (later: learn good start times from a time-of-day heat map)
def maybe_start_trip(day, cfg, rng, G):
    """Determine if a poacher starts a trip on this day, and if so, return a Trip object."""
    if rng.random() < cfg.trip_start_chance_per_day:
        entry = rng.choice(entry_points(G))
        target = rng.choice([n for n, d in G.nodes(data=True) if d["Type"] == "mountain"])
        start_tick = day * cfg.ticks_per_day
        if cfg.trip_start == "random":
            start_tick += rng.randrange(cfg.ticks_per_day)
        elif cfg.trip_start != "midnight":
            raise ValueError(f"Unknown trip_start: {cfg.trip_start!r}")
        return Trip(entry=entry, target=target, start_tick=start_tick)
    return None

def current_goal(trip: Trip):
    """Return the current goal node for the poacher based on the trip phase."""
    if trip.phase in (Phase.INBOUND, Phase.HUNTING):
        return trip.target
    if trip.phase == Phase.OUTBOUND:
        return trip.exit
    return None

def advance_trip(trip, now, cfg, rng, G, poacher):
    """Move the trip to its next phase when the poacher's situation calls for it.

    Phases change on what the body did (arrivals), not on timers. Captures can happen in any
    phase, so the runner handles them and sets Phase.CAUGHT itself.
    """
    if trip.phase == Phase.INBOUND:
        if poacher.just_arrived and poacher.current_node == trip.target:
            trip.phase = Phase.HUNTING
            trip.hunting_since = now

    elif trip.phase == Phase.HUNTING:
        find_chance_per_tick = 1 - (1 - cfg.leopard_find_chance_per_hour) ** (cfg.tick_minutes / 60)
        gave_up = now >= trip.hunting_since + cfg.max_hunt_minutes / cfg.tick_minutes
        if rng.random() < find_chance_per_tick:
            trip.leopard_found = True
        if trip.leopard_found or gave_up:
            trip.phase = Phase.OUTBOUND
            trip.exit = nearest_exit(G, poacher.current_node)

    elif trip.phase == Phase.OUTBOUND:
        # Any entry point counts: the brain picks the route and may reach a different exit first
        if poacher.just_arrived and G.nodes[poacher.current_node]["Type"] == "poacherh":
            trip.phase = Phase.ESCAPED
            trip.outcome = "escaped_with_leopard" if trip.leopard_found else "escaped_empty"
