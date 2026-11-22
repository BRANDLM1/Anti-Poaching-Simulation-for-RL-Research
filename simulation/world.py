"""The simulation world

The world is the keeper of the clock, weather, agent trips, ranger shifts, event log, and game rules (captures, random events, rewards)
Simulation agents consult a ControlPanel for decision-making.
The world only ever talks to agents through ControlPanel hooks, ensuring reproducible format for performance testing.
"""
from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np

# Physical recorded state of the reserve at any point of the simulation
from simulation.reserve import Simulation
# Agent state and location at any point of the simulation
from simulation.agent_instance import AgentInstance
# Control panel, simulation information access point for agents
from agents.base import ControlPanel

# Simulation clock
from simulation.clock import is_night, minute_of_day, shift_windows
# Simulation-level event log
from simulation.event_log import EventLog
# Event engine responsible for weather, random events
from simulation.random_events import EVENTS, matching, roll, apply

# Manages poacher agent's poaching trip status/states
from simulation.trip import Phase, Trip, maybe_start_trip, current_goal, advance_trip, nearest_exit


# Observed stalling behavior in early tests, so 'timeed_out' reward is set equal to 'caught'
# step() advances exactly 1 tick with reward of -1
REWARDS = {"tick": -1, "caught": -5000, "retreat": -1000, "leopard": 2000, "escaped": 3000, "timed_out": -5000}
ACTIVE_PHASES = (Phase.INBOUND, Phase.HUNTING, Phase.OUTBOUND)


# Create clock tracking time of day 'minute of day' from tracked ticks
def clock_str(tick, cfg):
    m = int(minute_of_day(tick, cfg))
    return f"{m // 60:02d}:{m % 60:02d}"

# Tracking current world conditions
def build_situation(agent, tick, sim, cfg, arriving):
    situation = {
        "night": is_night(tick, cfg),  # Determine day/night at a given moment
        "weather": sim.weather,        # Determine weather at a given moment 
        "headlights": True,            # Determine whether vehicle headlights are on (currently fixed to true)
        "motor_bike": True,            # Determine vehicle type for firing of vehicle based events (currently fixed to True)
        "node_type": None,             # Node type identified for firing of 'node_type' based events
        "herder_activity": None,       # Refers to population/level of activity at a point of civilization. May alter event chance of poacher being caught upon arrival
    }
    # Record node based intelligence upon arrival
    if arriving:
        situation["node_type"] = sim.graph.nodes[agent.current_node].get("Type")
        situation["herder_activity"] = sim.graph.nodes[agent.current_node].get("herder_activity")
    return situation

# Function to roll event engine and log in EventLog given the current built situation
def fire_events(when, role, name, agent, tick, sim, cfg, ctx, log):
    """Roll the event table for one agent. `role` matches the table's `who`; `name` goes in the log."""
    situation = build_situation(agent, tick, sim, cfg, arriving=(when == "on_arrival"))
    for rule in matching(EVENTS, when, role, situation):
        if roll(rule, ctx, situation):
            record = apply(rule, agent, tick, ctx)
            log.event(tick, record["event"], name, node=agent.current_node, pos=agent.get_current_pos(),
                      random=True, effect=record["effect"])

# Intentionally left both 'Party' and 'Ranger Unit' handles ambiguous for when multiple agent instances become allowed
@dataclass
class Party:
    """The class encapsulating a complete poacher or 'poacher party': a brain that persists across trips in a simulation, plus the current trip and poacher body."""
    name: str
    brain: ControlPanel
    trip: Trip | None = None
    body: AgentInstance | None = None
    trip_reward: float = 0
    closest_gap: float | None = None
    end_reason: str | None = None

    @property
    def active(self):
        return self.trip is not None and self.trip.phase in ACTIVE_PHASES


@dataclass
class RangerUnit:
    """The class encapsulating a complete ranger or 'ranger unit': a brain that persists across trips in a simulation, plus shift info and the ranger body"""
    name: str
    brain: ControlPanel
    body: AgentInstance
    shift_end: int | None = None
    # shift_end is set while on shift, None while off duty.


class World:
    def __init__(self, cfg, make_poacher_brain, make_ranger_brain, station="station"):
        """make_*_brain(world, index) -> a ControlPanel. Factories let each brain get its own rng."""
        self.cfg = cfg
        # Establish ranger station
        self.station = station
        # Load reserve graph
        self.G = nx.read_gml(cfg.graph_file)
        # Establish initial physical state of the reserve
        self.sim = Simulation(self.G, cfg.rng("weather"), cfg.rng("obstacles"))

        # Establish run log to record moves and events
        self.log = EventLog(cfg)
        # Captures context at a given instance
        self.ctx = {"rng": cfg.rng("events"), "graph": self.G, "tick_minutes": cfg.tick_minutes}
        # Stores rngs for world reproducability across various agents
        self.trips_rng = cfg.rng("trips")
        self.leopard_rng = cfg.rng("leopards")
        self.station_rng = cfg.rng("station")

        # A single world can simulate multiple poacher parties and ranger units. 
        # Creates complete agents for each learner
        self.parties = [Party(f"poacher{i + 1}", make_poacher_brain(self, i)) for i in range(cfg.poacher_parties)]
        n_rangers = max(spec.rangers for spec in cfg.shifts)
        self.rangers = [RangerUnit(f"ranger{i + 1}", make_ranger_brain(self, i), self._new_body(station))
                        for i in range(n_rangers)]

        # Initializing run records and tracking vars
        self.now = 0
        self.trips = []                 # one finished-trip record per trip
        self.shift_starts = {}          # start tick -> (end tick, rangers on that shift)
        # List index storing position for first event of the day given prior days event records
        self.day_first_event_index = 0

    #--------------------------------------------------------------------------#
    ###--------------------------------Public--------------------------------###

    @property
    # Returns whether sim has reached it's configured length
    def done(self):
        return self.now >= self.cfg.days * self.cfg.ticks_per_day

    def step(self):
        """Advance the world by one tick. Returns the events logged during it."""
        first = len(self.log.events)
        if self.now % self.cfg.ticks_per_day == 0:
            self._start_day(self.now // self.cfg.ticks_per_day)

        self._start_scheduled_trips()
        self._start_shifts()
        self._choose_destinations()
        self._advance_movement()
        for party in self.parties:
            if party.active:
                self._resolve(party)
        self.scheduled_end_tick()

        if (self.now + 1) % self.cfg.ticks_per_day == 0:
            self._print_day_summary()
        self.now += 1
        return self.log.events[first:]
    # Function called by run_simulation.py to start sim
    def run(self):
        while not self.done:
            self.step()

    #--------------------------------------------------------------------------#
    ###---------Manages day, ranger shift, and poacher trip triggers---------###

    # Creates new agent instance
    def _new_body(self, node):
        return AgentInstance(start_node=node, graph=self.G, speed_m_per_tick=self.cfg.bike_m_per_tick)

    # Function call for every agent brain in simulation
    def _all_brains(self):
        return [p.brain for p in self.parties] + [r.brain for r in self.rangers]

    # Initialize reserve condition and log events at the start of each day
    def _start_day(self, day):
        if day > 0:
            for brain in self._all_brains():
                brain.end_day()
        self.sim.update_environment()
        self.day_first_event_index = len(self.log.events)
        self.log.event(self.now, "weather", "world", weather=self.sim.weather)
        for u, v in sorted(self.sim.blocked_edges):
            self.log.event(self.now, "blocked_path", "world", edge=[u, v])

        for party in self.parties:
            if party.trip is None:
                # Decided now, begins at trip.start_tick (any time today when cfg.trip_start == "random")
                party.trip = maybe_start_trip(day, self.cfg, self.trips_rng, self.G)
        # Establish scheduled rangers shift/s for the day
        self.shift_starts = {start: (end, rangers) for start, end, rangers in shift_windows(day, self.cfg)}

    def _start_scheduled_trips(self):
        """A planned trip begins at its start tick, while the agent body appears at the entry point and the episode starts."""
        for party in self.parties:
            trip = party.trip
            if trip is None or trip.phase != Phase.PLANNED or self.now < trip.start_tick:
                continue
            # Set initial trip phase
            trip.phase = Phase.INBOUND
            # Create physical instance of poaacher agent
            party.body = self._new_body(trip.entry)
            party.trip_reward, party.closest_gap, party.end_reason = 0, None, None
            # One poacher trip is one learning episode, with the same policy and seed persisting across trips for each ablation run.
            party.brain.start_episode(trip.entry, trip.target)
            self.log.event(self.now, "trip_start", party.name, node=trip.entry,
                           pos=party.body.get_current_pos(), target=trip.target)
            print(f"  {party.name} trip starts {clock_str(self.now, self.cfg)} at {trip.entry}, target {trip.target}")

    def _start_shifts(self):
        if self.now not in self.shift_starts:
            return
        # Recording current tick and shift end tick
        end, n_on_shift = self.shift_starts[self.now]
        for ranger in self.rangers[:n_on_shift]:
            # Deploys rangers to ranger station
            ranger.body = self._new_body(self.station)
            # Assigns shift end, starts episode with tick budget
            ranger.shift_end = end
            ranger.brain.start_episode(self.station, None, {"budget_ticks": end - self.now})
            route = getattr(ranger.brain, "route", None)
            # Log start of shift
            self.log.event(self.now, "patrol_start", ranger.name, node=self.station,
                           pos=ranger.body.get_current_pos(), route=route)
            if route:
                print(f"  {ranger.name} shift starts {clock_str(self.now, self.cfg)}: patrol {' -> '.join(route)}")
    # Conditions for shift end
    def scheduled_end_tick(self):
        for ranger in self.rangers:
            if ranger.shift_end is None or self.now < ranger.shift_end:
                continue

            body = ranger.body
            if body.destination_node is not None or body.is_busy(self.now):
                continue
            
            ranger.brain.end_episode("shift_over")
            ranger.shift_end = None

    #--------------------------------------------------------------------------#
    ###----------------------Movement and trip resolution--------------------###

    # Requesting move decisions from each participating agent brain
    def _choose_destinations(self):
        """Ask every idle agent's brain for its next move."""
        for party in self.parties:
            trip = party.trip
            if party.active and trip is not None and trip.phase != Phase.HUNTING:   # hunting = stay put and search
                self._ask_for_move(party.name, party.body, party.brain, current_goal(trip))
        for ranger in self.rangers:
            if ranger.shift_end is not None:
                self._ask_for_move(ranger.name, ranger.body, ranger.brain, None)

    # Agent brain decides next move given known information and current situation
    def _ask_for_move(self, name, body, brain, goal):
        if body.destination_node is not None or body.is_busy(self.now):
            return
        # If a random event forced a wrong turn
        if body.forced_next is not None:
            # Agent moves to forced next, event is logged, brain is made aware of event
            to = body.take_forced_next()
            self.log.event(self.now, "forced_move", name, node=body.current_node, pos=body.get_current_pos(), to=to)
            brain.observe({"kind": "forced_move", "node": body.current_node, "to": to, "pos": body.get_current_pos()})
            return
        situation = build_situation(body, self.now, self.sim, self.cfg, arriving=False)
        nxt = brain.next_move(body.current_node, goal, situation)
        if nxt is not None:
            body.destination_node = nxt

    # Check agent movement, trigger movement-based event rolls
    def _advance_movement(self):
        # Adding active agents currently capable of moving their position
        movers = [(p.body, "poacher", p.name, p.brain) for p in self.parties if p.active and p.body is not None]
        movers += [(u.body, "ranger", u.name, u.brain) for u in self.rangers]
        # For each mover
        for body, role, name, brain in movers:
            # Update movement given world conditions
            body.update_movement(self.now, self.cfg, self.sim.blocked_edges)
            self.log.track(self.now, name, body.get_current_pos())
            # Check if edge is blocked
            if body.just_blocked:
                # If true, record blocked edge and log event
                edge = (body.current_node, body.destination_node)
                self.log.event(self.now, "hit_block", name, node=body.current_node, pos=body.get_current_pos(),
                               edge=list(edge))
                # Make agent aware of this occurance
                brain.observe({"kind": "blocked", "node": body.current_node, "edge": edge,
                               "pos": body.get_current_pos()})
            # Check if agent just returned to previous node
            elif body.just_returned:
                self.log.event(self.now, "turned_back", name, node=body.current_node, pos=body.get_current_pos())
                brain.observe({"kind": "returned", "node": body.current_node, "pos": body.get_current_pos()})
            # Check if agent just arrived at a node and fire appropriate events
            elif body.just_arrived:
                self.log.event(self.now, "arrived", name, node=body.current_node, pos=body.get_current_pos())
                brain.observe({"kind": "arrived", "node": body.current_node, "pos": body.get_current_pos()})
                fire_events("on_arrival", role, name, body, self.now, self.sim, self.cfg, self.ctx, self.log)
            # Check if agent is in transit and fire appropriate events
            elif body.destination_node is not None and not body.is_busy(self.now):
                fire_events("while_moving", role, name, body, self.now, self.sim, self.cfg, self.ctx, self.log)

    def _resolve(self, party):
        """Apply the rules to one active poacher party for this tick: cawptures, retreats, phases, rewards."""
        body, trip, brain, now = party.body, party.trip, party.brain, self.now
        reward = REWARDS["tick"]

        capture_event, capture_reason = False, ""
        # Headlights are always on for now (5f makes them a choice): visible from further at night
        radius_factor = self.cfg.night_detection_factor_headlights if is_night(now, self.cfg) else 1.0
        for ranger in self.rangers:
            gap = float(np.linalg.norm(np.array(body.get_current_pos()) - np.array(ranger.body.get_current_pos())))
            party.closest_gap = gap if party.closest_gap is None else min(party.closest_gap, gap)
            if not capture_event:
                capture_event, capture_reason = self.sim.check_for_capture(body, ranger.body, radius_factor)
                if capture_event and ranger.shift_end is None:
                    # Off duty at the station: a passer-by is only spotted with a chance per minute in sight
                    p_tick = 1 - (1 - self.cfg.off_duty_spot_chance_per_min) ** self.cfg.tick_minutes
                    capture_event = self.station_rng.random() < p_tick
                    capture_reason = "Spotted from the station (off duty)" if capture_event else ""

        if body.outcome is not None:
            kind, why = body.outcome
            body.outcome = None # handled now, so it can't fire again next tick
            if kind == "capture":
                capture_event, capture_reason = True, why
            elif kind == "retreat" and trip.phase != Phase.OUTBOUND:
                # Abandon the trip: head for the nearest exit without a leopard
                trip.phase, trip.exit = Phase.OUTBOUND, nearest_exit(self.G, body.current_node)
                reward += REWARDS["retreat"]
                self.log.event(now, "retreat", party.name, node=body.current_node, pos=body.get_current_pos(),
                               reason=why, heading_for=trip.exit)
                print(f"    {clock_str(now, self.cfg)} {party.name} retreats ({why}), heading for {trip.exit}")

        if capture_event:
            brain.observe({"kind": "captured", "node": body.current_node, "pos": body.get_current_pos(),
                           "reason": capture_reason})
            trip.phase = Phase.CAUGHT
            trip.outcome = "caught_after_kill" if trip.leopard_found else "caught"
            reward += REWARDS["caught"]
            party.end_reason = capture_reason
            self.log.event(now, "capture", party.name, node=body.current_node, pos=body.get_current_pos(),
                           reason=capture_reason, after_kill=trip.leopard_found)
            print(f"    {clock_str(now, self.cfg)} {party.name} caught near {body.current_node}: {capture_reason}")
        else:
            was_hunting = trip.phase == Phase.HUNTING
            advance_trip(trip, now, self.cfg, self.leopard_rng, self.G, body)
            if was_hunting and trip.phase == Phase.OUTBOUND:
                if trip.leopard_found:
                    reward += REWARDS["leopard"]
                    self.log.event(now, "leopard_killed", party.name, node=trip.target, pos=body.get_current_pos())
                    print(f"    {clock_str(now, self.cfg)} LEOPARD KILLED at {trip.target}; {party.name} heading for {trip.exit}")
                else:
                    self.log.event(now, "gave_up_hunting", party.name, node=trip.target, pos=body.get_current_pos())
                    print(f"    {clock_str(now, self.cfg)} {party.name} gave up hunting, heading for {trip.exit}")
            elif trip.phase == Phase.HUNTING and trip.hunting_since == now:
                self.log.event(now, "hunting_started", party.name, node=trip.target, pos=body.get_current_pos())
                print(f"    {clock_str(now, self.cfg)} {party.name} reached {trip.target}, hunting")
            if trip.phase == Phase.ESCAPED:
                if trip.leopard_found:
                    reward += REWARDS["escaped"]
                party.end_reason = trip.outcome
                self.log.event(now, "escaped", party.name, node=body.current_node, pos=body.get_current_pos(),
                               with_leopard=trip.leopard_found)
                print(f"    {clock_str(now, self.cfg)} {party.name} escaped via {body.current_node}: {trip.outcome}")
            elif now - trip.start_tick >= self.cfg.max_trip_hours * 60 / self.cfg.tick_minutes:
                trip.phase, trip.outcome = Phase.TIMED_OUT, "timed_out"
                reward += REWARDS["timed_out"]
                party.end_reason = "timed_out"
                self.log.event(now, "timed_out", party.name, node=body.current_node, pos=body.get_current_pos())
                print(f"    {clock_str(now, self.cfg)} {party.name} ran out of time near {body.current_node}")

        done = trip.phase not in ACTIVE_PHASES
        party.trip_reward += reward
        brain.feedback(reward, done)

        if done:
            brain.end_episode(trip.outcome)
            self.trips.append({
                "trip": len(self.trips) + 1, "party": party.name, "entry": trip.entry, "target": trip.target,
                "start_tick": trip.start_tick, "end_tick": now,
                "hours": round((now - trip.start_tick) * self.cfg.tick_minutes / 60, 1),
                "outcome": trip.outcome, "end_reason": party.end_reason, "leopard_found": trip.leopard_found,
                "closest_approach_m": None if party.closest_gap is None else round(party.closest_gap),
                "poacher_success": trip.outcome == "escaped_with_leopard",
                "ranger_failed": trip.leopard_found,
                "reward": party.trip_reward, "epsilon": getattr(brain, "epsilon", None),
                "path": body.path_taken,
                **getattr(brain, "last_trip_stats", {}),   # a learning brain's own numbers (loss, mean Q...)
            })
            party.trip, party.body = None, None

    def _print_day_summary(self):
        counts = Counter(e["kind"] for e in self.log.events[self.day_first_event_index:]
                         if e["kind"] not in ("weather", "arrived", "patrol_start"))
        running = [f"{p.name} {p.trip.phase.value}" for p in self.parties if p.trip is not None]
        status = ", ".join(running) or "no trip running"
        print(f"Day {self.sim.current_day}: {status}. Events: {dict(counts) or 'none'}")
