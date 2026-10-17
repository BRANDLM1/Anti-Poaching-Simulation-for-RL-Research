"""The world: everything that isn't an agent's own decision.

The world owns the clock, weather, trips, shifts, the rules (captures, random events, rewards)
and the event log. Agents are a body (where they are) plus a brain (a ControlPanel that decides).
The world only ever talks to brains through the ControlPanel hooks, so any brain can be plugged in.

step() advances exactly one tick, which is what the UI will call once per frame.
"""
from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np

from agents.base import ControlPanel
from simulation.agent_instance import AgentInstance
from simulation.clock import is_night, minute_of_day, shift_windows
from simulation.event_log import EventLog
from simulation.random_events import EVENTS, matching, roll, apply
from simulation.reserve import Simulation
from simulation.trip import Phase, Trip, maybe_start_trip, current_goal, advance_trip, nearest_exit

# timed_out = caught, so stalling can never be a cheaper strategy than trying
REWARDS = {"tick": -1, "caught": -5000, "retreat": -1000, "leopard": 2000, "escaped": 3000, "timed_out": -5000}
ACTIVE_PHASES = (Phase.INBOUND, Phase.HUNTING, Phase.OUTBOUND)


def clock_str(tick, cfg):
    m = int(minute_of_day(tick, cfg))
    return f"{m // 60:02d}:{m % 60:02d}"


def build_situation(agent, tick, sim, cfg, arriving):
    situation = {
        "night": is_night(tick, cfg),
        "weather": sim.weather,
        "headlights": True,      # 5f makes this a poacher decision
        "motor_bike": True,      # later: foot or bike chosen at trip start
        "node_type": None,           # later: node type
        "herder_activity": None,      # later: civ point activity level
    }
    if arriving:
        situation["node_type"] = sim.graph.nodes[agent.current_node].get("Type")
        situation["herder_activity"] = sim.graph.nodes[agent.current_node].get("herder_activity")
    return situation


def fire_events(when, role, name, agent, tick, sim, cfg, ctx, log):
    """Roll the event table for one agent. `role` matches the table's `who`; `name` goes in the log."""
    situation = build_situation(agent, tick, sim, cfg, arriving=(when == "on_arrival"))
    for rule in matching(EVENTS, when, role, situation):
        if roll(rule, ctx, situation):
            record = apply(rule, agent, tick, ctx)
            log.event(tick, record["event"], name, node=agent.current_node, pos=agent.get_current_pos(),
                      random=True, effect=record["effect"])


@dataclass
class Party:
    """One poacher group: a brain that persists across trips, plus the current trip and body."""
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
    """One ranger: a brain and a body. shift_end is set while on shift, None while off duty."""
    name: str
    brain: ControlPanel
    body: AgentInstance
    shift_end: int | None = None


class World:
    def __init__(self, cfg, make_poacher_brain, make_ranger_brain, station="station"):
        """make_*_brain(world, index) -> a ControlPanel. Factories let each brain get its own rng."""
        self.cfg = cfg
        self.station = station
        self.G = nx.read_gml(cfg.graph_file)
        self.sim = Simulation(self.G, cfg.rng("weather"), cfg.rng("obstacles"))
        self.log = EventLog(cfg)
        self.ctx = {"rng": cfg.rng("events"), "graph": self.G, "tick_minutes": cfg.tick_minutes}
        self.trips_rng = cfg.rng("trips")
        self.leopard_rng = cfg.rng("leopards")
        self.station_rng = cfg.rng("station")     # off-duty spotting: its own stream, so it can be switched off

        self.parties = [Party(f"poacher{i + 1}", make_poacher_brain(self, i)) for i in range(cfg.poacher_parties)]
        n_rangers = max(spec.rangers for spec in cfg.shifts)
        self.rangers = [RangerUnit(f"ranger{i + 1}", make_ranger_brain(self, i), self._new_body(station))
                        for i in range(n_rangers)]

        self.now = 0
        self.trips = []                 # one finished-trip record per trip
        self.shift_starts = {}          # start tick -> (end tick, rangers on that shift)
        self.day_first_event = 0

    # ---------------- public ----------------

    @property
    def done(self):
        return self.now >= self.cfg.days * self.cfg.ticks_per_day

    def step(self):
        """Advance the world by one tick. Returns the events logged during it."""
        first = len(self.log.events)
        if self.now % self.cfg.ticks_per_day == 0:
            self._start_day(self.now // self.cfg.ticks_per_day)

        self._begin_trips()
        self._start_shifts()
        self._decide()
        self._move()
        for party in self.parties:
            if party.active:
                self._resolve(party)
        self._end_shifts()

        if (self.now + 1) % self.cfg.ticks_per_day == 0:
            self._print_day_summary()
        self.now += 1
        return self.log.events[first:]

    def run(self):
        while not self.done:
            self.step()

    # ---------------- day, shifts ----------------

    def _new_body(self, node):
        return AgentInstance(start_node=node, graph=self.G, speed_m_per_tick=self.cfg.bike_m_per_tick)

    def _all_brains(self):
        return [p.brain for p in self.parties] + [r.brain for r in self.rangers]

    def _start_day(self, day):
        if day > 0:
            for brain in self._all_brains():
                brain.end_day()
        self.sim.update_environment()
        self.day_first_event = len(self.log.events)
        self.log.event(self.now, "weather", "world", weather=self.sim.weather)
        for u, v in sorted(self.sim.blocked_edges):
            self.log.event(self.now, "blocked_path", "world", edge=[u, v])

        for party in self.parties:
            if party.trip is None:
                # Decided now, begins at trip.start_tick (any time today when cfg.trip_start == "random")
                party.trip = maybe_start_trip(day, self.cfg, self.trips_rng, self.G)

        self.shift_starts = {start: (end, rangers) for start, end, rangers in shift_windows(day, self.cfg)}

    def _begin_trips(self):
        """A planned trip begins at its start tick: the body appears at the entry and the episode starts."""
        for party in self.parties:
            trip = party.trip
            if trip is None or trip.phase != Phase.PLANNED or self.now < trip.start_tick:
                continue
            trip.phase = Phase.INBOUND
            party.body = self._new_body(trip.entry)
            party.trip_reward, party.closest_gap, party.end_reason = 0, None, None
            party.brain.start_episode(trip.entry, trip.target)
            self.log.event(self.now, "trip_start", party.name, node=trip.entry,
                           pos=party.body.get_current_pos(), target=trip.target)
            print(f"  {party.name} trip starts {clock_str(self.now, self.cfg)} at {trip.entry}, target {trip.target}")

    def _start_shifts(self):
        if self.now not in self.shift_starts:
            return
        end, n_on_shift = self.shift_starts[self.now]
        for unit in self.rangers[:n_on_shift]:
            unit.body = self._new_body(self.station)
            unit.shift_end = end
            unit.brain.start_episode(self.station, None, {"budget_ticks": end - self.now})
            route = getattr(unit.brain, "route", None)
            self.log.event(self.now, "patrol_start", unit.name, node=self.station,
                           pos=unit.body.get_current_pos(), route=route)
            if route:
                print(f"  {unit.name} shift starts {clock_str(self.now, self.cfg)}: patrol {' -> '.join(route)}")

    def _end_shifts(self):
        for unit in self.rangers:
            body = unit.body
            if unit.shift_end is not None and self.now >= unit.shift_end \
                    and body.destination_node is None and not body.is_busy(self.now):
                unit.brain.end_episode("shift_over")
                unit.shift_end = None

    # ---------------- each tick ----------------

    def _decide(self):
        """Ask every idle agent's brain for its next move."""
        for party in self.parties:
            trip = party.trip
            if party.active and trip is not None and trip.phase != Phase.HUNTING:   # hunting = stay put and search
                self._ask_for_move(party.name, party.body, party.brain, current_goal(trip))
        for unit in self.rangers:
            if unit.shift_end is not None:
                self._ask_for_move(unit.name, unit.body, unit.brain, None)

    def _ask_for_move(self, name, body, brain, goal):
        if body.destination_node is not None or body.is_busy(self.now):
            return
        if body.forced_next is not None:
            # A random event forced a wrong turn: the brain isn't asked, only told what happened
            to = body.take_forced_next()
            self.log.event(self.now, "forced_move", name, node=body.current_node, pos=body.get_current_pos(), to=to)
            brain.observe({"kind": "forced_move", "node": body.current_node, "to": to, "pos": body.get_current_pos()})
            return
        situation = build_situation(body, self.now, self.sim, self.cfg, arriving=False)
        nxt = brain.next_move(body.current_node, goal, situation)
        if nxt is not None:
            body.destination_node = nxt

    def _move(self):
        movers = [(p.body, "poacher", p.name, p.brain) for p in self.parties if p.active and p.body is not None]
        movers += [(u.body, "ranger", u.name, u.brain) for u in self.rangers]
        for body, role, name, brain in movers:
            body.update_movement(self.now, self.cfg, self.sim.blocked_edges)
            self.log.track(self.now, name, body.get_current_pos())
            if body.just_blocked:
                edge = (body.current_node, body.destination_node)
                self.log.event(self.now, "hit_block", name, node=body.current_node, pos=body.get_current_pos(),
                               edge=list(edge))
                brain.observe({"kind": "blocked", "node": body.current_node, "edge": edge,
                               "pos": body.get_current_pos()})
            elif body.just_returned:
                self.log.event(self.now, "turned_back", name, node=body.current_node, pos=body.get_current_pos())
                brain.observe({"kind": "returned", "node": body.current_node, "pos": body.get_current_pos()})
            elif body.just_arrived:
                self.log.event(self.now, "arrived", name, node=body.current_node, pos=body.get_current_pos())
                brain.observe({"kind": "arrived", "node": body.current_node, "pos": body.get_current_pos()})
                fire_events("on_arrival", role, name, body, self.now, self.sim, self.cfg, self.ctx, self.log)
            elif body.destination_node is not None and not body.is_busy(self.now):
                fire_events("while_moving", role, name, body, self.now, self.sim, self.cfg, self.ctx, self.log)

    def _resolve(self, party):
        """Apply the rules to one active poacher party for this tick: captures, retreats, phases, rewards."""
        body, trip, brain, now = party.body, party.trip, party.brain, self.now
        reward = REWARDS["tick"]

        capture_event, capture_reason = False, ""
        # Headlights are always on for now (5f makes them a choice): visible from further at night
        radius_factor = self.cfg.night_detection_factor_headlights if is_night(now, self.cfg) else 1.0
        for unit in self.rangers:
            gap = float(np.linalg.norm(np.array(body.get_current_pos()) - np.array(unit.body.get_current_pos())))
            party.closest_gap = gap if party.closest_gap is None else min(party.closest_gap, gap)
            if not capture_event:
                capture_event, capture_reason = self.sim.check_for_capture(body, unit.body, radius_factor)
                if capture_event and unit.shift_end is None:
                    # Off duty at the station: a passer-by is only spotted with a chance per minute in sight
                    p_tick = 1 - (1 - self.cfg.off_duty_spot_chance_per_min) ** self.cfg.tick_minutes
                    capture_event = self.station_rng.random() < p_tick
                    capture_reason = "Spotted from the station (off duty)" if capture_event else ""

        if body.outcome is not None:
            kind, why = body.outcome
            body.outcome = None                       # handled now, so it can't fire again next tick
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
        counts = Counter(e["kind"] for e in self.log.events[self.day_first_event:]
                         if e["kind"] not in ("weather", "arrived", "patrol_start"))
        running = [f"{p.name} {p.trip.phase.value}" for p in self.parties if p.trip is not None]
        status = ", ".join(running) or "no trip running"
        print(f"Day {self.sim.current_day}: {status}. Events: {dict(counts) or 'none'}")
