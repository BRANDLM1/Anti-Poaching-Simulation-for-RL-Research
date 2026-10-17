from dataclasses import dataclass, field
from simulation.weather import WEATHER_STATES
import math


WHO = {"ranger", "poacher", "any"}
WHEN = {"on_arrival", "while_moving"}

# Condition keys and what the engine compares them against.
CONDITIONS = {
    "node_type",    # Node type (on_arrival only)
    "night",        # True / False
    "headlights",   # True / False (poacher's choice at night)
    "motor_bike",   # True / False
    "weather",      # one of WEATHER_STATES
}
SCALES = {
    "herder_activity", # Float value representing the activity level of a point of civilization. Impacts chance of capture on arrival and scaled by centrality score

}

# Effect keys and what they mean. Units go in the name so a number can't be misread.
EFFECTS = {
    "delay_min",    # agent stops for this many minutes
    "slow",         # speed multiplier, e.g. 0.5 = half speed ...
    "slow_min",     # ... for this many minutes
    "detour",       # "random_neighbor": agent ends up at a different nearby node
    "capture",      # reason string: ends the trip as a capture
    "retreat",      # reason string: poacher abandons the trip and heads for the nearest exit
}


@dataclass
class EventRule:
    name: str
    who: str    # "ranger" "poacher" or "any"
    when: str    # "on_arrival" / "while_moving"    
    rate: float # Probability of occurance
    scale_by: str | None = None  # Allows the probability of an event occurace to be scaled by a working simulation variable or graph node attr variable
    per_hour: bool = False #False refers to single event checks, True refers to rate per hour of exposure (while moving)
    conditions: dict = field(default_factory=dict)
    effect: dict = field(default_factory=dict)
    enabled: bool = True #Whether this event rule is currently active. Controlled for ablation

    def __post_init__(self):
        """Reject a bad row when the file loads, not hours into a run."""
        if self.who not in WHO:
            raise ValueError(f"{self.name}: unknown who={self.who!r}")
        if self.when not in WHEN:
            raise ValueError(f"{self.name}: unknown when={self.when!r}")
        if self.per_hour is False:
            if not 0.0 <= self.rate <= 1.0:
                    raise ValueError(f"{self.name}: rate {self.rate} must be between 0 and 1")
        if self.per_hour and self.when != "while_moving":
            raise ValueError(f"{self.name}: per_hour only makes sense for while_moving events")
        wanted = self.conditions.get("weather")
        if wanted is not None:
            names = wanted if isinstance(wanted, (set, list, tuple)) else [wanted] 
            for w in names:
                if w not in WEATHER_STATES:
                    raise ValueError(f"{self.name}: unknown weather condition {w!r}")
        for key in self.conditions:
            if key not in CONDITIONS:
                raise ValueError(f"{self.name}: unknown condition {key!r}")
        if self.scale_by is not None and self.scale_by not in SCALES:
            raise ValueError(f"{self.name}: unknown scale not in SCALES: {self.scale_by!r}")
        if self.scale_by is not None and self.when != "on_arrival": 
            raise ValueError(f"{self.name}: scale_by {self.scale_by!r} is only applicable for on_arrival events")
        if "node_type" in self.conditions and self.when != "on_arrival":
            raise ValueError(f"{self.name}: node_type can only be checked on arrival")
        for key in self.effect:
            if key not in EFFECTS:
                raise ValueError(f"{self.name}: unknown effect {key!r}")
        if not self.effect:
            raise ValueError(f"{self.name}: has no effect")


# Placeholder rates: tune these. Hourly rates are chances per hour of riding.
EVENTS = [
    EventRule(
        "tea_with_herders",
        "ranger",
        "on_arrival",
        0.3,
        conditions={"node_type": "civilization"},
        effect={"delay_min": 20}
    ),
    EventRule(
        "bathroom_stop",
        "any",
        "while_moving",
        0.10,
        per_hour=True,
        effect={"delay_min": 5}
    ),
    EventRule(
        "herder_capture",
        "poacher",
        "on_arrival",
        0.8,
        scale_by="herder_activity",
        conditions={"node_type": "civilization"},
        effect={"capture": "Caught by a local herder"}
    ),
    EventRule(
        "bike_breakdown",
        "any",
        "while_moving",
        0.02,
        per_hour=True,
        conditions={"motor_bike": True},
        effect={"delay_min": 30}
    ),
    EventRule(
        "bike_crash_in_dark",
        "poacher",
        "while_moving",
        0.05,
        per_hour=True,
        conditions={"night": True, "headlights": False, "motor_bike": True},
        effect={"delay_min": 60, "slow": 0.5, "slow_min": 120}
    ),
    EventRule(
        "bike_crash_in_dark_bad_weather",
        "poacher",
        "while_moving",
        0.20,
        per_hour=True,
        conditions={"night": True, "headlights": False, "motor_bike": True, "weather": {"Snow", "Rain"}},
        effect={"retreat": "Abandoned infiltration attempt"}
        ),
    EventRule(
        "bike_breakdown_bad_weather",
        "poacher",
        "while_moving",
        0.13,
        per_hour=True,
        conditions={"motor_bike": True, "weather": {"Snow", "Rain"}},
        effect={"delay_min": 30, "slow": 0.75, "slow_min": 60}
            ),
    EventRule(
        "lost_in_dark",
        "poacher",
        "on_arrival",
        0.15,
        conditions={"night": True, "headlights": False},
        effect={"detour": "random_neighbor"}
    ),
    EventRule(
        "weather_turns_back",
        "poacher",
        "while_moving",
        0.03,
        per_hour=True,
        conditions={"weather": {"Snow", "Fog"}},
        effect={"retreat": "Abandoned infiltration attempt"}
    ),
]

def condition_holds(wanted, actual):
    if isinstance(wanted, (set, list, tuple)):
        return actual in wanted
    return actual == wanted
    
def applies (rule, when, who, situation):
    if not rule.enabled:
        return False
    if rule.when != when:
        return False
    if rule.who != "any" and rule.who != who:
        return False
    for key, wanted in rule.conditions.items():
        if key not in situation:
            raise KeyError(f"Condition key '{key}' not found in situation")
        if not condition_holds(wanted, situation[key]):
            return False
    return True

def matching(events, when, who, situation):
    return [rule for rule in events if applies(rule, when, who, situation)]


def chance_each_roll(rule, tick_minutes):
    "Probability this rule fires upon rolling"
    if not rule.per_hour:
        return rule.rate
    else:
        return 1 - math.exp(-rule.rate * tick_minutes / 60)
   
def roll(rule, ctx, situation):
    "True if the rule fires on this roll."
    rate = chance_each_roll(rule, ctx["tick_minutes"])
    if rule.scale_by is not None and rule.scale_by in situation:
        rate *= situation[rule.scale_by]
    return ctx["rng"].random() < rate


def _ticks(minutes, tick_minutes):
    return math.ceil(minutes / tick_minutes)

def do_delay(agent, effect, now, ctx):
    delay_min = effect.get("delay_min", 0)
    delay = _ticks(delay_min, ctx["tick_minutes"])
    if delay_min > 0:
        agent.busy_until = max(agent.busy_until, now + delay)
    return agent.busy_until

def do_slow(agent, effect, now, ctx):
    slow = effect.get("slow", 1)
    slow_min = effect.get("slow_min", 0)
    if slow < 1:
        agent.speed_factor = slow
    if slow_min > 0:
        agent.slow_until = now + _ticks(slow_min, ctx["tick_minutes"])
    return agent.slow_until, agent.speed_factor

def do_detour(agent, effect, now, ctx):
    neighbors = list(ctx["graph"].neighbors(agent.current_node))
    if neighbors:
        agent.forced_next = ctx["rng"].choice(neighbors)
    return agent.forced_next

def do_capture(agent, effect, now, ctx):
    agent.outcome = ("capture", effect["capture"])
    return agent.outcome

def do_retreat(agent, effect, now, ctx):
    agent.outcome = ("retreat", effect["retreat"])
    return agent.outcome

HANDLERS = {
    "delay_min": do_delay,
    "slow": do_slow,
    "detour": do_detour,
    "capture": do_capture,
    "retreat": do_retreat,
}

def apply(rule, agent, now, ctx):
    "Record what happens for event log"
    for key in rule.effect:
        if key in HANDLERS:
            HANDLERS[key](agent, rule.effect, now, ctx)
    return {"tick": now, "event": rule.name, "who": rule.who, "effect": dict(rule.effect)}