from dataclasses import dataclass, field
import math
import random

@dataclass
class ShiftSpec:
    anchor: str = "sunrise"  # "sunrise", "sunset", or "clock" (absolute time in minutes)
    offset_minutes: float = 0  # minutes from anchor" pos = after, neg = before
    duration_minutes: float = 240  # length of shift in minutes
    rangers: int = 1  # number of rangers on shift

@dataclass
class SimConfig:
    tick_minutes: float = 1
    start_day_of_year: int = 152 # june 1st
    #location coordinates
    latitude: float = 43.2
    longitude: float = 100.7
    timezone_meridian: float = 120 # UTC+8

    #poacher hunting 
    leopard_find_chance_per_hour: float = 0.1
    max_hunt_minutes: float = 360
    max_trip_hours: float = 48     # a trip still going after this ends as "timed_out" (no endless wandering)
    trip_start: str = "random"     # "random" = any minute of the day; "midnight" = the old fixed start

    #detection
    off_duty_spot_chance_per_min: float = 0.1   # off-duty station: chance per minute in sight that a passer-by is spotted
    night_detection_factor_headlights: float = 1.5  # at night, headlights are visible from further (part 5 design)
    night_detection_factor_dark: float = 0.5        # at night without headlights (5f makes headlights a choice)

    #travel
    bike_speed_kmh: float = 40
    foot_speed_kmh: float = 4
    visit_minutes: dict = field(default_factory=lambda: {"civilization":15, "mountain":35, "water":10, "camera":10, "rangerh": 5, "poacherh": 5})
    block_found_at: float = 0.5   # fraction along a blocked edge where the block is found (then drive back)

    #ranger patrol time
    shifts: list = field(default_factory = lambda:[ShiftSpec()])
    #poacher attempt
    trip_start_chance_per_day: float = 0.2
    poacher_parties: int = 1   # independent poacher groups, each with its own trips
    #random seed set
    seed: int = 1

    #run
    days: int = 10
    graph_file: str = "data/sparse_weighted_graph.gml"

    @property
    def ticks_per_day(self):
        return int(24 * 60 / self.tick_minutes)

    @property
    def bike_m_per_tick(self):
        return ((self.bike_speed_kmh * 1000 / 60) * self.tick_minutes)

    @property
    def foot_m_per_tick(self):
        return ((self.foot_speed_kmh * 1000 / 60) * self.tick_minutes)

    @property
    def solar_noon_minute(self):
        return (720 +(self.timezone_meridian - self.longitude) * 4)

    def rng(self, stream):
        """A separate, repeatable random generator per stream ("weather", "events", ...).

        Seeding with the stream's name means each stream's sequence depends only on the seed,
        so switching one feature off (an ablation) can't change another stream's draws.
        """
        return random.Random(f"{self.seed}:{stream}")
