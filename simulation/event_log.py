"""Everything that happens in a run, in one place.

Two kinds of record:
- events: one row per thing that happened (a trip starting, an arrival, a capture, a breakdown...)
- tracks: every agent's position on every tick

Everything downstream reads these instead of reaching into the sim: the UI replays tracks + events,
the metrics summarise them, and later (parts 10/12) agents can be shown only the events their
intelligence level would realistically know about.
"""
import csv
import json
import pathlib

from simulation.clock import day_index, minute_of_day

EVENT_FIELDS = ["tick", "day", "time", "kind", "who", "node", "x", "y", "detail"]


class EventLog:
    def __init__(self, cfg):
        self.cfg = cfg
        self.events = []      # list of dicts, in the order they happened
        self.tracks = []      # (tick, who, x, y)
        self.keep_tracks = True   # False for long training runs: tracks are only needed for replays

    def event(self, tick, kind, who, node=None, pos=None, **detail):
        """Record one event. Anything extra (reason, route, effect...) goes in `detail`."""
        minute = int(minute_of_day(tick, self.cfg))
        x, y = (None, None) if pos is None else (round(float(pos[0]), 1), round(float(pos[1]), 1))
        record = {
            "tick": tick, "day": day_index(tick, self.cfg) + 1, "time": f"{minute // 60:02d}:{minute % 60:02d}",
            "kind": kind, "who": who, "node": node, "x": x, "y": y, "detail": detail,
        }
        self.events.append(record)
        return record

    def track(self, tick, who, pos):
        """Record where an agent is on this tick (for replay in the UI)."""
        if not self.keep_tracks:
            return
        self.tracks.append((tick, who, round(float(pos[0]), 1), round(float(pos[1]), 1)))

    def save(self, folder):
        folder = pathlib.Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "events.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
            writer.writeheader()
            for e in self.events:
                writer.writerow({**e, "detail": json.dumps(e["detail"], default=str)})
        with open(folder / "tracks.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["tick", "who", "x", "y"])
            writer.writerows(self.tracks)
