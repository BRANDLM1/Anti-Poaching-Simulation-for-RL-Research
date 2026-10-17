"""Summary numbers for a run, and saving a run's outputs together.

Each run gets its own folder under runs/ holding:
- config.json   every setting used (so any result can be reproduced, and ablations compared)
- trips.csv     one row per poacher trip
- events.csv    everything that happened (from EventLog)
- tracks.csv    every agent's position each tick (for the UI replay)
- summary.json  the headline numbers below
"""
import json
import pathlib
from collections import Counter
from dataclasses import asdict

import pandas as pd


def _share(count, total):
    return round(count / total, 3) if total else None


def summarize(trips, log):
    """Headline numbers for a run, from the per-trip rows and the event log."""
    n = len(trips)
    caught = [t for t in trips if t["outcome"].startswith("caught")]
    hours = [t["hours"] for t in trips]
    gaps = [t["closest_approach_m"] for t in trips if t["closest_approach_m"] is not None]
    ranger_nodes = {e["node"] for e in log.events if e["who"].startswith("ranger") and e["kind"] == "arrived"}
    return {
        "trips": n,
        "poacher_success_rate": _share(sum(t["poacher_success"] for t in trips), n),   # leopard + escape
        "capture_rate": _share(len(caught), n),
        "timeout_rate": _share(sum(t["outcome"] == "timed_out" for t in trips), n),
        "leopards_killed": sum(t["leopard_found"] for t in trips),
        "ranger_fail_rate": _share(sum(t["ranger_failed"] for t in trips), n),          # any leopard found
        "retreats": sum(1 for e in log.events if e["kind"] == "retreat"),
        "mean_trip_hours": round(sum(hours) / n, 2) if n else None,
        "mean_closest_approach_m": round(sum(gaps) / len(gaps)) if gaps else None,
        "outcomes": dict(Counter(t["outcome"] for t in trips)),
        "capture_reasons": dict(Counter(t["end_reason"] for t in caught)),
        "random_events": dict(Counter(e["kind"] for e in log.events if e["detail"].get("random"))),
        "ranger_nodes_patrolled": len(ranger_nodes),
    }


def save_run(folder, cfg, config, trips, log):
    """Write every output of a run into one folder and return the summary."""
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "config.json", "w", encoding="utf-8") as f:
        json.dump({"sim": asdict(cfg), "agents": config}, f, indent=2, default=str)
    pd.DataFrame(trips).to_csv(folder / "trips.csv", index=False)
    log.save(folder)
    summary = summarize(trips, log)
    with open(folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
