# build_dataset.py
"""Rerun after new experiments to store results into tables for analysis

    analysis/trips_all.csv   one row per trip, training and evaluation from every run
    analysis/runs_all.csv    one row per run (arm x seed x experiment): summary numbers + LLM usage

Each row: `experiment`, `world`, `arm`, `seed` and `phase` ("train" / "eval")
"""
import glob
import json
import os

import pandas as pd

# What changed between each experiment
EXPERIMENTS = {
    "runs/ablation_v1_baseline":        "v1: plain DQN, greedy evaluation",
    "runs/ablation_v2_ddqn":            "v2: Double DQN, eval epsilon 0.05",
    "runs/ablation_move_road":          "v2b: v2 with real edge awareness",
    "runs/ablation_v3":                 "v3: no immediate return, greedy eval, RB v1 (lessons as inputs)",
    "runs/ablation_v4_realistic":       "v4: as v3, realistic world",
    "runs/ablation_v4_realistic_dusk":  "v4: as v3, realistic world + dusk shift",
    "runs/ablation_v5_rb2":             "v5: RB v2 (lessons as score bonus), realistic world",
}


def world_name(sim):
    """Which world a run used"""
    if sim.get("trip_start", "midnight") == "midnight":
        return "old"
    return "realistic_dusk" if len(sim["shifts"]) > 1 else "realistic"


def main():
    trips, runs = [], []
    for folder, description in EXPERIMENTS.items():
        experiment = os.path.basename(folder)
        results = pd.read_csv(f"{folder}/results.csv")
        for run_dir in sorted(glob.glob(f"{folder}/*_s*/")):
            config = json.load(open(f"{run_dir}/config.json"))
            agents = config["agents"]
            arm, seed = agents["arm"], agents["seed"]
            switches = agents.get("switches", {})
            labels = {"experiment": experiment, "description": description, "world": world_name(config["sim"]),
                      "arm": arm, "seed": seed, "bonus_strength": switches.get("bonus_strength",
                      2.0 if switches.get("lesson_use") == "bonus" else None)}

            for phase, name in [("train", "train_trips.csv"), ("eval", "trips.csv")]:
                path = f"{run_dir}/{name}"
                if os.path.exists(path):
                    t = pd.read_csv(path).drop(columns=["path"], errors="ignore")
                    t.insert(0, "phase", phase)
                    for i, (k, v) in enumerate(labels.items()):
                        t.insert(i, k, v)
                    t["trip_index"] = range(1, len(t) + 1)
                    t["caught"] = t["outcome"].str.startswith("caught")
                    t["timed_out"] = t["outcome"] == "timed_out"
                    t["start_hour"] = (t["start_tick"] % (24 * 60)) // 60
                    trips.append(t)

            row = results[(results.arm == arm) & (results.seed == seed)]
            runs.append({**labels, **(row.iloc[0].drop(["arm", "seed"]).to_dict() if len(row) else {})})

    os.makedirs("analysis", exist_ok=True)
    trips_all = pd.concat(trips, ignore_index=True)
    runs_all = pd.DataFrame(runs)
    trips_all.to_csv("analysis/trips_all.csv", index=False)
    runs_all.to_csv("analysis/runs_all.csv", index=False)
    print(f"trips_all.csv: {len(trips_all):,} trips | runs_all.csv: {len(runs_all)} runs")
    print(runs_all.groupby(["experiment", "world"]).arm.nunique().rename("arms").to_string())
    print("\ntrips_all columns:", list(trips_all.columns))


if __name__ == "__main__":
    main()
