# run_ablation.py
"""Ablation study to assess which parts of the poacher's intelligence actually helps

Each arm changes one thing:
    wanderer        random moves                       
    shortest_path   scripted fastest route, no learning
    dqn_move_only   DQN only aware of move geometry
    dqn_knowledge   aware of move geometry and what it learns about each place
    dqn_full        dqn_knowledge with threat grid memory                 
    dqn_full_road   dqn_full and knows the graph network
    dqn_move_road   aware of move geometry and graph network
    rb_every_trip   move-only + LLM lessons after every trip           (reasoning bank: faster learning?)
    rb_outcomes     move-only + LLM lessons after captures/successes   (fewer, more pointed lessons)
    rb2_every_trip  reasoning bank v2: lessons lower move scores directly when choosing
    rb2_outcomes    reasoning bank v2 + LLM lessons after captures/successes

The rb* arms call Claude (needs ANTHROPIC_API_KEY)
Skipped without key

Learning arms train first then equal evaluations:
Greedy moves (no immediate return to the node just left), no learning on days it never trained on. 

Outputs in runs/ablation/:
    <arm>_s<seed>/       the evaluation run (trips, events, tracks, summary) and train_trips.csv
    results.csv          one row per arm x seed
    results_summary.csv  mean and spread per arm
"""
import contextlib
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import networkx as nx
import pandas as pd

from agents.base import Wanderer
from agents.dqn_poacher import DQNAgent
from agents.dqnrb_poacher import DQNRBAgent
from agents.shortest_path_poacher import ShortestPathPoacher
from agents.planner_ranger import PlannerRanger
from simulation.config import ShiftSpec, SimConfig
from simulation.metrics import save_run, summarize
from simulation.world import World

ARMS = {
    "wanderer":      (Wanderer, {}),
    "shortest_path": (ShortestPathPoacher, {}),
    "dqn_move_only": (DQNAgent, {"use_knowledge": False, "use_threat": False}),
    "dqn_knowledge": (DQNAgent, {"use_knowledge": True, "use_threat": False}),
    "dqn_full":      (DQNAgent, {"use_knowledge": True, "use_threat": True}),
    "dqn_full_road": (DQNAgent, {"use_knowledge": True, "use_threat": True, "use_road_distance": True}),
    "dqn_move_road": (DQNAgent, {"use_knowledge": False, "use_threat": False, "use_road_distance": True}),

    "rb_every_trip": (DQNRBAgent, {"use_knowledge": False, "use_threat": False, "lesson_trigger": "every_trip"}),
    "rb_outcomes":   (DQNRBAgent, {"use_knowledge": False, "use_threat": False, "lesson_trigger": "outcomes"}),
    # Reasoning bank v2 lessons act directly on choices (score bonus), and lessons judged only after exploration (epsilon < 0.3)
    "rb2_every_trip": (DQNRBAgent, {"use_knowledge": False, "use_threat": False, "lesson_trigger": "every_trip",
                                    "lesson_use": "bonus", "judge_below_epsilon": 0.3}),
    "rb2_outcomes":   (DQNRBAgent, {"use_knowledge": False, "use_threat": False, "lesson_trigger": "outcomes",
                                    "lesson_use": "bonus", "judge_below_epsilon": 0.3}),
}

SETTINGS = {
    "train_days": 1000,         # one trip per day while training
    "eval_days": 200,
    "trip_chance": 1.0,
    "seeds": [1, 2, 3, 4, 5],
    "eval_seed_offset": 10_000, # evaluation days use different weather/events/trips than training
    "dqn": {"epsilon_decay": 0.995, "min_epsilon": 0.05, "update_target_every": 10},
    "ranger": {"initial_temp": 100, "cooling_rate": 0.995, "max_iterations": 2000},
    "out": "runs/ablation",
    "world": {},                # SimConfig overrides shared by training and evaluation (see WORLDS)
}

# World variants for robustness checks. 
# v3 and earlier used the old world (trips at midnight, certain off-duty spotting, same range day and night
# pass {"trip_start": "midnight", "off_duty_spot_chance_per_min": 1.0, "night_detection_factor_headlights": 1.0} to reproduce it.
DAWN = ShiftSpec(anchor="sunrise")
DUSK = ShiftSpec(anchor="sunset", offset_minutes=-120)      # 4 h centred on sunset
WORLDS = {
    "realistic": {},                                         # random starts, off-duty chance, headlights at night
    "realistic_dusk": {"shifts": [DAWN, DUSK]},              # + a second shift at dusk
}


def play(cfg, brain, settings, keep_tracks):
    """Create a world with a poacher brain. Ranger rebuilt per play"""
    def make_poacher(world, i):
        return brain

    def make_ranger(world, i):
        return PlannerRanger(world.G, world.cfg.rng(f"ranger{i + 1}_brain"), world.sim, world.cfg,
                             **settings["ranger"])

    world = World(cfg, make_poacher, make_ranger)
    world.log.keep_tracks = keep_tracks
    world.run()
    return world


def run_one(arm, seed, settings):
    """Train (learning arms only), then evaluate. Returns one results row."""
    import torch
    torch.set_num_threads(1) # one core per run; the runs themselves are in parallel

    brain_cls, switches = ARMS[arm]
    kwargs = {**settings["dqn"], **switches} if issubclass(brain_cls, DQNAgent) else {}
    graph = nx.read_gml(SimConfig().graph_file)
    brain = brain_cls(graph, random.Random(f"{seed}:{arm}:brain"), **kwargs)
    folder = f"{settings['out']}/{arm}_s{seed}"
    start = time.time()

    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        learns = hasattr(brain, "training")
        if learns:
            train_cfg = SimConfig(days=settings["train_days"], trip_start_chance_per_day=settings["trip_chance"],
                                  seed=seed, **settings["world"])
            train_world = play(train_cfg, brain, settings, keep_tracks=False)
            os.makedirs(folder, exist_ok=True)
            pd.DataFrame(train_world.trips).to_csv(f"{folder}/train_trips.csv", index=False)
            if isinstance(brain, DQNRBAgent):
                pd.DataFrame(brain.bank.log).to_csv(f"{folder}/lessons.csv", index=False)
            brain.training = False

        eval_cfg = SimConfig(days=settings["eval_days"], trip_start_chance_per_day=settings["trip_chance"],
                             seed=seed + settings["eval_seed_offset"], **settings["world"])
        eval_world = play(eval_cfg, brain, settings, keep_tracks=True)
        config = {"arm": arm, "seed": seed, "brain": brain_cls.__name__, "switches": switches,
                  "settings": {k: v for k, v in settings.items() if k != "seeds"}}
        summary = save_run(folder, eval_cfg, config, eval_world.trips, eval_world.log)

    minutes = (time.time() - start) / 60
    print(f"  done {arm} seed {seed} in {minutes:.1f} min: success {summary['poacher_success_rate']}, "
          f"caught {summary['capture_rate']}", flush=True)
    keep = ["trips", "poacher_success_rate", "capture_rate", "timeout_rate", "leopards_killed", "ranger_fail_rate",
            "mean_trip_hours", "mean_closest_approach_m"]
    llm = brain.bank.writer.usage() if isinstance(brain, DQNRBAgent) else {}
    return {"arm": arm, "seed": seed, **{k: summary[k] for k in keep}, "minutes": round(minutes, 1), **llm}


def main(settings, arms=None, workers=None):
    arms = arms or list(ARMS)
    llm_arms = [a for a in arms if ARMS[a][0] is DQNRBAgent]
    if llm_arms and not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"ANTHROPIC_API_KEY is not set, skipping the reasoning-bank arms: {llm_arms}")
        arms = [a for a in arms if a not in llm_arms]
    jobs = [(arm, seed) for arm in arms for seed in settings["seeds"]]
    workers = workers or min(len(jobs), max(1, (os.cpu_count() or 2) - 2))   # leave 2 cores for you
    print(f"{len(jobs)} runs on {workers} processes: {arms} x seeds {settings['seeds']}")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run_one, *zip(*jobs), [settings] * len(jobs)))

    results = pd.DataFrame(rows)
    os.makedirs(settings["out"], exist_ok=True)
    results.to_csv(f"{settings['out']}/results.csv", index=False)
    metrics = ["poacher_success_rate", "capture_rate", "timeout_rate", "ranger_fail_rate", "mean_trip_hours"]
    summary = results.groupby("arm", sort=False)[metrics].agg(["mean", "std"]).round(3)
    summary.to_csv(f"{settings['out']}/results_summary.csv")
    print(summary.to_string())
    return results


if __name__ == "__main__":
    # Smoke tests settings and functionality before committing
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        smoke = {**SETTINGS, "train_days": 6, "eval_days": 3, "seeds": [1], "out": "runs/ablation_smoke"}
        main(smoke)
    else:
        main(SETTINGS)
