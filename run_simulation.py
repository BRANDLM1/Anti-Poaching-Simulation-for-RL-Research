# run_simulation.py
# Builds a World, plugs in the brains, runs it, and saves the outputs.
# All the rules live in simulation/world.py; all the decisions live in the brains (agents/).
from agents.dqn_poacher import DQNAgent
from agents.planner_ranger import PlannerRanger
from simulation.config import SimConfig
from simulation.metrics import save_run
from simulation.world import World


def run_sim(cfg, config, poacher_brain=DQNAgent, ranger_brain=PlannerRanger):
    """Run one simulation. Swap `poacher_brain` / `ranger_brain` to compare intelligence levels."""

    def make_poacher(world, i):
        return poacher_brain(world.G, world.cfg.rng(f"poacher{i + 1}_brain"), **config.get("poacher_brain", {}))

    def make_ranger(world, i):
        return ranger_brain(world.G, world.cfg.rng(f"ranger{i + 1}_brain"), world.sim, world.cfg,
                            **config.get("ranger_brain", {}))

    world = World(cfg, make_poacher, make_ranger, station=config.get("station", "station"))
    world.run()

    folder = f"runs/{config['run_name']}"
    summary = save_run(folder, cfg, config, world.trips, world.log)
    print(f"Simulation complete. {len(world.trips)} trips. Outputs saved to {folder}/")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


# --- Run the script ---
if __name__ == "__main__":

    cfg = SimConfig(days=20, trip_start_chance_per_day=0.5)

    CONFIG = {
        "run_name": "dqn_vs_sa",
        "station": "station",
        "poacher_brain": {"epsilon_decay": 0.995, "min_epsilon": 0.01, "update_target_every": 10},
        "ranger_brain": {"initial_temp": 100, "cooling_rate": 0.995, "max_iterations": 2000},
    }
    run_sim(cfg, CONFIG)
