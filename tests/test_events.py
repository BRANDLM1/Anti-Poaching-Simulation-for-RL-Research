from simulation.random_events import EVENTS, matching

arriving = {"node_type": "civilization", "night": True, "headlights": False,
            "motor_bike": True, "weather": "Clear"}
print("poacher arrives:", [r.name for r in matching(EVENTS, "on_arrival", "poacher", arriving)])
print("ranger arrives: ", [r.name for r in matching(EVENTS, "on_arrival", "ranger", arriving)])

riding = {"night": True, "headlights": False, "motor_bike": True, "weather": "Snow"}
print("poacher riding: ", [r.name for r in matching(EVENTS, "while_moving", "poacher", riding)])