import random
from simulation.random_events import EVENTS, chance_each_roll, roll


def roll_once(rule, rng, tick_minutes):
    return roll(rule, {"rng": rng, "tick_minutes": tick_minutes}, {})

bathroom = next(r for r in EVENTS if r.name == "bathroom_stop")    # 10% per hour
tea = next(r for r in EVENTS if r.name == "tea_with_herders")       # 30% per arrival

# 1. The maths, exact
print("bathroom per 1-min tick:", round(chance_each_roll(bathroom, 1), 5))   # expect 0.00167 = 1 - exp(-0.1/60) (Poisson)
print("tea per arrival:        ", chance_each_roll(tea, 1))                  # expect 0.3

# 2. Same seed, same results
r1, r2 = random.Random(7), random.Random(7)
print("reproducible:", [roll_once(tea, r1, 1) for _ in range(10)] == [roll_once(tea, r2, 1) for _ in range(10)])

# 3. Over many hours, about 10% of hours should contain a bathroom stop, whatever the tick length
for tick in (0.5, 1, 5):
    rng = random.Random(0)
    checks_per_hour = int(60 / tick)
    hours_with_stop = sum(
        any(roll_once(bathroom, rng, tick) for _ in range(checks_per_hour))
        for _ in range(20000)
    )
    print(f"tick {tick} min: {hours_with_stop / 20000:.3f} of hours had a stop")   # each ≈ 0.095 = 1 - exp(-0.1)