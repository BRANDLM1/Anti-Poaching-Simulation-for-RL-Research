from simulation.config import SimConfig
from simulation.clock import sun_times, minute_of_day, day_of_year, is_night

cfg = SimConfig()
for name, day in [("Jun 21", 172), ("Sep 22", 265), ("Dec 21", 355)]:
    rise, set_ = sun_times(day, cfg)
    print(name, round(rise), round(set_))
# expect: Jun 21 341 1253 | Sep 22 439 1155 | Dec 21 533 1061
#         (05:41 / 20:53,     07:19 / 19:14,     08:53 / 17:41)

print(cfg.ticks_per_day, round(cfg.bike_m_per_tick, 1))   # 1440 666.7
print(minute_of_day(1500, cfg), day_of_year(1500, cfg))  # 60 153 (1 a.m. on day 2)
print(is_night(60, cfg), is_night(720, cfg))             # True False