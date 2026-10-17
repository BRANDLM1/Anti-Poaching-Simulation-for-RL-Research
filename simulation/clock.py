import math

def day_index(tick, cfg):
    return math.floor(tick // cfg.ticks_per_day)

def minute_of_day(tick, cfg):
    return math.floor(tick % cfg.ticks_per_day * cfg.tick_minutes)

def day_of_year(tick, cfg):
    return (cfg.start_day_of_year - 1 + day_index(tick, cfg)) % 365 + 1

def sun_times(day, cfg):
    declination = math.radians(23.44) * math.sin(2 * math.pi * (284 + day) / 365)
    half_sun_deg_H = math.degrees(math.acos(-math.tan(math.radians(cfg.latitude)) * math.tan(declination)))
    #Equation of time EoT
    B = 2 * math.pi * (day - 81) / 365
    EoT = 9.87 * math.sin(2 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)
    
    solar_noon = cfg.solar_noon_minute - EoT
    sunrise = solar_noon - (half_sun_deg_H * 4)
    sunset = solar_noon + (half_sun_deg_H * 4)
    return sunrise, sunset

def shift_windows(day, cfg):

    doy = (cfg.start_day_of_year - 1 + day) % 365 + 1
    sunrise, sunset = sun_times(doy, cfg)
    
    patrol_start_stop_windows = []
    anchors = {"sunrise": sunrise, "sunset": sunset, "clock": 0}

    for shift in cfg.shifts:
        duration = shift.duration_minutes
        anchor_minute = anchors[shift.anchor]
        offset = shift.offset_minutes
        rangers = shift.rangers
        absolute = day * cfg.ticks_per_day

        start_tick = absolute + round((anchor_minute + offset) / cfg.tick_minutes)
        end_tick = start_tick + round((duration / cfg.tick_minutes))

        patrol_start_stop_windows.append((start_tick, end_tick, rangers))

    return patrol_start_stop_windows

def is_night(tick, cfg):
    day = day_of_year(tick, cfg)
    sunrise, sunset = sun_times(day, cfg)
    minute = minute_of_day(tick, cfg)
    return sunrise > minute or minute > sunset
