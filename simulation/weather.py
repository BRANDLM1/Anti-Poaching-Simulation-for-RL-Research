CLEAR, RAIN, FOG, SNOW = "Clear", "Rain", "Fog", "Snow"
WEATHER_STATES = [CLEAR, RAIN, FOG, SNOW]          # DQN's one-hot order

TRANSITIONS = {
    CLEAR: {CLEAR: 0.75, RAIN: 0.10, FOG: 0.10, SNOW: 0.05},
    RAIN:  {CLEAR: 0.40, RAIN: 0.50, FOG: 0.10, SNOW: 0.00},
    FOG:   {CLEAR: 0.30, RAIN: 0.30, FOG: 0.40, SNOW: 0.00},
    SNOW:  {CLEAR: 0.20, RAIN: 0.10, FOG: 0.00, SNOW: 0.70},
}
DETECTION_RADIUS_M = {CLEAR: 1400, RAIN: 1000, FOG: 400, SNOW: 700}
POACHER_COST_MULT  = {CLEAR: 1.0,  RAIN: 1.2,  FOG: 1.3, SNOW: 1.5}
DISPLAY = {CLEAR: "Sunny!", RAIN: "Rainy :(", FOG: "Foggy", SNOW: "SNOW STORM"}   # keep the fun labels for printing

def next_weather(current, rng):
    row = TRANSITIONS[current]
    return rng.choices(list(row), weights=list(row.values()))[0]