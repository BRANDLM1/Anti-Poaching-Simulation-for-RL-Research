# test_llm_lesson.py: ONE real call to Claude Haiku 5.5 (costs a fraction of a cent). Needs ANTHROPIC_API_KEY.
from agents.reasoning_bank import LLMLessonWriter

DEBRIEF = """Trip outcome: caught.
Route (8 moves): border15 (entry point) -> cam2 (camera) -> border4 (entry point) -> civ2 (village) -> peak2 (mountain)
Caught near: civ2 (village).
Conditions: mostly Rain; 2 of 8 moves at night.
Places known so far: entry point 21, village 6, mountain 9, camera 4, water 1.
Places where I have been caught before: civ2 x2, civ3 x1.
Current weights: {"avoid_villages": 0.3, "avoid_station_area": 0.2, "avoid_high_centrality": 0.2, "prefer_low_viewshed": 0.2, "avoid_capture_sites": 0.3, "avoid_threat_cells": 0.2}
Return updated weights for every concern and a one-sentence lesson."""

writer = LLMLessonWriter()
result = writer(DEBRIEF)
print("result:", result)
usage = writer.usage()
print("usage:", usage)

# Haiku 5.5: $0.10 per million input tokens, $0.50 per million output tokens
per_call = usage["llm_input_tokens"] * 0.10 / 1e6 + usage["llm_output_tokens"] * 0.50 / 1e6
print(f"cost of this call ${per_call:.5f}; a full run (~8,500 calls) ≈ ${per_call * 8500:.2f}")
