"""Reasoning bank: an LLM turns a trip debrief into a lesson the DQN can use.

A lesson is a weight from 0 to 1 on each concern in LESSON_CONCERNS (plus one sentence explaining it).
The poacher's network sees, for every possible move, weight x how much that move hits the concern.

Lessons are judged by results: a lesson active on a trip that ends in capture gets a strike, a successful
trip removes one, and STRIKES_TO_DROP strikes drop it. At most MAX_LESSONS are kept (oldest go first).

The debrief contains only what the poacher knows (its knowledge map and its own trip), never hidden facts.
"""
import json
import os
import time
from typing import Any

from agents.dqn_poacher import LESSON_CONCERNS

MODEL = "claude-haiku-5-5"
MAX_LESSONS = 10
STRIKES_TO_DROP = 3

CONCERN_MEANINGS = {
    "avoid_villages": "stay away from villages (herders may report or catch you)",
    "avoid_station_area": "stay away from the ranger station and places near it",
    "avoid_high_centrality": "avoid busy junction places that many routes pass through",
    "prefer_low_viewshed": "avoid exposed places that can be seen from far away",
    "avoid_capture_sites": "avoid places where you have been caught before",
    "avoid_threat_cells": "avoid map areas where you were spotted or caught recently",
}

SYSTEM_PROMPT = (
    "You advise a poacher agent in a wildlife-protection simulation used to test ranger patrol strategies. "
    "After each trip you read a debrief and update how much the agent should weigh each concern when "
    "choosing its next move. Weights run from 0 (ignore) to 1 (top priority). Change the current weights "
    "only as far as the evidence supports; a single trip is weak evidence. Concerns:\n"
    + "\n".join(f"- {name}: {meaning}" for name, meaning in CONCERN_MEANINGS.items())
)

LESSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{name: {"type": "number"} for name in LESSON_CONCERNS},
        "lesson": {"type": "string"},
    },
    "required": [*LESSON_CONCERNS, "lesson"],
    "additionalProperties": False,
}


class LLMLessonWriter:
    """Calls Claude to turn a debrief into {concern: weight} plus a one-sentence lesson. Tracks cost and time."""

    def __init__(self, model=MODEL, effort="low"):
        import anthropic                          # imported here so plain DQN runs don't need the SDK
        self.anthropic = anthropic
        # Keys not tied to a workspace must name one per request (ANTHROPIC_WORKSPACE_ID, "wrkspc_...")
        workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        headers = {"anthropic-workspace-id": workspace} if workspace else None
        # Reads ANTHROPIC_API_KEY; fails loudly if missing
        self.client = anthropic.Anthropic(max_retries=5, default_headers=headers)
        self.model = model
        self.effort = effort
        self.calls = self.failures = 0
        self.input_tokens = self.output_tokens = 0
        self.seconds = 0.0

    def __call__(self, debrief):
        start = time.time()
        self.calls += 1
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": debrief}],
                output_config=self._output_config(),
            )
        except (self.anthropic.APIConnectionError, self.anthropic.RateLimitError):
            self.failures += 1                    # transient (after the SDK's own retries): skip this lesson
            return None
        except self.anthropic.APIStatusError as e:
            if e.status_code < 500:
                raise                             # bad request / auth / model name: fail loudly
            self.failures += 1                    # server-side trouble: skip this lesson
            return None
        finally:
            self.seconds += time.time() - start

        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        if response.stop_reason in ("refusal", "max_tokens"):
            self.failures += 1
            return None
        data = json.loads(next(b.text for b in response.content if b.type == "text"))
        weights = {name: min(1.0, max(0.0, float(data[name]))) for name in LESSON_CONCERNS}
        return weights, data["lesson"]

    def _output_config(self) -> Any:
        return {"effort": self.effort, "format": {"type": "json_schema", "schema": LESSON_SCHEMA}}

    def usage(self):
        return {"llm_calls": self.calls, "llm_failures": self.failures, "llm_input_tokens": self.input_tokens,
                "llm_output_tokens": self.output_tokens, "llm_seconds": round(self.seconds, 1)}


class ReasoningBank:
    """The lessons currently trusted, their strikes, and a log of everything written and dropped."""

    def __init__(self, writer, max_lessons=MAX_LESSONS, strikes_to_drop=STRIKES_TO_DROP):
        self.writer = writer
        self.max_lessons = max_lessons
        self.strikes_to_drop = strikes_to_drop
        self.lessons = []       # {"id", "weights", "text", "strikes", "trip"}
        self.log = []           # one row per lesson written or dropped (for the write-up)
        self.next_id = 1

    def weights(self):
        """Current weight per concern: the average over trusted lessons (all 0 with no lessons)."""
        if not self.lessons:
            return {name: 0.0 for name in LESSON_CONCERNS}
        return {name: sum(l["weights"][name] for l in self.lessons) / len(self.lessons) for name in LESSON_CONCERNS}

    def judge(self, outcome, trip):
        """Every lesson active this trip is judged by its outcome: capture = strike, success = one strike off."""
        for lesson in self.lessons:
            if outcome.startswith("caught"):
                lesson["strikes"] += 1
            elif outcome == "escaped_with_leopard":
                lesson["strikes"] = max(0, lesson["strikes"] - 1)
        for lesson in [l for l in self.lessons if l["strikes"] >= self.strikes_to_drop]:
            self.lessons.remove(lesson)
            self.log.append({"trip": trip, "event": "dropped", "id": lesson["id"], "text": lesson["text"]})

    def write(self, debrief, trip):
        result = self.writer(debrief)
        if result is None:
            return
        weights, text = result
        lesson = {"id": self.next_id, "weights": weights, "text": text, "strikes": 0, "trip": trip}
        self.next_id += 1
        self.lessons.append(lesson)
        self.log.append({"trip": trip, "event": "written", "id": lesson["id"], "text": text, **weights})
        if len(self.lessons) > self.max_lessons:
            old = self.lessons.pop(0)
            self.log.append({"trip": trip, "event": "retired", "id": old["id"], "text": old["text"]})
