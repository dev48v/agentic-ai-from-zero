"""ReAct Planning Agent — reason and act in a loop, with guardrails.

Four ideas, hand-rolled (no framework):

  1. observe -> think -> act -> reflect loop
        Each iteration the model emits a Thought + an Action (a tool call). The
        tool runs and returns an Observation. A distinct self-critic step then
        Reflects on whether that Observation moved us toward the goal. The
        reflection is fed back into the next Thought, so the agent can change
        course. (`ReActAgent.run` — the `for step in range(max_steps)` loop.)

  2. max iteration limits
        A hard cap (`max_steps`, default 6) so the loop can NEVER run forever. On
        hitting the cap we synthesise a best-so-far answer from the transcript and
        return status MAX_STEPS with a clear "stopped: max steps" reason.

  3. self-critic
        `_reflect` is a separate LLM call that grades the last action: on_track?
        what did we learn? what next? Its `next_hint` is written into the
        scratchpad so a wrong turn (a tool error, an irrelevant result) gets
        corrected on the following Thought.

  4. graceful degradation
        Tools never crash the loop (see tools.Tool.run) — a failure becomes an
        ERROR observation. If the goal turns out to be unreachable the model can
        emit the `give_up` action to return a PARTIAL answer + the reason
        (status DEGRADED), and the max-step cap is the backstop. Either way we
        return an honest partial result instead of hallucinating success.

The Thought/Action/Action-Input contract is a strict JSON object per step, parsed
robustly (fenced-JSON tolerant). temperature=0 for reproducibility.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from common.client import DEFAULT_MODEL, get_client
from tools import Tool, default_tools

logger = logging.getLogger("react_planning_agent")

# Terminal statuses.
SOLVED = "SOLVED"          # model reached a final answer
DEGRADED = "DEGRADED"      # model gave up cleanly with a partial answer + reason
MAX_STEPS = "MAX_STEPS"    # hit the iteration cap (backstop) -> best-so-far

# Actions the model may emit.
_FINISH = "final"          # -> SOLVED, action_input is the answer
_GIVE_UP = "give_up"       # -> DEGRADED, action_input is a partial answer + reason


@dataclass
class Step:
    n: int
    thought: str
    action: str
    action_input: str
    observation: str
    observation_ok: bool
    reflection: str = ""
    on_track: bool | None = None
    next_hint: str = ""


@dataclass
class ReActResult:
    goal: str
    status: str
    final_answer: str
    reason: str
    steps: list[Step] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    max_steps: int = 6
    elapsed_s: float = 0.0

    @property
    def iterations(self) -> int:
        return len(self.steps)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model reply, tolerating ```json fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    return brace.group(0) if brace else text.strip()


class ReActAgent:
    def __init__(
        self,
        tools: dict[str, Tool] | None = None,
        model: str = DEFAULT_MODEL,
        max_steps: int = 6,
    ) -> None:
        self.tools = tools if tools is not None else default_tools()
        self.model = model
        self.max_steps = max_steps
        self.client = get_client()

    # ------------------------------------------------------------------ #
    # Prompts
    # ------------------------------------------------------------------ #
    def _tool_menu(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())

    def _system_prompt(self) -> str:
        return (
            "You are a ReAct planning agent. You solve a goal by reasoning in a "
            "loop: think, then take ONE action, observe the result, and repeat.\n\n"
            "Available actions:\n"
            f"{self._tool_menu()}\n"
            f"- {_FINISH}: you have enough to answer. action_input = the final answer.\n"
            f"- {_GIVE_UP}: the goal cannot be met (e.g. a tool keeps failing). "
            "action_input = the best PARTIAL answer you can give PLUS the reason it is incomplete.\n\n"
            "Rules:\n"
            "- Take exactly ONE action per step. Prefer tools over guessing; the "
            "knowledge_lookup facts are about a FICTIONAL company, so you cannot know "
            "them without calling the tool.\n"
            "- Look at the Observations and Reflections already in the transcript. "
            "Do NOT repeat an action that already failed the same way — change course.\n"
            "- Tool Observations are AUTHORITATIVE and correct. If the transcript "
            "already contains the Observation(s) that fully answer the goal (e.g. a "
            f"completed calculation), your action MUST be '{_FINISH}' — do NOT re-look-up "
            "or re-compute facts you already have.\n"
            "- When a tool returns an ERROR, adapt: try a different key/tool, or if it "
            f"is hopeless use {_GIVE_UP}. Never invent a tool result.\n\n"
            "Reply with ONLY a JSON object, no prose around it:\n"
            '{"thought": "<your reasoning for this step>", '
            '"action": "<one action name from the list above>", '
            '"action_input": "<the tool input, or the final/partial answer>"}\n\n'
            "Example of finishing: if the goal is 'what is 6 desks times 4 chairs?' and "
            "the transcript already shows 'Observation: 6 * 4 = 24', do NOT compute "
            "again — reply exactly:\n"
            '{"thought": "The calculator already returned 24, which answers the goal.", '
            '"action": "final", "action_input": "24 chairs in total."}'
        )

    # ------------------------------------------------------------------ #
    # observe -> think -> act (one LLM call that yields Thought + Action)
    # ------------------------------------------------------------------ #
    def _think_act(self, goal: str, transcript: str) -> tuple[str, str, str]:
        user = (
            f"Goal: {goal}\n\n"
            f"Transcript so far:\n{transcript or '(nothing yet — this is step 1)'}\n\n"
            "Decide the next single step now."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        try:
            data = json.loads(_extract_json(raw))
            thought = str(data.get("thought", "")).strip()
            action = str(data.get("action", "")).strip()
            action_input = str(data.get("action_input", "")).strip()
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            # Parsing degradation: don't crash — treat as a malformed step the
            # loop can recover from (the reflection will nudge a retry).
            logger.warning("think/act reply was not clean JSON (%s)", exc)
            return "", "", ""
        return thought, action, action_input

    # ------------------------------------------------------------------ #
    # reflect (the self-critic: a separate LLM call grading the last action)
    # ------------------------------------------------------------------ #
    def _reflect(
        self, goal: str, thought: str, action: str, action_input: str, observation: str
    ) -> tuple[str, bool, str]:
        system = (
            "You are the self-critic for a ReAct agent. Given the goal and the agent's "
            "most recent Thought/Action/Observation, judge whether that step moved the "
            "agent CLOSER to the goal, and say what to do next. Be concise and honest.\n"
            "IMPORTANT: tool Observations are GROUND TRUTH — the calculator's arithmetic "
            "and the knowledge base's facts are authoritative and correct. NEVER claim a "
            "tool's output is wrong or needs re-verifying. Judge only whether the agent "
            "is gathering the right information and making progress. If the transcript "
            "now contains every fact and any computation needed to answer the goal, set "
            "on_track=true and next_hint='ready to give the final answer'. Do NOT invent "
            "extra busywork (re-checking numbers, unit conversions) once the answer is "
            "present. Only set on_track=false when a step failed (an ERROR) or was "
            "genuinely off-topic.\n"
            'Reply with ONLY a JSON object: {"reflection": "<1-2 sentence critique>", '
            '"on_track": true|false, "next_hint": "<what the agent should do next; '
            "say 'ready to give the final answer' if there is now enough information>\"}"
        )
        user = (
            f"Goal: {goal}\n\n"
            f"Last Thought: {thought}\n"
            f"Last Action: {action}\n"
            f"Last Action Input: {action_input}\n"
            f"Observation: {observation}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
            )
            data = json.loads(_extract_json(resp.choices[0].message.content or ""))
            reflection = str(data.get("reflection", "")).strip()
            on_track = bool(data.get("on_track", False))
            next_hint = str(data.get("next_hint", "")).strip()
            return reflection, on_track, next_hint
        except Exception as exc:  # noqa: BLE001 — critic must never break the loop
            logger.warning("reflect step failed (%s); continuing without a critique", exc)
            return "(self-critic unavailable this step)", True, ""

    # ------------------------------------------------------------------ #
    # graceful degradation: synthesise a best-so-far answer from the transcript
    # ------------------------------------------------------------------ #
    def _best_so_far(self, goal: str, transcript: str, reason: str) -> str:
        system = (
            "The agent ran out of steps before finishing. Using ONLY the transcript "
            "below, give the best PARTIAL answer you honestly can and state plainly "
            "what is still missing. Do NOT invent facts. Start your reply with "
            "'[PARTIAL] '."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Goal: {goal}\n\nTranscript:\n{transcript}"},
                ],
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip() or f"[PARTIAL] {reason}"
        except Exception:  # noqa: BLE001
            return f"[PARTIAL] Could not complete the goal. {reason}"

    # ------------------------------------------------------------------ #
    # scratchpad rendering (what the model sees as the running transcript)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render(steps: list[Step]) -> str:
        blocks: list[str] = []
        for s in steps:
            block = (
                f"Step {s.n}:\n"
                f"Thought: {s.thought}\n"
                f"Action: {s.action}\n"
                f"Action Input: {s.action_input}\n"
                f"Observation: {s.observation}"
            )
            if s.reflection:
                block += (
                    f"\nReflection: {s.reflection} "
                    f"(on_track={s.on_track}; next: {s.next_hint})"
                )
            blocks.append(block)
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ #
    # The loop
    # ------------------------------------------------------------------ #
    def run(self, goal: str) -> ReActResult:
        started = time.time()
        steps: list[Step] = []
        logger.info("GOAL: %s", goal)

        for n in range(1, self.max_steps + 1):
            transcript = self._render(steps)
            thought, action, action_input = self._think_act(goal, transcript)
            logger.info("step %d | thought: %s", n, thought or "(none)")
            logger.info("step %d | action: %s(%r)", n, action or "(none)", action_input)

            # --- terminal actions -------------------------------------- #
            if action == _FINISH:
                return ReActResult(
                    goal=goal, status=SOLVED, final_answer=action_input,
                    reason=f"model returned a final answer at step {n}",
                    steps=steps, model=self.model, max_steps=self.max_steps,
                    elapsed_s=round(time.time() - started, 2),
                )
            if action == _GIVE_UP:
                logger.info("step %d | model chose to degrade gracefully", n)
                return ReActResult(
                    goal=goal, status=DEGRADED, final_answer=action_input,
                    reason=f"model determined the goal was unreachable at step {n} "
                           "and returned a partial answer",
                    steps=steps, model=self.model, max_steps=self.max_steps,
                    elapsed_s=round(time.time() - started, 2),
                )

            # --- act: run the chosen tool ------------------------------ #
            if not action:
                obs, ok = "ERROR: could not parse a valid action from the model", False
            elif action not in self.tools:
                obs = (
                    f"ERROR: unknown action '{action}'. valid actions: "
                    f"{', '.join([*self.tools, _FINISH, _GIVE_UP])}"
                )
                ok = False
            else:
                result = self.tools[action].run(action_input)
                obs, ok = result.as_observation(), result.ok
            logger.info("step %d | observation: %s", n, obs)

            # --- reflect: the self-critic ------------------------------ #
            reflection, on_track, next_hint = self._reflect(
                goal, thought, action, action_input, obs
            )
            logger.info("step %d | reflect: on_track=%s | %s", n, on_track, reflection)

            steps.append(
                Step(
                    n=n, thought=thought, action=action, action_input=action_input,
                    observation=obs, observation_ok=ok, reflection=reflection,
                    on_track=on_track, next_hint=next_hint,
                )
            )

        # --- max-iteration cap fired: degrade to best-so-far ----------- #
        reason = f"stopped: max steps ({self.max_steps}) reached without a final answer"
        logger.info(reason)
        partial = self._best_so_far(goal, self._render(steps), reason)
        return ReActResult(
            goal=goal, status=MAX_STEPS, final_answer=partial, reason=reason,
            steps=steps, model=self.model, max_steps=self.max_steps,
            elapsed_s=round(time.time() - started, 2),
        )
