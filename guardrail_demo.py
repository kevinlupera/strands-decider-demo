"""Tool-call guardrail: stop an agent from calling a tool with invented arguments.

This is the scenario from the Strands Decider launch post, made self-contained. An
eager agent is asked "What's the weather?" with no city. Instead of asking which city,
it guesses one and proposes `get_weather(location="Seattle")`. Before that call runs,
Strands Decider reads the conversation plus the proposed call and answers two narrow
yes/no questions:

  1. args_grounded -- are the argument values grounded in what the user actually said?
  2. premature     -- is it too early to call this tool, before clarifying?

A few lines of plain Python turn those probabilities into a typed action. `Guide` hands
the model its turn back with feedback (so it asks the user) instead of a hard allow/deny.

Why a decision model here instead of letting the LLM check itself? The gate is a local
forward pass of a couple hundred milliseconds, the conversation never leaves the machine,
and each verdict is a *number your `if` branches on* rather than prose the model can argue
with. This decision is cheap enough to sit in a path where an LLM call never could.

Run:
    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099
    python guardrail_demo.py            # needs AWS Bedrock creds for the agent
    python guardrail_demo.py --fallback # run the pattern without the model/Bedrock

Credit: adapted from the official example at
https://github.com/strands-labs/strands-decider/tree/main/examples/strands (Apache-2.0).
"""

from __future__ import annotations

import argparse
import json
import sys

from decider_client import NAME, Decider, ServerUnavailable

# Two narrow yes/no (noul) questions. Strands Decider answers each with P(true) in 0..1;
# it never decides the action. The true/false descriptions carry most of the signal.
QUESTIONS = {
    "args_grounded": Decider.noul(
        "Are the tool's argument values grounded in facts the user actually provided?",
        {
            "true": "every argument value traces back to something the user said",
            "false": "an argument value was guessed or invented, not stated by the user",
        },
    ),
    "premature": Decider.noul(
        "Is it premature to call this tool now, before clarifying with the user?",
        {
            "true": "the assistant should ask a clarifying question before calling the tool",
            "false": "there is nothing left to clarify; calling now is appropriate",
        },
    ),
}

# A probability at or above this reads as a confident "yes". Policy knob, owned by the
# code -- not the model. 0.45 is illustrative; tune it to your own traffic.
YES = 0.45


def _render(messages: list[dict]) -> str:
    """Flatten Strands' message list to plain text: only human-readable blocks."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        for block in message.get("content", []):
            if isinstance(block, dict) and "text" in block:
                lines.append(f"{role}: {block['text']}")
    return "\n".join(lines) if lines else "(no conversation yet)"


def build_handler(decider: "Decider"):
    """Build the InterventionHandler. Imported lazily so --fallback works without Bedrock."""
    from strands.interventions import Guide, InterventionHandler, Proceed

    class ToolCallReviewer(InterventionHandler):
        """Review each proposed tool call before it runs, using Strands Decider.

        The model classifies; the code decides. That separation is the whole point.
        """

        name = f"{NAME}-tool-call-reviewer"

        def __init__(self, decider: "Decider") -> None:
            self._decider = decider
            self.log: list[dict[str, object]] = []

        def before_tool_call(self, event, **kwargs):
            tool_name = event.tool_use["name"]
            arguments = event.tool_use.get("input") or {}
            state = (
                "A conversation between a user and an AI assistant is below, followed by "
                "a tool call the assistant now wants to make.\n\n"
                f"--- CONVERSATION ---\n{_render(event.agent.messages)}\n\n"
                f"--- PROPOSED TOOL CALL ---\ntool: {tool_name}\n"
                f"arguments: {json.dumps(arguments)}"
            )

            answers = self._decider.ask(state, QUESTIONS)
            args_grounded = answers["args_grounded"]["noul"]
            premature = answers["premature"]["noul"]
            fb = answers["args_grounded"].get("fallback")

            tag = " [FALLBACK heuristic, not the model]" if fb else ""
            print(f"[intervention] classifications (P of 'yes'){tag}:")
            print(f"    args_grounded : {args_grounded:.2f}")
            print(f"    premature     : {premature:.2f}")
            if self._decider.last_latency_ms is not None:
                print(f"    decided in    : {self._decider.last_latency_ms:.0f} ms")

            if args_grounded < YES:
                verdict, action = "guided", Guide(
                    feedback=(
                        "The arguments are not grounded in anything the user said -- they "
                        "look guessed. Ask the user to confirm the values instead of "
                        "inventing them."
                    )
                )
            elif premature >= YES:
                verdict, action = "guided", Guide(
                    feedback=(
                        "It is too early to call this tool. Clarify with the user first, "
                        "then try again."
                    )
                )
            else:
                verdict, action = "proceeded", Proceed()

            self.log.append(
                {"tool": tool_name, "args_grounded": round(args_grounded, 2),
                 "premature": round(premature, 2), "verdict": verdict}
            )
            arrow = {"proceeded": "->", "guided": "~>"}[verdict]
            print(f"    {arrow} {verdict}\n")
            return action

    return ToolCallReviewer(decider)


SYSTEM_PROMPT = (
    "You are an eager weather assistant. Always answer weather questions by calling the "
    "get_weather tool immediately. Never ask the user for clarification -- if no city is "
    "given, just assume Seattle and call the tool with location='Seattle'."
)
USER_REQUEST = "What's the weather?"


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool-call guardrail with Strands Decider")
    parser.add_argument(
        "--fallback", action="store_true",
        help="run the gate with the offline heuristic (no server, no Bedrock agent)",
    )
    args = parser.parse_args()

    decider = Decider(allow_fallback=args.fallback)
    try:
        print(decider.banner(), "\n")
    except ServerUnavailable as exc:
        print(exc)
        return 1

    if args.fallback:
        return _fallback_walkthrough(decider)

    # Real path: drive a Strands agent on Bedrock; the gate runs before the tool.
    from strands import Agent, tool
    from strands.interventions import InterventionHandler  # noqa: F401 (import check)

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a specific city.

        Args:
            location: The city to look up, e.g. "Seattle" or "Paris".
        """
        return f"It's 21C and sunny in {location}."

    gate = build_handler(decider)
    agent = Agent(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        tools=[get_weather],
        interventions=[gate],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )

    print(f"the agent proposes each tool call; {NAME} decides whether it is ready\n")
    print(f"USER: {USER_REQUEST}\n")
    result = agent(USER_REQUEST)

    guided = sum(1 for row in gate.log if row["verdict"] == "guided")
    print(f"{decider}; guided {guided} of {len(gate.log)} proposed calls back to clarify")
    print(f"\n--- what the agent finally replied ---\n{result}")
    return 0


def _fallback_walkthrough(decider: "Decider") -> int:
    """Show the gate's logic directly, without an LLM agent or the model server."""
    print("Simulating the proposed tool call the eager agent would make:\n")
    guessed = {"name": "get_weather", "input": {"location": "Seattle"}}
    state = (
        "--- CONVERSATION ---\n"
        f"user: {USER_REQUEST}\n\n"
        "--- PROPOSED TOOL CALL ---\n"
        f"tool: {guessed['name']}\narguments: {json.dumps(guessed['input'])} (guessed; "
        "the user never said a city)"
    )
    answers = decider.ask(state, QUESTIONS)
    g = answers["args_grounded"]["noul"]
    p = answers["premature"]["noul"]
    print(f"  args_grounded : {g:.2f}  [FALLBACK heuristic]")
    print(f"  premature     : {p:.2f}  [FALLBACK heuristic]")
    verdict = "GUIDE back to ask the user" if g < YES or p >= YES else "PROCEED"
    print(f"  verdict       : {verdict}")
    print(
        "\nWith the real model, these two numbers come from one ~150 ms local forward "
        "pass; the conversation never leaves the machine."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
