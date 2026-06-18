"""
evaluate.py — programmatic evaluation of the agent, scored in Langfuse.

What it does:
  1. Runs a fixed set of test questions through the SAME StreamingAgent the app
     uses (so the traces look identical to real usage and appear in Langfuse).
  2. Collects each final answer.
  3. Uses an LLM-as-judge (a separate OpenAI call) to score each answer 0-1 on
     relevance + helpfulness, with a short rationale.
  4. Attaches that score to the run's Langfuse trace via create_score, so the
     scores show up on the traces in the Langfuse dashboard.

Run it (backend venv active, Langfuse + OpenAI keys in .env, Langfuse container up):
    python evaluate.py

Then open the Langfuse UI (http://localhost:3000) → Traces / Scores to review.
"""
from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()

import observability as obs
from agent import StreamingAgent, Steering
from events import EventType, RunAgentInput

# Wrapped client so eval runs are traced too.
if obs.is_enabled():
    from langfuse.openai import AsyncOpenAI
else:
    from openai import AsyncOpenAI

# ---- the evaluation dataset ----
TEST_CASES = [
    {"q": "Explain what a Python list comprehension is, with one example.",
     "expects": "a clear definition and a correct code example"},
    {"q": "What is 17 * 23? Show your reasoning.",
     "expects": "the correct answer 391"},
    {"q": "Give me three tips for writing readable code.",
     "expects": "three concrete, sensible tips"},
    {"q": "What's the difference between TCP and UDP?",
     "expects": "accurate contrast of reliability/ordering vs. speed"},
]

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a user question, the criteria it should "
    "meet, and an assistant answer, score the answer from 0.0 to 1.0 for how "
    "well it satisfies the criteria (correctness + helpfulness). Respond ONLY "
    "with JSON: {\"score\": <float 0-1>, \"reason\": \"<one sentence>\"}."
)


async def run_agent_once(client, question: str) -> tuple[str, str | None]:
    """Run the agent on one question; return (final_answer, trace_id)."""
    steering = Steering()
    agent = StreamingAgent(client, steering)
    run_input = RunAgentInput(
        thread_id="eval-thread",
        run_id=f"eval_{abs(hash(question)) % 10_000}",
        messages=[{"id": "u1", "role": "user", "content": question}],
        forwarded_props={"model": "gpt-4o-mini"},
    )
    answer = ""
    async for ev in agent.run(run_input):
        if ev.type == EventType.TEXT_MESSAGE_CONTENT:
            answer += getattr(ev, "delta", "") or ""
    return answer, steering.trace_id


async def judge(client, question: str, expects: str, answer: str) -> dict:
    """LLM-as-judge scoring of one answer."""
    resp = await client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content":
                f"QUESTION:\n{question}\n\nCRITERIA:\n{expects}\n\nANSWER:\n{answer}"},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        return {"score": float(data.get("score", 0)), "reason": data.get("reason", "")}
    except Exception:
        return {"score": 0.0, "reason": f"unparseable judge output: {raw[:80]}"}


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Add it to backend/.env.")
        return
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"Running {len(TEST_CASES)} eval cases "
          f"(Langfuse {'ENABLED' if obs.is_enabled() else 'DISABLED'})\n")

    total = 0.0
    for i, case in enumerate(TEST_CASES, 1):
        answer, trace_id = await run_agent_once(client, case["q"])
        verdict = await judge(client, case["q"], case["expects"], answer)
        total += verdict["score"]

        print(f"[{i}] {case['q']}")
        print(f"    score: {verdict['score']:.2f} — {verdict['reason']}")

        # Attach the judge score to the agent run's trace in Langfuse.
        if trace_id and obs.is_enabled():
            obs.score_trace(
                trace_id=trace_id,
                name="llm-judge-relevance",
                value=verdict["score"],
                comment=verdict["reason"],
            )

    obs.flush()
    avg = total / len(TEST_CASES) if TEST_CASES else 0
    print(f"\nAverage score: {avg:.2f}")
    if obs.is_enabled():
        print("Scores attached to traces — view them at http://localhost:3000")


if __name__ == "__main__":
    asyncio.run(main())
