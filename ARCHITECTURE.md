# Langfuse Prompt Management + Evaluation Architecture

## The Three Features (Production-Grade)

### 1️⃣ **Fetch Prompts from Langfuse**
**What:** At runtime, fetch the system prompt from Langfuse instead of hardcoding it in `agent.py`.

**Why:** 
- Non-technical team members can edit prompts without touching code
- Instant deployments (no redeploy needed)
- Fallback to hardcoded defaults if Langfuse is unavailable

**How it works:**
```python
# Instead of:
SYSTEM_PROMPT = """You are a helpful assistant..."""

# You call:
system_prompt = obs.get_system_prompt()  # Fetches from Langfuse
```

The Langfuse SDK caches prompts client-side, so first fetch is 50ms, subsequent are 1ms.

---

### 2️⃣ **Maintain Versioned Prompts**
**What:** In Langfuse, create multiple versions of the same prompt with labels (production, experimental, etc.).

**Why:**
- A/B test prompts before going live
- Rollback to previous versions if something breaks
- Track which version was used for each run
- Iterate on improvements without breaking production

**How it works:**
```
Langfuse Prompts tab
  └─ agent-system
      ├─ v1 [production] ← current
      ├─ v2 [experimental]
      └─ v3 [archive]
```

Edit in the UI, create a new version, test it, then move the "production" label.

---

### 3️⃣ **Sample Evaluation of Each Agent Run**
**What:** After each response, run an LLM-as-judge evaluator that scores the response.

**Why:**
- Measure response quality objectively
- Detect regressions (if new prompt causes lower scores)
- Understand why one prompt version is better than another
- Build a dataset of scored outputs for future improvements

**How it works:**
```
User: "What's the weather in Tokyo?"
Agent: "It's sunny and 22°C..."

Evaluation (automatic, after response):
  ├─ Quality: 0.92 (well-formatted, clear)
  ├─ Relevance: 0.95 (directly answers)
  └─ Correctness: 0.88 (facts accurate based on search)

Scores attached to trace in Langfuse ✓
```

Three evaluation types:
- **Quality:** Is the response well-written and helpful?
- **Relevance:** Does it answer the user's question?
- **Correctness:** Are the facts accurate? (runs if search used)

---

## File Structure

```
agui-pro/backend/
├── agent.py              (MODIFIED: fetch prompts, trigger evaluation)
├── main.py               (MODIFIED: initialize observability)
├── observability.py      (REPLACED: new version with prompts + eval)
├── prompt_manager.py     (NEW: fetch & cache prompts)
├── evaluators.py         (NEW: LLM-as-judge scoring)
├── init_prompts.py       (NEW: bootstrap prompts to Langfuse)
├── tools.py              (unchanged)
├── events.py             (unchanged)
├── requirements.txt      (ensure langfuse>=3.0.0)
├── .env                  (LANGFUSE keys already here)
└── IMPLEMENTATION_GUIDE.md (NEW: step-by-step setup)
```

---

## Data Flow

### Prompt Flow
```
┌──────────────────────────────┐
│ Langfuse Dashboard           │
│ Prompts: agent-system [prod] │
└────────────┬─────────────────┘
             │ (at runtime)
             ▼
┌──────────────────────────────┐
│ prompt_manager.get_prompt()  │
│ - Fetches from Langfuse      │
│ - Caches locally             │
│ - Fallback to hardcoded      │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ obs.get_system_prompt()      │
│ → "You are a helpful..."     │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ agent.py _run_impl()         │
│ Uses it for the OpenAI call  │
└──────────────────────────────┘
```

### Evaluation Flow
```
┌──────────────────────────────┐
│ Agent finishes response       │
│ final_response = "..."       │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ _evaluate_run()              │
│ - Calls obs.evaluate_and_score() × 3 │
└────────────┬─────────────────┘
             │
     ┌───────┼───────┬─────────┐
     ▼       ▼       ▼         ▼
  Quality Relevance Correctness (if search used)
     │       │       │         │
     └───────┴───────┴─────────┘
             │
             ▼
┌──────────────────────────────┐
│ LLMJudgeEvaluator            │
│ - Calls gpt-4o-mini          │
│ - Gets score + reasoning     │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ obs.score_trace()            │
│ - Attaches to Langfuse trace │
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│ Langfuse Traces tab          │
│ Shows scores with reasoning  │
└──────────────────────────────┘
```

---

## Design Decisions (Why These Choices)

### 1. **Client-side Caching (Not Custom Cache)**
The Langfuse SDK handles caching automatically via in-memory storage. 
- ✅ Simple (no extra code)
- ✅ Fast (1ms after first fetch)
- ✅ No stale data issues

### 2. **LLM-as-Judge (Not Rule-Based)**
Uses Claude/GPT to evaluate instead of simple heuristics.
- ✅ More accurate (understands nuance)
- ✅ Works for any response length/format
- ✅ Easy to explain results to stakeholders

Alternative: `SimpleRulesEvaluator` in evaluators.py (free, but less accurate).

### 3. **Async Evaluation (Not Blocking)**
Evaluation runs after response is sent to user.
- ✅ No latency impact (user sees answer immediately)
- ✅ Scores appear in Langfuse shortly after
- ✅ If eval fails, user doesn't notice

### 4. **Three Evaluation Types (Not Just One)**
Quality + Relevance + Correctness give a complete picture.
- Quality: "is this well-written?"
- Relevance: "does it answer the question?"
- Correctness: "is it factually accurate?" (needs search context)

### 5. **Prompt Manager Singleton**
Global instance initialized once in `main.py`, used everywhere.
- ✅ Simple API: `get_prompt_manager().get_prompt(...)`
- ✅ Single source of truth
- ✅ Works even if Langfuse is down (falls back gracefully)

---

## Integration Points

### In agent.py:
```python
from prompt_manager import get_prompt_manager
from observability import get_observability

class StreamingAgent:
    def __init__(self, client):
        self.prompt_manager = get_prompt_manager()
        self.obs = get_observability()
    
    async def _run_impl(self):
        # Fetch prompt from Langfuse
        system_prompt = self.obs.get_system_prompt()
        messages = [{"role": "system", "content": system_prompt}, ...]
        # Rest of logic...
    
    async def run(self):
        # After response completes:
        await self._evaluate_run(trace_id, query, response, used_search)
```

### In main.py:
```python
from observability import init_observability

# Initialize after creating OpenAI client
openai_client = AsyncOpenAI(api_key=...)
obs = init_observability(openai_client)

# Now obs is available everywhere via get_observability()
```

### In Langfuse UI:
1. Prompts tab: Create/edit/version prompts
2. Traces tab: See evaluation scores attached
3. Scores tab: Filter by score ranges, identify patterns

---

## Production Checklist

### Before Going Live:
- [ ] Test locally (run a query, see evaluation scores)
- [ ] Create 2-3 prompt versions
- [ ] A/B test them (compare average scores)
- [ ] Set best version as "production" label
- [ ] Monitor evaluation metrics for a week
- [ ] Set up alerts if quality score drops below 0.7

### Monitoring:
```
Langfuse Dashboard Metrics:
- Average quality score per day
- Average relevance score per day
- Correlation between prompt version and scores
- Cost of evaluations (should be <$1/day for typical usage)
```

### Cost Estimate:
- Evaluation: gpt-4o-mini @ $0.00015/call
- 1,000 messages/day × 3 evals = $0.45/day
- Prompt fetching: free (caching included in SDK)

---

## What You Get After Implementation

✅ **Non-technical prompt editing** — Team members can update prompts in Langfuse UI without touching code

✅ **Instant A/B testing** — Create a new prompt version, test it, compare scores, deploy

✅ **Quality metrics** — Every response scored on quality, relevance, correctness

✅ **Audit trail** — Every version tracked with timestamps and who made changes

✅ **Rollback capability** — If a new prompt scores worse, revert the label to the old version

✅ **Data for improvement** — Dataset of scored outputs to analyze what works

---

## Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| "Prompt not found" | init_prompts.py not run | `python init_prompts.py` |
| "Evaluation disabled" | Langfuse keys missing | Set LANGFUSE_PUBLIC_KEY/SECRET in .env |
| "Scores not showing" | Traces haven't flushed | Check `obs.flush()` in finally block |
| "Slow evaluations" | Waiting for results | Evaluations run async, shouldn't slow user |
| "Langfuse down" | Server crashed | Agent falls back to hardcoded prompt, works fine |

---

## Next: Implementation Steps

1. **Copy files** to backend/
2. **Run init_prompts.py** to create initial prompt
3. **Update agent.py** with prompt fetching + evaluation
4. **Initialize observability** in main.py
5. **Test locally** with a few queries
6. **Create prompt variants** in Langfuse UI
7. **Monitor scores** in Traces tab
8. **Iterate** on prompts based on evaluation results

Full step-by-step guide in IMPLEMENTATION_GUIDE.md →
