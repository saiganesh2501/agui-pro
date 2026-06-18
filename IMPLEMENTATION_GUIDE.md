# Langfuse Prompt Management + Evaluation Implementation Guide

This guide walks you through implementing three production features:

1. **Fetch prompts from Langfuse** (decouple from code)
2. **Maintain versioned prompts** (A/B test, rollback)
3. **Evaluate agent runs** (LLM-as-judge scoring)

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│ Langfuse Dashboard (localhost:3000)             │
│  Prompts tab: Create/edit/version prompts      │
│  Scores tab: View evaluation results            │
└─────────────────────────────────────────────────┘
         ▲
         │ (fetch at runtime)
         │ (attach scores)
         │
┌────────┴──────────────────────────────────────┐
│ Your Agent (agent.py)                          │
│  - Uses prompt_manager.get_prompt()            │
│  - Triggers obs.evaluate_and_score() after run │
└─────────────────────────────────────────────────┘
```

## Step-by-Step Implementation

### Step 1: Add the new modules to your project

Copy these files to your `backend/` folder:

```bash
cp prompt_manager.py backend/
cp evaluators.py backend/
cp init_prompts.py backend/
cp observability_updated.py backend/observability.py  # Replace the old one
```

### Step 2: Update requirements.txt

Ensure you have the latest Langfuse SDK:

```
langfuse>=3.0.0
```

(Note: You already have this if Langfuse is working.)

### Step 3: Create initial prompts in Langfuse

Run the initialization script:

```bash
cd backend/
python init_prompts.py
```

This creates the "agent-system" prompt in Langfuse with:
- Type: text
- Content: the default system prompt
- Version: 1
- Label: "production"

**Expected output:**
```
INFO:__main__:Creating prompt 'agent-system'...
INFO:__main__:✓ Created prompt 'agent-system' version 1
INFO:__main__:✓ Prompt initialization complete!

Next steps:
1. Go to Langfuse dashboard: http://localhost:3000
2. Click 'Prompts' tab
3. You should see 'agent-system' prompt
...
```

### Step 4: Verify in Langfuse UI

1. Go to **http://localhost:3000** (your Langfuse dashboard)
2. Click the **Prompts** tab (left sidebar, under "Prompt Management")
3. You should see **"agent-system"** listed
4. Click it to see:
   - Version 1
   - Label: "production"
   - Type: "text"
   - Content: the system prompt

### Step 5: Update agent.py to use prompts

In your `backend/agent.py`, make these changes:

**Add imports at the top:**
```python
from prompt_manager import get_prompt_manager
from observability import get_observability
```

**In the `StreamingAgent.__init__` method, add:**
```python
def __init__(self, client, model: str = "gpt-4o-mini"):
    self.client = client
    self.model = model
    self.steering = Steering()
    
    # NEW: Initialize managers
    self.prompt_manager = get_prompt_manager()
    self.obs = get_observability()
```

**In `_run_impl()` method, replace the hardcoded system prompt:**

BEFORE:
```python
system_prompt = """You are a helpful AI assistant..."""
```

AFTER:
```python
# Fetch from Langfuse (falls back to hardcoded if unavailable)
system_prompt = self.obs.get_system_prompt()
```

### Step 6: Enable evaluation after each run

Add this method to your `StreamingAgent` class:

```python
async def _evaluate_run(
    self,
    trace_id: str,
    query: str,
    response: str,
    used_search: bool,
):
    """Evaluate the agent's response and score in Langfuse."""
    if not self.obs.is_enabled():
        return

    logger.info(f"Evaluating trace {trace_id[:8]}...")

    try:
        # Run quality evaluation
        await self.obs.evaluate_and_score(
            trace_id=trace_id,
            user_query=query,
            response=response,
            eval_type="quality",
        )

        # Run relevance evaluation
        await self.obs.evaluate_and_score(
            trace_id=trace_id,
            user_query=query,
            response=response,
            eval_type="relevance",
        )

        # Run correctness evaluation if search was used
        if used_search:
            await self.obs.evaluate_and_score(
                trace_id=trace_id,
                user_query=query,
                response=response,
                eval_type="correctness",
            )

        logger.info(f"✓ Evaluations complete for {trace_id[:8]}...")
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
```

**Then, in your `run()` method, call evaluation after the response is complete:**

```python
async def run(self, run_input: RunAgentInput) -> AsyncGenerator[BaseEvent, None]:
    trace_id = self.steering.trace_id or "unknown"
    used_search = False
    final_response = ""

    try:
        async for ev in self._run_impl(run_input):
            yield ev
            
            # Track if search was used
            if isinstance(ev, ToolCallEndEvent) and ev.tool_name == "web_search":
                used_search = True
            
            # Collect response text
            if isinstance(ev, TextMessageContentEvent):
                final_response += ev.delta

        # NEW: Evaluate the run
        if final_response.strip():
            await self._evaluate_run(
                trace_id=trace_id,
                query=run_input.user_message,
                response=final_response,
                used_search=used_search,
            )

    finally:
        self.obs.flush()
```

### Step 7: Initialize observability in main.py

In `backend/main.py`, ensure observability is initialized:

```python
from observability import init_observability

# After creating AsyncOpenAI client:
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# NEW: Initialize observability (prompts + evaluation)
obs = init_observability(openai_client)

# Then create the agent with this client
agent = StreamingAgent(client=openai_client)
```

## Usage: Versioning and A/B Testing

### Create a new prompt version

1. Go to **Langfuse dashboard** → **Prompts**
2. Click **"agent-system"**
3. Click **"New version"** button
4. Edit the prompt content
5. Optionally set config (temperature, max_tokens, etc.)
6. Click **"Create version"**

### Label a version as "production"

Once you've tested a new version and it's working:

1. Go to **Prompts** → **"agent-system"**
2. See the list of versions (v1, v2, v3, etc.)
3. Click the version you want to make production
4. Click **"Add label"** → type "production"
5. Save

Now `obs.get_system_prompt()` will fetch this new version.

### A/B test two prompt versions

To run two versions in parallel:

**Version A (production):**
```python
prompt_a = obs.get_system_prompt(label="production")
```

**Version B (experimental):**
```python
prompt_b = obs.get_system_prompt(label="experimental")
```

Then compare evaluation scores in Langfuse:

1. Go to **Traces**
2. Filter by traces using each prompt
3. Compare average **quality**, **relevance**, **correctness** scores

## Evaluation Scoring Explained

After each agent run, three evaluations run (if enabled):

### 1. **Quality** (0.0-1.0)
Evaluates: clarity, structure, helpfulness

- 1.0 = excellent response
- 0.7-0.9 = good, minor issues
- 0.4-0.6 = acceptable but could be better
- 0.0-0.3 = poor quality

### 2. **Relevance** (0.0-1.0)
Evaluates: does it answer the question?

- 1.0 = directly answers the question
- 0.8-0.9 = answers most of it
- 0.5-0.7 = partially relevant
- 0.0-0.4 = off-topic

### 3. **Correctness** (0.0-1.0)
Evaluates: factual accuracy (only if search was used)

- 1.0 = all facts accurate
- 0.8-0.9 = mostly accurate
- 0.5-0.7 = some uncertain claims
- 0.0-0.4 = significant errors

## Viewing Results in Langfuse

1. Go to **Traces** tab
2. Click any trace (a user message)
3. Scroll to **Evaluations** section
4. You'll see:
   - Score name (e.g., "llm-judge-quality")
   - Score value (0.0-1.0)
   - Reasoning (why it got that score)

## Production Considerations

### 1. **Evaluation Cost**

Each evaluation makes an LLM call (gpt-4o-mini), which costs ~$0.00015 per call.

For a 1,000 messages/day with 3 evaluations each = $0.45/day.

**To reduce cost:**
- Only evaluate certain messages (e.g., if score_sample_rate=0.1, only evaluate 10%)
- Use SimpleRulesEvaluator for free heuristic scoring

### 2. **Latency**

Evaluations run asynchronously after the response is sent to the user, so they don't block.

Total: ~500ms per evaluation (gpt-4o-mini is fast).

### 3. **Prompt Caching**

The Langfuse SDK automatically caches prompts client-side. First fetch is ~50ms, subsequent fetches are ~1ms (memory).

### 4. **Fallbacks**

If Langfuse is down:
- Prompts fall back to hardcoded defaults
- Evaluation is skipped
- Agent still works normally

## Troubleshooting

### "Prompt not found in Langfuse"

Run `init_prompts.py` again to create the prompt.

### "Evaluation disabled"

Check that `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`.

### "No scores appearing in Langfuse"

Make sure:
1. `obs.flush()` is called (happens in `_run_impl()` finally block)
2. Evaluate runs after response (check agent.py has `_evaluate_run()` call)
3. Traces are visible first (check Traces tab for the message)

### Slow evaluation?

Evaluations run asynchronously and don't block the user. If still slow, reduce to 1 evaluation type (e.g., quality only).

## Next Steps

1. **Test locally**: Run a query, see evaluation scores in Langfuse
2. **Create variants**: Make 2-3 prompt versions, A/B test them
3. **Monitor metrics**: Track average scores over time in Langfuse
4. **Iterate**: Based on evaluation results, improve the system prompt
5. **Deploy**: Push to production once you're confident in the prompts

## Reference: File Map

| File | Purpose |
|------|---------|
| `prompt_manager.py` | Fetches versioned prompts from Langfuse with caching |
| `evaluators.py` | LLM-as-judge scoring (quality, relevance, correctness) |
| `observability.py` | Orchestrates prompts, evaluation, and Langfuse tracing |
| `init_prompts.py` | Bootstrap script to create initial prompts in Langfuse |
| `agent.py` | Call `obs.get_system_prompt()` and `_evaluate_run()` |
| `main.py` | Initialize observability with `init_observability()` |

## Final Checklist

- [ ] Copied new files to `backend/`
- [ ] Ran `init_prompts.py` (created agent-system in Langfuse)
- [ ] Verified prompt exists in Langfuse UI
- [ ] Updated `agent.py` to use `obs.get_system_prompt()`
- [ ] Updated `agent.py` to call `_evaluate_run()` after response
- [ ] Initialized observability in `main.py`
- [ ] Tested with a query, saw evaluation scores in Langfuse
- [ ] Created a test prompt version to verify versioning works

You're done! You now have:
✅ Decoupled prompts (in Langfuse, not code)
✅ Versioned prompt management
✅ Automatic evaluation + scoring
