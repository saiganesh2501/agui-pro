"""
Observability and Langfuse integration with Prompt Management and Evaluation.

Initializes:
- Langfuse client (for tracing)
- Prompt manager (fetch versioned prompts from Langfuse)
- Evaluator (LLM-as-judge for scoring runs)

This module exposes BOTH:
- A singleton Observability instance (via get_observability())
- Module-level functions (is_enabled, flush, score_trace, evaluate_and_score,
  get_system_prompt) so code can do `import observability as obs` and call
  `obs.is_enabled()` directly.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Observability:
    """Central observability hub for Langfuse, prompts, and evaluation."""

    def __init__(self):
        self.langfuse = None
        self.prompt_manager = None
        self.evaluator = None
        self._enabled = False

    def init(self, openai_client=None):
        """
        Initialize all observability components.

        Args:
            openai_client: AsyncOpenAI client (for evaluation)
        """
        try:
            # Import here to avoid import-time failures
            from langfuse import get_client
            from prompt_manager import init_prompt_manager
            from evaluators import init_evaluator

            # Initialize Langfuse
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
            secret_key = os.getenv("LANGFUSE_SECRET_KEY")

            if not public_key or not secret_key:
                logger.warning(
                    "Langfuse keys not set. Observability disabled. "
                    "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable."
                )
                self._enabled = False
                return

            self.langfuse = get_client()
            self._enabled = True
            logger.info("✓ Langfuse initialized")

            # Initialize prompt manager (will fetch prompts from Langfuse)
            self.prompt_manager = init_prompt_manager(self.langfuse)
            logger.info("✓ Prompt manager initialized")

            # Initialize evaluator (for scoring agent runs)
            self.evaluator = init_evaluator(openai_client)
            logger.info("✓ Evaluator initialized")

        except ImportError as e:
            logger.error(f"Langfuse or evaluators import failed: {e}")
            self._enabled = False
        except Exception as e:
            logger.error(f"Observability initialization failed: {e}")
            self._enabled = False

    def is_enabled(self) -> bool:
        """Check if observability is enabled."""
        return self._enabled

    def get_current_trace_id(self) -> Optional[str]:
        """
        Return the ID of the trace currently active in the Langfuse context.

        The Langfuse-wrapped OpenAI client creates a trace per call; this reads
        that trace's ID so evaluation scores attach to the real trace shown in
        the dashboard (rather than a made-up id).
        """
        if not self._enabled or self.langfuse is None:
            return None
        try:
            return self.langfuse.get_current_trace_id()
        except Exception as e:
            logger.debug(f"Could not get current trace id: {e}")
            return None

    def get_system_prompt(self) -> str:
        """
        Fetch the agent's system prompt from Langfuse.

        Falls back to hardcoded default if Langfuse unavailable.
        """
        if not self._enabled or self.prompt_manager is None:
            return _FALLBACK_SYSTEM_PROMPT

        try:
            prompt_template = self.prompt_manager.get_prompt(
                "agent-system",
                label="production",
                fallback_name="agent-system",
            )
            return prompt_template.content
        except Exception as e:
            logger.warning(f"Failed to fetch system prompt: {e}. Using fallback.")
            return _FALLBACK_SYSTEM_PROMPT

    def score_trace(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: Optional[str] = None,
    ):
        """Attach a score to a trace in Langfuse."""
        if not self._enabled or self.langfuse is None:
            logger.debug(f"Scoring disabled for {name}")
            return

        try:
            self.langfuse.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
            logger.debug(f"✓ Scored trace {str(trace_id)[:8]}... {name}={value:.2f}")
        except Exception as e:
            logger.warning(f"Failed to score trace: {e}")

    async def evaluate_and_score(
        self,
        trace_id: str,
        user_query: str,
        response: str,
        search_results: Optional[list] = None,
        eval_type: str = "quality",
    ):
        """Run evaluation and attach scores to trace."""
        if not self._enabled or self.evaluator is None:
            logger.debug("Evaluation disabled")
            return

        result = await self.evaluator.evaluate_response(
            user_query=user_query,
            agent_response=response,
            trace_id=trace_id,
            search_results=search_results,
            eval_type=eval_type,
        )

        self.score_trace(
            trace_id=trace_id,
            name=f"llm-judge-{eval_type}",
            value=result.score,
            comment=result.reasoning,
        )

        return result

    def flush(self):
        """Flush all pending traces to Langfuse."""
        if self._enabled and self.langfuse:
            try:
                self.langfuse.flush()
                logger.debug("✓ Traces flushed to Langfuse")
            except Exception as e:
                logger.warning(f"Flush failed: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# Hardcoded fallback system prompt
# ════════════════════════════════════════════════════════════════════════════════
_FALLBACK_SYSTEM_PROMPT = """You are a helpful AI assistant that can search the web for real-time information.

When answering questions:
- Be concise and accurate
- If you need current information, search the web
- Format your responses clearly with sections if needed
- Always cite your sources

You have access to a web search tool. Use it when:
- The question requires current/real-time information
- You need to verify facts
- The user asks about recent events
- You're uncertain about recent developments

Respond naturally and conversationally."""


# ════════════════════════════════════════════════════════════════════════════════
# Global singleton
# ════════════════════════════════════════════════════════════════════════════════
_obs: Optional[Observability] = None


def init_observability(openai_client=None) -> Observability:
    """Initialize the global observability instance."""
    global _obs
    _obs = Observability()
    _obs.init(openai_client)
    return _obs


def get_observability() -> Observability:
    """Get the global observability instance (creates a disabled one if needed)."""
    global _obs
    if _obs is None:
        _obs = Observability()
    return _obs


# ════════════════════════════════════════════════════════════════════════════════
# Module-level wrapper functions
# These let code do `import observability as obs` then call `obs.is_enabled()`,
# `obs.flush()`, `obs.score_trace(...)`, etc. directly on the module.
# They all delegate to the singleton instance.
# ════════════════════════════════════════════════════════════════════════════════

def is_enabled() -> bool:
    """Module-level: is observability enabled?"""
    return get_observability().is_enabled()


def flush():
    """Module-level: flush traces to Langfuse."""
    return get_observability().flush()


def score_trace(trace_id: str, name: str, value: float, comment: Optional[str] = None):
    """Module-level: attach a score to a trace."""
    return get_observability().score_trace(trace_id, name, value, comment)


async def evaluate_and_score(
    trace_id: str,
    user_query: str,
    response: str,
    search_results: Optional[list] = None,
    eval_type: str = "quality",
):
    """Module-level: run evaluation and attach scores."""
    return await get_observability().evaluate_and_score(
        trace_id, user_query, response, search_results, eval_type
    )


def get_system_prompt() -> str:
    """Module-level: fetch the system prompt."""
    return get_observability().get_system_prompt()


def get_current_trace_id() -> Optional[str]:
    """Module-level: get the current Langfuse trace id."""
    return get_observability().get_current_trace_id()