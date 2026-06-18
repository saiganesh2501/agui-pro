"""
Langfuse Evaluation integration.

Implements LLM-as-judge evaluation of agent runs.
Scores are attached to traces in Langfuse automatically.

Evaluators:
- relevance: does the answer address the user's question?
- quality: is the answer well-formatted, clear, and helpful?
- correctness: is the factual information accurate?
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating an agent run."""
    trace_id: str
    evaluator_name: str
    score: float  # 0.0 to 1.0
    reasoning: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class LLMJudgeEvaluator:
    """
    LLM-as-judge evaluator using Claude (or another LLM) to score responses.
    
    Works by:
    1. Taking user question, agent response, and search results
    2. Prompting an LLM to score on relevance/quality/correctness
    3. Returning a 0-1 score with reasoning
    4. Attaching the score to the trace in Langfuse
    """

    def __init__(self, openai_client=None):
        """
        Initialize evaluator.
        
        Args:
            openai_client: OpenAI AsyncOpenAI client for making eval calls
        """
        self.client = openai_client
        self._enabled = openai_client is not None

    async def evaluate_response(
        self,
        user_query: str,
        agent_response: str,
        trace_id: str,
        search_results: Optional[list] = None,
        eval_type: str = "quality",
    ) -> EvaluationResult:
        """
        Evaluate an agent's response using LLM-as-judge.
        
        Args:
            user_query: The original user question
            agent_response: The agent's answer
            trace_id: Langfuse trace ID to attach the score to
            search_results: Optional list of search results used
            eval_type: Type of evaluation (quality, relevance, correctness)
        
        Returns:
            EvaluationResult with score (0-1) and reasoning
        
        Example:
            result = await evaluator.evaluate_response(
                user_query="What's the weather in Tokyo?",
                agent_response="It's sunny and 22°C in Tokyo.",
                trace_id="trace_abc123",
                eval_type="relevance"
            )
            # result.score = 0.95, result.reasoning = "..."
        """
        if not self._enabled:
            logger.debug("Evaluation disabled (no OpenAI client)")
            return EvaluationResult(
                trace_id=trace_id,
                evaluator_name=f"llm-judge-{eval_type}",
                score=0.5,  # Neutral score when disabled
                reasoning="Evaluation disabled",
                timestamp=datetime.now(),
            )

        try:
            score, reasoning = await self._judge(
                user_query=user_query,
                response=agent_response,
                search_results=search_results,
                eval_type=eval_type,
            )

            result = EvaluationResult(
                trace_id=trace_id,
                evaluator_name=f"llm-judge-{eval_type}",
                score=score,
                reasoning=reasoning,
                timestamp=datetime.now(),
                metadata={
                    "eval_type": eval_type,
                    "query_length": len(user_query),
                    "response_length": len(agent_response),
                },
            )

            logger.info(
                f"Evaluation '{eval_type}' for trace {trace_id}: "
                f"score={score:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Evaluation failed for trace {trace_id}: {e}")
            return EvaluationResult(
                trace_id=trace_id,
                evaluator_name=f"llm-judge-{eval_type}",
                score=0.5,
                reasoning=f"Evaluation error: {str(e)}",
                timestamp=datetime.now(),
            )

    async def _judge(
        self,
        user_query: str,
        response: str,
        search_results: Optional[list],
        eval_type: str,
    ) -> tuple[float, str]:
        """
        Use an LLM to judge the response.
        
        Returns:
            (score, reasoning) where score is 0.0-1.0
        """
        # Build the evaluation prompt based on eval_type
        if eval_type == "relevance":
            prompt = self._build_relevance_prompt(user_query, response)
        elif eval_type == "quality":
            prompt = self._build_quality_prompt(user_query, response)
        elif eval_type == "correctness":
            prompt = self._build_correctness_prompt(
                user_query, response, search_results
            )
        else:
            prompt = self._build_quality_prompt(user_query, response)

        # Call LLM with eval prompt
        try:
            completion = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast, cheap eval model
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert evaluator of AI responses. "
                        "Respond with ONLY a single JSON object on one line: "
                        '{"score": <0.0-1.0>, "reasoning": "<brief explanation>"}',
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=200,
            )

            # Parse the response
            response_text = completion.choices[0].message.content.strip()
            import json

            try:
                parsed = json.loads(response_text)
                score = float(parsed.get("score", 0.5))
                reasoning = str(parsed.get("reasoning", ""))
                # Clamp score to 0-1 range
                score = max(0.0, min(1.0, score))
                return score, reasoning
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse eval response: {response_text}")
                return 0.5, f"Parse error: {str(e)}"

        except Exception as e:
            logger.error(f"LLM eval call failed: {e}")
            raise

    def _build_relevance_prompt(self, query: str, response: str) -> str:
        """Evaluate if the response answers the user's question."""
        return f"""
User question: {query}

Agent response: {response}

Evaluate the relevance of the response to the user's question.
- Score 1.0 if the response directly and completely answers the question
- Score 0.8-0.9 if it answers most of the question with minor gaps
- Score 0.5-0.7 if it's partially relevant but misses key aspects
- Score 0.0-0.4 if it's off-topic or irrelevant

Respond with only valid JSON:
{{"score": <float 0-1>, "reasoning": "<brief explanation>"}}
"""

    def _build_quality_prompt(self, query: str, response: str) -> str:
        """Evaluate the overall quality and clarity of the response."""
        return f"""
User question: {query}

Agent response: {response}

Evaluate the quality of this response on these criteria:
- Clarity: Is it well-written and easy to understand?
- Helpfulness: Does it provide actionable information?
- Structure: Is it well-organized (paragraphs, bullet points)?
- Tone: Is it conversational and friendly?

Score 1.0 for excellent response, 0.5 for average, 0.0 for poor.

Respond with only valid JSON:
{{"score": <float 0-1>, "reasoning": "<brief explanation>"}}
"""

    def _build_correctness_prompt(
        self, query: str, response: str, search_results: Optional[list]
    ) -> str:
        """Evaluate factual correctness of the response."""
        context = ""
        if search_results:
            context = "Search results context:\n" + "\n".join(
                [f"- {r.get('title', 'N/A')}: {r.get('body', 'N/A')[:200]}"
                 for r in search_results[:3]]
            )

        return f"""
User question: {query}

Agent response: {response}

{context}

Evaluate the factual correctness of the response:
- Score 1.0 if all facts are accurate and well-supported
- Score 0.8-0.9 if mostly accurate with minor issues
- Score 0.5-0.7 if some facts are uncertain or unsupported
- Score 0.0-0.4 if contains significant factual errors

Respond with only valid JSON:
{{"score": <float 0-1>, "reasoning": "<brief explanation>"}}
"""


class SimpleRulesEvaluator:
    """
    Simple rule-based evaluator (alternative to LLM judge).
    
    Fast and cheap, uses heuristics instead of LLM calls:
    - Length: longer responses generally better
    - Contains code: presence of code blocks scores higher
    - Cites sources: if search was used, cites them
    """

    @staticmethod
    def evaluate(
        user_query: str,
        response: str,
        used_search: bool = False,
        search_results: Optional[list] = None,
    ) -> float:
        """
        Simple heuristic score without LLM call.
        
        Args:
            user_query: Original question
            response: Agent's response
            used_search: Whether web search was used
            search_results: Results from search (if any)
        
        Returns:
            Score 0.0-1.0
        """
        score = 0.5  # Start at middle

        # Length: responses under 50 chars are probably incomplete
        if len(response) < 50:
            score -= 0.2
        elif len(response) > 5000:
            score -= 0.1  # Too verbose

        # Has code block: good sign of quality
        if "```" in response:
            score += 0.15

        # References sources: good sign
        if "[" in response and "]" in response:
            score += 0.1

        # Actually used search when needed (science/current topics)
        current_keywords = [
            "weather", "today", "recent", "latest", "2024", "2025",
            "current", "now", "breaking"
        ]
        if any(kw in user_query.lower() for kw in current_keywords):
            if used_search:
                score += 0.2
            else:
                score -= 0.2

        # Clamp to 0-1
        score = max(0.0, min(1.0, score))
        return score


# Singleton instance
_evaluator: Optional[LLMJudgeEvaluator] = None


def init_evaluator(openai_client) -> LLMJudgeEvaluator:
    """Initialize the global evaluator."""
    global _evaluator
    _evaluator = LLMJudgeEvaluator(openai_client)
    return _evaluator


def get_evaluator() -> LLMJudgeEvaluator:
    """Get the global evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = LLMJudgeEvaluator(None)
    return _evaluator