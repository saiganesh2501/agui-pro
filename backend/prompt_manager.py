"""
Langfuse Prompt Management integration.

Fetches prompts from Langfuse at runtime. Supports:
- Version selection (production label recommended)
- Client-side caching (built into Langfuse SDK)
- Fallback to hardcoded defaults if Langfuse unavailable
- Chat-type prompts (system + user templates)
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Represents a fetched prompt template."""
    name: str
    version: int
    type: str  # "chat" or "text"
    content: str  # for text prompts
    messages: Optional[List[Dict[str, str]]] = None  # for chat prompts [{role, content}]
    config: Optional[Dict[str, Any]] = None  # model config, temperature, etc.


# Hardcoded fallback prompts (used if Langfuse is unavailable)
FALLBACK_SYSTEM_PROMPT = """You are a helpful AI assistant that can search the web for real-time information.

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

FALLBACK_PROMPTS = {
    "agent-system": PromptTemplate(
        name="agent-system",
        version=1,
        type="text",
        content=FALLBACK_SYSTEM_PROMPT,
        config={
            "temperature": 0.4,
            "model": "gpt-4o-mini",
            "max_tokens": 2048,
        },
    ),
}


class PromptManager:
    """
    Manages fetching and caching prompts from Langfuse.
    
    Langfuse SDK handles client-side caching automatically, so prompts
    are fetched once and then served from memory. No additional caching needed.
    """

    def __init__(self, langfuse_client=None):
        """
        Initialize prompt manager.
        
        Args:
            langfuse_client: Langfuse client instance (from observability.py)
        """
        self.client = langfuse_client
        self._fetch_enabled = langfuse_client is not None and self._is_enabled()

    def _is_enabled(self) -> bool:
        """Check if Langfuse is enabled and credentials are set."""
        if self.client is None:
            return False
        try:
            # Check if required env vars are set
            pk = os.getenv("LANGFUSE_PUBLIC_KEY")
            sk = os.getenv("LANGFUSE_SECRET_KEY")
            return bool(pk and sk)
        except Exception as e:
            logger.warning(f"Langfuse prompt management disabled: {e}")
            return False

    def get_prompt(
        self,
        prompt_name: str,
        label: str = "production",
        fallback_name: Optional[str] = None,
    ) -> PromptTemplate:
        """
        Fetch a prompt from Langfuse with fallback to hardcoded default.
        
        The Langfuse SDK automatically caches prompts client-side, so this call
        is fast after the first fetch (reads from memory).
        
        Args:
            prompt_name: Name of the prompt in Langfuse (e.g., "agent-system")
            label: Version label to fetch (default: "production")
            fallback_name: Key to use for fallback if fetch fails (e.g., "agent-system")
        
        Returns:
            PromptTemplate with the prompt content, version, and config
            
        Example:
            prompt = prompt_manager.get_prompt(
                "agent-system",
                label="production",
                fallback_name="agent-system"
            )
            system_message = prompt.content
        """
        if not self._fetch_enabled:
            logger.info(f"Langfuse disabled; using fallback for '{prompt_name}'")
            return self._get_fallback(fallback_name or prompt_name)

        try:
            logger.debug(f"Fetching prompt '{prompt_name}' with label '{label}'")
            
            # Fetch from Langfuse using the SDK (handles caching automatically)
            langfuse_prompt = self.client.get_prompt(
                name=prompt_name,
                label=label,
            )

            # Parse the Langfuse prompt into our PromptTemplate
            template = self._parse_langfuse_prompt(langfuse_prompt)
            logger.info(
                f"✓ Fetched prompt '{prompt_name}' v{template.version} "
                f"({template.type})"
            )
            return template

        except Exception as e:
            logger.warning(
                f"Failed to fetch prompt '{prompt_name}': {e}. "
                f"Using fallback."
            )
            return self._get_fallback(fallback_name or prompt_name)

    def _parse_langfuse_prompt(self, langfuse_prompt) -> PromptTemplate:
        """
        Parse a Langfuse prompt response into our PromptTemplate.
        
        Langfuse prompts can be:
        - Text prompts: raw string content
        - Chat prompts: array of {role, content} objects
        """
        prompt_type = langfuse_prompt.type  # "text" or "chat"
        
        if prompt_type == "chat":
            # Chat-type prompt: messages with role/content
            template = PromptTemplate(
                name=langfuse_prompt.name,
                version=langfuse_prompt.version,
                type="chat",
                content=None,
                messages=langfuse_prompt.prompt,  # list of {role, content}
                config=langfuse_prompt.config or {},
            )
        else:
            # Text-type prompt: raw string
            template = PromptTemplate(
                name=langfuse_prompt.name,
                version=langfuse_prompt.version,
                type="text",
                content=langfuse_prompt.prompt,  # string
                messages=None,
                config=langfuse_prompt.config or {},
            )

        return template

    def _get_fallback(self, prompt_name: str) -> PromptTemplate:
        """Get hardcoded fallback prompt."""
        if prompt_name in FALLBACK_PROMPTS:
            return FALLBACK_PROMPTS[prompt_name]
        
        # Ultimate fallback if name not found
        logger.error(f"Prompt '{prompt_name}' not found in Langfuse or fallbacks")
        return FALLBACK_PROMPTS["agent-system"]

    def get_config(self, prompt_template: PromptTemplate) -> Dict[str, Any]:
        """
        Extract model config from a prompt.
        
        Args:
            prompt_template: PromptTemplate from get_prompt()
        
        Returns:
            Dict with temperature, max_tokens, etc. for OpenAI
            
        Example:
            prompt = pm.get_prompt("agent-system")
            config = pm.get_config(prompt)
            # {temperature: 0.4, max_tokens: 2048, ...}
        """
        return prompt_template.config or {}


# Singleton instance (initialized in observability.py)
_prompt_manager: Optional[PromptManager] = None


def init_prompt_manager(langfuse_client) -> PromptManager:
    """Initialize the global prompt manager."""
    global _prompt_manager
    _prompt_manager = PromptManager(langfuse_client)
    return _prompt_manager


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        # Fallback: create a disabled manager if not initialized
        _prompt_manager = PromptManager(None)
    return _prompt_manager