#!/usr/bin/env python3
"""
Initialize Langfuse with the default agent-system prompt.

Uses the Langfuse Python SDK (already installed) instead of raw HTTP,
so it works correctly with your Langfuse version.

Usage:
    python init_prompts.py

Requires (in backend/.env):
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
    LANGFUSE_HOST (or LANGFUSE_BASE_URL)
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful AI assistant that can search the web for real-time information.

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


def main():
    # Verify credentials are present
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")

    if not all([public_key, secret_key, host]):
        logger.error(
            "Missing Langfuse credentials. Set LANGFUSE_PUBLIC_KEY, "
            "LANGFUSE_SECRET_KEY, and LANGFUSE_HOST in backend/.env"
        )
        return False

    logger.info(f"Connecting to Langfuse at {host}")

    try:
        from langfuse import get_client
        langfuse = get_client()
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")
        return False

    # Check if the prompt already exists
    try:
        existing = langfuse.get_prompt("agent-system", label="production")
        logger.info(
            f"Prompt 'agent-system' already exists (version {existing.version}). "
            "Skipping creation. To update, create a new version in the UI."
        )
        return True
    except Exception:
        # Not found (or fetch failed) -> proceed to create it
        logger.info("Prompt 'agent-system' not found. Creating it...")

    # Create the prompt
    try:
        result = langfuse.create_prompt(
            name="agent-system",
            type="text",
            prompt=SYSTEM_PROMPT,
            labels=["production"],   # mark this version as production
            config={
                "temperature": 0.4,
                "model": "gpt-4o-mini",
                "max_tokens": 2048,
            },
        )
        version = getattr(result, "version", "?")
        logger.info(f"✓ Created prompt 'agent-system' version {version}")
    except Exception as e:
        logger.error(f"Failed to create prompt: {e}")
        return False

    logger.info("\n✓ Prompt initialization complete!")
    logger.info("\nNext steps:")
    logger.info("1. Open Langfuse: http://localhost:3000")
    logger.info("2. Click the 'Prompts' tab")
    logger.info("3. You should see 'agent-system' with a 'production' label")
    logger.info("4. Edit it in the UI to create new versions")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)