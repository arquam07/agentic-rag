"""
LLM Factory — returns a LangChain ChatModel based on config.

Supports two providers:
  • Ollama  → runs locally, no API key needed
  • DeepSeek → cloud API, needs DEEPSEEK_API_KEY in .env

The factory returns a standard LangChain BaseChatModel, so the rest
of the code (agent, chains) doesn't care which provider is active.
Switching is just an env var change.
"""

import os
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def get_llm() -> BaseChatModel:
    """
    Build and return the configured LLM.

    Env vars used:
        LLM_PROVIDER     — "ollama" (default) or "deepseek"
        OLLAMA_MODEL      — model name for Ollama   (default: "llama3.2")
        OLLAMA_BASE_URL   — Ollama server URL        (default: "http://localhost:11434")
        DEEPSEEK_API_KEY  — API key for DeepSeek
        DEEPSEEK_MODEL    — model name for DeepSeek  (default: "deepseek-chat")
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3.5:27b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.3,
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI

        # DeepSeek uses an OpenAI-compatible API
        return ChatOpenAI(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
            temperature=0.3,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. Use 'ollama' or 'deepseek'."
        )