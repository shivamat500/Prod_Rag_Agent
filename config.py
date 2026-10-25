"""Central configuration: typed settings plus the live provider objects.

Settings reads environment variables (and a local .env file). Providers builds
the expensive objects (LLM, embeddings, tracing) lazily, so importing this
module never touches the network or needs an API key.
"""
from __future__ import annotations

from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Mode ---------------------------------------------------------------
    offline_mode: bool = False

    # --- Google Gemini ------------------------------------------------------
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    temperature: float = 0.3
    # Gemini "thinking" models count their reasoning tokens against this limit,
    # so 1024 can be used up before any answer text is produced.
    max_tokens: int = 4096

    # --- Embeddings (local, no API key) -------------------------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Inside the project folder so the model downloaded at build time on Render
    # is still there when the service starts.
    embedding_cache_dir: str = "./models_cache"

    # --- Vector store (FAISS) -----------------------------------------------
    faiss_index_path: str = "./faiss_index"
    top_k: int = 4
    retrieval_min_score: float = 0.35
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Memory -------------------------------------------------------------
    memory_dir: str = "./memory_store"
    memory_window: int = 10

    # --- Agent behaviour ----------------------------------------------------
    # The self-critique pass costs one extra Gemini call per question. Turn it
    # off if you are on a tight free-tier quota.
    enable_self_critique: bool = True

    # --- Web search fallback (Tavily) ---------------------------------------
    tavily_api_key: str | None = None

    # --- Optional observability (Langfuse) ----------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Guardrails ---------------------------------------------------------
    use_guardrails_ai: bool = False

    # --- Web layer (app.py) -------------------------------------------------
    max_web_sessions: int = 200
    max_stored_messages: int = 40
    rate_limit_per_minute: int = 20
    max_message_chars: int = 1000

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def tavily_enabled(self) -> bool:
        return bool(self.tavily_api_key)


class Providers:
    """Builds the live objects the rest of the app needs, once, on demand."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    @cached_property
    def llm(self):
        # Checked before any import so offline mode never even loads the client.
        if self.settings.offline_mode:
            # A guard, not a code path anyone should hit: in offline mode the
            # agent never builds an LLM chain. Reaching here means a caller
            # bypassed that, and silently making a billed API call would be
            # worse than failing loudly.
            raise RuntimeError(
                "OFFLINE_MODE is enabled, so no LLM is available. "
                "Set OFFLINE_MODE=false to use Gemini."
            )
        if not self.settings.gemini_api_key:
            raise RuntimeError(
                "[config][providers] Gemini API key is not set. "
                "Please set the 'GEMINI_API_KEY' environment variable."
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=self.settings.gemini_model,
            google_api_key=self.settings.gemini_api_key,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_tokens,
        )

    @cached_property
    def embeddings(self):
        from embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(
            model_name=self.settings.embedding_model,
            cache_dir=self.settings.embedding_cache_dir,
        )

    @cached_property
    def langfuse_handler(self):
        if not self.settings.langfuse_enabled:
            return None
        try:
            # Langfuse v3 API. Install it with: pip install "langfuse>=3,<4"
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            Langfuse(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
            return CallbackHandler()
        except Exception as exc:  # optional feature: never take the app down
            print(f"[config][providers] Langfuse disabled: {exc}")
            return None

    def callbacks(self) -> list:
        handler = self.langfuse_handler
        return [handler] if handler else []
