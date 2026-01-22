from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AsrConfig(BaseModel):
    """
    Configuration for ASR (Automatic Speech Recognition) in Memory.
    
    Includes options for post-processing ASR output using an LLM to clean
    the transcription before use in memory operations.
    """
    
    provider: str = Field(
        description="Provider of the ASR (e.g., 'openai_whisper', 'google_stt', 'assemblyai', 'local')",
        default="openai_whisper"
    )
    config: Optional[dict] = Field(
        description="Configuration for the specific ASR provider",
        default={}
    )
    
    # LLM-based cleanup options
    enable_cleanup: bool = Field(
        description="Whether to use LLM to clean ASR output before use",
        default=True
    )
    cleanup_prompt: Optional[str] = Field(
        description="Custom prompt for cleaning ASR output for memory ingestion. "
                    "If None, uses default ASR_CLEANUP_PROMPT.",
        default=None
    )
    query_cleanup_prompt: Optional[str] = Field(
        description="Custom prompt for cleaning ASR output for search queries. "
                    "If None, uses default ASR_QUERY_CLEANUP_PROMPT.",
        default=None
    )

    @field_validator("config")
    def validate_config(cls, v, values):
        provider = values.data.get("provider")
        supported_providers = (
            "openai_whisper",
            "google_stt",
            "assemblyai",
            "local",
        )
        if provider in supported_providers:
            return v
        else:
            raise ValueError(
                f"Unsupported ASR provider: {provider}. "
                f"Supported providers: {supported_providers}"
            )

