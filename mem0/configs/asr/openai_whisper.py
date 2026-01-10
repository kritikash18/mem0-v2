from typing import Dict, Literal, Optional, Union

from mem0.configs.asr.base import BaseAsrConfig


class OpenAIWhisperConfig(BaseAsrConfig):
    """
    Configuration class for OpenAI Whisper-specific parameters.
    Inherits from BaseAsrConfig and adds Whisper-specific settings.
    
    Supports both OpenAI API and local Whisper models.
    """

    def __init__(
        self,
        # Base parameters
        model: str = "whisper-1",
        api_key: Optional[str] = None,
        language: str = "en",
        sample_rate: int = 16000,
        http_client_proxies: Optional[Union[Dict, str]] = None,
        # OpenAI Whisper-specific parameters
        use_local: bool = False,
        device: str = "auto",
        compute_type: str = "float16",
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "text",
        temperature: float = 0.0,
        prompt: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openrouter_base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI Whisper configuration.

        Args:
            model: Whisper model to use.
                For API: 'whisper-1'.
                For local: 'tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'.
                Defaults to "whisper-1"
            api_key: OpenAI API key (not needed for local models).
                Defaults to None
            language: Language code for transcription.
                Defaults to "en"
            sample_rate: Audio sample rate in Hz.
                Defaults to 16000
            http_client_proxies: Proxy settings for HTTP client.
                Defaults to None
            use_local: Whether to use local Whisper model instead of API.
                Defaults to False
            device: Device for local model inference.
                Options: 'auto', 'cpu', 'cuda', 'mps'. Defaults to "auto"
            compute_type: Compute type for local inference.
                Options: 'float16', 'float32', 'int8'. Defaults to "float16"
            response_format: Response format for transcription.
                Options: 'json', 'text', 'srt', 'verbose_json', 'vtt'. Defaults to "text"
            temperature: Sampling temperature for transcription.
                Range: 0.0 to 1.0. Defaults to 0.0
            prompt: Optional prompt to guide transcription style.
                Defaults to None
            openai_base_url: Custom base URL for OpenAI API.
                Defaults to None
            openrouter_base_url: Custom base URL for OpenRouter API.
                Defaults to None
        """
        # Initialize base parameters
        super().__init__(
            model=model,
            api_key=api_key,
            language=language,
            sample_rate=sample_rate,
            http_client_proxies=http_client_proxies,
        )

        # OpenAI Whisper-specific parameters
        self.use_local = use_local
        self.device = device
        self.compute_type = compute_type
        self.response_format = response_format
        self.temperature = temperature
        self.prompt = prompt
        self.openai_base_url = openai_base_url
        self.openrouter_base_url = openrouter_base_url
