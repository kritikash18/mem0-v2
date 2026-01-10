from abc import ABC
from typing import Dict, List, Optional, Union

import httpx


class BaseAsrConfig(ABC):
    """
    Base configuration for ASR (Automatic Speech Recognition) providers.
    Provider-specific configurations should be handled by separate config classes.

    This class contains only the parameters that are common across all ASR providers.
    For provider-specific parameters, use the appropriate provider config class.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        language: str = "en",
        sample_rate: int = 16000,
        supported_formats: Optional[List[str]] = None,
        max_audio_duration: Optional[int] = None,
        enable_timestamps: bool = False,
        enable_diarization: bool = False,
        http_client_proxies: Optional[Union[Dict, str]] = None,
    ):
        """
        Initialize a base configuration class instance for ASR.

        Args:
            model: The ASR model identifier to use (e.g., "whisper-1", "facebook/wav2vec2-base-960h").
                Defaults to None (will be set by provider-specific configs)
            api_key: API key for the ASR provider. If None, will try to get from environment variables.
                Defaults to None
            language: Language code for transcription (e.g., 'en', 'es', 'fr', 'en-US').
                Defaults to "en"
            sample_rate: Audio sample rate in Hz.
                Common values: 8000, 16000, 22050, 44100, 48000. Defaults to 16000
            supported_formats: List of supported audio file formats.
                Defaults to ["wav", "mp3", "flac", "ogg", "m4a", "webm"]
            max_audio_duration: Maximum audio duration in seconds.
                None for no limit. Defaults to None
            enable_timestamps: Whether to include word-level timestamps in transcription.
                Defaults to False
            enable_diarization: Whether to enable speaker diarization (identify different speakers).
                Defaults to False
            http_client_proxies: Proxy settings for HTTP client.
                Can be a dict or string. Defaults to None
        """
        self.model = model
        self.api_key = api_key
        self.language = language
        self.sample_rate = sample_rate
        self.supported_formats = supported_formats or ["wav", "mp3", "flac", "ogg", "m4a", "webm"]
        self.max_audio_duration = max_audio_duration
        self.enable_timestamps = enable_timestamps
        self.enable_diarization = enable_diarization
        self.http_client = httpx.Client(proxies=http_client_proxies) if http_client_proxies else None
