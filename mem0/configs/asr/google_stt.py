from typing import Dict, List, Literal, Optional, Union

from mem0.configs.asr.base import BaseAsrConfig


class GoogleSTTConfig(BaseAsrConfig):
    """
    Configuration class for Google Cloud Speech-to-Text-specific parameters.
    Inherits from BaseAsrConfig and adds Google STT-specific settings.
    """

    def __init__(
        self,
        # Base parameters
        model: str = "latest_long",
        api_key: Optional[str] = None,
        language: str = "en-US",
        sample_rate: int = 16000,
        http_client_proxies: Optional[Union[Dict, str]] = None,
        # Google STT-specific parameters
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
        encoding: Literal[
            "ENCODING_UNSPECIFIED",
            "LINEAR16",
            "FLAC",
            "MULAW",
            "AMR",
            "AMR_WB",
            "OGG_OPUS",
            "SPEEX_WITH_HEADER_BYTE",
            "MP3",
            "WEBM_OPUS"
        ] = "LINEAR16",
        enable_automatic_punctuation: bool = True,
        enable_word_time_offsets: bool = False,
        enable_speaker_diarization: bool = False,
        diarization_speaker_count: int = 2,
        speech_contexts: Optional[List[dict]] = None,
        use_enhanced: bool = False,
    ):
        """
        Initialize Google Cloud Speech-to-Text configuration.

        Args:
            model: Google STT model.
                Options: 'latest_long', 'latest_short', 'phone_call', 'video', 'command_and_search'.
                Defaults to "latest_long"
            api_key: API key (alternative to credentials file).
                Defaults to None
            language: BCP-47 language code (e.g., 'en-US', 'es-ES').
                Defaults to "en-US"
            sample_rate: Audio sample rate in Hz.
                Defaults to 16000
            http_client_proxies: Proxy settings for HTTP client.
                Defaults to None
            credentials_path: Path to Google Cloud credentials JSON file.
                Defaults to None
            project_id: Google Cloud project ID.
                Defaults to None
            encoding: Audio encoding format.
                Defaults to "LINEAR16"
            enable_automatic_punctuation: Whether to add punctuation to transcription.
                Defaults to True
            enable_word_time_offsets: Whether to include word-level timestamps.
                Defaults to False
            enable_speaker_diarization: Whether to enable speaker diarization.
                Defaults to False
            diarization_speaker_count: Expected number of speakers for diarization.
                Defaults to 2
            speech_contexts: Speech contexts for improved recognition (phrases, boost values).
                Defaults to None
            use_enhanced: Whether to use enhanced models (premium feature).
                Defaults to False
        """
        # Initialize base parameters
        super().__init__(
            model=model,
            api_key=api_key,
            language=language,
            sample_rate=sample_rate,
            enable_timestamps=enable_word_time_offsets,
            enable_diarization=enable_speaker_diarization,
            http_client_proxies=http_client_proxies,
        )

        # Google STT-specific parameters
        self.credentials_path = credentials_path
        self.project_id = project_id
        self.encoding = encoding
        self.enable_automatic_punctuation = enable_automatic_punctuation
        self.enable_word_time_offsets = enable_word_time_offsets
        self.enable_speaker_diarization = enable_speaker_diarization
        self.diarization_speaker_count = diarization_speaker_count
        self.speech_contexts = speech_contexts
        self.use_enhanced = use_enhanced
