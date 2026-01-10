import os
import logging
from typing import Any, Dict, List, Optional, Union

from mem0.asr.base import ASRBase, AudioInput
from mem0.configs.asr.base import BaseAsrConfig
from mem0.configs.asr.google_stt import GoogleSTTConfig

logger = logging.getLogger(__name__)


class GoogleSTTASR(ASRBase):
    """
    Google Cloud Speech-to-Text ASR provider.
    """

    def __init__(self, config: Optional[Union[BaseAsrConfig, GoogleSTTConfig, Dict]] = None):
        # Convert to GoogleSTTConfig if needed
        if config is None:
            config = GoogleSTTConfig()
        elif isinstance(config, dict):
            config = GoogleSTTConfig(**config)
        elif isinstance(config, BaseAsrConfig) and not isinstance(config, GoogleSTTConfig):
            # Convert BaseAsrConfig to GoogleSTTConfig
            config = GoogleSTTConfig(
                model=config.model,
                language=config.language,
                sample_rate=config.sample_rate,
            )

        super().__init__(config)

        self._init_client()

    def _init_client(self):
        """Initialize Google Cloud Speech client."""
        try:
            from google.cloud import speech_v1 as speech
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "The 'google-cloud-speech' library is required. "
                "Install with: pip install google-cloud-speech"
            )

        # Set up credentials
        if self.config.credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                self.config.credentials_path
            )
            self.client = speech.SpeechClient(credentials=credentials)
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            self.client = speech.SpeechClient()
        else:
            raise ValueError(
                "Google Cloud credentials are required. Either set GOOGLE_APPLICATION_CREDENTIALS "
                "environment variable or provide credentials_path in config."
            )

        self.speech = speech

    def _get_encoding(self, format: str):
        """
        Map file format to Google Speech encoding.

        Args:
            format: Audio file format.

        Returns:
            Google Speech encoding enum value.
        """
        encoding_map = {
            "wav": self.speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "flac": self.speech.RecognitionConfig.AudioEncoding.FLAC,
            "mp3": self.speech.RecognitionConfig.AudioEncoding.MP3,
            "ogg": self.speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            "webm": self.speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            "amr": self.speech.RecognitionConfig.AudioEncoding.AMR,
            "amr_wb": self.speech.RecognitionConfig.AudioEncoding.AMR_WB,
        }
        return encoding_map.get(format.lower(), self.speech.RecognitionConfig.AudioEncoding.LINEAR16)

    def transcribe(
        self,
        audio: Union[str, bytes, "np.ndarray", Dict[str, Any], AudioInput],
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe audio using Google Cloud Speech-to-Text.

        Args:
            audio: Audio input (file path, URL, bytes, numpy array, etc.)
            sample_rate: Sample rate for numpy array input. Defaults to None.
            **kwargs: Additional parameters.

        Returns:
            str: Transcribed text.
        """
        audio_input = self._prepare_audio(audio, sample_rate)

        try:
            # Get audio content
            audio_bytes = audio_input.get_audio_bytes()

            # Determine encoding from source
            source_type = audio_input.source_type
            if source_type == "file":
                ext = os.path.splitext(audio_input.source)[1].lower().lstrip(".")
                encoding = self._get_encoding(ext)
            else:
                encoding = self._get_encoding("wav")

            # Create recognition config
            config = self.speech.RecognitionConfig(
                encoding=encoding,
                sample_rate_hertz=audio_input.sample_rate or self.config.sample_rate,
                language_code=kwargs.get("language", self.config.language),
                enable_automatic_punctuation=self.config.enable_automatic_punctuation,
                enable_word_time_offsets=self.config.enable_word_time_offsets,
                model=self.config.model,
                use_enhanced=self.config.use_enhanced,
            )

            # Add diarization config if enabled
            if self.config.enable_speaker_diarization:
                config.diarization_config = self.speech.SpeakerDiarizationConfig(
                    enable_speaker_diarization=True,
                    min_speaker_count=1,
                    max_speaker_count=self.config.diarization_speaker_count,
                )

            # Add speech contexts if provided
            if self.config.speech_contexts:
                config.speech_contexts = [
                    self.speech.SpeechContext(**ctx) for ctx in self.config.speech_contexts
                ]

            # Create audio object
            audio_obj = self.speech.RecognitionAudio(content=audio_bytes)

            # Perform recognition
            response = self.client.recognize(config=config, audio=audio_obj)

            # Extract text from results
            texts = []
            for result in response.results:
                if result.alternatives:
                    texts.append(result.alternatives[0].transcript)

            return " ".join(texts).strip()

        finally:
            audio_input.cleanup()

    def transcribe_long_audio(
        self,
        audio: Union[str, bytes, "np.ndarray", Dict[str, Any], AudioInput],
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe long audio files (>1 minute) using async recognition.
        Requires the audio to be uploaded to Google Cloud Storage.

        Args:
            audio: Audio input - must be a GCS URI (gs://bucket/file.wav)
            sample_rate: Sample rate.
            **kwargs: Additional parameters.

        Returns:
            str: Transcribed text.
        """
        if not isinstance(audio, str) or not audio.startswith("gs://"):
            raise ValueError(
                "Long audio transcription requires audio to be uploaded to Google Cloud Storage. "
                "Provide a GCS URI like 'gs://bucket-name/audio-file.wav'"
            )

        # Determine encoding from URI
        ext = os.path.splitext(audio)[1].lower().lstrip(".")
        encoding = self._get_encoding(ext)

        # Create recognition config
        config = self.speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=sample_rate or self.config.sample_rate,
            language_code=kwargs.get("language", self.config.language),
            enable_automatic_punctuation=self.config.enable_automatic_punctuation,
            enable_word_time_offsets=self.config.enable_word_time_offsets,
            model=self.config.model,
            use_enhanced=self.config.use_enhanced,
        )

        # Create audio object with GCS URI
        audio_obj = self.speech.RecognitionAudio(uri=audio)

        # Perform async recognition
        operation = self.client.long_running_recognize(config=config, audio=audio_obj)

        logger.info("Waiting for long audio transcription to complete...")
        response = operation.result(timeout=kwargs.get("timeout", 3600))

        # Extract text from results
        texts = []
        for result in response.results:
            if result.alternatives:
                texts.append(result.alternatives[0].transcript)

        return " ".join(texts).strip()

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages (BCP-47 codes).

        Returns:
            List[str]: Major supported language codes.
        """
        return [
            "en-US", "en-GB", "en-AU", "en-IN",
            "es-ES", "es-MX", "es-US",
            "fr-FR", "fr-CA",
            "de-DE", "de-AT", "de-CH",
            "it-IT",
            "pt-BR", "pt-PT",
            "ja-JP",
            "ko-KR",
            "zh-CN", "zh-TW", "zh-HK",
            "ar-SA", "ar-EG",
            "hi-IN",
            "ru-RU",
            "nl-NL",
            "pl-PL",
            "tr-TR",
            "vi-VN",
            "th-TH",
            "id-ID",
            "ms-MY",
            "sv-SE",
            "da-DK",
            "fi-FI",
            "no-NO",
            "cs-CZ",
            "el-GR",
            "he-IL",
            "uk-UA",
            "ro-RO",
            "hu-HU",
            "bg-BG",
            "hr-HR",
            "sk-SK",
            "sl-SI",
            "lt-LT",
            "lv-LV",
            "et-EE",
        ]
