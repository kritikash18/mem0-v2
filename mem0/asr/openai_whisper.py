import os
import logging
from typing import Any, Dict, List, Optional, Union

from mem0.asr.base import ASRBase, AudioInput
from mem0.configs.asr.base import BaseAsrConfig
from mem0.configs.asr.openai_whisper import OpenAIWhisperConfig

logger = logging.getLogger(__name__)


class OpenAIWhisperASR(ASRBase):
    """
    OpenAI Whisper ASR provider.
    Supports both OpenAI API and local Whisper models.
    """

    def __init__(self, config: Optional[Union[BaseAsrConfig, OpenAIWhisperConfig, Dict]] = None):
        # Convert to OpenAIWhisperConfig if needed
        if config is None:
            config = OpenAIWhisperConfig()
        elif isinstance(config, dict):
            config = OpenAIWhisperConfig(**config)
        elif isinstance(config, BaseAsrConfig) and not isinstance(config, OpenAIWhisperConfig):
            # Convert BaseAsrConfig to OpenAIWhisperConfig
            config = OpenAIWhisperConfig(
                model=config.model,
                api_key=config.api_key,
                language=config.language,
                sample_rate=config.sample_rate,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "whisper-1"

        if self.config.use_local:
            self._init_local_model()
        else:
            self._init_api_client()

    def _init_api_client(self):
        """Initialize OpenAI API client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' library is required for OpenAI Whisper API. "
                "Install with: pip install openai"
            )

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.config.openai_base_url or os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or provide api_key in config."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self._local_model = None
        self._use_faster_whisper = False

    def _init_local_model(self):
        """Initialize local Whisper model."""
        try:
            import whisper

            self._use_faster_whisper = False
        except ImportError:
            try:
                from faster_whisper import WhisperModel

                self._use_faster_whisper = True
            except ImportError:
                raise ImportError(
                    "Local Whisper requires either 'openai-whisper' or 'faster-whisper'. "
                    "Install with: pip install openai-whisper  OR  pip install faster-whisper"
                )

        device = self.config.device
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = self.config.model
        if model_name == "whisper-1":
            model_name = "base"  # Default for local

        if self._use_faster_whisper:
            from faster_whisper import WhisperModel

            self._local_model = WhisperModel(
                model_name,
                device=device,
                compute_type=self.config.compute_type,
            )
        else:
            import whisper

            self._local_model = whisper.load_model(model_name, device=device)

        self.client = None

    def transcribe(
        self,
        audio: Union[str, bytes, "np.ndarray", Dict[str, Any], AudioInput],
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe audio using OpenAI Whisper.

        Args:
            audio: Audio input (file path, URL, bytes, numpy array, etc.)
            sample_rate: Sample rate for numpy array input. Defaults to None.
            **kwargs: Additional parameters (language, prompt, temperature, etc.)

        Returns:
            str: Transcribed text.
        """
        audio_input = self._prepare_audio(audio, sample_rate)

        try:
            if self.config.use_local:
                return self._transcribe_local(audio_input, **kwargs)
            else:
                return self._transcribe_api(audio_input, **kwargs)
        finally:
            audio_input.cleanup()

    def _transcribe_api(self, audio_input: AudioInput, **kwargs) -> str:
        """Transcribe using OpenAI API."""
        # Get file path for API upload
        file_path = audio_input.get_temp_file_path(format="wav")

        with open(file_path, "rb") as audio_file:
            params = {
                "model": self.config.model,
                "file": audio_file,
                "response_format": kwargs.get("response_format", self.config.response_format),
            }

            # Add optional parameters
            if self.config.language:
                params["language"] = self.config.language
            if self.config.prompt:
                params["prompt"] = self.config.prompt
            if self.config.temperature > 0:
                params["temperature"] = self.config.temperature

            # Override with kwargs
            for key in ["language", "prompt", "temperature"]:
                if key in kwargs:
                    params[key] = kwargs[key]

            response = self.client.audio.transcriptions.create(**params)

        # Handle different response formats
        if isinstance(response, str):
            return response
        elif hasattr(response, "text"):
            return response.text
        else:
            return str(response)

    def _transcribe_local(self, audio_input: AudioInput, **kwargs) -> str:
        """Transcribe using local Whisper model."""
        if self._use_faster_whisper:
            return self._transcribe_faster_whisper(audio_input, **kwargs)
        else:
            return self._transcribe_openai_whisper(audio_input, **kwargs)

    def _transcribe_openai_whisper(self, audio_input: AudioInput, **kwargs) -> str:
        """Transcribe using openai-whisper library."""
        file_path = audio_input.get_temp_file_path(format="wav")

        params = {
            "fp16": self.config.compute_type == "float16",
        }

        if self.config.language and self.config.language != "auto":
            params["language"] = self.config.language
        if self.config.prompt:
            params["initial_prompt"] = self.config.prompt
        if self.config.temperature > 0:
            params["temperature"] = self.config.temperature

        # Override with kwargs
        params.update(kwargs)

        result = self._local_model.transcribe(file_path, **params)
        return result["text"].strip()

    def _transcribe_faster_whisper(self, audio_input: AudioInput, **kwargs) -> str:
        """Transcribe using faster-whisper library."""
        file_path = audio_input.get_temp_file_path(format="wav")

        params = {}

        if self.config.language and self.config.language != "auto":
            params["language"] = self.config.language
        if self.config.prompt:
            params["initial_prompt"] = self.config.prompt
        if self.config.temperature > 0:
            params["temperature"] = self.config.temperature

        # Override with kwargs
        params.update(kwargs)

        segments, _ = self._local_model.transcribe(file_path, **params)

        # Combine all segments
        text_parts = [segment.text for segment in segments]
        return " ".join(text_parts).strip()

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.

        Returns:
            List[str]: Whisper supports 99 languages.
        """
        return [
            "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
            "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
            "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
            "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
            "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
            "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru",
            "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
            "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
            "yi", "yo", "zh", "yue",
        ]
