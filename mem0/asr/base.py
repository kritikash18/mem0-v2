import io
import os
import re
import base64
import logging
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

from mem0.configs.asr.base import BaseAsrConfig

logger = logging.getLogger(__name__)


class AudioInput:
    """
    Represents audio input from various sources.
    Handles conversion between different audio formats for ASR processing.

    Supports:
        - File paths (local files)
        - URLs (remote files)
        - Base64 encoded audio
        - Raw audio bytes
        - NumPy arrays with sample rate
        - HuggingFace datasets audio format
    """

    def __init__(
        self,
        source: Union[str, bytes, "np.ndarray", Dict[str, Any]],
        sample_rate: Optional[int] = None,
        format: Optional[str] = None,
    ):
        """Initialize an AudioInput instance.

        :param source: Audio source (file path, URL, bytes, numpy array, or dict)
        :type source: Union[str, bytes, np.ndarray, Dict[str, Any]]
        :param sample_rate: Audio sample rate in Hz, defaults to None
        :type sample_rate: Optional[int], optional
        :param format: Audio format (e.g., 'wav', 'mp3'), defaults to None
        :type format: Optional[str], optional
        """
        self.source = source
        self.sample_rate = sample_rate
        self.format = format
        self._audio_bytes: Optional[bytes] = None
        self._audio_array: Optional["np.ndarray"] = None
        self._temp_file: Optional[str] = None

    @property
    def source_type(self) -> str:
        """
        Determine the type of audio source.

        Returns:
            str: Source type ('url', 'file', 'base64', 'bytes', 'numpy', 'huggingface')
        """
        if isinstance(self.source, str):
            if self.source.startswith(("http://", "https://")):
                return "url"
            elif self.source.startswith("data:audio"):
                return "base64"
            elif os.path.isfile(self.source):
                return "file"
            else:
                # Assume it's base64 without data URI prefix
                try:
                    base64.b64decode(self.source)
                    return "base64_raw"
                except Exception:
                    raise ValueError(f"Invalid audio source: {self.source}")
        elif isinstance(self.source, bytes):
            return "bytes"
        elif isinstance(self.source, dict):
            if "array" in self.source and "sampling_rate" in self.source:
                return "huggingface"
            elif "audio" in self.source:
                return "huggingface_nested"
            else:
                raise ValueError(f"Invalid dict audio source format: {self.source.keys()}")
        else:
            # Assume numpy array
            try:
                import numpy as np

                if isinstance(self.source, np.ndarray):
                    return "numpy"
            except ImportError:
                pass
            raise ValueError(f"Unsupported audio source type: {type(self.source)}")

    def get_audio_bytes(self) -> bytes:
        """
        Get audio as bytes, downloading or converting if necessary.

        Returns:
            bytes: Audio content as bytes
        """
        if self._audio_bytes is not None:
            return self._audio_bytes

        source_type = self.source_type

        if source_type == "file":
            with open(self.source, "rb") as f:
                self._audio_bytes = f.read()

        elif source_type == "url":
            response = requests.get(self.source, timeout=60)
            response.raise_for_status()
            self._audio_bytes = response.content

        elif source_type == "base64":
            # Extract base64 data from data URI
            match = re.match(r"data:audio/[^;]+;base64,(.+)", self.source)
            if match:
                self._audio_bytes = base64.b64decode(match.group(1))
            else:
                raise ValueError("Invalid base64 data URI format")

        elif source_type == "base64_raw":
            self._audio_bytes = base64.b64decode(self.source)

        elif source_type == "bytes":
            self._audio_bytes = self.source

        elif source_type in ("numpy", "huggingface", "huggingface_nested"):
            # Convert numpy array to WAV bytes
            self._audio_bytes = self._numpy_to_wav_bytes()

        else:
            raise ValueError(f"Cannot get bytes for source type: {source_type}")

        return self._audio_bytes

    def get_audio_array(self) -> Tuple["np.ndarray", int]:
        """
        Get audio as numpy array with sample rate.

        Returns:
            Tuple[np.ndarray, int]: Audio array and sample rate
        """
        import numpy as np

        if self._audio_array is not None:
            return self._audio_array, self.sample_rate

        source_type = self.source_type

        if source_type == "numpy":
            self._audio_array = self.source
            if self.sample_rate is None:
                raise ValueError("sample_rate must be provided for numpy array input")
            return self._audio_array, self.sample_rate

        elif source_type == "huggingface":
            self._audio_array = np.array(self.source["array"])
            self.sample_rate = self.source["sampling_rate"]
            return self._audio_array, self.sample_rate

        elif source_type == "huggingface_nested":
            audio_data = self.source["audio"]
            self._audio_array = np.array(audio_data["array"])
            self.sample_rate = audio_data["sampling_rate"]
            return self._audio_array, self.sample_rate

        else:
            # Load from bytes using soundfile or librosa
            audio_bytes = self.get_audio_bytes()
            try:
                import soundfile as sf

                audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
                self._audio_array = audio_array
                self.sample_rate = sample_rate
            except ImportError:
                try:
                    import librosa

                    # Write to temp file and read with librosa
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        f.write(audio_bytes)
                        temp_path = f.name
                    audio_array, sample_rate = librosa.load(temp_path, sr=self.sample_rate)
                    os.unlink(temp_path)
                    self._audio_array = audio_array
                    self.sample_rate = sample_rate
                except ImportError:
                    raise ImportError(
                        "Either 'soundfile' or 'librosa' is required for audio processing. "
                        "Install with: pip install soundfile librosa"
                    )

            return self._audio_array, self.sample_rate

    def get_temp_file_path(self, format: str = "wav") -> str:
        """
        Get a temporary file path with the audio content.

        Args:
            format: Audio format for the temp file. Defaults to "wav".

        Returns:
            str: Path to temporary audio file
        """
        if self._temp_file is not None and os.path.exists(self._temp_file):
            return self._temp_file

        source_type = self.source_type

        # If already a file with correct format, return the path
        if source_type == "file":
            file_ext = os.path.splitext(self.source)[1].lower().lstrip(".")
            if file_ext == format:
                return self.source

        # Otherwise, create temp file
        audio_bytes = self.get_audio_bytes()

        with tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False) as f:
            f.write(audio_bytes)
            self._temp_file = f.name

        return self._temp_file

    def _numpy_to_wav_bytes(self) -> bytes:
        """
        Convert numpy array to WAV bytes.

        Returns:
            bytes: WAV audio data
        """
        import numpy as np

        source_type = self.source_type

        if source_type == "numpy":
            audio_array = self.source
            sample_rate = self.sample_rate
        elif source_type == "huggingface":
            audio_array = np.array(self.source["array"])
            sample_rate = self.source["sampling_rate"]
        elif source_type == "huggingface_nested":
            audio_data = self.source["audio"]
            audio_array = np.array(audio_data["array"])
            sample_rate = audio_data["sampling_rate"]
        else:
            raise ValueError(f"Cannot convert {source_type} to numpy")

        if sample_rate is None:
            raise ValueError("sample_rate is required for numpy array conversion")

        try:
            import soundfile as sf

            buffer = io.BytesIO()
            sf.write(buffer, audio_array, sample_rate, format="WAV")
            return buffer.getvalue()
        except ImportError:
            # Fallback to scipy
            try:
                from scipy.io import wavfile

                buffer = io.BytesIO()
                # Ensure audio is in correct format
                if audio_array.dtype != np.int16:
                    audio_array = (audio_array * 32767).astype(np.int16)
                wavfile.write(buffer, sample_rate, audio_array)
                return buffer.getvalue()
            except ImportError:
                raise ImportError(
                    "Either 'soundfile' or 'scipy' is required for audio conversion. "
                    "Install with: pip install soundfile scipy"
                )

    def cleanup(self):
        """Clean up temporary files."""
        if self._temp_file and os.path.exists(self._temp_file):
            try:
                os.unlink(self._temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")
            self._temp_file = None


class ASRBase(ABC):
    """
    Base class for all ASR (Automatic Speech Recognition) providers.
    Handles common functionality and delegates provider-specific logic to subclasses.
    """

    def __init__(self, config: Optional[Union[BaseAsrConfig, Dict]] = None):
        """Initialize a base ASR class.

        :param config: ASR configuration option class or dict, defaults to None
        :type config: Optional[Union[BaseAsrConfig, Dict]], optional
        """
        if config is None:
            self.config = BaseAsrConfig()
        elif isinstance(config, dict):
            # Handle dict-based configuration (backward compatibility)
            self.config = BaseAsrConfig(**config)
        else:
            self.config = config

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """
        Validate the configuration.
        Override in subclasses to add provider-specific validation.
        """
        if not hasattr(self.config, "model"):
            raise ValueError("Configuration must have a 'model' attribute")

    @abstractmethod
    def transcribe(
        self,
        audio: Union[str, bytes, "np.ndarray", Dict[str, Any], AudioInput],
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio input - can be:
                - str: File path, URL, or base64 encoded audio
                - bytes: Raw audio bytes
                - np.ndarray: Audio samples as numpy array
                - dict: HuggingFace audio format {"array": [...], "sampling_rate": 16000}
                - AudioInput: Pre-processed audio input object
            sample_rate: Sample rate in Hz (required for numpy arrays). Defaults to None.
            **kwargs: Additional provider-specific parameters.

        Returns:
            str: Transcribed text.
        """
        pass

    def _prepare_audio(
        self,
        audio: Union[str, bytes, "np.ndarray", Dict[str, Any], AudioInput],
        sample_rate: Optional[int] = None,
    ) -> AudioInput:
        """
        Prepare audio input for transcription.

        Args:
            audio: Various audio input formats.
            sample_rate: Optional sample rate.

        Returns:
            AudioInput: Prepared audio input object.
        """
        if isinstance(audio, AudioInput):
            return audio

        return AudioInput(
            source=audio,
            sample_rate=sample_rate or self.config.sample_rate,
        )

    def transcribe_batch(
        self,
        audio_list: List[Union[str, bytes, "np.ndarray", Dict[str, Any], AudioInput]],
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """
        Transcribe multiple audio inputs.

        Default implementation calls transcribe() for each input.
        Override for providers that support batch processing.

        Args:
            audio_list: List of audio inputs.
            sample_rate: Default sample rate.
            **kwargs: Additional parameters.

        Returns:
            List[str]: List of transcribed texts.
        """
        results = []
        for audio in audio_list:
            text = self.transcribe(audio, sample_rate=sample_rate, **kwargs)
            results.append(text)
        return results

    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported audio formats.

        Returns:
            List[str]: Supported audio formats.
        """
        return self.config.supported_formats

    def get_supported_languages(self) -> Optional[List[str]]:
        """
        Get list of supported languages.
        Override in subclasses to return provider-specific language list.

        Returns:
            Optional[List[str]]: List of supported language codes, or None.
        """
        return None
