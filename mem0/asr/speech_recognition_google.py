import logging
from typing import Any, Dict, List, Optional, Union

from mem0.asr.base import ASRBase, AudioInput
from mem0.configs.asr.base import BaseAsrConfig

logger = logging.getLogger(__name__)


class SpeechRecognitionGoogleASR(ASRBase):
    """
    Simple Google Speech Recognition ASR provider using the speech_recognition library.
    
    This uses the free Google Speech Recognition API (no credentials required).
    Note: This is different from Google Cloud Speech-to-Text (google_stt).
    
    Advantages:
    - No API key or credentials required
    - Easy to set up
    - Good for testing and prototyping
    
    Limitations:
    - Rate limited (50 requests per day per IP)
    - Less features than Google Cloud STT
    - Not suitable for production
    """

    def __init__(self, config: Optional[Union[BaseAsrConfig, Dict]] = None):
        if config is None:
            config = BaseAsrConfig(model="google")
        elif isinstance(config, dict):
            config = BaseAsrConfig(**config)

        super().__init__(config)
        self._init_recognizer()

    def _init_recognizer(self):
        """Initialize speech recognition client."""
        try:
            import speech_recognition as sr
        except ImportError:
            raise ImportError(
                "The 'SpeechRecognition' library is required. "
                "Install with: pip install SpeechRecognition"
            )

        self.sr = sr
        self.recognizer = sr.Recognizer()
        
        # Optional: Adjust recognizer settings for better accuracy
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

    def transcribe(
        self,
        audio: Union[str, bytes, Any, Dict[str, Any], AudioInput],
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe audio using Google Speech Recognition (free API).

        Args:
            audio: Audio input (file path, URL, bytes, numpy array, etc.)
            sample_rate: Sample rate for numpy array input. Defaults to None.
            **kwargs: Additional parameters:
                - language: Language code (default: from config)
                - show_all: Return all alternatives (default: False)

        Returns:
            str: Transcribed text.
        """
        audio_input = self._prepare_audio(audio, sample_rate)

        try:
            # Get audio file path (speech_recognition works with files)
            if audio_input.source_type == "file":
                audio_file = audio_input.source
            else:
                # Convert to wav file for speech_recognition
                audio_file = audio_input.get_temp_file_path(format="wav")

            # Load audio file
            with self.sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record audio from file
                audio_data = self.recognizer.record(source)

            # Perform recognition
            language = kwargs.get("language", self.config.language)
            show_all = kwargs.get("show_all", False)
            
            try:
                result = self.recognizer.recognize_google(
                    audio_data,
                    language=language,
                    show_all=show_all
                )
                
                if show_all:
                    # Return first alternative if available
                    if result and "alternative" in result:
                        return result["alternative"][0].get("transcript", "")
                    return ""
                else:
                    return result or ""
                    
            except self.sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
                return ""
            except self.sr.RequestError as e:
                logger.error(f"Could not request results from Google Speech Recognition: {e}")
                raise RuntimeError(f"Google Speech Recognition API error: {e}")

        finally:
            audio_input.cleanup()

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Google Speech Recognition supports many languages.
        See: https://cloud.google.com/speech-to-text/docs/languages

        Returns:
            List[str]: Common supported language codes.
        """
        return [
            "en-US", "en-GB", "en-AU", "en-CA", "en-IN", "en-NZ",
            "es-ES", "es-MX", "es-AR", "es-CO",
            "fr-FR", "fr-CA",
            "de-DE",
            "it-IT",
            "pt-BR", "pt-PT",
            "ja-JP",
            "ko-KR",
            "zh-CN", "zh-TW", "zh-HK",
            "ar-SA",
            "hi-IN",
            "ru-RU",
            "nl-NL",
            "pl-PL",
            "tr-TR",
            "vi-VN",
            "th-TH",
            "id-ID",
            "sv-SE",
            "da-DK",
            "fi-FI",
            "no-NO",
            "cs-CZ",
            "el-GR",
            "he-IL",
            "uk-UA",
        ]

