import os
import logging
from typing import Any, Dict, List, Optional, Union

from mem0.asr.base import ASRBase, AudioInput
from mem0.configs.asr.base import BaseAsrConfig
from mem0.configs.asr.assemblyai import AssemblyAIConfig

logger = logging.getLogger(__name__)


class AssemblyAIASR(ASRBase):
    """
    AssemblyAI ASR provider.
    Provides high-accuracy transcription with additional features like
    speaker diarization, entity detection, and summarization.
    """

    def __init__(self, config: Optional[Union[BaseAsrConfig, AssemblyAIConfig, Dict]] = None):
        # Convert to AssemblyAIConfig if needed
        if config is None:
            config = AssemblyAIConfig()
        elif isinstance(config, dict):
            config = AssemblyAIConfig(**config)
        elif isinstance(config, BaseAsrConfig) and not isinstance(config, AssemblyAIConfig):
            # Convert BaseAsrConfig to AssemblyAIConfig
            config = AssemblyAIConfig(
                model=config.model if config.model in ["best", "nano"] else "best",
                api_key=config.api_key,
                language=config.language,
            )

        super().__init__(config)

        self._init_client()

    def _init_client(self):
        """Initialize AssemblyAI client."""
        try:
            import assemblyai as aai
        except ImportError:
            raise ImportError(
                "The 'assemblyai' library is required. "
                "Install with: pip install assemblyai"
            )

        api_key = self.config.api_key or os.getenv("ASSEMBLYAI_API_KEY")

        if not api_key:
            raise ValueError(
                "AssemblyAI API key is required. Set ASSEMBLYAI_API_KEY environment variable "
                "or provide api_key in config."
            )

        aai.settings.api_key = api_key
        self.aai = aai
        self.transcriber = aai.Transcriber()

    def _build_transcription_config(self, **kwargs):
        """
        Build transcription config from settings and kwargs.

        Args:
            **kwargs: Override parameters.

        Returns:
            TranscriptionConfig: AssemblyAI transcription configuration.
        """
        config_params = {
            "speech_model": (
                self.aai.SpeechModel.best
                if self.config.model == "best"
                else self.aai.SpeechModel.nano
            ),
            "punctuate": self.config.punctuate,
            "format_text": self.config.format_text,
            "speaker_labels": self.config.speaker_labels,
            "filter_profanity": self.config.filter_profanity,
            "redact_pii": self.config.redact_pii,
            "auto_chapters": self.config.auto_chapters,
            "entity_detection": self.config.entity_detection,
            "sentiment_analysis": self.config.sentiment_analysis,
        }

        # Language detection
        if self.config.language and self.config.language != "en":
            config_params["language_code"] = self.config.language
        else:
            config_params["language_detection"] = True

        # Speaker count
        if self.config.speakers_expected:
            config_params["speakers_expected"] = self.config.speakers_expected

        # Word boost
        if self.config.word_boost:
            config_params["word_boost"] = self.config.word_boost
            config_params["boost_param"] = self.config.boost_param

        # PII redaction policies
        if self.config.redact_pii and self.config.redact_pii_policies:
            config_params["redact_pii_policies"] = [
                getattr(self.aai.PIIRedactionPolicy, policy.lower())
                for policy in self.config.redact_pii_policies
            ]

        # Summarization
        if self.config.summarization:
            config_params["summarization"] = True
            config_params["summary_model"] = getattr(
                self.aai.SummarizationModel, self.config.summary_model
            )
            config_params["summary_type"] = getattr(
                self.aai.SummarizationType, self.config.summary_type
            )

        # Webhook
        if self.config.webhook_url:
            config_params["webhook_url"] = self.config.webhook_url

        # Override with kwargs
        config_params.update(kwargs)

        return self.aai.TranscriptionConfig(**config_params)

    def transcribe(
        self,
        audio: Union[str, bytes, Any, Dict[str, Any], AudioInput],
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe audio using AssemblyAI.

        Args:
            audio: Audio input (file path, URL, bytes, numpy array, etc.)
            sample_rate: Sample rate for numpy array input. Defaults to None.
            **kwargs: Additional parameters for TranscriptionConfig.

        Returns:
            str: Transcribed text.
        """
        audio_input = self._prepare_audio(audio, sample_rate)

        try:
            # Build config
            config = self._build_transcription_config(**kwargs)

            # Determine input type
            source_type = audio_input.source_type

            if source_type == "url":
                # Direct URL transcription
                transcript = self.transcriber.transcribe(
                    audio_input.source,
                    config=config,
                )
            elif source_type == "file":
                # Local file transcription
                transcript = self.transcriber.transcribe(
                    audio_input.source,
                    config=config,
                )
            else:
                # Upload audio bytes
                file_path = audio_input.get_temp_file_path(format="wav")
                transcript = self.transcriber.transcribe(
                    file_path,
                    config=config,
                )

            # Check for errors
            if transcript.status == self.aai.TranscriptStatus.error:
                raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

            return transcript.text or ""

        finally:
            audio_input.cleanup()

    def transcribe_with_details(
        self,
        audio: Union[str, bytes, Any, Dict[str, Any], AudioInput],
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Transcribe audio and return detailed results including metadata.

        Args:
            audio: Audio input.
            sample_rate: Sample rate.
            **kwargs: Additional parameters.

        Returns:
            Dict[str, Any]: Dict containing text, words, speakers, chapters, entities, etc.
        """
        audio_input = self._prepare_audio(audio, sample_rate)

        try:
            config = self._build_transcription_config(**kwargs)

            source_type = audio_input.source_type
            if source_type == "url":
                transcript = self.transcriber.transcribe(audio_input.source, config=config)
            elif source_type == "file":
                transcript = self.transcriber.transcribe(audio_input.source, config=config)
            else:
                file_path = audio_input.get_temp_file_path(format="wav")
                transcript = self.transcriber.transcribe(file_path, config=config)

            if transcript.status == self.aai.TranscriptStatus.error:
                raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

            result = {
                "text": transcript.text,
                "id": transcript.id,
                "audio_duration": transcript.audio_duration,
                "confidence": transcript.confidence,
            }

            # Add optional fields if available
            if transcript.words:
                result["words"] = [
                    {"text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence}
                    for w in transcript.words
                ]

            if transcript.utterances:
                result["utterances"] = [
                    {"speaker": u.speaker, "text": u.text, "start": u.start, "end": u.end}
                    for u in transcript.utterances
                ]

            if transcript.chapters:
                result["chapters"] = [
                    {"headline": c.headline, "summary": c.summary, "start": c.start, "end": c.end}
                    for c in transcript.chapters
                ]

            if transcript.entities:
                result["entities"] = [
                    {"text": e.text, "entity_type": e.entity_type, "start": e.start, "end": e.end}
                    for e in transcript.entities
                ]

            if transcript.summary:
                result["summary"] = transcript.summary

            if transcript.sentiment_analysis_results:
                result["sentiment"] = [
                    {"text": s.text, "sentiment": s.sentiment.value, "confidence": s.confidence}
                    for s in transcript.sentiment_analysis_results
                ]

            return result

        finally:
            audio_input.cleanup()

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.

        Returns:
            List[str]: Supported language codes.
        """
        return [
            "en", "en_au", "en_uk", "en_us",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "nl",
            "hi",
            "ja",
            "zh",
            "fi",
            "ko",
            "pl",
            "ru",
            "tr",
            "uk",
            "vi",
        ]
