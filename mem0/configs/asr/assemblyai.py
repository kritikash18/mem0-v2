from typing import Dict, List, Literal, Optional, Union

from mem0.configs.asr.base import BaseAsrConfig


class AssemblyAIConfig(BaseAsrConfig):
    """
    Configuration class for AssemblyAI-specific parameters.
    Inherits from BaseAsrConfig and adds AssemblyAI-specific settings.
    """

    def __init__(
        self,
        # Base parameters
        model: Literal["best", "nano"] = "best",
        api_key: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: int = 16000,
        http_client_proxies: Optional[Union[Dict, str]] = None,
        # AssemblyAI-specific parameters
        punctuate: bool = True,
        format_text: bool = True,
        speaker_labels: bool = False,
        speakers_expected: Optional[int] = None,
        word_boost: Optional[List[str]] = None,
        boost_param: Literal["low", "default", "high"] = "default",
        filter_profanity: bool = False,
        redact_pii: bool = False,
        redact_pii_policies: Optional[List[str]] = None,
        auto_chapters: bool = False,
        entity_detection: bool = False,
        sentiment_analysis: bool = False,
        summarization: bool = False,
        summary_model: Literal["informative", "conversational", "catchy"] = "informative",
        summary_type: Literal["bullets", "bullets_verbose", "gist", "headline", "paragraph"] = "bullets",
        webhook_url: Optional[str] = None,
    ):
        """
        Initialize AssemblyAI configuration.

        Args:
            model: AssemblyAI model tier.
                Options: 'best' for highest accuracy, 'nano' for speed.
                Defaults to "best"
            api_key: AssemblyAI API key.
                Defaults to None
            language: Language code (None for auto-detection).
                Defaults to None
            sample_rate: Audio sample rate in Hz.
                Defaults to 16000
            http_client_proxies: Proxy settings for HTTP client.
                Defaults to None
            punctuate: Whether to add punctuation.
                Defaults to True
            format_text: Whether to format text (numbers, dates, etc.).
                Defaults to True
            speaker_labels: Whether to enable speaker diarization.
                Defaults to False
            speakers_expected: Expected number of speakers (improves diarization).
                Defaults to None
            word_boost: List of words/phrases to boost recognition.
                Defaults to None
            boost_param: Boost intensity for word_boost.
                Options: 'low', 'default', 'high'. Defaults to "default"
            filter_profanity: Whether to filter profanity from transcription.
                Defaults to False
            redact_pii: Whether to redact personally identifiable information.
                Defaults to False
            redact_pii_policies: PII types to redact (e.g., 'email_address', 'phone_number').
                Defaults to None
            auto_chapters: Whether to enable auto chapters.
                Defaults to False
            entity_detection: Whether to enable entity detection.
                Defaults to False
            sentiment_analysis: Whether to enable sentiment analysis.
                Defaults to False
            summarization: Whether to enable summarization.
                Defaults to False
            summary_model: Summary model style.
                Options: 'informative', 'conversational', 'catchy'. Defaults to "informative"
            summary_type: Summary output type.
                Options: 'bullets', 'bullets_verbose', 'gist', 'headline', 'paragraph'.
                Defaults to "bullets"
            webhook_url: Webhook URL for async transcription notifications.
                Defaults to None
        """
        # Initialize base parameters
        super().__init__(
            model=model,
            api_key=api_key,
            language=language or "en",
            sample_rate=sample_rate,
            enable_diarization=speaker_labels,
            http_client_proxies=http_client_proxies,
        )

        # AssemblyAI-specific parameters
        self.punctuate = punctuate
        self.format_text = format_text
        self.speaker_labels = speaker_labels
        self.speakers_expected = speakers_expected
        self.word_boost = word_boost
        self.boost_param = boost_param
        self.filter_profanity = filter_profanity
        self.redact_pii = redact_pii
        self.redact_pii_policies = redact_pii_policies
        self.auto_chapters = auto_chapters
        self.entity_detection = entity_detection
        self.sentiment_analysis = sentiment_analysis
        self.summarization = summarization
        self.summary_model = summary_model
        self.summary_type = summary_type
        self.webhook_url = webhook_url
