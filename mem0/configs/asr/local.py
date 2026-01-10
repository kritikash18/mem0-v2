from typing import Any, Callable, Dict, List, Optional, Union

from mem0.configs.asr.base import BaseAsrConfig


class LocalAsrConfig(BaseAsrConfig):
    """
    Configuration class for locally hosted ASR model parameters.
    Inherits from BaseAsrConfig and adds local/custom model-specific settings.
    
    This provides an open-ended interface for supporting custom ASR models
    like Wav2Vec2, HuBERT, Conformer, or any other locally hosted model.
    """

    def __init__(
        self,
        # Base parameters
        model: str = "facebook/wav2vec2-base-960h",
        api_key: Optional[str] = None,
        language: str = "en",
        sample_rate: int = 16000,
        http_client_proxies: Optional[Union[Dict, str]] = None,
        # Local model-specific parameters
        model_type: str = "wav2vec2",
        device: str = "auto",
        torch_dtype: str = "float32",
        processor_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        use_pipeline: bool = True,
        chunk_length_s: float = 30.0,
        stride_length_s: float = 5.0,
        batch_size: int = 1,
        return_timestamps: bool = False,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        # Custom model support
        custom_model_class: Optional[str] = None,
        custom_transcribe_fn: Optional[Callable] = None,
        # Server-based local ASR support
        server_url: Optional[str] = None,
        server_auth_token: Optional[str] = None,
        server_timeout: int = 300,
    ):
        """
        Initialize local ASR configuration.

        Args:
            model: HuggingFace model ID or local model path.
                Defaults to "facebook/wav2vec2-base-960h"
            api_key: API key (for server-based local ASR if needed).
                Defaults to None
            language: Language code for transcription.
                Defaults to "en"
            sample_rate: Audio sample rate in Hz.
                Defaults to 16000
            http_client_proxies: Proxy settings for HTTP client.
                Defaults to None
            model_type: Type of model.
                Options: 'wav2vec2', 'hubert', 'whisper', 'seamless', 'custom'.
                Defaults to "wav2vec2"
            device: Device for inference.
                Options: 'auto', 'cpu', 'cuda', 'cuda:0', 'mps'. Defaults to "auto"
            torch_dtype: Torch dtype.
                Options: 'float16', 'float32', 'bfloat16'. Defaults to "float32"
            processor_id: HuggingFace processor ID (if different from model).
                Defaults to None
            tokenizer_id: HuggingFace tokenizer ID (if different from model).
                Defaults to None
            use_pipeline: Whether to use HuggingFace pipeline for inference.
                Defaults to True
            chunk_length_s: Audio chunk length in seconds for long audio processing.
                Defaults to 30.0
            stride_length_s: Stride length in seconds for overlapping chunks.
                Defaults to 5.0
            batch_size: Batch size for inference.
                Defaults to 1
            return_timestamps: Whether to return word/chunk timestamps.
                Defaults to False
            generate_kwargs: Additional kwargs for model.generate() method.
                Defaults to None
            custom_model_class: Fully qualified class name for custom model
                (e.g., 'mypackage.models.CustomASR'). Defaults to None
            custom_transcribe_fn: Custom transcription function that takes
                (audio_array, sample_rate) and returns text. Defaults to None
            server_url: URL for locally hosted ASR server
                (e.g., 'http://localhost:8000/transcribe'). Defaults to None
            server_auth_token: Authentication token for local ASR server.
                Defaults to None
            server_timeout: Timeout in seconds for server requests.
                Defaults to 300
        """
        # Initialize base parameters
        super().__init__(
            model=model,
            api_key=api_key,
            language=language,
            sample_rate=sample_rate,
            enable_timestamps=return_timestamps,
            http_client_proxies=http_client_proxies,
        )

        # Local model-specific parameters
        self.model_type = model_type
        self.device = device
        self.torch_dtype = torch_dtype
        self.processor_id = processor_id
        self.tokenizer_id = tokenizer_id
        self.use_pipeline = use_pipeline
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.batch_size = batch_size
        self.return_timestamps = return_timestamps
        self.generate_kwargs = generate_kwargs

        # Custom model support
        self.custom_model_class = custom_model_class
        self.custom_transcribe_fn = custom_transcribe_fn

        # Server-based local ASR support
        self.server_url = server_url
        self.server_auth_token = server_auth_token
        self.server_timeout = server_timeout
