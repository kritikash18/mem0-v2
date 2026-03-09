import os
import logging
import importlib
from typing import Any, Dict, List, Optional, Union

import requests

from mem0.asr.base import ASRBase, AudioInput
from mem0.configs.asr.base import BaseAsrConfig
from mem0.configs.asr.local import LocalAsrConfig

logger = logging.getLogger(__name__)


class LocalASR(ASRBase):
    """
    Local ASR provider for self-hosted models.

    Supports:
        - HuggingFace Transformers models (Wav2Vec2, HuBERT, Whisper, etc.)
        - Custom model classes
        - Custom transcription functions
        - Local ASR servers via HTTP
    """

    def __init__(self, config: Optional[Union[BaseAsrConfig, LocalAsrConfig, Dict]] = None):
        # Convert to LocalAsrConfig if needed
        if config is None:
            config = LocalAsrConfig()
        elif isinstance(config, dict):
            config = LocalAsrConfig(**config)
        elif isinstance(config, BaseAsrConfig) and not isinstance(config, LocalAsrConfig):
            # Convert BaseAsrConfig to LocalAsrConfig
            config = LocalAsrConfig(
                model=config.model,
                language=config.language,
                sample_rate=config.sample_rate,
            )

        super().__init__(config)

        self._pipeline = None
        self._model = None
        self._processor = None
        self._custom_fn = None
        self._device = None
        self._torch_dtype = None

        self._init_model()

    def _init_model(self):
        """Initialize the ASR model based on configuration."""
        # Priority: custom function > server > custom class > HuggingFace pipeline

        if self.config.custom_transcribe_fn:
            self._custom_fn = self.config.custom_transcribe_fn
            logger.info("Using custom transcription function")
            return

        if self.config.server_url:
            logger.info(f"Using local ASR server at {self.config.server_url}")
            return

        if self.config.custom_model_class:
            self._init_custom_model()
            return

        if self.config.use_pipeline:
            self._init_pipeline()
        else:
            self._init_manual_model()

    def _init_pipeline(self):
        """Initialize HuggingFace pipeline."""
        try:
            from transformers import pipeline
            import torch
        except ImportError:
            raise ImportError(
                "The 'transformers' and 'torch' libraries are required. "
                "Install with: pip install transformers torch"
            )

        device = self.config.device
        if device == "auto":
            device = 0 if torch.cuda.is_available() else -1
        elif device == "cpu":
            device = -1
        elif device == "cuda":
            device = 0
        elif device.startswith("cuda:"):
            device = int(device.split(":")[1])
        elif device == "mps":
            device = "mps"

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float32)

        # Create pipeline
        pipeline_kwargs = {
            "task": "automatic-speech-recognition",
            "model": self.config.model,
            "device": device,
            "torch_dtype": torch_dtype,
        }

        is_ctc = self.config.model_type in ("wav2vec2", "hubert")

        if self.config.return_timestamps:
            # CTC models require 'char' or 'word', not True
            pipeline_kwargs["return_timestamps"] = "word" if is_ctc else self.config.return_timestamps
            pipeline_kwargs["chunk_length_s"] = self.config.chunk_length_s
            pipeline_kwargs["stride_length_s"] = (self.config.stride_length_s, self.config.stride_length_s)
        elif not is_ctc:
            # Seq2seq models (Whisper) support chunked long-form with timestamps
            pipeline_kwargs["chunk_length_s"] = self.config.chunk_length_s
            pipeline_kwargs["stride_length_s"] = (self.config.stride_length_s, self.config.stride_length_s)

        self._pipeline = pipeline(**pipeline_kwargs)
        logger.info(f"Initialized ASR pipeline with model: {self.config.model}")

    def _init_manual_model(self):
        """Initialize model and processor manually (without pipeline)."""
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCTC
            import torch
        except ImportError:
            raise ImportError(
                "The 'transformers' and 'torch' libraries are required. "
                "Install with: pip install transformers torch"
            )

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float32)

        processor_id = self.config.processor_id or self.config.model

        # Load processor
        self._processor = AutoProcessor.from_pretrained(processor_id)

        # Load model based on type
        model_type = self.config.model_type.lower()
        if model_type in ["whisper", "seamless"]:
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.config.model,
                torch_dtype=torch_dtype,
                device_map=device,
            )
        else:
            # CTC models like Wav2Vec2, HuBERT
            self._model = AutoModelForCTC.from_pretrained(
                self.config.model,
                torch_dtype=torch_dtype,
            ).to(device)

        self._device = device
        self._torch_dtype = torch_dtype
        logger.info(f"Initialized ASR model: {self.config.model}")

    def _init_custom_model(self):
        """Initialize custom model class."""
        class_path = self.config.custom_model_class
        module_path, class_name = class_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not import custom model class '{class_path}': {e}")

        # Instantiate the model with config
        self._model = model_class(self.config)
        logger.info(f"Initialized custom ASR model: {class_path}")

    def transcribe(
        self,
        audio: Union[str, bytes, "np.ndarray", Dict[str, Any], AudioInput],
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Transcribe audio using local ASR model.

        Args:
            audio: Audio input (file path, URL, bytes, numpy array, HuggingFace format, etc.)
            sample_rate: Sample rate for numpy array input. Defaults to None.
            **kwargs: Additional parameters.

        Returns:
            str: Transcribed text.
        """
        audio_input = self._prepare_audio(audio, sample_rate)

        try:
            # Route to appropriate method
            if self._custom_fn:
                return self._transcribe_custom_fn(audio_input, **kwargs)
            elif self.config.server_url:
                return self._transcribe_server(audio_input, **kwargs)
            elif self._pipeline:
                return self._transcribe_pipeline(audio_input, **kwargs)
            elif self._model:
                return self._transcribe_manual(audio_input, **kwargs)
            else:
                raise RuntimeError("No transcription method available")

        finally:
            audio_input.cleanup()

    def _transcribe_custom_fn(self, audio_input: AudioInput, **kwargs) -> str:
        """Transcribe using custom function."""
        audio_array, sample_rate = audio_input.get_audio_array()
        result = self._custom_fn(audio_array, sample_rate, **kwargs)

        if isinstance(result, str):
            return result
        elif isinstance(result, dict) and "text" in result:
            return result["text"]
        else:
            return str(result)

    def _transcribe_server(self, audio_input: AudioInput, **kwargs) -> str:
        """Transcribe using local ASR server."""
        # Get audio bytes
        audio_bytes = audio_input.get_audio_bytes()

        # Build request
        headers = {}
        if self.config.server_auth_token:
            headers["Authorization"] = f"Bearer {self.config.server_auth_token}"

        # Send as multipart form data
        files = {"audio": ("audio.wav", audio_bytes, "audio/wav")}
        data = {
            "language": self.config.language,
            **kwargs,
        }

        response = requests.post(
            self.config.server_url,
            files=files,
            data=data,
            headers=headers,
            timeout=self.config.server_timeout,
        )
        response.raise_for_status()

        result = response.json()

        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return result.get("text", result.get("transcription", str(result)))
        else:
            return str(result)

    def _transcribe_pipeline(self, audio_input: AudioInput, **kwargs) -> str:
        """Transcribe using HuggingFace pipeline."""
        import numpy as np

        # Get audio array
        audio_array, sample_rate = audio_input.get_audio_array()

        # Resample if needed
        target_sr = self.config.sample_rate
        if sample_rate != target_sr:
            audio_array = self._resample(audio_array, sample_rate, target_sr)

        # Prepare input
        inputs = {"array": audio_array, "sampling_rate": target_sr}

        # Build pipeline call kwargs
        pipeline_call_kwargs = {
            "batch_size": self.config.batch_size,
        }

        # generate_kwargs and language are only supported by seq2seq models (Whisper),
        # not CTC models (Wav2Vec2, HuBERT) which decode directly from logits
        is_ctc = self.config.model_type in ("wav2vec2", "hubert")
        if not is_ctc:
            generate_kwargs = self.config.generate_kwargs or {}
            if self.config.language:
                generate_kwargs["language"] = self.config.language
            generate_kwargs.update(kwargs.get("generate_kwargs", {}))
            if generate_kwargs:
                pipeline_call_kwargs["generate_kwargs"] = generate_kwargs

        # Run pipeline
        result = self._pipeline(inputs, **pipeline_call_kwargs)

        if isinstance(result, dict):
            return result.get("text", "").strip()
        elif isinstance(result, list):
            return " ".join(r.get("text", "") for r in result).strip()
        else:
            return str(result).strip()

    def _transcribe_manual(self, audio_input: AudioInput, **kwargs) -> str:
        """Transcribe using manually loaded model and processor."""
        import torch
        import numpy as np

        # Get audio array
        audio_array, sample_rate = audio_input.get_audio_array()

        # Resample if needed
        target_sr = self.config.sample_rate
        if sample_rate != target_sr:
            audio_array = self._resample(audio_array, sample_rate, target_sr)

        # Process input
        inputs = self._processor(
            audio_array,
            sampling_rate=target_sr,
            return_tensors="pt",
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        model_type = self.config.model_type.lower()

        if model_type in ["whisper", "seamless"]:
            # Seq2Seq models
            generate_kwargs = self.config.generate_kwargs or {}
            if self.config.language:
                generate_kwargs["language"] = self.config.language
            generate_kwargs.update(kwargs.get("generate_kwargs", {}))

            with torch.no_grad():
                generated_ids = self._model.generate(
                    inputs["input_features"],
                    **generate_kwargs,
                )

            transcription = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            return transcription[0].strip() if transcription else ""

        else:
            # CTC models
            with torch.no_grad():
                logits = self._model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self._processor.batch_decode(predicted_ids)
            return transcription[0].strip() if transcription else ""

    def _resample(self, audio: "np.ndarray", orig_sr: int, target_sr: int) -> "np.ndarray":
        """
        Resample audio to target sample rate.

        Args:
            audio: Audio array.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            np.ndarray: Resampled audio.
        """
        try:
            import librosa

            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            try:
                import torchaudio.functional as F
                import torch
                import numpy as np

                audio_tensor = torch.from_numpy(audio).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

                resampled = F.resample(audio_tensor, orig_sr, target_sr)
                return resampled.squeeze().numpy()
            except ImportError:
                raise ImportError(
                    "Either 'librosa' or 'torchaudio' is required for resampling. "
                    "Install with: pip install librosa  OR  pip install torchaudio"
                )

    def get_supported_languages(self) -> Optional[List[str]]:
        """
        Get list of supported languages (model-dependent).

        Returns:
            Optional[List[str]]: List of language codes, or None if unknown.
        """
        # This depends on the specific model being used
        if "whisper" in self.config.model.lower():
            # Return Whisper's language list
            return [
                "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
                "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
                "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
                "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
                "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
                "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru",
                "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
                "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
                "yi", "yo", "zh",
            ]

        # For other models, return None (language support is model-specific)
        return None
