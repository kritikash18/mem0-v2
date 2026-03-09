"""
Configuration for Audio Memory Evaluation.

Edit these settings to customize your evaluation.

Required Environment Variables:
    OPENAI_API_KEY: Required for OpenAI LLM, embedder, and ASR (whisper)
    HF_TOKEN: Optional, for private HuggingFace datasets

Required Python Packages:
    qdrant-client: For Qdrant vector storage (pip install qdrant-client)
    
Optional Environment Variables (if using other providers):
    ANTHROPIC_API_KEY: For Anthropic LLM
    ASSEMBLYAI_API_KEY: For AssemblyAI ASR
    GROQ_API_KEY: For Groq LLM
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# =============================================================================
# API KEYS (from environment or set directly)
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
HF_TOKEN = os.getenv("HF_TOKEN")

# Validate required API key
if not OPENAI_API_KEY:
    import warnings
    warnings.warn(
        "OPENAI_API_KEY not found in environment. "
        "Set it via: export OPENAI_API_KEY='your-key' or in .env file"
    )

# =============================================================================
# DATASET
# =============================================================================

DATASET_NAME = "byteCode18/spoken-squad-1k-memory-eval"
DATASET_SPLIT = "test"

# Original columns from Spoken SQuAD
AUDIO_COLUMN = "context"
QUESTION_COLUMN = "instruction"
ANSWER_COLUMN = "answer"

# =============================================================================
# ASR (Automatic Speech Recognition)
# =============================================================================

ASR_PROVIDER = "openai_whisper"  # Options: "openai_whisper", "speech_recognition_google", "assemblyai", "google_stt", "local"
ASR_MODEL = "whisper-1"          # whisper-1 | best/nano (AssemblyAI) | default/phone_call/video/command_and_search (Google STT v1) | openai/whisper-* (local)
ASR_MODEL_TYPE = "wav2vec2"      # Model architecture type for local ASR: "wav2vec2", "hubert", "whisper". Only used when ASR_PROVIDER="local"
ASR_LANGUAGE = None              # None = auto-detect (Whisper/AssemblyAI) or defaults to "en-US" (Google STT). Use "en-US", "es-ES", etc. for explicit language
ASR_ENABLE_CLEANUP = True        # Use LLM to clean ASR output

# =============================================================================
# LLM
# =============================================================================

LLM_PROVIDER = "openai"          # Options: "openai", "anthropic", "ollama", "groq", "together"
LLM_MODEL = "gpt-4o-mini"        # gpt-4o-mini | claude-3-5-sonnet-20241022 | llama3.2 | llama3.1:70b | etc.
LLM_TEMPERATURE = 0.0
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama server URL (only used when provider="ollama")

# LLM Judge — separate provider/model for evaluation scoring
# Defaults to the same as the main LLM; override to use a cheaper/different model for judging
LLM_JUDGE_PROVIDER = None        # None = same as LLM_PROVIDER. Options: "openai", "ollama", etc.
LLM_JUDGE_MODEL = None           # None = "gpt-4o-mini". Examples: "qwen2.5", "llama3.2", "gpt-4o"

# =============================================================================
# EMBEDDER
# =============================================================================

EMBEDDER_PROVIDER = "openai"
EMBEDDER_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536            # 1536 for text-embedding-3-small, 3072 for text-embedding-3-large

# =============================================================================
# VECTOR STORE
# =============================================================================

VECTOR_STORE_PROVIDER = "qdrant"  # Options: "faiss", "qdrant", "chroma"
COLLECTION_NAME = "audio_eval_memories"

# Qdrant local storage settings
QDRANT_PATH = "./qdrant_data"      # Local storage path for Qdrant database
QDRANT_ON_DISK = True              # True = persistent storage, False = in-memory (lost on restart)

# =============================================================================
# MEMORY SETTINGS
# =============================================================================

TOP_K = 10                        # Number of memories to retrieve
INFER_MEMORIES = True            # True = extract facts, False = store raw transcript
CLEANUP_AFTER_SAMPLE = True      # True = delete memories after each sample (isolated evaluation)
                                  # False = accumulate memories across samples (persistent evaluation)

# Use custom reading comprehension prompt for memory extraction
USE_READING_COMPREHENSION_PROMPT = True  # True = use custom RC prompt, False = use default mem0 prompt

# Maximum facts per audio passage (prompt will auto-consolidate if exceeded)
MAX_FACTS_PER_PASSAGE = 30  # Prevents too many memories from long audio

# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = "results/audio_eval"
EXPERIMENT_NAME = "default"


# =============================================================================
# CONFIG BUILDER (matches mem0 MemoryConfig structure)
# =============================================================================

def _get_asr_config() -> dict:
    """
    Build ASR configuration based on provider.
    
    Returns:
        dict: ASR configuration with provider, config, and enable_cleanup
    """
    asr_config = {
        "provider": ASR_PROVIDER,
        "config": {
            "model": ASR_MODEL,
            "language": ASR_LANGUAGE,
        },
        "enable_cleanup": ASR_ENABLE_CLEANUP,
    }
    
    # Set API key/credentials based on provider
    if ASR_PROVIDER == "openai_whisper":
        asr_config["config"]["api_key"] = OPENAI_API_KEY
    elif ASR_PROVIDER == "assemblyai":
        asr_config["config"]["api_key"] = ASSEMBLYAI_API_KEY
        # AssemblyAI supports "best" and "nano" models
        if ASR_MODEL not in ["best", "nano"]:
            import warnings
            warnings.warn(
                f"ASR_MODEL '{ASR_MODEL}' is not a standard AssemblyAI model. "
                f"Valid options: 'best', 'nano'. Defaulting to 'best'."
            )
            asr_config["config"]["model"] = "best"
    
    elif ASR_PROVIDER == "google_stt":
        # Google Cloud Speech-to-Text v1 uses credentials file
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path:
            asr_config["config"]["credentials_path"] = credentials_path
        
        # Google STT v1 requires a valid language code (doesn't support None/auto-detect)
        # Default to "en-US" if None
        if ASR_LANGUAGE is None:
            asr_config["config"]["language"] = "en-US"
            import warnings
            warnings.warn(
                "Google STT requires explicit language code. Defaulting to 'en-US'. "
                "Set ASR_LANGUAGE='es-ES' (or other BCP-47 code) for other languages."
            )
        
        # Valid models for v1 API: default, phone_call, video, command_and_search
        # Note: Use empty string or "default" for the default model
        valid_models = ["default", "", "phone_call", "video", "command_and_search"]
        if ASR_MODEL not in valid_models:
            import warnings
            warnings.warn(
                f"ASR_MODEL '{ASR_MODEL}' is not a standard Google STT v1 model. "
                f"Valid options: 'default', 'phone_call', 'video', 'command_and_search'. "
                f"Defaulting to 'default'."
            )
            asr_config["config"]["model"] = "default"
    
    elif ASR_PROVIDER == "local":
        asr_config["config"]["model_type"] = ASR_MODEL_TYPE
        asr_config["config"]["use_pipeline"] = True
        asr_config["config"]["device"] = "auto"
        # Local models don't need an API key
        asr_config["config"].pop("api_key", None)
    
    return asr_config


def get_mem0_config() -> dict:
    """
    Build configuration dictionary for mem0 Memory.from_config().
    
    This matches the structure expected by mem0.configs.base.MemoryConfig:
    - llm: LlmConfig with provider and config dict
    - embedder: EmbedderConfig with provider and config dict
    - vector_store: VectorStoreConfig with provider and config dict
    - asr: AsrConfig with provider, config dict, and enable_cleanup
    - custom_fact_extraction_prompt: Optional custom prompt for memory extraction
    - version: API version string
    
    Returns:
        Configuration dictionary compatible with Memory.from_config()
    """
    # Build LLM config based on provider
    llm_config = {
        "model": LLM_MODEL,
        "temperature": LLM_TEMPERATURE,
    }
    
    # Add provider-specific settings
    if LLM_PROVIDER == "openai":
        llm_config["api_key"] = OPENAI_API_KEY
    elif LLM_PROVIDER == "ollama":
        llm_config["ollama_base_url"] = OLLAMA_BASE_URL
    # Other providers (anthropic, groq, etc.) use API keys from environment
    
    config = {
        "llm": {
            "provider": LLM_PROVIDER,
            "config": llm_config,
        },
        "embedder": {
            "provider": EMBEDDER_PROVIDER,
            "config": {
                "model": EMBEDDER_MODEL,
                "embedding_dims": EMBEDDING_DIMS,
                "api_key": OPENAI_API_KEY,
            },
        },
        "vector_store": {
            "provider": VECTOR_STORE_PROVIDER,
            "config": {
                "collection_name": COLLECTION_NAME,
                "embedding_model_dims": EMBEDDING_DIMS,
                "path": QDRANT_PATH,           # Local storage path
                "on_disk": QDRANT_ON_DISK,     # Persistent storage
            },
        },
        "asr": _get_asr_config(),
        "version": "v1.1",
    }
    
    # Add custom reading comprehension prompts if enabled
    if USE_READING_COMPREHENSION_PROMPT:
        from .prompts import READING_COMPREHENSION_MEMORY_PROMPT, READING_COMPREHENSION_UPDATE_MEMORY_PROMPT
        config["custom_fact_extraction_prompt"] = READING_COMPREHENSION_MEMORY_PROMPT
        config["custom_update_memory_prompt"] = READING_COMPREHENSION_UPDATE_MEMORY_PROMPT
    
    return config


def get_output_path(experiment_name: str, suffix: str = "") -> str:
    """
    Get output file path for results.
    
    Args:
        experiment_name: Name of the experiment
        suffix: Optional suffix (e.g., 'intermediate', 'final')
        
    Returns:
        Full path to output JSON file
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{experiment_name}_{suffix}.json" if suffix else f"{experiment_name}.json"
    return os.path.join(OUTPUT_DIR, filename)


def validate_config():
    """
    Validate configuration before running evaluation.
    
    Raises:
        ValueError: If required configuration is missing
    """
    errors = []
    
    if not OPENAI_API_KEY and LLM_PROVIDER == "openai":
        errors.append("OPENAI_API_KEY required for OpenAI LLM provider")
    
    if LLM_PROVIDER == "ollama":
        # Check if Ollama is running
        try:
            import requests
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            if response.status_code != 200:
                errors.append(f"Ollama server not responding at {OLLAMA_BASE_URL}. Start with: ollama serve")
        except Exception as e:
            errors.append(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Ensure Ollama is running: ollama serve")
    
    if not OPENAI_API_KEY and EMBEDDER_PROVIDER == "openai":
        errors.append("OPENAI_API_KEY required for OpenAI embedder provider")
    
    if not OPENAI_API_KEY and ASR_PROVIDER == "openai_whisper":
        errors.append("OPENAI_API_KEY required for OpenAI Whisper ASR")
    
    if not ASSEMBLYAI_API_KEY and ASR_PROVIDER == "assemblyai":
        errors.append("ASSEMBLYAI_API_KEY required for AssemblyAI ASR. Set via: export ASSEMBLYAI_API_KEY='your-key'")
    
    if ASR_PROVIDER == "speech_recognition_google":
        try:
            import speech_recognition
        except ImportError:
            errors.append("SpeechRecognition library required for speech_recognition_google. Install via: pip install SpeechRecognition")
    
    if ASR_PROVIDER == "google_stt":
        if not GOOGLE_APPLICATION_CREDENTIALS:
            errors.append(
                "GOOGLE_APPLICATION_CREDENTIALS required for Google Cloud Speech-to-Text. "
                "Set via: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/credentials.json'"
            )
        try:
            import google.cloud.speech_v1
        except ImportError:
            errors.append("google-cloud-speech library required for google_stt. Install via: pip install google-cloud-speech")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True
