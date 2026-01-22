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

ASR_PROVIDER = "openai_whisper"  # Options: "openai_whisper", "local", "assemblyai", "google_stt"
ASR_MODEL = "whisper-1"          # For local: "openai/whisper-base", "openai/whisper-large-v3"
ASR_LANGUAGE = None              # None = auto-detect, or "en", "es", etc.
ASR_ENABLE_CLEANUP = True        # Use LLM to clean ASR output

# =============================================================================
# LLM
# =============================================================================

LLM_PROVIDER = "openai"          # Options: "openai", "anthropic", "ollama", "groq", "together"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0

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
    config = {
        "llm": {
            "provider": LLM_PROVIDER,
            "config": {
                "model": LLM_MODEL,
                "temperature": LLM_TEMPERATURE,
                "api_key": OPENAI_API_KEY,  # Explicit, falls back to env var in mem0
            },
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
        "asr": {
            "provider": ASR_PROVIDER,
            "config": {
                "model": ASR_MODEL,
                "language": ASR_LANGUAGE,
                "api_key": OPENAI_API_KEY,
            },
            "enable_cleanup": ASR_ENABLE_CLEANUP,
        },
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
    
    if not OPENAI_API_KEY and EMBEDDER_PROVIDER == "openai":
        errors.append("OPENAI_API_KEY required for OpenAI embedder provider")
    
    if not OPENAI_API_KEY and ASR_PROVIDER == "openai_whisper":
        errors.append("OPENAI_API_KEY required for OpenAI Whisper ASR")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True
