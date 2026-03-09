"""
Configuration for Cognee Memory Evaluation.

Cognee is a competing AI memory framework that organizes data into a
knowledge graph with embeddings. Audio is passed directly to Cognee as
a .wav file — Cognee handles transcription internally.

Required Environment Variables:
    LLM_API_KEY: Required for Cognee's LLM (defaults to OpenAI)
    OPENAI_API_KEY: Required for answer generation and LLM judge
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# API KEYS
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_API_KEY = os.getenv("LLM_API_KEY", OPENAI_API_KEY)

if not OPENAI_API_KEY:
    import warnings
    warnings.warn(
        "OPENAI_API_KEY not found in environment. "
        "Set it via: export OPENAI_API_KEY='your-key' or in .env file"
    )

# =============================================================================
# DATASET (same as audio_eval for fair comparison)
# =============================================================================

DATASET_NAME = "byteCode18/spoken-squad-1k-memory-eval"
DATASET_SPLIT = "test"

AUDIO_COLUMN = "context"
QUESTION_COLUMN = "instruction"
ANSWER_COLUMN = "answer"

HF_TOKEN = os.getenv("HF_TOKEN")

# =============================================================================
# COGNEE SETTINGS
# =============================================================================

# LLM used by Cognee for knowledge graph construction and internal transcription
COGNEE_LLM_PROVIDER = "openai"
COGNEE_LLM_MODEL = "gpt-4o-mini"

# Embedding used by Cognee
COGNEE_EMBEDDING_PROVIDER = "openai"
COGNEE_EMBEDDING_MODEL = "text-embedding-3-small"
COGNEE_EMBEDDING_DIMENSIONS = 1536   # text-embedding-3-small max; Cognee defaults to 3072 which causes a 400

# Cognee search type
# Options: "GRAPH_COMPLETION", "RAG_COMPLETION", "CHUNKS", "SUMMARIES",
#          "GRAPH_SUMMARY_COMPLETION", "GRAPH_COMPLETION_COT"
COGNEE_SEARCH_TYPE = "GRAPH_COMPLETION"

# =============================================================================
# ANSWER GENERATION (external LLM for generating answers from Cognee results)
# =============================================================================

LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0

# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

TOP_K = 5                        # Number of results to retrieve from Cognee
RESET_BETWEEN_SAMPLES = True     # Reset Cognee state between samples for isolation

# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR = "results/cognee_eval"
EXPERIMENT_NAME = "default"


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def get_cognee_config() -> dict:
    """
    Build configuration summary dictionary (for result metadata).

    Returns:
        Dictionary describing the evaluation configuration
    """
    return {
        "framework": "cognee",
        "cognee_llm": f"{COGNEE_LLM_PROVIDER}/{COGNEE_LLM_MODEL}",
        "cognee_embedding": f"{COGNEE_EMBEDDING_PROVIDER}/{COGNEE_EMBEDDING_MODEL}@{COGNEE_EMBEDDING_DIMENSIONS}",
        "search_type": COGNEE_SEARCH_TYPE,
        "answer_llm": f"{LLM_PROVIDER}/{LLM_MODEL}",
        "top_k": TOP_K,
    }


def get_output_path(experiment_name: str, suffix: str = "") -> str:
    """Get output file path for results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{experiment_name}_{suffix}.json" if suffix else f"{experiment_name}.json"
    return os.path.join(OUTPUT_DIR, filename)


def validate_config():
    """Validate configuration before running evaluation."""
    errors = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY required")

    try:
        import cognee
    except ImportError:
        errors.append("cognee package not installed. Install with: pip install cognee")

    try:
        import soundfile
    except ImportError:
        errors.append("soundfile package not installed. Install with: pip install soundfile")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    return True
