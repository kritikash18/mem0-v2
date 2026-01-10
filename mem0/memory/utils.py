import hashlib
import logging
import re

from mem0.configs.prompts import (
    FACT_RETRIEVAL_PROMPT,
    USER_MEMORY_EXTRACTION_PROMPT,
    AGENT_MEMORY_EXTRACTION_PROMPT,
    ASR_CLEANUP_PROMPT,
    ASR_QUERY_CLEANUP_PROMPT,
)

logger = logging.getLogger(__name__)


def get_fact_retrieval_messages(message, is_agent_memory=False):
    """Get fact retrieval messages based on the memory type.
    
    Args:
        message: The message content to extract facts from
        is_agent_memory: If True, use agent memory extraction prompt, else use user memory extraction prompt
        
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    if is_agent_memory:
        return AGENT_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"
    else:
        return USER_MEMORY_EXTRACTION_PROMPT, f"Input:\n{message}"


def get_fact_retrieval_messages_legacy(message):
    """Legacy function for backward compatibility."""
    return FACT_RETRIEVAL_PROMPT, f"Input:\n{message}"


def parse_messages(messages):
    response = ""
    for msg in messages:
        if msg["role"] == "system":
            response += f"system: {msg['content']}\n"
        if msg["role"] == "user":
            response += f"user: {msg['content']}\n"
        if msg["role"] == "assistant":
            response += f"assistant: {msg['content']}\n"
    return response


def format_entities(entities):
    if not entities:
        return ""

    formatted_lines = []
    for entity in entities:
        simplified = f"{entity['source']} -- {entity['relationship']} -- {entity['destination']}"
        formatted_lines.append(simplified)

    return "\n".join(formatted_lines)


def remove_code_blocks(content: str) -> str:
    """
    Removes enclosing code block markers ```[language] and ``` from a given string.

    Remarks:
    - The function uses a regex pattern to match code blocks that may start with ``` followed by an optional language tag (letters or numbers) and end with ```.
    - If a code block is detected, it returns only the inner content, stripping out the markers.
    - If no code block markers are found, the original content is returned as-is.
    """
    pattern = r"^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, content.strip())
    match_res=match.group(1).strip() if match else content.strip()
    return re.sub(r"<think>.*?</think>", "", match_res, flags=re.DOTALL).strip()



def extract_json(text):
    """
    Extracts JSON content from a string, removing enclosing triple backticks and optional 'json' tag if present.
    If no code block is found, returns the text as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text  # assume it's raw JSON
    return json_str


def get_image_description(image_obj, llm, vision_details):
    """
    Get the description of the image
    """

    if isinstance(image_obj, str):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "A user is providing an image. Provide a high level description of the image and do not include any additional text.",
                    },
                    {"type": "image_url", "image_url": {"url": image_obj, "detail": vision_details}},
                ],
            },
        ]
    else:
        messages = [image_obj]

    response = llm.generate_response(messages=messages)
    return response


def parse_vision_messages(messages, llm=None, vision_details="auto"):
    """
    Parse the vision messages from the messages
    """
    returned_messages = []
    for msg in messages:
        if msg["role"] == "system":
            returned_messages.append(msg)
            continue

        # Handle message content
        if isinstance(msg["content"], list):
            # Multiple image URLs in content
            description = get_image_description(msg, llm, vision_details)
            returned_messages.append({"role": msg["role"], "content": description})
        elif isinstance(msg["content"], dict) and msg["content"].get("type") == "image_url":
            # Single image content
            image_url = msg["content"]["image_url"]["url"]
            try:
                description = get_image_description(image_url, llm, vision_details)
                returned_messages.append({"role": msg["role"], "content": description})
            except Exception:
                raise Exception(f"Error while downloading {image_url}.")
        else:
            # Regular text content
            returned_messages.append(msg)

    return returned_messages


def process_telemetry_filters(filters):
    """
    Process the telemetry filters
    """
    if filters is None:
        return {}

    encoded_ids = {}
    if "user_id" in filters:
        encoded_ids["user_id"] = hashlib.md5(filters["user_id"].encode()).hexdigest()
    if "agent_id" in filters:
        encoded_ids["agent_id"] = hashlib.md5(filters["agent_id"].encode()).hexdigest()
    if "run_id" in filters:
        encoded_ids["run_id"] = hashlib.md5(filters["run_id"].encode()).hexdigest()

    return list(filters.keys()), encoded_ids


def sanitize_relationship_for_cypher(relationship) -> str:
    """Sanitize relationship text for Cypher queries by replacing problematic characters."""
    char_map = {
        "...": "_ellipsis_",
        "…": "_ellipsis_",
        "。": "_period_",
        "，": "_comma_",
        "；": "_semicolon_",
        "：": "_colon_",
        "！": "_exclamation_",
        "？": "_question_",
        "（": "_lparen_",
        "）": "_rparen_",
        "【": "_lbracket_",
        "】": "_rbracket_",
        "《": "_langle_",
        "》": "_rangle_",
        "'": "_apostrophe_",
        '"': "_quote_",
        "\\": "_backslash_",
        "/": "_slash_",
        "|": "_pipe_",
        "&": "_ampersand_",
        "=": "_equals_",
        "+": "_plus_",
        "*": "_asterisk_",
        "^": "_caret_",
        "%": "_percent_",
        "$": "_dollar_",
        "#": "_hash_",
        "@": "_at_",
        "!": "_bang_",
        "?": "_question_",
        "(": "_lparen_",
        ")": "_rparen_",
        "[": "_lbracket_",
        "]": "_rbracket_",
        "{": "_lbrace_",
        "}": "_rbrace_",
        "<": "_langle_",
        ">": "_rangle_",
    }

    # Apply replacements and clean up
    sanitized = relationship
    for old, new in char_map.items():
        sanitized = sanitized.replace(old, new)

    return re.sub(r"_+", "_", sanitized).strip("_")


# =============================================================================
# Audio Processing Utilities
# =============================================================================

def is_audio_content(content):
    """
    Check if the message content represents audio data.
    
    Args:
        content: Message content to check
        
    Returns:
        bool: True if content is audio data
    """
    if isinstance(content, dict):
        content_type = content.get("type", "")
        if content_type == "audio_url":
            return True
        if content_type == "audio":
            return True
        # HuggingFace audio format
        if "array" in content and "sampling_rate" in content:
            return True
        if "audio" in content and isinstance(content["audio"], dict):
            return True
    elif isinstance(content, list):
        # Check if any item in list is audio
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("audio_url", "audio"):
                    return True
    return False


def get_audio_transcription(audio_obj, asr, language=None, llm=None, asr_config=None):
    """
    Get the transcription of audio content, optionally cleaning with LLM.
    
    Args:
        audio_obj: Audio content - can be:
            - str: URL or file path
            - dict: {"type": "audio_url", "audio_url": {"url": "..."}} or
                    {"type": "audio", "audio": {"data": "base64...", "format": "wav"}} or
                    {"array": [...], "sampling_rate": 16000} (HuggingFace format)
            - Full message dict with audio content
        asr: ASR instance for transcription
        language: Optional language code
        llm: Optional LLM instance for cleaning transcription
        asr_config: Optional ASR config with cleanup settings
        
    Returns:
        str: Transcribed (and optionally cleaned) text
    """
    # Extract audio source from various formats
    audio_source = None
    
    if isinstance(audio_obj, str):
        # Direct URL or file path
        audio_source = audio_obj
        
    elif isinstance(audio_obj, dict):
        # Check for various audio formats
        content_type = audio_obj.get("type")
        
        if content_type == "audio_url":
            # Format: {"type": "audio_url", "audio_url": {"url": "..."}}
            audio_url_obj = audio_obj.get("audio_url", {})
            audio_source = audio_url_obj.get("url")
            
        elif content_type == "audio":
            # Format: {"type": "audio", "audio": {"data": "base64...", "format": "wav"}}
            audio_data = audio_obj.get("audio", {})
            if "data" in audio_data:
                # Base64 encoded audio
                audio_format = audio_data.get("format", "wav")
                audio_source = f"data:audio/{audio_format};base64,{audio_data['data']}"
            elif "url" in audio_data:
                audio_source = audio_data["url"]
            elif "path" in audio_data:
                audio_source = audio_data["path"]
                
        elif "array" in audio_obj and "sampling_rate" in audio_obj:
            # HuggingFace audio format
            audio_source = audio_obj
            
        elif "audio" in audio_obj and isinstance(audio_obj["audio"], dict):
            # Nested HuggingFace format
            audio_source = audio_obj
            
        elif "content" in audio_obj:
            # Full message dict - recursively extract
            return get_audio_transcription(audio_obj["content"], asr, language, llm, asr_config)
            
    elif isinstance(audio_obj, list):
        # List of content items - find audio and transcribe
        transcriptions = []
        for item in audio_obj:
            if isinstance(item, dict) and item.get("type") in ("audio_url", "audio"):
                text = get_audio_transcription(item, asr, language, llm, asr_config)
                if text:
                    transcriptions.append(text)
            elif isinstance(item, dict) and item.get("type") == "text":
                # Include text content as-is
                transcriptions.append(item.get("text", ""))
        return " ".join(transcriptions)
    
    if audio_source is None:
        raise ValueError(f"Could not extract audio source from: {audio_obj}")
    
    # Transcribe using ASR
    kwargs = {}
    if language:
        kwargs["language"] = language
    
    transcription = asr.transcribe(audio_source, **kwargs)
    transcription = transcription.strip()
    
    # Clean transcription using LLM if enabled
    enable_cleanup = True
    custom_prompt = None
    
    if asr_config is not None:
        enable_cleanup = getattr(asr_config, 'enable_cleanup', True)
        custom_prompt = getattr(asr_config, 'cleanup_prompt', None)
    
    if enable_cleanup and llm is not None:
        transcription = clean_asr_output(
            transcription, 
            llm, 
            custom_prompt=custom_prompt,
            is_query=False  # Use memory ingestion prompt
        )
    
    return transcription


def parse_audio_messages(messages, asr=None, language=None, llm=None, asr_config=None):
    """
    Parse audio messages and convert them to text using ASR.
    
    Similar to parse_vision_messages but for audio content.
    Optionally cleans transcription using LLM if configured.
    
    Args:
        messages: List of message dicts
        asr: ASR instance for transcription (required if audio content is present)
        language: Optional language code for transcription
        llm: Optional LLM instance for cleaning transcription
        asr_config: Optional ASR config with cleanup settings
        
    Returns:
        list: Messages with audio content replaced by transcribed text
    """
    if asr is None:
        # No ASR configured, return messages as-is but log warning for audio content
        has_audio = any(
            is_audio_content(msg.get("content")) 
            for msg in messages 
            if isinstance(msg, dict)
        )
        if has_audio:
            logger.warning(
                "Audio content detected but no ASR is configured. "
                "Audio messages will be skipped. Enable ASR in config to process audio."
            )
        return messages
    
    returned_messages = []
    
    for msg in messages:
        if not isinstance(msg, dict):
            returned_messages.append(msg)
            continue
            
        if msg.get("role") == "system":
            returned_messages.append(msg)
            continue
        
        content = msg.get("content")
        
        # Check if content is audio
        if is_audio_content(content):
            try:
                transcription = get_audio_transcription(
                    content, asr, language, 
                    llm=llm, asr_config=asr_config
                )
                returned_messages.append({
                    "role": msg["role"],
                    "content": transcription
                })
            except Exception as e:
                raise Exception(f"Error transcribing audio: {e}")
        else:
            # Regular text or other content
            returned_messages.append(msg)
    
    return returned_messages


def parse_multimodal_messages(messages, llm=None, asr=None, vision_details="auto", audio_language=None, asr_config=None):
    """
    Parse messages that may contain text, images, and/or audio.
    
    This is a unified function that handles all multimodal content types.
    Audio transcription can optionally be cleaned using the same LLM.
    
    Args:
        messages: List of message dicts
        llm: LLM instance for vision processing and ASR cleanup (required for image content)
        asr: ASR instance for audio transcription (required for audio content)
        vision_details: Detail level for image processing ("auto", "low", "high")
        audio_language: Language code for audio transcription
        asr_config: Optional ASR config with cleanup settings
        
    Returns:
        list: Messages with multimodal content converted to text
    """
    # First pass: handle audio (with optional LLM cleanup)
    messages = parse_audio_messages(
        messages, asr=asr, language=audio_language, 
        llm=llm, asr_config=asr_config
    )
    
    # Second pass: handle images
    messages = parse_vision_messages(messages, llm=llm, vision_details=vision_details)
    
    return messages


# =============================================================================
# Audio Query Processing
# =============================================================================

def is_audio_query(query):
    """
    Check if a query is an audio input rather than text.
    
    Args:
        query: The query to check - can be str, bytes, dict, or other audio formats
        
    Returns:
        bool: True if the query represents audio data
    """
    if isinstance(query, bytes):
        return True
    
    if isinstance(query, str):
        # Check for audio URLs
        if query.startswith(("http://", "https://")):
            audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm", ".opus")
            if any(query.lower().endswith(ext) for ext in audio_extensions):
                return True
            # Check for common audio hosting patterns
            if any(pattern in query.lower() for pattern in ["audio", "speech", "voice", "recording"]):
                return True
        # Check for base64 audio data URI
        if query.startswith("data:audio"):
            return True
        # Check for local file paths with audio extensions
        if any(query.lower().endswith(ext) for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm")):
            import os
            if os.path.isfile(query):
                return True
        return False
    
    if isinstance(query, dict):
        # Check for audio content formats
        content_type = query.get("type", "")
        if content_type in ("audio_url", "audio"):
            return True
        # HuggingFace audio format
        if "array" in query and "sampling_rate" in query:
            return True
        if "audio" in query and isinstance(query.get("audio"), dict):
            return True
        return False
    
    # Check for numpy array (common for audio data)
    try:
        import numpy as np
        if isinstance(query, np.ndarray):
            return True
    except ImportError:
        pass
    
    return False


def clean_asr_output(transcription, llm, custom_prompt=None, is_query=False):
    """
    Clean ASR transcription output using an LLM.
    
    This function uses an LLM to remove filler words, fix grammar,
    and improve the quality of transcribed text.
    
    Args:
        transcription: Raw transcription from ASR
        llm: LLM instance for cleanup
        custom_prompt: Optional custom cleanup prompt
        is_query: If True, uses query-optimized cleanup prompt
        
    Returns:
        str: Cleaned transcription text
    """
    if not transcription or not transcription.strip():
        return transcription
    
    if llm is None:
        logger.warning("No LLM provided for ASR cleanup, returning raw transcription")
        return transcription
    
    # Select appropriate prompt
    if custom_prompt:
        system_prompt = custom_prompt
    elif is_query:
        system_prompt = ASR_QUERY_CLEANUP_PROMPT
    else:
        system_prompt = ASR_CLEANUP_PROMPT
    
    # Build messages for LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcription}
    ]
    
    try:
        response = llm.generate_response(messages=messages)
        
        # Handle different response types
        if isinstance(response, str):
            cleaned = response.strip()
        elif isinstance(response, dict) and "content" in response:
            cleaned = response["content"].strip() if response["content"] else transcription
        else:
            cleaned = str(response).strip()
        
        # Validate output - if empty or too different, use original
        if not cleaned:
            logger.warning("LLM returned empty response for ASR cleanup, using original")
            return transcription
        
        logger.debug(f"ASR cleanup: '{transcription[:50]}...' -> '{cleaned[:50]}...'")
        return cleaned
        
    except Exception as e:
        logger.warning(f"ASR cleanup failed: {e}, using original transcription")
        return transcription


def transcribe_audio_query(query, asr, language=None, llm=None, asr_config=None):
    """
    Transcribe an audio query to text using ASR, optionally cleaning with LLM.
    
    Args:
        query: Audio input - can be:
            - str: File path, URL, or base64 encoded audio
            - bytes: Raw audio bytes
            - dict: Audio format dict (e.g., {"type": "audio_url", "audio_url": {"url": "..."}})
            - np.ndarray: Audio samples as numpy array
            - dict: HuggingFace audio format {"array": [...], "sampling_rate": 16000}
        asr: ASR instance for transcription
        language: Optional language code
        llm: Optional LLM instance for cleaning transcription
        asr_config: Optional ASR config with cleanup settings
        
    Returns:
        str: Transcribed (and optionally cleaned) text query
    """
    if asr is None:
        raise ValueError(
            "ASR is required to process audio queries. "
            "Configure ASR in MemoryConfig to enable audio query support."
        )
    
    # Extract audio source from various formats
    audio_source = query
    
    if isinstance(query, dict):
        content_type = query.get("type", "")
        
        if content_type == "audio_url":
            audio_url_obj = query.get("audio_url", {})
            audio_source = audio_url_obj.get("url", query)
            
        elif content_type == "audio":
            audio_data = query.get("audio", {})
            if "data" in audio_data:
                audio_format = audio_data.get("format", "wav")
                audio_source = f"data:audio/{audio_format};base64,{audio_data['data']}"
            elif "url" in audio_data:
                audio_source = audio_data["url"]
            elif "path" in audio_data:
                audio_source = audio_data["path"]
                
        elif "array" in query and "sampling_rate" in query:
            # HuggingFace audio format - pass as is
            audio_source = query
            
        elif "audio" in query and isinstance(query["audio"], dict):
            # Nested HuggingFace format - pass as is
            audio_source = query
    
    # Transcribe using ASR
    kwargs = {}
    if language:
        kwargs["language"] = language
    
    transcription = asr.transcribe(audio_source, **kwargs)
    transcription = transcription.strip()
    
    # Clean transcription using LLM if enabled
    enable_cleanup = True
    custom_prompt = None
    
    if asr_config is not None:
        enable_cleanup = getattr(asr_config, 'enable_cleanup', True)
        custom_prompt = getattr(asr_config, 'query_cleanup_prompt', None)
    
    if enable_cleanup and llm is not None:
        transcription = clean_asr_output(
            transcription, 
            llm, 
            custom_prompt=custom_prompt,
            is_query=True
        )
    
    return transcription

