"""
Test script for modality-weighted retrieval feature.

This demonstrates how to:
1. Store memories with different modalities (audio, image, text)
2. Configure modality weights to prioritize certain modalities
3. Retrieve memories with weighted scores based on modality
"""

from mem0 import Memory

def test_basic_modality_weighting():
    """Test basic modality weighting with default weights (1.0 for all)"""
    print("=" * 60)
    print("Test 1: Basic Modality Storage (Default Weights)")
    print("=" * 60)
    
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "test_modality",
                "path": "./test_qdrant_modality",
                "on_disk": True,
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4",
                "temperature": 0,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            }
        },
    }
    
    m = Memory.from_config(config)
    
    # Add text memory
    print("\n1. Adding text memory...")
    m.add("Python is a programming language", user_id="user1")
    
    # Add audio memory (simulated - in real usage you'd pass audio file/bytes)
    # For this demo, we'll add text but the modality would be detected from actual audio input
    print("2. Adding audio-based memory...")
    # In real usage: m.add(audio_file_path, user_id="user1")
    # For demo: we'll manually show the concept
    
    # Add image memory (simulated)
    print("3. Adding image-based memory...")
    # In real usage: m.add([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "..."}}]}], user_id="user1")
    
    # Search
    print("\n4. Searching memories...")
    results = m.search("programming", user_id="user1")
    
    print(f"\nFound {len(results['results'])} memories:")
    for i, mem in enumerate(results['results']):
        print(f"\n  Memory {i+1}:")
        print(f"    Content: {mem['memory']}")
        print(f"    Modality: {mem.get('modality', 'unknown')}")
        print(f"    Score: {mem.get('score', 'N/A')}")
        print(f"    Original Score: {mem.get('original_score', 'N/A')}")
        print(f"    Modality Weight: {mem.get('modality_weight', 'N/A')}")


def test_weighted_modality_retrieval():
    """Test with custom modality weights to prioritize audio"""
    print("\n" + "=" * 60)
    print("Test 2: Weighted Modality Retrieval (Prioritize Audio)")
    print("=" * 60)
    
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "test_weighted_modality",
                "path": "./test_qdrant_weighted",
                "on_disk": True,
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4",
                "temperature": 0,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            }
        },
        # Custom modality weights: prioritize audio > image > text
        "modality_weights": {
            "audio": 2.0,   # 2x boost for audio memories
            "image": 1.5,   # 1.5x boost for image memories
            "text": 1.0,    # baseline for text memories
        }
    }
    
    m = Memory.from_config(config)
    
    print("\nModality weights configured:")
    print(f"  Audio: 2.0x")
    print(f"  Image: 1.5x")
    print(f"  Text: 1.0x")
    
    # Add memories of different modalities
    print("\n1. Adding text memory...")
    m.add("The United Methodist Church is a denomination", user_id="user2")
    
    print("2. Adding more memories...")
    m.add("Religious organizations have various structures", user_id="user2")
    
    # Search
    print("\n3. Searching with weighted modality...")
    results = m.search("church", user_id="user2")
    
    print(f"\nFound {len(results['results'])} memories (sorted by weighted score):")
    for i, mem in enumerate(results['results']):
        print(f"\n  Memory {i+1}:")
        print(f"    Content: {mem['memory'][:80]}...")
        print(f"    Modality: {mem.get('modality', 'text')}")
        print(f"    Original Score: {mem.get('original_score', 0):.4f}")
        print(f"    Modality Weight: {mem.get('modality_weight', 1.0)}x")
        print(f"    Final Score: {mem.get('score', 0):.4f}")
        print(f"    → {'Audio memories boosted!' if mem.get('modality') == 'audio' else 'Standard weight'}")


def test_extreme_weighting():
    """Test with extreme weights to demonstrate filtering effect"""
    print("\n" + "=" * 60)
    print("Test 3: Extreme Weighting (Filter Out Text)")
    print("=" * 60)
    
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "test_extreme_weighting",
                "path": "./test_qdrant_extreme",
                "on_disk": True,
            }
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": "gpt-4",
                "temperature": 0,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
            }
        },
        # Extreme weights: heavily favor audio/image, de-prioritize text
        "modality_weights": {
            "audio": 5.0,   # 5x boost for audio
            "image": 3.0,   # 3x boost for images
            "text": 0.1,    # 90% penalty for text
        }
    }
    
    m = Memory.from_config(config)
    
    print("\nExtreme modality weights configured:")
    print(f"  Audio: 5.0x (heavily prioritized)")
    print(f"  Image: 3.0x (prioritized)")
    print(f"  Text: 0.1x (heavily de-prioritized)")
    
    print("\nThis configuration would cause audio/image memories to rank")
    print("much higher than text memories, even if text has better semantic match.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MODALITY-WEIGHTED RETRIEVAL TEST SUITE")
    print("=" * 60)
    print("\nThis feature allows you to:")
    print("  • Store memories with modality metadata (audio/image/text)")
    print("  • Configure weights per modality in the config")
    print("  • Retrieve memories with scores adjusted by modality weight")
    print("  • Re-rank results to prioritize preferred modalities")
    
    try:
        # Note: These tests require OpenAI API key and will create local Qdrant instances
        print("\n\nTo run these tests, ensure:")
        print("  1. OPENAI_API_KEY is set in environment")
        print("  2. Local file write permissions for Qdrant storage")
        print("  3. mem0 is installed: pip install -e .")
        
        # Uncomment to run actual tests:
        # test_basic_modality_weighting()
        # test_weighted_modality_retrieval()
        # test_extreme_weighting()
        
        print("\n\n" + "=" * 60)
        print("Tests are ready to run (uncomment in script)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running tests: {e}")
        print("\nMake sure you have:")
        print("  • OpenAI API key configured")
        print("  • Qdrant dependencies installed")
        print("  • Sufficient permissions for local file storage")

