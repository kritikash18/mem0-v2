"""
Test script for Qdrant Local setup.

Verifies Qdrant is working correctly by:
1. Initializing local Qdrant client
2. Creating a test collection
3. Adding random vectors
4. Searching/fetching vectors
5. Cleaning up

Usage:
    python test_qdrant_local.py
"""

import os
import shutil
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


def test_qdrant_local():
    """Test Qdrant local setup with random data."""
    
    # Configuration
    test_path = "./test_qdrant_data"
    collection_name = "test_collection"
    vector_dim = 1536  # OpenAI text-embedding-3-small dimension
    num_vectors = 10
    
    print("=" * 60)
    print("QDRANT LOCAL TEST")
    print("=" * 60)
    
    # Clean up any existing test data
    if os.path.exists(test_path):
        print(f"\n[Cleanup] Removing existing test data at {test_path}")
        shutil.rmtree(test_path)
    
    # Step 1: Initialize Qdrant client (local mode)
    print(f"\n[1/5] Initializing Qdrant client (local mode)")
    print(f"      Path: {test_path}")
    client = QdrantClient(path=test_path)
    print("      ✓ Client initialized")
    
    # Step 2: Create collection
    print(f"\n[2/5] Creating collection: {collection_name}")
    print(f"      Vector dimensions: {vector_dim}")
    print(f"      Distance metric: Cosine")
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
            on_disk=True  # Persistent storage
        )
    )
    print("      ✓ Collection created")
    
    # Verify collection exists
    collections = client.get_collections()
    print(f"      Collections: {[c.name for c in collections.collections]}")
    
    # Step 3: Add random vectors with metadata
    print(f"\n[3/5] Adding {num_vectors} random vectors")
    
    points = []
    for i in range(num_vectors):
        # Generate random vector
        vector = np.random.rand(vector_dim).tolist()
        
        # Create payload (metadata)
        payload = {
            "id": i,
            "text": f"Test memory {i}",
            "user_id": f"user_{i % 3}",  # 3 different users
            "timestamp": f"2024-01-{i+1:02d}",
        }
        
        # Create point
        point = PointStruct(
            id=i,
            vector=vector,
            payload=payload
        )
        points.append(point)
    
    # Insert all points
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"      ✓ Added {num_vectors} vectors")
    
    # Verify count
    collection_info = client.get_collection(collection_name)
    print(f"      Collection size: {collection_info.points_count} vectors")
    
    # Step 4: Fetch and search data
    print(f"\n[4/5] Fetching and searching data")
    
    # 4a. Retrieve specific vector by ID
    print("      [4a] Retrieve vector by ID (id=5):")
    result = client.retrieve(
        collection_name=collection_name,
        ids=[5],
        with_payload=True,
        with_vectors=False  # Don't return full vector (too long)
    )
    if result:
        print(f"           ID: {result[0].id}")
        print(f"           Text: {result[0].payload['text']}")
        print(f"           User: {result[0].payload['user_id']}")
        print(f"           Timestamp: {result[0].payload['timestamp']}")
        print("           ✓ Retrieved successfully")
    
    # 4b. Search for similar vectors
    print("\n      [4b] Search for similar vectors:")
    query_vector = np.random.rand(vector_dim).tolist()  # Random query
    
    search_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3  # Top 3 results
    )
    
    print(f"           Found {len(search_results.points)} similar vectors:")
    for i, point in enumerate(search_results.points, 1):
        print(f"           {i}. ID={point.id}, Score={point.score:.4f}, Text='{point.payload['text']}'")
    print("           ✓ Search successful")
    
    # 4c. Filter by user_id
    print("\n      [4c] Filter by user_id='user_1':")
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    filtered_results = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value="user_1")
                )
            ]
        ),
        limit=10
    )
    
    vectors_found = filtered_results[0]  # First element is list of points
    print(f"           Found {len(vectors_found)} vectors for user_1:")
    for point in vectors_found:
        print(f"           - ID={point.id}, Text='{point.payload['text']}'")
    print("           ✓ Filtering successful")
    
    # Step 5: Cleanup
    print(f"\n[5/5] Cleaning up")
    
    # Delete collection
    print(f"      Deleting collection: {collection_name}")
    client.delete_collection(collection_name)
    print("      ✓ Collection deleted")
    
    # Verify deletion
    collections = client.get_collections()
    print(f"      Remaining collections: {[c.name for c in collections.collections]}")
    
    # Remove test data directory
    print(f"      Removing test data directory: {test_path}")
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    print("      ✓ Directory removed")
    
    # Final verification
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY ✓")
    print("=" * 60)
    print("\nQdrant local setup is working correctly!")
    print("You can now run the audio evaluation with confidence.")
    print("\nNext step:")
    print("  cd ../evaluation")
    print("  python -m audio_eval.evaluator --num_samples 5")


if __name__ == "__main__":
    try:
        test_qdrant_local()
    except ImportError as e:
        print("\n" + "=" * 60)
        print("ERROR: Missing dependency")
        print("=" * 60)
        print(f"\n{e}")
        print("\nPlease install qdrant-client:")
        print("  pip install qdrant-client")
    except Exception as e:
        print("\n" + "=" * 60)
        print("ERROR: Test failed")
        print("=" * 60)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()

