"""
Qdrant Vector Database Example
===============================
This example demonstrates using Qdrant, a high-performance vector database
that can run locally or in the cloud.

Requirements:
- qdrant-client>=1.7.0
- sentence-transformers>=2.0.0

What is Qdrant?
Qdrant is a vector similarity search engine with extended filtering support.
It can run locally (no setup needed) or as a cloud service.

Best Practices:
- Use collections to organize different datasets
- Leverage payload filtering for hybrid search
- Use quantization for memory efficiency
- Batch operations for better performance
"""



def qdrant_basics_example():
    """
    Basic Qdrant operations using in-memory mode.
    Perfect for development and testing.
    """
    print("=== Qdrant Basics (In-Memory) ===")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from sentence_transformers import SentenceTransformer
        
        # Initialize client (in-memory - no server needed!)
        print("\nInitializing Qdrant client (in-memory)...")
        client = QdrantClient(":memory:")
        
        # Load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create collection
        collection_name = "my_collection"
        print(f"\nCreating collection: {collection_name}")
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("✓ Collection created")
        
        # Sample documents
        documents = [
            {"id": 1, "text": "Qdrant is a vector database", "category": "database"},
            {"id": 2, "text": "Python is great for AI development", "category": "programming"},
            {"id": 3, "text": "Vector search enables semantic similarity", "category": "database"},
            {"id": 4, "text": "Machine learning models need training data", "category": "ai"},
        ]
        
        # Generate embeddings and create points
        print("\nUpserting documents...")
        points = []
        for doc in documents:
            embedding = model.encode(doc["text"])
            point = PointStruct(
                id=doc["id"],
                vector=embedding.tolist(),
                payload={
                    "text": doc["text"],
                    "category": doc["category"]
                }
            )
            points.append(point)
        
        # Upsert points
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"✓ Upserted {len(points)} points")
        
        # Search
        print("\n--- Searching ---")
        query_text = "Tell me about vector databases"
        query_vector = model.encode(query_text).tolist()
        
        search_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=2
        ).points
        
        print(f"Query: '{query_text}'")
        print("\nTop matches:")
        for result in search_results:
            print(f"  Score: {result.score:.4f}")
            print(f"  Text: {result.payload['text']}")
            print(f"  Category: {result.payload['category']}\n")
        
        return client, collection_name
        
    except ImportError:
        print("❌ Qdrant not installed. Install with: uv pip install qdrant-client")
        return None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


def persistent_storage_example():
    """
    Using Qdrant with persistent storage.
    Data is saved to disk and survives restarts.
    """
    print("\n=== Persistent Storage ===")
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from sentence_transformers import SentenceTransformer
        
        # Initialize with persistent storage
        storage_path = "./qdrant_storage"
        print(f"\nInitializing Qdrant with persistent storage: {storage_path}")
        
        client = QdrantClient(path=storage_path)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        collection_name = "persistent_collection"
        
        # Create or use existing collection
        collections = [c.name for c in client.get_collections().collections]
        
        if collection_name not in collections:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print("✓ Created new collection")
            
            # Add some data
            documents = [
                {"id": 1, "text": "Data persists across restarts"},
                {"id": 2, "text": "Qdrant saves to disk automatically"}
            ]
            
            points = []
            for doc in documents:
                embedding = model.encode(doc["text"])
                points.append(PointStruct(
                    id=doc["id"],
                    vector=embedding.tolist(),
                    payload={"text": doc["text"]}
                ))
            
            client.upsert(collection_name=collection_name, points=points)
            print(f"✓ Added {len(points)} points")
        else:
            print("✓ Using existing collection")
            
            # Show that data persisted
            count = client.count(collection_name=collection_name)
            print(f"  Collection has {count.count} points")
        
        print("\n💡 Data is saved to disk and will persist across restarts!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def payload_filtering_example(client, collection_name):
    """
    Advanced payload filtering for hybrid search.
    """
    print("\n=== Payload Filtering ===")
    
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Search with filter
        query_text = "programming and development"
        query_vector = model.encode(query_text).tolist()
        
        print(f"\nQuery: '{query_text}'")
        print("Filter: category = 'programming'")
        
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value="programming")
                    )
                ]
            ),
            limit=3
        ).points
        
        print("\nFiltered results:")
        for result in results:
            print(f"  [{result.score:.4f}] {result.payload['text']}")
        
        # Complex filters
        print("\n--- Filter Types ---")
        print("• must: All conditions must match (AND)")
        print("• should: At least one condition must match (OR)")
        print("• must_not: Conditions must not match (NOT)")
        print("\n--- Condition Types ---")
        print("• match: Exact match")
        print("• range: Numeric range (gt, gte, lt, lte)")
        print("• geo_radius: Geographic distance")
        print("• values_count: Count of values in array")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def scroll_and_batch_example(client, collection_name):
    """
    Scrolling through large result sets and batch operations.
    """
    print("\n=== Scroll & Batch Operations ===")
    
    try:
        from qdrant_client.models import PointStruct
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Add more points for demonstration
        print("\nAdding batch of points...")
        batch_size = 20
        points = []
        
        for i in range(batch_size):
            text = f"Batch document number {i} with various content"
            embedding = model.encode(text)
            points.append(PointStruct(
                id=100 + i,
                vector=embedding.tolist(),
                payload={"text": text, "batch": "demo"}
            ))
        
        client.upsert(collection_name=collection_name, points=points)
        print(f"✓ Added {batch_size} points")
        
        # Scroll through all points
        print("\nScrolling through collection...")
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False  # Don't return vectors to save bandwidth
        )
        
        points, next_offset = scroll_result
        print(f"✓ Retrieved {len(points)} points")
        print(f"  First point: {points[0].payload.get('text', 'N/A')[:50]}...")
        
        # Get collection info
        collection_info = client.get_collection(collection_name=collection_name)
        print(f"\nCollection stats:")
        print(f"  Total points: {collection_info.points_count}")
        print(f"  Vector size: {collection_info.config.params.vectors.size}")
        print(f"  Distance: {collection_info.config.params.vectors.distance}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def update_and_delete_example(client, collection_name):
    """
    Updating and deleting points.
    """
    print("\n=== Update & Delete Operations ===")
    
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Update payload
        print("\nUpdating payload...")
        client.set_payload(
            collection_name=collection_name,
            payload={"updated": True, "timestamp": "2024-01-01"},
            points=[1]
        )
        print("✓ Updated point 1")
        
        # Delete by ID
        print("\nDeleting by ID...")
        client.delete(
            collection_name=collection_name,
            points_selector=[100, 101]
        )
        print("✓ Deleted points 100, 101")
        
        # Delete by filter
        print("\nDeleting by filter...")
        client.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="batch",
                        match=MatchValue(value="demo")
                    )
                ]
            )
        )
        print("✓ Deleted points matching filter")
        
        # Clear payload field
        print("\nClearing payload field...")
        client.delete_payload(
            collection_name=collection_name,
            keys=["updated"],
            points=[1]
        )
        print("✓ Cleared 'updated' field from point 1")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def quantization_example():
    """
    Using quantization for memory efficiency.
    Reduces memory usage with minimal quality loss.
    """
    print("\n=== Quantization (Advanced) ===")
    
    print("""
Quantization reduces memory usage by compressing vectors:

• Scalar Quantization: Convert float32 to int8
  - 4x memory reduction
  - Minimal quality loss (<1%)
  - Faster search

• Product Quantization: More aggressive compression
  - 8-16x memory reduction
  - Some quality trade-off
  - Best for very large datasets

Enable during collection creation:
```python
from qdrant_client.models import ScalarQuantization, ScalarType

client.create_collection(
    collection_name="quantized_collection",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    quantization_config=ScalarQuantization(
        scalar=ScalarType.INT8,
        quantile=0.99
    )
)
```

When to use:
• Large datasets (millions of vectors)
• Memory-constrained environments
• When search speed is critical
    """)


if __name__ == "__main__":
    print("=" * 70)
    print("Qdrant Vector Database Examples")
    print("=" * 70)
    
    try:
        # Run examples
        client, collection_name = qdrant_basics_example()
        
        if client and collection_name:
            print("\n" + "=" * 70 + "\n")
            persistent_storage_example()
            
            print("\n" + "=" * 70 + "\n")
            payload_filtering_example(client, collection_name)
            
            print("\n" + "=" * 70 + "\n")
            scroll_and_batch_example(client, collection_name)
            
            print("\n" + "=" * 70 + "\n")
            update_and_delete_example(client, collection_name)
            
            print("\n" + "=" * 70 + "\n")
            quantization_example()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed!")
        print("=" * 70)
        
        print("\nNext Steps:")
        print("1. Try running Qdrant server locally: docker run -p 6333:6333 qdrant/qdrant")
        print("2. Explore Qdrant Cloud for production deployments")
        print("3. Experiment with different distance metrics")
        print("4. Try quantization for large datasets")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have installed: uv pip install qdrant-client sentence-transformers")
