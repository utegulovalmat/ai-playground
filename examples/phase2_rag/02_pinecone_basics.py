"""
Pinecone Vector Database Example
=================================
This example demonstrates using Pinecone, a cloud-native vector database
optimized for similarity search at scale.

Requirements:
- pinecone-client>=3.0.0
- sentence-transformers>=2.0.0
- PINECONE_API_KEY environment variable

What is Pinecone?
Pinecone is a fully managed vector database service that makes it easy to
build high-performance vector search applications without infrastructure management.

Best Practices:
- Use namespaces for multi-tenancy
- Leverage metadata filtering for hybrid search
- Choose appropriate index types (serverless vs pod-based)
- Monitor usage and costs
"""

import os
import time
from typing import List, Dict
import numpy as np


def pinecone_basics_example():
    """
    Basic Pinecone operations: create index, upsert, query.
    """
    print("=== Pinecone Basics ===")
    
    try:
        from pinecone import Pinecone, ServerlessSpec
        from sentence_transformers import SentenceTransformer
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        
        # Create index (serverless - pay per use)
        index_name = "quickstart-index"
        
        print(f"\nCreating index: {index_name}")
        
        # Check if index exists
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=384,  # Match embedding model dimensions
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("✓ Index created")
        else:
            print("✓ Index already exists")
        
        # Connect to index
        index = pc.Index(index_name)
        
        # Wait for index to be ready
        time.sleep(1)
        
        # Load embedding model
        print("\nLoading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample documents
        documents = [
            {"id": "doc1", "text": "Python is a versatile programming language", "category": "programming"},
            {"id": "doc2", "text": "Machine learning enables computers to learn", "category": "ai"},
            {"id": "doc3", "text": "JavaScript is essential for web development", "category": "programming"},
            {"id": "doc4", "text": "Deep learning uses neural networks", "category": "ai"},
        ]
        
        # Generate embeddings and prepare for upsert
        print("\nUpserting documents...")
        vectors = []
        for doc in documents:
            embedding = model.encode(doc["text"]).tolist()
            vectors.append({
                "id": doc["id"],
                "values": embedding,
                "metadata": {
                    "text": doc["text"],
                    "category": doc["category"]
                }
            })
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        print(f"✓ Upserted {len(vectors)} vectors")
        
        # Query
        print("\n--- Querying ---")
        query_text = "Tell me about AI and neural networks"
        query_embedding = model.encode(query_text).tolist()
        
        results = index.query(
            vector=query_embedding,
            top_k=2,
            include_metadata=True
        )
        
        print(f"Query: '{query_text}'")
        print("\nTop matches:")
        for match in results['matches']:
            print(f"  Score: {match['score']:.4f}")
            print(f"  Text: {match['metadata']['text']}")
            print(f"  Category: {match['metadata']['category']}\n")
        
        return index
        
    except ImportError:
        print("❌ Pinecone not installed. Install with: uv pip install pinecone-client")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure PINECONE_API_KEY is set")
        return None


def namespace_example():
    """
    Using namespaces for multi-tenancy.
    Namespaces allow you to partition your index.
    """
    print("\n=== Namespaces Example ===")
    
    try:
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer
        
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index("quickstart-index")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Upsert to different namespaces (e.g., different users/tenants)
        print("\nUpserting to different namespaces...")
        
        # User A's documents
        user_a_docs = [
            {"id": "a1", "text": "User A's first document"},
            {"id": "a2", "text": "User A's second document"}
        ]
        
        vectors_a = []
        for doc in user_a_docs:
            embedding = model.encode(doc["text"]).tolist()
            vectors_a.append({"id": doc["id"], "values": embedding, "metadata": {"text": doc["text"]}})
        
        index.upsert(vectors=vectors_a, namespace="user-a")
        
        # User B's documents
        user_b_docs = [
            {"id": "b1", "text": "User B's first document"},
            {"id": "b2", "text": "User B's second document"}
        ]
        
        vectors_b = []
        for doc in user_b_docs:
            embedding = model.encode(doc["text"]).tolist()
            vectors_b.append({"id": doc["id"], "values": embedding, "metadata": {"text": doc["text"]}})
        
        index.upsert(vectors=vectors_b, namespace="user-b")
        
        print("✓ Upserted to user-a and user-b namespaces")
        
        # Query specific namespace
        query_embedding = model.encode("first document").tolist()
        
        results_a = index.query(
            vector=query_embedding,
            top_k=1,
            namespace="user-a",
            include_metadata=True
        )
        
        print(f"\nQuerying user-a namespace:")
        print(f"  Result: {results_a['matches'][0]['metadata']['text']}")
        
        # Get stats
        stats = index.describe_index_stats()
        print(f"\nIndex stats:")
        print(f"  Total vectors: {stats['total_vector_count']}")
        print(f"  Namespaces: {list(stats.get('namespaces', {}).keys())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def metadata_filtering_example():
    """
    Advanced metadata filtering for hybrid search.
    """
    print("\n=== Metadata Filtering ===")
    
    try:
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer
        
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index("quickstart-index")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Query with metadata filter
        query_text = "programming language"
        query_embedding = model.encode(query_text).tolist()
        
        print(f"\nQuery: '{query_text}'")
        print("Filter: category = 'programming'")
        
        results = index.query(
            vector=query_embedding,
            top_k=3,
            filter={"category": {"$eq": "programming"}},
            include_metadata=True
        )
        
        print("\nFiltered results:")
        for match in results['matches']:
            print(f"  [{match['score']:.4f}] {match['metadata']['text']}")
        
        # Complex filters
        print("\n--- Complex Filters ---")
        print("Available operators:")
        print("  $eq: equals")
        print("  $ne: not equals")
        print("  $in: in list")
        print("  $nin: not in list")
        print("  $gt, $gte: greater than (or equal)")
        print("  $lt, $lte: less than (or equal)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def batch_operations_example():
    """
    Efficient batch operations for large datasets.
    """
    print("\n=== Batch Operations ===")
    
    try:
        from pinecone import Pinecone
        from sentence_transformers import SentenceTransformer
        
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index("quickstart-index")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate large batch
        print("\nPreparing large batch...")
        batch_size = 100
        documents = [f"Document number {i} with various content" for i in range(batch_size)]
        
        # Batch encode
        embeddings = model.encode(documents, batch_size=32, show_progress_bar=False)
        
        # Prepare vectors
        vectors = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            vectors.append({
                "id": f"batch-{i}",
                "values": emb.tolist(),
                "metadata": {"text": doc}
            })
        
        # Upsert in batches (Pinecone recommends batches of 100-200)
        print(f"Upserting {len(vectors)} vectors...")
        index.upsert(vectors=vectors, namespace="batch-test")
        
        print(f"✓ Batch upsert complete")
        
        # Batch query
        print("\nBatch querying...")
        query_texts = ["document content", "various information", "number data"]
        query_embeddings = model.encode(query_texts, show_progress_bar=False)
        
        for query_text, query_emb in zip(query_texts, query_embeddings):
            results = index.query(
                vector=query_emb.tolist(),
                top_k=1,
                namespace="batch-test",
                include_metadata=True
            )
            print(f"  Query: '{query_text}' -> {results['matches'][0]['id']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def cleanup_example():
    """
    Cleanup operations: delete vectors, namespaces, indexes.
    """
    print("\n=== Cleanup Operations ===")
    
    try:
        from pinecone import Pinecone
        
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index("quickstart-index")
        
        # Delete by ID
        print("\nDeleting specific vectors...")
        index.delete(ids=["batch-0", "batch-1"], namespace="batch-test")
        print("✓ Deleted specific vectors")
        
        # Delete by filter
        print("\nDeleting by metadata filter...")
        index.delete(filter={"category": "ai"})
        print("✓ Deleted vectors matching filter")
        
        # Delete entire namespace
        print("\nDeleting namespace...")
        index.delete(delete_all=True, namespace="batch-test")
        print("✓ Deleted namespace")
        
        # Note: To delete entire index, use:
        # pc.delete_index("index-name")
        
        print("\n--- Cleanup Best Practices ---")
        print("• Delete by filter for selective cleanup")
        print("• Use namespaces for easy tenant data deletion")
        print("• Monitor costs - deleted vectors stop incurring charges")
        print("• Consider index deletion for complete cleanup")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Check API key
    if not os.environ.get("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY environment variable not set")
        print("\nTo get started:")
        print("1. Sign up at https://www.pinecone.io")
        print("2. Create an API key")
        print("3. Set it: export PINECONE_API_KEY='your-key-here'")
        exit(1)
    
    print("=" * 70)
    print("Pinecone Vector Database Examples")
    print("=" * 70)
    
    try:
        # Run examples
        index = pinecone_basics_example()
        
        if index:
            print("\n" + "=" * 70 + "\n")
            namespace_example()
            
            print("\n" + "=" * 70 + "\n")
            metadata_filtering_example()
            
            print("\n" + "=" * 70 + "\n")
            batch_operations_example()
            
            print("\n" + "=" * 70 + "\n")
            cleanup_example()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed!")
        print("=" * 70)
        
        print("\nNext Steps:")
        print("1. Explore different index configurations")
        print("2. Try hybrid search with metadata filtering")
        print("3. Benchmark performance for your use case")
        print("4. Consider pod-based indexes for dedicated resources")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have a valid Pinecone API key and internet connection")
