"""
Vector Database Comparison Guide
=================================
This guide helps you choose the right vector database for your use case.

Databases Covered:
- ChromaDB: Simple, embedded database
- Pinecone: Managed cloud service
- Qdrant: High-performance, flexible deployment
- FAISS: Facebook's similarity search library
- Weaviate: GraphQL-based with hybrid search

Requirements:
- All vector database clients (see individual examples)
- sentence-transformers>=2.0.0
"""

import time


def feature_comparison():
    """
    Feature comparison matrix for all vector databases.
    """
    print("=== Feature Comparison ===\n")
    
    print("""
┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ Feature      │ ChromaDB │ Pinecone │ Qdrant   │ FAISS    │ Weaviate │
├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
│ Deployment   │ Embedded │ Cloud    │ Both     │ Local    │ Both     │
│ Setup        │ Easy     │ Easy     │ Medium   │ Easy     │ Medium   │
│ Scalability  │ Small    │ Large    │ Large    │ Large    │ Large    │
│ Cost         │ Free     │ Paid     │ Free/Paid│ Free     │ Free/Paid│
│ Metadata     │ Yes      │ Yes      │ Yes      │ Limited  │ Yes      │
│ Filtering    │ Basic    │ Advanced │ Advanced │ Basic    │ Advanced │
│ Persistence  │ Yes      │ Yes      │ Yes      │ Manual   │ Yes      │
│ Multi-tenant │ No       │ Yes      │ Yes      │ No       │ Yes      │
│ Hybrid Search│ No       │ Limited  │ Yes      │ No       │ Yes      │
│ GraphQL      │ No       │ No       │ No       │ No       │ Yes      │
│ GPU Support  │ No       │ Yes      │ Yes      │ Yes      │ Limited  │
└──────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
    """)


def performance_benchmark():
    """
    Simple performance benchmark comparing vector databases.
    """
    print("\n=== Performance Benchmark ===\n")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Setup
        model = SentenceTransformer('all-MiniLM-L6-v2')
        n_vectors = 1000
        dimension = 384
        
        print(f"Benchmark setup:")
        print(f"  Vectors: {n_vectors}")
        print(f"  Dimensions: {dimension}")
        print(f"  Queries: 100\n")
        
        # Generate test data
        print("Generating test data...")
        texts = [f"Sample document {i} with various content" for i in range(n_vectors)]
        embeddings = model.encode(texts, show_progress_bar=False)
        query_embeddings = embeddings[:100]  # Use first 100 as queries
        
        results = {}
        
        # ChromaDB
        print("\n--- ChromaDB ---")
        try:
            import chromadb
            
            client = chromadb.Client()
            collection = client.create_collection("benchmark")
            
            start = time.time()
            collection.add(
                ids=[str(i) for i in range(n_vectors)],
                embeddings=embeddings.tolist(),
                documents=texts
            )
            insert_time = time.time() - start
            
            start = time.time()
            for query in query_embeddings:
                collection.query(query_embeddings=[query.tolist()], n_results=10)
            query_time = time.time() - start
            
            results['ChromaDB'] = {
                'insert': insert_time,
                'query': query_time,
                'qps': 100 / query_time
            }
            print(f"  Insert: {insert_time:.3f}s")
            print(f"  Query (100): {query_time:.3f}s")
            print(f"  QPS: {100/query_time:.1f}")
            
        except Exception as e:
            print(f"  Skipped: {e}")
        
        # FAISS
        print("\n--- FAISS ---")
        try:
            import faiss
            
            index = faiss.IndexFlatL2(dimension)
            
            start = time.time()
            index.add(embeddings.astype('float32'))
            insert_time = time.time() - start
            
            start = time.time()
            for query in query_embeddings:
                index.search(query.reshape(1, -1).astype('float32'), 10)
            query_time = time.time() - start
            
            results['FAISS'] = {
                'insert': insert_time,
                'query': query_time,
                'qps': 100 / query_time
            }
            print(f"  Insert: {insert_time:.3f}s")
            print(f"  Query (100): {query_time:.3f}s")
            print(f"  QPS: {100/query_time:.1f}")
            
        except Exception as e:
            print(f"  Skipped: {e}")
        
        # Qdrant
        print("\n--- Qdrant ---")
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            client = QdrantClient(":memory:")
            client.create_collection(
                collection_name="benchmark",
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
            )
            
            points = [
                PointStruct(id=i, vector=emb.tolist(), payload={"text": text})
                for i, (emb, text) in enumerate(zip(embeddings, texts))
            ]
            
            start = time.time()
            client.upsert(collection_name="benchmark", points=points)
            insert_time = time.time() - start
            
            start = time.time()
            for query in query_embeddings:
                client.query_points(
                    collection_name="benchmark",
                    query=query.tolist(),
                    limit=10
                )
            query_time = time.time() - start
            
            results['Qdrant'] = {
                'insert': insert_time,
                'query': query_time,
                'qps': 100 / query_time
            }
            print(f"  Insert: {insert_time:.3f}s")
            print(f"  Query (100): {query_time:.3f}s")
            print(f"  QPS: {100/query_time:.1f}")
            
        except Exception as e:
            print(f"  Skipped: {e}")
        
        # Summary
        if results:
            print("\n--- Summary ---")
            fastest_insert = min(results.items(), key=lambda x: x[1]['insert'])
            fastest_query = min(results.items(), key=lambda x: x[1]['query'])
            
            print(f"Fastest insert: {fastest_insert[0]} ({fastest_insert[1]['insert']:.3f}s)")
            print(f"Fastest query: {fastest_query[0]} ({fastest_query[1]['qps']:.1f} QPS)")
            
            print("\n💡 Note: Results vary by dataset size, hardware, and configuration")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def use_case_recommendations():
    """
    Recommendations for different use cases.
    """
    print("\n=== Use Case Recommendations ===\n")
    
    print("""
🎯 GETTING STARTED / PROTOTYPING
├─ Best choice: ChromaDB or FAISS
├─ Why: Zero setup, runs locally, easy to learn
└─ Example: Building a proof-of-concept RAG system

🎯 SMALL TO MEDIUM SCALE (<1M vectors)
├─ Best choice: ChromaDB or Qdrant (local)
├─ Why: Simple deployment, good performance, low cost
└─ Example: Company knowledge base, document search

🎯 LARGE SCALE (1M-100M vectors)
├─ Best choice: Qdrant or Pinecone
├─ Why: Optimized for scale, managed infrastructure
└─ Example: E-commerce product search, content recommendation

🎯 VERY LARGE SCALE (100M+ vectors)
├─ Best choice: FAISS (with GPU) or Pinecone
├─ Why: Handles billions of vectors, GPU acceleration
└─ Example: Web-scale search, image similarity

🎯 MULTI-TENANT APPLICATIONS
├─ Best choice: Pinecone or Qdrant
├─ Why: Built-in namespace support, isolation
└─ Example: SaaS platforms, multi-customer apps

🎯 HYBRID SEARCH (Keyword + Semantic)
├─ Best choice: Weaviate or Qdrant
├─ Why: Native hybrid search support
└─ Example: Advanced search engines, research tools

🎯 BUDGET CONSTRAINED
├─ Best choice: FAISS or Qdrant (self-hosted)
├─ Why: Free, open-source, no API costs
└─ Example: Startups, personal projects

🎯 PRIVACY SENSITIVE
├─ Best choice: FAISS or Qdrant (self-hosted)
├─ Why: Data stays on your infrastructure
└─ Example: Healthcare, financial services

🎯 RAPID DEVELOPMENT
├─ Best choice: Pinecone or ChromaDB
├─ Why: Minimal setup, managed service
└─ Example: Hackathons, MVPs, demos

🎯 COMPLEX FILTERING NEEDS
├─ Best choice: Qdrant or Weaviate
├─ Why: Advanced metadata filtering, GraphQL
└─ Example: E-commerce with many filters
    """)


def cost_comparison():
    """
    Cost comparison for different scales.
    """
    print("\n=== Cost Comparison ===\n")
    
    print("""
💰 COST BREAKDOWN (Approximate, as of 2024)

ChromaDB (Self-hosted)
├─ Infrastructure: $10-100/month (cloud VM)
├─ Storage: $0.02/GB/month
├─ Compute: Included in VM cost
└─ Total: ~$20-200/month for small-medium scale

Pinecone (Managed)
├─ Starter: $70/month (100K vectors, 1 pod)
├─ Standard: $0.096/hour per pod (~$70/month)
├─ Storage: Included
└─ Total: ~$70-500+/month depending on scale

Qdrant Cloud (Managed)
├─ Free tier: 1GB storage
├─ Paid: $25/month minimum
├─ Storage: $0.25/GB/month
└─ Total: $0-100+/month

FAISS (Self-hosted)
├─ Infrastructure: $10-100/month (cloud VM)
├─ GPU (optional): +$200-1000/month
├─ Storage: $0.02/GB/month
└─ Total: ~$20-1000+/month

Weaviate (Self-hosted or Cloud)
├─ Self-hosted: $10-100/month (cloud VM)
├─ Cloud: Custom pricing
├─ Storage: Varies
└─ Total: ~$20-500+/month

💡 COST OPTIMIZATION TIPS:
• Start with free/cheap options (ChromaDB, FAISS)
• Use quantization to reduce storage costs
• Implement caching to reduce query costs
• Monitor usage and scale appropriately
• Consider spot instances for self-hosted options
    """)


def migration_guide():
    """
    Guide for migrating between vector databases.
    """
    print("\n=== Migration Guide ===\n")
    
    print("""
🔄 MIGRATION STRATEGIES

1. EXPORT-IMPORT PATTERN
   ├─ Export vectors and metadata from source
   ├─ Transform to target format
   └─ Import to destination database
   
2. DUAL-WRITE PATTERN
   ├─ Write to both old and new databases
   ├─ Gradually shift reads to new database
   └─ Deprecate old database once validated
   
3. SNAPSHOT-RESTORE PATTERN
   ├─ Take snapshot of source database
   ├─ Process offline
   └─ Load into new database

EXAMPLE: ChromaDB → Pinecone

```python
# 1. Export from ChromaDB
collection = chromadb_client.get_collection("my_collection")
data = collection.get(include=["embeddings", "documents", "metadatas"])

# 2. Transform and import to Pinecone
from pinecone import Pinecone
pc = Pinecone(api_key="...")
index = pc.Index("my-index")

vectors = []
for id, emb, meta in zip(data['ids'], data['embeddings'], data['metadatas']):
    vectors.append({
        "id": id,
        "values": emb,
        "metadata": meta
    })

# Batch upsert
index.upsert(vectors=vectors, batch_size=100)
```

⚠️  MIGRATION CHECKLIST:
□ Backup source database
□ Test with small dataset first
□ Verify vector dimensions match
□ Map metadata fields correctly
□ Test query results match
□ Monitor performance
□ Plan for downtime or dual-write period
    """)


def decision_tree():
    """
    Decision tree for choosing a vector database.
    """
    print("\n=== Decision Tree ===\n")
    
    print("""
START HERE: What's your primary concern?

├─ 💰 COST
│  ├─ Free only → FAISS or ChromaDB
│  └─ Budget available → Qdrant or Pinecone
│
├─ ⚡ SPEED
│  ├─ <1M vectors → FAISS or Qdrant
│  └─ >1M vectors → FAISS (GPU) or Pinecone
│
├─ 🔧 EASE OF USE
│  ├─ Beginner → ChromaDB
│  └─ Production → Pinecone or Qdrant Cloud
│
├─ 📊 SCALE
│  ├─ <100K vectors → ChromaDB or FAISS
│  ├─ 100K-10M vectors → Qdrant or Pinecone
│  └─ >10M vectors → FAISS or Pinecone
│
├─ 🔒 PRIVACY
│  ├─ Must be on-premise → FAISS or Qdrant (self-hosted)
│  └─ Cloud OK → Any
│
├─ 🎯 FEATURES
│  ├─ Need hybrid search → Weaviate or Qdrant
│  ├─ Need multi-tenancy → Pinecone or Qdrant
│  ├─ Need GraphQL → Weaviate
│  └─ Simple vector search → ChromaDB or FAISS
│
└─ 🚀 DEPLOYMENT
   ├─ Embedded in app → ChromaDB or FAISS
   ├─ Managed service → Pinecone or Qdrant Cloud
   └─ Self-hosted → Qdrant or Weaviate

QUICK RECOMMENDATIONS:
• Just starting? → ChromaDB
• Building MVP? → Pinecone
• Need control? → Qdrant (self-hosted)
• Maximum performance? → FAISS (with GPU)
• Advanced features? → Weaviate
    """)


if __name__ == "__main__":
    print("=" * 70)
    print("Vector Database Comparison Guide")
    print("=" * 70)
    
    feature_comparison()
    print("\n" + "=" * 70)
    
    performance_benchmark()
    print("\n" + "=" * 70)
    
    use_case_recommendations()
    print("\n" + "=" * 70)
    
    cost_comparison()
    print("\n" + "=" * 70)
    
    migration_guide()
    print("\n" + "=" * 70)
    
    decision_tree()
    
    print("\n" + "=" * 70)
    print("✓ Comparison guide complete!")
    print("=" * 70)
    
    print("\nNext Steps:")
    print("1. Review the decision tree to choose your database")
    print("2. Try the corresponding example file")
    print("3. Run benchmarks with your actual data")
    print("4. Start with a free option, scale as needed")
