# Phase 2: RAG & Vector Databases

Build intelligent document Q&A systems using Retrieval-Augmented Generation.

## Files

1. **[01_chromadb_basics.py](01_chromadb_basics.py)** - ChromaDB & RAG
   - Vector database setup
   - Similarity search
   - RAG Q&A
   - Document chunking
   - Metadata filtering

2. **[02_pinecone_basics.py](02_pinecone_basics.py)** - Pinecone (Cloud)
   - Managed vector database
   - Namespaces for multi-tenancy
   - Metadata filtering
   - Batch operations
   - Production patterns

3. **[03_qdrant_basics.py](03_qdrant_basics.py)** - Qdrant
   - In-memory and persistent storage
   - Advanced payload filtering
   - Quantization for efficiency
   - Scroll operations
   - Update and delete

4. **[04_faiss_basics.py](04_faiss_basics.py)** - FAISS
   - Multiple index types (Flat, IVF, HNSW)
   - Performance optimization
   - Save and load indexes
   - Batch search
   - GPU acceleration

5. **[05_vector_db_comparison.py](05_vector_db_comparison.py)** - Comparison Guide
   - Feature comparison matrix
   - Performance benchmarks
   - Use case recommendations
   - Cost analysis
   - Migration guide

6. **[06_document_qa_system.py](06_document_qa_system.py)** - Complete Q&A System
   - Production-ready class
   - File loading (TXT, PDF)
   - Conversation memory
   - Interactive chat mode

## Quick Start

```bash
# Install dependencies
uv pip install chromadb sentence-transformers pypdf langchain langchain-google-genai \
  pinecone-client qdrant-client faiss-cpu cohere scikit-learn

# Set API key
export GEMINI_API_KEY='your-key'

# Optional: Set for cloud vector databases
export PINECONE_API_KEY='your-key'  # For Pinecone examples
export COHERE_API_KEY='your-key'    # For Cohere embeddings

# Run examples
python 01_chromadb_basics.py
python 02_pinecone_basics.py
python 03_qdrant_basics.py
python 04_faiss_basics.py
python 05_vector_db_comparison.py
python 06_document_qa_system.py
```

## What You'll Learn

- Vector databases (ChromaDB, Pinecone, Qdrant, FAISS)
- Embeddings and similarity search
- Retrieval-Augmented Generation (RAG)
- Document processing and chunking
- Conversation memory
- Building reusable Q&A systems
- Choosing the right vector database
- Production deployment patterns
- Cost optimization strategies

## Project Ideas

- PDF chatbot
- Knowledge base Q&A
- Semantic search engine
- Research assistant

## Next Steps

After completing Phase 2, move to [Phase 3: Production](../phase3_production/) to learn about deploying APIs and demos.
