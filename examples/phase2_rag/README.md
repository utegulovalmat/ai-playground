# Phase 2: RAG & Vector Databases

Build intelligent document Q&A systems using Retrieval-Augmented Generation.

## Files

1. **[01_chromadb_basics.py](01_chromadb_basics.py)** - ChromaDB & RAG
   - Vector database setup
   - Similarity search
   - RAG Q&A
   - Document chunking
   - Metadata filtering

2. **[02_document_qa_system.py](02_document_qa_system.py)** - Complete Q&A System
   - Production-ready class
   - File loading (TXT, PDF)
   - Conversation memory
   - Interactive chat mode

## Quick Start

```bash
# Install dependencies
uv pip install chromadb sentence-transformers pypdf langchain langchain-google-genai

# Set API key
export GEMINI_API_KEY='your-key'

# Run examples
python 01_chromadb_basics.py
python 02_document_qa_system.py
```

## What You'll Learn

- Vector databases (ChromaDB)
- Embeddings and similarity search
- Retrieval-Augmented Generation (RAG)
- Document processing and chunking
- Conversation memory
- Building reusable Q&A systems

## Project Ideas

- PDF chatbot
- Knowledge base Q&A
- Semantic search engine
- Research assistant

## Next Steps

After completing Phase 2, move to [Phase 3: Production](../phase3_production/) to learn about deploying APIs and demos.
