"""
Embeddings Basics - Multi-Provider Examples
============================================
This example demonstrates how to generate embeddings using different providers
and perform similarity search operations.

Requirements:
- openai>=1.0.0
- google-genai>=1.0.0
- cohere>=5.0.0
- sentence-transformers>=2.0.0
- scikit-learn>=1.3.0
- numpy

API Keys (set as needed):
- OPENAI_API_KEY
- GEMINI_API_KEY
- COHERE_API_KEY

What are Embeddings?
Embeddings are numerical representations of text that capture semantic meaning.
Similar texts have similar embeddings, enabling semantic search and comparison.

Best Practices:
- Choose embedding models based on your use case (cost, performance, language)
- Use batch processing for multiple texts
- Cache embeddings to avoid redundant API calls
- Normalize embeddings for cosine similarity
"""

import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# OpenAI Embeddings
# ============================================================================

def openai_embeddings_example():
    """
    OpenAI embeddings example.
    Models: text-embedding-3-small (1536 dims), text-embedding-3-large (3072 dims)
    """
    print("=== OpenAI Embeddings ===")
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Sample texts
        texts = [
            "Python is a programming language",
            "JavaScript is used for web development",
            "Machine learning uses neural networks",
            "Python is great for data science"
        ]
        
        # Generate embeddings (small model - cost effective)
        print("\nGenerating embeddings with text-embedding-3-small...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  Dimensions: {len(embeddings[0])}")
        print(f"  Total tokens: {response.usage.total_tokens}")
        
        # Calculate similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        print("\nSimilarity Matrix:")
        print("Text 0 vs Text 3 (both about Python):", f"{similarity_matrix[0][3]:.4f}")
        print("Text 0 vs Text 1 (different topics):", f"{similarity_matrix[0][1]:.4f}")
        
        # Dimensionality reduction (new feature in v3)
        print("\n--- Dimensionality Reduction ---")
        response_reduced = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts[0],
            dimensions=512  # Reduce from 1536 to 512
        )
        
        print(f"✓ Reduced dimensions: {len(response_reduced.data[0].embedding)}")
        print("  Benefit: Faster search, less storage, minimal quality loss")
        
        return embeddings
        
    except ImportError:
        print("❌ OpenAI not installed. Install with: uv pip install openai")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure OPENAI_API_KEY is set")
        return None


# ============================================================================
# Google Gemini Embeddings
# ============================================================================

def google_embeddings_example():
    """
    Google Gemini embeddings example.
    Model: text-embedding-004 (768 dims)
    Supports task-type optimization
    """
    print("\n=== Google Gemini Embeddings ===")
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        
        # Sample texts
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A fast auburn fox leaps above a sleepy canine",
            "Python programming is fun and powerful"
        ]
        
        query = "What did the fox do?"
        
        # Generate embeddings with task types
        print("\nGenerating embeddings with task-type optimization...")
        
        # Document embeddings (for storage)
        doc_embeddings = []
        for doc in documents:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=doc,
                task_type="retrieval_document"  # Optimized for storage
            )
            doc_embeddings.append(result['embedding'])
        
        # Query embedding (for search)
        query_result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"  # Optimized for queries
        )
        query_embedding = query_result['embedding']
        
        print(f"✓ Generated {len(doc_embeddings)} document embeddings")
        print(f"  Dimensions: {len(doc_embeddings[0])}")
        
        # Find most similar document
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        
        print(f"\nQuery: '{query}'")
        print(f"Best match: '{documents[best_match_idx]}'")
        print(f"Similarity: {similarities[best_match_idx]:.4f}")
        
        # Show all similarities
        print("\nAll similarities:")
        for i, (doc, sim) in enumerate(zip(documents, similarities)):
            print(f"  {i+1}. [{sim:.4f}] {doc}")
        
        # Other task types
        print("\n--- Available Task Types ---")
        print("• retrieval_query: For search queries")
        print("• retrieval_document: For documents to be retrieved")
        print("• semantic_similarity: For comparing text similarity")
        print("• classification: For text classification")
        print("• clustering: For grouping similar texts")
        
        return doc_embeddings
        
    except ImportError:
        print("❌ Google GenAI not installed. Install with: uv pip install google-genai")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure GEMINI_API_KEY is set")
        return None


# ============================================================================
# Cohere Embeddings
# ============================================================================

def cohere_embeddings_example():
    """
    Cohere embeddings example.
    Models: embed-english-v3.0, embed-multilingual-v3.0
    Supports input types for better performance
    """
    print("\n=== Cohere Embeddings ===")
    
    try:
        import cohere
        
        co = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))
        
        # Sample texts
        documents = [
            "Artificial intelligence is transforming technology",
            "Machine learning models require large datasets",
            "The weather is sunny today"
        ]
        
        query = "Tell me about AI and ML"
        
        print("\nGenerating embeddings with input types...")
        
        # Document embeddings
        doc_response = co.embed(
            texts=documents,
            model="embed-english-v3.0",
            input_type="search_document"  # For documents
        )
        doc_embeddings = doc_response.embeddings
        
        # Query embedding
        query_response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"  # For queries
        )
        query_embedding = query_response.embeddings[0]
        
        print(f"✓ Generated {len(doc_embeddings)} embeddings")
        print(f"  Dimensions: {len(doc_embeddings[0])}")
        
        # Find best match
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        
        print(f"\nQuery: '{query}'")
        print(f"Best match: '{documents[best_match_idx]}'")
        print(f"Similarity: {similarities[best_match_idx]:.4f}")
        
        # Multilingual example
        print("\n--- Multilingual Support ---")
        multilingual_texts = [
            "Hello, how are you?",
            "Bonjour, comment allez-vous?",
            "Hola, ¿cómo estás?",
            "Hallo, wie geht es dir?"
        ]
        
        multi_response = co.embed(
            texts=multilingual_texts,
            model="embed-multilingual-v3.0",
            input_type="clustering"
        )
        
        print(f"✓ Multilingual embeddings: {len(multi_response.embeddings)}")
        print("  Supports 100+ languages!")
        
        # Show cross-language similarity
        multi_embeddings = multi_response.embeddings
        cross_sim = cosine_similarity(multi_embeddings)
        print(f"\nEnglish vs French similarity: {cross_sim[0][1]:.4f}")
        print(f"English vs Spanish similarity: {cross_sim[0][2]:.4f}")
        
        return doc_embeddings
        
    except ImportError:
        print("❌ Cohere not installed. Install with: uv pip install cohere")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure COHERE_API_KEY is set")
        return None


# ============================================================================
# Sentence Transformers (Local)
# ============================================================================

def sentence_transformers_example():
    """
    Sentence Transformers example (local, no API required).
    Models: all-MiniLM-L6-v2 (fast), all-mpnet-base-v2 (quality)
    """
    print("\n=== Sentence Transformers (Local) ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load model (downloads on first use)
        print("\nLoading model: all-MiniLM-L6-v2 (fast, 384 dims)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample texts
        sentences = [
            "The cat sits on the mat",
            "A feline rests on a rug",
            "Dogs are great pets",
            "Cats and dogs are popular animals"
        ]
        
        # Generate embeddings (runs locally!)
        print("Generating embeddings locally...")
        embeddings = model.encode(sentences)
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"  Dimensions: {embeddings.shape[1]}")
        print(f"  No API costs! Runs on your machine")
        
        # Calculate similarities
        similarities = cosine_similarity(embeddings)
        
        print("\nSimilarity Matrix:")
        print(f"Sentence 0 vs 1 (similar meaning): {similarities[0][1]:.4f}")
        print(f"Sentence 0 vs 2 (different topics): {similarities[0][2]:.4f}")
        
        # Batch processing example
        print("\n--- Batch Processing ---")
        large_batch = [f"This is sentence number {i}" for i in range(100)]
        
        import time
        start = time.time()
        model.encode(large_batch, batch_size=32, show_progress_bar=False)
        elapsed = time.time() - start
        
        print(f"✓ Processed {len(large_batch)} sentences in {elapsed:.2f}s")
        print(f"  Speed: {len(large_batch)/elapsed:.1f} sentences/sec")
        
        # Model comparison
        print("\n--- Popular Models ---")
        print("• all-MiniLM-L6-v2: Fast, 384 dims, good quality")
        print("• all-mpnet-base-v2: Best quality, 768 dims, slower")
        print("• paraphrase-multilingual: 50+ languages")
        print("• all-MiniLM-L12-v2: Balanced speed/quality")
        
        return embeddings
        
    except ImportError:
        print("❌ Sentence Transformers not installed.")
        print("   Install with: uv pip install sentence-transformers")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# ============================================================================
# Practical Similarity Search
# ============================================================================

def similarity_search_demo():
    """
    Practical similarity search demonstration.
    Shows how to find similar items in a knowledge base.
    """
    print("\n=== Practical Similarity Search ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Knowledge base
        knowledge_base = [
            "How to install Python packages using pip",
            "Creating virtual environments in Python",
            "Introduction to machine learning with scikit-learn",
            "Building web applications with Flask",
            "Data analysis using pandas and numpy",
            "Writing unit tests in Python with pytest",
            "Deploying Python applications to production",
            "Understanding Python decorators and generators"
        ]
        
        # Generate embeddings for knowledge base
        print("\nIndexing knowledge base...")
        kb_embeddings = model.encode(knowledge_base)
        
        # User queries
        queries = [
            "How do I set up a Python environment?",
            "What's the best way to test my code?",
            "I want to build a website"
        ]
        
        print("\nSearching knowledge base...\n")
        
        for query in queries:
            # Generate query embedding
            query_embedding = model.encode([query])
            
            # Find similarities
            similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
            
            # Get top 2 results
            top_indices = np.argsort(similarities)[-2:][::-1]
            
            print(f"Query: '{query}'")
            print("Top matches:")
            for idx in top_indices:
                print(f"  [{similarities[idx]:.4f}] {knowledge_base[idx]}")
            print()
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# Batch Processing Optimization
# ============================================================================

def batch_processing_example():
    """
    Demonstrates efficient batch processing for large datasets.
    """
    print("\n=== Batch Processing Optimization ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        import time
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate test data
        texts = [f"Sample document number {i} about various topics" for i in range(500)]
        
        # Method 1: One at a time (slow)
        print("\nMethod 1: Processing one at a time...")
        start = time.time()
        _ = [model.encode([text])[0] for text in texts[:50]]
        time_slow = time.time() - start
        print(f"  Time for 50 texts: {time_slow:.2f}s")
        
        # Method 2: Batch processing (fast)
        print("\nMethod 2: Batch processing...")
        start = time.time()
        _ = model.encode(texts[:50], batch_size=32, show_progress_bar=False)
        time_fast = time.time() - start
        print(f"  Time for 50 texts: {time_fast:.2f}s")
        print(f"  Speedup: {time_slow/time_fast:.1f}x faster!")
        
        # Best practices
        print("\n--- Batch Processing Tips ---")
        print("• Use batch_size=32 or 64 for best performance")
        print("• Larger batches = more memory, diminishing returns")
        print("• Enable show_progress_bar for long operations")
        print("• Consider GPU acceleration for very large datasets")
        
    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# Provider Comparison
# ============================================================================

def provider_comparison():
    """
    Compare different embedding providers.
    """
    print("\n=== Provider Comparison ===")
    
    print("""
┌─────────────────┬──────────┬───────────┬─────────────┬──────────────┐
│ Provider        │ Dims     │ Cost      │ Speed       │ Best For     │
├─────────────────┼──────────┼───────────┼─────────────┼──────────────┤
│ OpenAI          │ 1536     │ Low       │ Fast        │ General use  │
│ text-emb-3-sm   │          │ $0.02/1M  │             │              │
├─────────────────┼──────────┼───────────┼─────────────┼──────────────┤
│ OpenAI          │ 3072     │ Medium    │ Fast        │ High quality │
│ text-emb-3-lg   │          │ $0.13/1M  │             │              │
├─────────────────┼──────────┼───────────┼─────────────┼──────────────┤
│ Google Gemini   │ 768      │ Free*     │ Fast        │ Free tier    │
│ text-emb-004    │          │           │             │              │
├─────────────────┼──────────┼───────────┼─────────────┼──────────────┤
│ Cohere          │ 1024     │ Low       │ Fast        │ Multilingual │
│ embed-v3.0      │          │ $0.10/1M  │             │              │
├─────────────────┼──────────┼───────────┼─────────────┼──────────────┤
│ Sentence Trans. │ 384-768  │ FREE      │ Medium      │ Privacy,     │
│ (Local)         │          │           │             │ No API costs │
└─────────────────┴──────────┴───────────┴─────────────┴──────────────┘

*Free tier available, check current pricing

Recommendations:
• Development/Testing: Sentence Transformers (free, local)
• Production (general): OpenAI text-embedding-3-small (best balance)
• High quality needs: OpenAI text-embedding-3-large
• Multilingual: Cohere embed-multilingual-v3.0
• Budget conscious: Google Gemini (free tier) or Sentence Transformers
• Privacy sensitive: Sentence Transformers (runs locally)
    """)


if __name__ == "__main__":
    print("=" * 70)
    print("Embeddings Basics - Multi-Provider Examples")
    print("=" * 70)
    
    # Check which API keys are available
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_cohere = bool(os.environ.get("COHERE_API_KEY"))
    
    print("\nAPI Keys Status:")
    print(f"  OpenAI: {'✓' if has_openai else '✗'}")
    print(f"  Gemini: {'✓' if has_gemini else '✗'}")
    print(f"  Cohere: {'✓' if has_cohere else '✗'}")
    print(f"  Local (Sentence Transformers): ✓ (no key needed)")
    
    print("\n" + "=" * 70 + "\n")
    
    # Run examples based on available keys
    if has_openai:
        openai_embeddings_example()
        print("\n" + "=" * 70 + "\n")
    
    if has_gemini:
        google_embeddings_example()
        print("\n" + "=" * 70 + "\n")
    
    if has_cohere:
        cohere_embeddings_example()
        print("\n" + "=" * 70 + "\n")
    
    # Always run local examples (no API key needed)
    sentence_transformers_example()
    print("\n" + "=" * 70 + "\n")
    
    similarity_search_demo()
    print("\n" + "=" * 70 + "\n")
    
    batch_processing_example()
    print("\n" + "=" * 70 + "\n")
    
    provider_comparison()
    
    print("\n" + "=" * 70)
    print("✓ Examples completed!")
    print("=" * 70)
    
    print("\nNext Steps:")
    print("1. Try different embedding models")
    print("2. Experiment with your own text data")
    print("3. Move to Phase 2 to learn about vector databases")
    print("4. Build a semantic search system")
