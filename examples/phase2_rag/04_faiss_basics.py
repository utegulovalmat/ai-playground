"""
FAISS Vector Database Example
==============================
This example demonstrates using FAISS (Facebook AI Similarity Search),
a library for efficient similarity search and clustering of dense vectors.

Requirements:
- faiss-cpu>=1.7.4 (or faiss-gpu for GPU support)
- sentence-transformers>=2.0.0
- numpy

What is FAISS?
FAISS is a library developed by Facebook Research for efficient similarity
search. It's optimized for speed and can handle billions of vectors.

Best Practices:
- Choose appropriate index type for your use case
- Use IVF for large datasets (millions of vectors)
- Consider GPU acceleration for very large datasets
- Save and load indexes to avoid recomputing
"""

import numpy as np
import time


def faiss_basics_example():
    """
    Basic FAISS operations with Flat index (exact search).
    """
    print("=== FAISS Basics (Flat Index) ===")
    
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        
        # Load embedding model
        print("\nLoading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        dimension = 384  # Model output dimension
        
        # Create Flat index (exact search, no compression)
        print(f"\nCreating Flat index (dimension: {dimension})...")
        index = faiss.IndexFlatL2(dimension)  # L2 distance
        
        print(f"✓ Index created")
        print(f"  Type: Flat (exact search)")
        print(f"  Is trained: {index.is_trained}")
        print(f"  Total vectors: {index.ntotal}")
        
        # Sample documents
        documents = [
            "FAISS is a library for similarity search",
            "Python is used for data science",
            "Vector databases enable semantic search",
            "Machine learning requires large datasets",
            "FAISS can handle billions of vectors"
        ]
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        embeddings = model.encode(documents)
        embeddings = embeddings.astype('float32')  # FAISS requires float32
        
        # Add to index
        print(f"Adding {len(embeddings)} vectors to index...")
        index.add(embeddings)
        
        print(f"✓ Vectors added")
        print(f"  Total vectors in index: {index.ntotal}")
        
        # Search
        print("\n--- Searching ---")
        query_text = "Tell me about similarity search"
        query_vector = model.encode([query_text]).astype('float32')
        
        k = 3  # Number of nearest neighbors
        distances, indices = index.search(query_vector, k)
        
        print(f"Query: '{query_text}'")
        print(f"\nTop {k} matches:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            print(f"  {i+1}. Distance: {dist:.4f}")
            print(f"     Text: {documents[idx]}\n")
        
        return index, embeddings, documents
        
    except ImportError:
        print("❌ FAISS not installed. Install with: uv pip install faiss-cpu")
        return None, None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None, None


def index_types_example():
    """
    Comparing different FAISS index types.
    """
    print("\n=== FAISS Index Types ===")
    
    try:
        import faiss
        
        dimension = 384
        
        # Generate sample data
        n_vectors = 10000
        print(f"\nGenerating {n_vectors} random vectors...")
        vectors = np.random.random((n_vectors, dimension)).astype('float32')
        
        # 1. Flat Index (exact, slow for large datasets)
        print("\n--- Flat Index (Exact Search) ---")
        index_flat = faiss.IndexFlatL2(dimension)
        
        start = time.time()
        index_flat.add(vectors)
        add_time = time.time() - start
        
        query = np.random.random((1, dimension)).astype('float32')
        start = time.time()
        D, I = index_flat.search(query, 10)
        search_time = time.time() - start
        
        print(f"  Add time: {add_time:.4f}s")
        print(f"  Search time: {search_time:.6f}s")
        print(f"  Memory: ~{vectors.nbytes / 1024 / 1024:.1f} MB")
        print(f"  Accuracy: 100% (exact)")
        
        # 2. IVF Index (approximate, faster)
        print("\n--- IVF Index (Approximate Search) ---")
        nlist = 100  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        # Train the index
        print("  Training index...")
        index_ivf.train(vectors)
        
        start = time.time()
        index_ivf.add(vectors)
        add_time = time.time() - start
        
        index_ivf.nprobe = 10  # Number of clusters to search
        start = time.time()
        D, I = index_ivf.search(query, 10)
        search_time = time.time() - start
        
        print(f"  Add time: {add_time:.4f}s")
        print(f"  Search time: {search_time:.6f}s")
        print(f"  Memory: ~{vectors.nbytes / 1024 / 1024:.1f} MB")
        print(f"  Accuracy: ~95-99% (approximate)")
        
        # 3. HNSW Index (hierarchical graph, very fast)
        print("\n--- HNSW Index (Graph-Based) ---")
        M = 32  # Number of connections per layer
        index_hnsw = faiss.IndexHNSWFlat(dimension, M)
        
        start = time.time()
        index_hnsw.add(vectors)
        add_time = time.time() - start
        
        start = time.time()
        D, I = index_hnsw.search(query, 10)
        search_time = time.time() - start
        
        print(f"  Add time: {add_time:.4f}s")
        print(f"  Search time: {search_time:.6f}s")
        print(f"  Memory: Higher (graph structure)")
        print(f"  Accuracy: ~99% (approximate)")
        
        # Recommendations
        print("\n--- Index Selection Guide ---")
        print("• Flat: <10K vectors, need exact results")
        print("• IVF: 10K-10M vectors, good balance")
        print("• HNSW: Need fastest search, have memory")
        print("• IVF+PQ: 10M+ vectors, limited memory")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def save_and_load_example(index, embeddings):
    """
    Saving and loading FAISS indexes.
    """
    print("\n=== Save & Load Index ===")
    
    try:
        import faiss
        
        # Save index
        index_file = "faiss_index.bin"
        print(f"\nSaving index to {index_file}...")
        faiss.write_index(index, index_file)
        print("✓ Index saved")
        
        # Load index
        print(f"\nLoading index from {index_file}...")
        loaded_index = faiss.read_index(index_file)
        print("✓ Index loaded")
        print(f"  Total vectors: {loaded_index.ntotal}")
        
        # Verify it works
        query = embeddings[0:1]  # Use first vector as query
        D, I = loaded_index.search(query, 3)
        print(f"\nTest search successful!")
        print(f"  Found {len(I[0])} results")
        
        # Cleanup
        import os
        if os.path.exists(index_file):
            os.remove(index_file)
            print(f"\n✓ Cleaned up {index_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def batch_search_example(index, embeddings):
    """
    Batch searching for efficiency.
    """
    print("\n=== Batch Search ===")
    
    try:
        
        # Single query
        print("\nSingle query search...")
        query = embeddings[0:1]
        start = time.time()
        D, I = index.search(query, 5)
        single_time = time.time() - start
        print(f"  Time: {single_time:.6f}s")
        
        # Batch queries
        print("\nBatch query search (100 queries)...")
        queries = embeddings[:100]
        start = time.time()
        D, I = index.search(queries, 5)
        batch_time = time.time() - start
        print(f"  Time: {batch_time:.6f}s")
        print(f"  Per query: {batch_time/100:.6f}s")
        print(f"  Speedup: {(single_time * 100) / batch_time:.1f}x")
        
        print("\n💡 Batch searching is much more efficient!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def range_search_example(index, embeddings):
    """
    Range search: find all vectors within a distance threshold.
    """
    print("\n=== Range Search ===")
    
    try:
        
        query = embeddings[0:1]
        threshold = 50.0  # Distance threshold
        
        print(f"\nFinding all vectors within distance {threshold}...")
        
        # Range search
        lims, D, I = index.range_search(query, threshold)
        
        n_results = lims[1] - lims[0]
        print(f"✓ Found {n_results} vectors within threshold")
        
        if n_results > 0:
            print(f"\nFirst few results:")
            for i in range(min(5, n_results)):
                print(f"  Distance: {D[i]:.4f}, Index: {I[i]}")
        
        print("\n💡 Range search is useful for finding all similar items,")
        print("   not just top-k.")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def gpu_acceleration_info():
    """
    Information about GPU acceleration.
    """
    print("\n=== GPU Acceleration ===")
    
    try:
        import faiss
        
        # Check if GPU is available
        gpu_available = faiss.get_num_gpus() > 0
        
        if gpu_available:
            print(f"\n✓ GPU available! Found {faiss.get_num_gpus()} GPU(s)")
            print("\nTo use GPU:")
            print("""
# Convert CPU index to GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# Or create GPU index directly
gpu_index = faiss.GpuIndexFlatL2(res, dimension)
            """)
        else:
            print("\n✗ No GPU detected")
            print("\nTo enable GPU support:")
            print("1. Install CUDA toolkit")
            print("2. Install faiss-gpu: pip install faiss-gpu")
            print("\nGPU benefits:")
            print("• 5-10x faster search")
            print("• Essential for billion-scale datasets")
            print("• Lower latency for real-time applications")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("=" * 70)
    print("FAISS Vector Database Examples")
    print("=" * 70)
    
    try:
        # Run examples
        index, embeddings, documents = faiss_basics_example()
        
        if index is not None:
            print("\n" + "=" * 70 + "\n")
            index_types_example()
            
            print("\n" + "=" * 70 + "\n")
            save_and_load_example(index, embeddings)
            
            print("\n" + "=" * 70 + "\n")
            batch_search_example(index, embeddings)
            
            print("\n" + "=" * 70 + "\n")
            range_search_example(index, embeddings)
            
            print("\n" + "=" * 70 + "\n")
            gpu_acceleration_info()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed!")
        print("=" * 70)
        
        print("\nNext Steps:")
        print("1. Experiment with different index types")
        print("2. Try GPU acceleration if available")
        print("3. Benchmark with your dataset size")
        print("4. Explore product quantization for compression")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have installed: uv pip install faiss-cpu sentence-transformers")
