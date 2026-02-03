"""
Phase 2: RAG with ChromaDB Example
===================================
This example demonstrates Retrieval-Augmented Generation (RAG) using ChromaDB.

Requirements:
- chromadb>=0.4.0
- langchain>=1.2.0
- langchain-google-genai>=4.2.0
- sentence-transformers>=2.0.0
- GEMINI_API_KEY environment variable

What is RAG?
RAG combines retrieval from a knowledge base with LLM generation.
This allows the LLM to answer questions using your custom documents.

Best Practices:
- Use appropriate chunk sizes (500-1000 tokens)
- Choose good embedding models
- Implement proper error handling
- Clean and preprocess documents
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.documents import Document


def setup_vector_db():
    """
    Set up ChromaDB with sample documents.
    In production, you'd load from files, databases, or APIs.
    """
    print("=== Setting Up Vector Database ===")
    
    # Sample documents about Python
    documents = [
        Document(
            page_content="Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
            metadata={"source": "python_intro", "topic": "basics"}
        ),
        Document(
            page_content="Python supports multiple programming paradigms including procedural, object-oriented, and functional programming. It has a comprehensive standard library.",
            metadata={"source": "python_features", "topic": "features"}
        ),
        Document(
            page_content="Popular Python frameworks include Django and Flask for web development, NumPy and Pandas for data science, and PyTorch and TensorFlow for machine learning.",
            metadata={"source": "python_frameworks", "topic": "ecosystem"}
        ),
        Document(
            page_content="Python's package manager pip makes it easy to install third-party libraries. The Python Package Index (PyPI) hosts hundreds of thousands of packages.",
            metadata={"source": "python_packages", "topic": "ecosystem"}
        ),
        Document(
            page_content="Python is widely used in data science, web development, automation, artificial intelligence, and scientific computing. Major companies like Google, Netflix, and NASA use Python.",
            metadata={"source": "python_uses", "topic": "applications"}
        ),
    ]
    
    # Initialize embeddings
    # Using Google's embedding model (free with Gemini API key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="python_docs"
    )
    
    print(f"✓ Created vector store with {len(documents)} documents")
    return vectorstore


def simple_similarity_search(vectorstore):
    """
    Simple similarity search example.
    Find documents most similar to a query.
    """
    print("\n=== Simple Similarity Search ===")
    
    query = "What is Python used for?"
    
    # Search for similar documents
    results = vectorstore.similarity_search(query, k=2)
    
    print(f"Query: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}\n")
    
    return results


def similarity_search_with_scores(vectorstore):
    """
    Similarity search with relevance scores.
    Useful for filtering low-quality results.
    """
    print("=== Similarity Search with Scores ===")
    
    query = "Tell me about Python frameworks"
    
    # Search with scores
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    print(f"Query: {query}\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} (Score: {score:.4f}):")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}\n")
    
    return results


def rag_qa_example(vectorstore):
    """
    Full RAG Q&A example.
    Combines retrieval with LLM generation.
    """
    print("=== RAG Q&A Example ===")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.3  # Lower temperature for factual answers
    )
    
    # Create retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" puts all docs in context
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )
    
    # Ask questions
    questions = [
        "What is Python?",
        "What frameworks are available for Python?",
        "Where is Python used?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"Answer: {result['result']}")
        print(f"Sources: {[doc.metadata['source'] for doc in result['source_documents']]}")


def custom_prompt_rag(vectorstore):
    """
    RAG with custom prompt template.
    Gives you more control over the response format.
    """
    print("\n=== RAG with Custom Prompt ===")
    
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    
    # Custom prompt template
    template = """You are a helpful Python programming assistant.
    Use the following context to answer the question.
    If you don't know the answer, say so - don't make up information.
    
    Context: {context}
    
    Question: {question}
    
    Answer in a clear, concise way suitable for beginners:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    
    question = "How do I install Python packages?"
    result = qa_chain.invoke({"query": question})
    
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")


def document_processing_example():
    """
    Example of processing longer documents.
    Shows how to split text into chunks.
    """
    print("\n=== Document Processing Example ===")
    
    # Sample long document
    long_document = """
    Python is a versatile programming language that has become one of the most popular
    languages in the world. It was created by Guido van Rossum and first released in 1991.
    
    Python's design philosophy emphasizes code readability with the use of significant
    indentation. Its language constructs aim to help programmers write clear, logical code
    for small and large-scale projects.
    
    Python is dynamically typed and garbage-collected. It supports multiple programming
    paradigms, including structured, object-oriented and functional programming.
    
    The language's core philosophy is summarized in the document The Zen of Python,
    which includes aphorisms such as "Beautiful is better than ugly" and
    "Simple is better than complex."
    """
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # Characters per chunk
        chunk_overlap=50,  # Overlap between chunks
        length_function=len,
    )
    
    chunks = text_splitter.create_documents([long_document])
    
    print(f"Split document into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk.page_content[:100] + "...")
    
    return chunks


def metadata_filtering_example(vectorstore):
    """
    Search with metadata filters.
    Useful for filtering by source, date, category, etc.
    """
    print("\n=== Metadata Filtering Example ===")
    
    # Search only in documents about "ecosystem"
    results = vectorstore.similarity_search(
        "packages and frameworks",
        k=2,
        filter={"topic": "ecosystem"}
    )
    
    print("Searching only in 'ecosystem' topic:")
    for doc in results:
        print(f"- {doc.page_content[:80]}...")
        print(f"  Topic: {doc.metadata['topic']}\n")


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
        exit(1)
    
    try:
        # Set up vector database
        vectorstore = setup_vector_db()
        
        # Run examples
        simple_similarity_search(vectorstore)
        similarity_search_with_scores(vectorstore)
        rag_qa_example(vectorstore)
        custom_prompt_rag(vectorstore)
        document_processing_example()
        metadata_filtering_example(vectorstore)
        
        print("\n✓ All examples completed successfully!")
        print("\nNext steps:")
        print("1. Try loading your own documents (PDFs, text files)")
        print("2. Experiment with different chunk sizes")
        print("3. Try different embedding models")
        print("4. Build a chatbot with conversation memory")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure you have installed: uv pip install chromadb langchain langchain-google-genai")
