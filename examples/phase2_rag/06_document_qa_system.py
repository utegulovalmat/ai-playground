"""
Phase 2: Document Q&A System
=============================
This example builds a complete document Q&A system that can answer questions
about uploaded documents (text files, PDFs, etc.).

Requirements:
- langchain>=1.2.0
- chromadb>=0.4.0
- langchain-google-genai>=4.2.0
- pypdf>=3.0.0 (for PDF support)
- GEMINI_API_KEY environment variable

This is a practical, production-ready example you can extend.
"""

import os
from pathlib import Path
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document


class DocumentQA:
    """
    A complete Document Q&A system with conversation memory.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize the Q&A system."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.3
        )
        
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def load_documents_from_text(self, texts: List[str], metadatas: List[dict] = None):
        """
        Load documents from text strings.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
        """
        print(f"Loading {len(texts)} documents...")
        
        # Create documents
        documents = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas or [{}] * len(texts))
        ]
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name="document_qa"
        )
        
        # Create QA chain
        self._create_qa_chain()
        print("✓ Documents loaded successfully!")
    
    def load_from_file(self, file_path: str):
        """
        Load a document from a file.
        Supports .txt files (PDF support requires pypdf).
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Loading file: {path.name}")
        
        if path.suffix == '.txt':
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.load_documents_from_text(
                [content],
                [{"source": path.name}]
            )
        elif path.suffix == '.pdf':
            # PDF support (requires pypdf)
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                texts = [page.extract_text() for page in reader.pages]
                metadatas = [{"source": path.name, "page": i} for i in range(len(texts))]
                self.load_documents_from_text(texts, metadatas)
            except ImportError:
                print("PDF support requires pypdf: uv pip install pypdf")
                raise
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    
    def _create_qa_chain(self):
        """Create the conversational QA chain."""
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
    
    def ask(self, question: str) -> dict:
        """
        Ask a question about the documents.
        
        Args:
            question: The question to ask
            
        Returns:
            dict with 'answer' and 'sources'
        """
        if not self.qa_chain:
            raise ValueError("No documents loaded. Call load_documents_from_text() first.")
        
        result = self.qa_chain.invoke({"question": question})
        
        return {
            "answer": result["answer"],
            "sources": [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        }
    
    def chat(self):
        """
        Interactive chat mode.
        Type 'quit' or 'exit' to stop.
        """
        if not self.qa_chain:
            raise ValueError("No documents loaded.")
        
        print("\n=== Document Q&A Chat ===")
        print("Ask questions about your documents. Type 'quit' to exit.\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                result = self.ask(question)
                print(f"\nAssistant: {result['answer']}")
                print(f"Sources: {', '.join(set(result['sources']))}\n")
            except Exception as e:
                print(f"Error: {e}\n")


def example_with_sample_documents():
    """
    Example using sample documents about AI.
    """
    print("=== Example: AI Knowledge Base ===\n")
    
    # Sample documents about AI
    documents = [
        "Machine Learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
        
        "Deep Learning is a subset of machine learning that uses neural networks with multiple layers. These neural networks attempt to simulate the behavior of the human brain, allowing it to learn from large amounts of data.",
        
        "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.",
        
        "Computer Vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.",
        
        "Reinforcement Learning is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximize cumulative reward. It's used in robotics, gaming, and autonomous vehicles."
    ]
    
    metadatas = [
        {"topic": "machine_learning", "difficulty": "beginner"},
        {"topic": "deep_learning", "difficulty": "intermediate"},
        {"topic": "nlp", "difficulty": "intermediate"},
        {"topic": "computer_vision", "difficulty": "intermediate"},
        {"topic": "reinforcement_learning", "difficulty": "advanced"}
    ]
    
    # Create QA system
    qa = DocumentQA()
    qa.load_documents_from_text(documents, metadatas)
    
    # Ask questions
    questions = [
        "What is machine learning?",
        "Tell me about deep learning",
        "What are the different types of AI mentioned?",
        "How does reinforcement learning work?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        result = qa.ask(question)
        print(f"A: {result['answer']}")
        print(f"Sources: {', '.join(set(result['sources']))}")


def example_with_conversation():
    """
    Example showing conversation memory.
    """
    print("\n\n=== Example: Conversation with Memory ===\n")
    
    documents = [
        "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability and simplicity.",
        "Python supports multiple programming paradigms including object-oriented, functional, and procedural programming.",
        "Popular Python web frameworks include Django and Flask. Django is a full-featured framework while Flask is lightweight and flexible."
    ]
    
    qa = DocumentQA()
    qa.load_documents_from_text(documents)
    
    # Conversation with follow-up questions
    print("Q: Who created Python?")
    result = qa.ask("Who created Python?")
    print(f"A: {result['answer']}\n")
    
    print("Q: When was it released?")  # Follow-up question
    result = qa.ask("When was it released?")
    print(f"A: {result['answer']}\n")
    
    print("Q: What frameworks are available?")
    result = qa.ask("What frameworks are available?")
    print(f"A: {result['answer']}\n")


def example_create_sample_file():
    """
    Create a sample text file for testing.
    """
    sample_content = """
    Artificial Intelligence: A Comprehensive Overview
    
    Introduction
    Artificial Intelligence (AI) refers to the simulation of human intelligence in machines
    that are programmed to think and learn like humans. The term may also be applied to any
    machine that exhibits traits associated with a human mind such as learning and problem-solving.
    
    History
    The field of AI research was founded at a workshop at Dartmouth College in 1956. The
    attendees became the leaders of AI research for decades. They and their students wrote
    programs that were, to most people, simply astonishing.
    
    Types of AI
    1. Narrow AI: AI that is designed to perform a specific task, such as facial recognition.
    2. General AI: AI that can perform any intellectual task that a human can do.
    3. Super AI: AI that surpasses human intelligence and ability.
    
    Applications
    AI is used in various fields including healthcare, finance, transportation, and entertainment.
    Some common applications include virtual assistants, recommendation systems, autonomous
    vehicles, and medical diagnosis.
    
    Future of AI
    The future of AI holds immense potential. As technology advances, AI systems will become
    more sophisticated and capable of handling increasingly complex tasks. However, this also
    raises important ethical and societal questions that need to be addressed.
    """
    
    # Create sample file
    file_path = Path("sample_ai_document.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"✓ Created sample file: {file_path}")
    return file_path


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        exit(1)
    
    try:
        # Run examples
        example_with_sample_documents()
        example_with_conversation()
        
        # Create and load from file
        print("\n\n=== Example: Loading from File ===\n")
        sample_file = example_create_sample_file()
        
        qa = DocumentQA()
        qa.load_from_file(str(sample_file))
        
        # Ask questions about the file
        result = qa.ask("What is the history of AI?")
        print(f"\nQ: What is the history of AI?")
        print(f"A: {result['answer']}")
        
        # Uncomment to try interactive chat
        # qa.chat()
        
        print("\n✓ All examples completed!")
        print("\nTry running qa.chat() for interactive mode!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure to install: uv pip install chromadb langchain langchain-google-genai")
