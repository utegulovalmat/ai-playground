"""
Phase 3: Gradio Demo Example
=============================
This example shows how to create interactive demos with Gradio.
Perfect for quickly prototyping and sharing AI applications.

Requirements:
- gradio>=4.0.0
- langchain>=1.2.0
- langchain-google-genai>=4.2.0
- chromadb>=0.4.0
- GEMINI_API_KEY environment variable

Run with:
    python phase3_gradio_demo.py

Then visit the URL shown in the terminal (usually http://localhost:7860)

Best Practices:
- Keep interfaces simple and intuitive
- Add clear examples
- Include error messages
- Use appropriate input/output types
"""

import os
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.7
)


def simple_chat(message, history):
    """
    Simple chatbot function.
    
    Args:
        message: Current user message
        history: List of [user_msg, bot_msg] pairs
    
    Returns:
        Bot response
    """
    try:
        # Convert history to messages
        messages = []
        for user_msg, bot_msg in history:
            messages.append(HumanMessage(content=user_msg))
            if bot_msg:
                messages.append(SystemMessage(content=bot_msg))
        
        # Add current message
        messages.append(HumanMessage(content=message))
        
        # Get response
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"Error: {str(e)}"


def custom_prompt_chat(message, system_prompt, temperature):
    """
    Chatbot with customizable system prompt and temperature.
    """
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]
        
        # Create LLM with custom temperature
        custom_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=temperature
        )
        
        response = custom_llm.invoke(messages)
        return response.content
        
    except Exception as e:
        return f"Error: {str(e)}"


def text_summarizer(text, style):
    """
    Summarize text in different styles.
    """
    try:
        style_prompts = {
            "Brief": "Summarize this in 2-3 sentences",
            "Detailed": "Provide a comprehensive summary with key points",
            "Bullet Points": "Summarize as bullet points",
            "ELI5": "Explain this like I'm 5 years old"
        }
        
        prompt = f"{style_prompts[style]}:\n\n{text}"
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
        
    except Exception as e:
        return f"Error: {str(e)}"


def code_explainer(code, language):
    """
    Explain code in simple terms.
    """
    try:
        prompt = f"""Explain this {language} code in simple terms:

```{language}
{code}
```

Provide:
1. What the code does
2. How it works
3. Any important concepts"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
        
    except Exception as e:
        return f"Error: {str(e)}"


# Create sample knowledge base for RAG demo
def create_sample_kb():
    """Create a sample knowledge base."""
    documents = [
        Document(
            page_content="Python is a high-level programming language created by Guido van Rossum in 1991.",
            metadata={"topic": "python"}
        ),
        Document(
            page_content="Machine learning is a subset of AI that enables systems to learn from data.",
            metadata={"topic": "ml"}
        ),
        Document(
            page_content="FastAPI is a modern web framework for building APIs with Python.",
            metadata={"topic": "fastapi"}
        ),
    ]
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="demo_kb"
    )
    
    return vectorstore


# Initialize knowledge base
kb = create_sample_kb()


def rag_qa(question):
    """
    Answer questions using RAG.
    """
    try:
        # Retrieve relevant documents
        docs = kb.similarity_search(question, k=2)
        
        # Create context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer
        prompt = f"""Answer the question based on this context:

Context:
{context}

Question: {question}

Answer:"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        
        # Format response with sources
        sources = [doc.metadata.get("topic", "unknown") for doc in docs]
        return f"{response.content}\n\n*Sources: {', '.join(sources)}*"
        
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interfaces
def create_demos():
    """Create all demo interfaces."""
    
    # Demo 1: Simple Chatbot
    chatbot_demo = gr.ChatInterface(
        fn=simple_chat,
        title="🤖 Simple Chatbot",
        description="A basic chatbot powered by Gemini",
        examples=[
            "What is Python?",
            "Explain machine learning",
            "Write a haiku about coding"
        ],
        theme="soft"
    )
    
    # Demo 2: Custom Prompt Chatbot
    custom_chat_demo = gr.Interface(
        fn=custom_prompt_chat,
        inputs=[
            gr.Textbox(label="Your Message", placeholder="Ask anything..."),
            gr.Textbox(
                label="System Prompt",
                value="You are a helpful assistant.",
                lines=3
            ),
            gr.Slider(0, 2, value=0.7, label="Temperature")
        ],
        outputs=gr.Textbox(label="Response", lines=10),
        title="⚙️ Custom Chatbot",
        description="Customize the bot's behavior with system prompts and temperature"
    )
    
    # Demo 3: Text Summarizer
    summarizer_demo = gr.Interface(
        fn=text_summarizer,
        inputs=[
            gr.Textbox(label="Text to Summarize", lines=10, placeholder="Paste your text here..."),
            gr.Radio(
                ["Brief", "Detailed", "Bullet Points", "ELI5"],
                label="Summary Style",
                value="Brief"
            )
        ],
        outputs=gr.Textbox(label="Summary", lines=8),
        title="📝 Text Summarizer",
        description="Summarize text in different styles",
        examples=[
            ["Artificial intelligence is transforming industries worldwide. From healthcare to finance, AI systems are being deployed to automate tasks, analyze data, and make predictions. Machine learning, a subset of AI, enables computers to learn from data without explicit programming.", "Brief"]
        ]
    )
    
    # Demo 4: Code Explainer
    code_demo = gr.Interface(
        fn=code_explainer,
        inputs=[
            gr.Code(label="Code", language="python"),
            gr.Dropdown(
                ["Python", "JavaScript", "Java", "C++"],
                label="Language",
                value="Python"
            )
        ],
        outputs=gr.Textbox(label="Explanation", lines=15),
        title="💻 Code Explainer",
        description="Get simple explanations of code",
        examples=[
            ["def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)", "Python"]
        ]
    )
    
    # Demo 5: RAG Q&A
    rag_demo = gr.Interface(
        fn=rag_qa,
        inputs=gr.Textbox(label="Question", placeholder="Ask about Python, ML, or FastAPI..."),
        outputs=gr.Textbox(label="Answer", lines=8),
        title="📚 Knowledge Base Q&A",
        description="Ask questions about our knowledge base (RAG demo)",
        examples=[
            "What is Python?",
            "Tell me about machine learning",
            "What is FastAPI?"
        ]
    )
    
    # Combine all demos in tabs
    demo = gr.TabbedInterface(
        [chatbot_demo, custom_chat_demo, summarizer_demo, code_demo, rag_demo],
        ["Chatbot", "Custom Chat", "Summarizer", "Code Explainer", "Knowledge Q&A"],
        title="🚀 LLM Demo Suite",
        theme="soft"
    )
    
    return demo


if __name__ == "__main__":
    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        exit(1)
    
    print("Creating Gradio demos...")
    demo = create_demos()
    
    print("\n" + "="*60)
    print("🚀 Launching Gradio Demo Suite")
    print("="*60)
    print("\nFeatures:")
    print("  • Simple Chatbot")
    print("  • Custom Prompt Chatbot")
    print("  • Text Summarizer")
    print("  • Code Explainer")
    print("  • RAG Knowledge Q&A")
    print("\nThe demo will open in your browser automatically.")
    print("="*60 + "\n")
    
    # Launch with sharing option
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",
        server_port=7860
    )
