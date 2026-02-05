# Phase 1: Foundation

Master the basics of LLM APIs and frameworks.

## Files

1. **[01_openai_basics.py](01_openai_basics.py)** - OpenAI Responses API
   - Simple examples with instructions
   - Detailed behavior guidance
   - Streaming responses
   - Temperature control
   - Multi-turn conversations
   - API overview

2. **[02_anthropic_basics.py](02_anthropic_basics.py)** - Anthropic Claude
   - Claude API usage
   - System prompts
   - Streaming
   - Tool use

3. **[03_gemini_basics.py](03_gemini_basics.py)** - Google Gemini
   - Gemini API
   - Multimodal support
   - Safety settings
   - Function calling

4. **[04_langchain_basics.py](04_langchain_basics.py)** - LangChain
   - Prompt templates
   - Chains
   - Conversation memory
   - Multi-step workflows

5. **[05_pydantic_ai_basics.py](05_pydantic_ai_basics.py)** - Pydantic AI
   - Type-safe agents
   - Structured outputs
   - Validation
   - Dependency injection

6. **[06_langgraph_basics.py](06_langgraph_basics.py)** - LangGraph
   - Graph-based workflows
   - Conditional branching
   - State management
   - Multi-agent collaboration
   - Checkpointing

7. **[07_embeddings_basics.py](07_embeddings_basics.py)** - Embeddings
   - OpenAI, Gemini, Cohere
   - Local embeddings (Sentence Transformers)
   - Similarity search & Batch processing

8. **[08_huggingface_basics.py](08_huggingface_basics.py)** - Hugging Face Ecosystem
   - Transformers Pipelines (Sentiment, Text Gen)
   - Inference API (Serverless)
   - Manual Model Loading

9. **[09_llama_basics.py](09_llama_basics.py)** - Local Llama 3
   - Llama 3.2 1B/3B Inference
   - Transformers & Torch integration
   - Gated model authentication

## Quick Start

```bash
# Install dependencies
uv pip install openai anthropic google-genai langchain langgraph langchain-google-genai pydantic-ai cohere sentence-transformers scikit-learn

# Set API key (choose one)
export OPENAI_API_KEY='your-key'
export ANTHROPIC_API_KEY='your-key'
export GEMINI_API_KEY='your-key'

# Run examples
python 01_openai_basics.py
python 03_gemini_basics.py
python 04_langchain_basics.py
```

## What You'll Learn

- API authentication and setup
- Message formats and conversations
- Streaming responses
- Function calling / tool use
- Framework abstractions
- Best practices

## Next Steps

After completing Phase 1, move to [Phase 2: RAG](../phase2_rag/) to learn about vector databases and document Q&A systems.
