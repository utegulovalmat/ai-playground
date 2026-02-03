# Phase 1: Foundation

Master the basics of LLM APIs and frameworks.

## Files

1. **[01_openai_basics.py](01_openai_basics.py)** - OpenAI API
   - Simple completions
   - System messages
   - Conversations
   - Streaming
   - Function calling

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

## Quick Start

```bash
# Install dependencies
uv pip install openai anthropic google-genai langchain langchain-google-genai pydantic-ai

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
