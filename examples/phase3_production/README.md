# Phase 3: Production

Deploy production-ready LLM applications with REST APIs and interactive demos.

## Files

1. **[01_fastapi_rest_api.py](01_fastapi_rest_api.py)** - FastAPI REST API
   - Async endpoints
   - Streaming responses (SSE)
   - Batch processing
   - Auto-generated docs

2. **[02_gradio_demos.py](02_gradio_demos.py)** - Interactive Demos
   - 5 demo interfaces
   - Chatbot
   - Summarizer
   - Code explainer
   - RAG Q&A

## Quick Start

```bash
# Install dependencies
uv pip install fastapi uvicorn gradio langchain langchain-google-genai

# Set API key
export GEMINI_API_KEY='your-key'

# Run FastAPI (visit http://localhost:8000/docs)
uvicorn 01_fastapi_rest_api:app --reload

# Run Gradio (opens in browser)
python 02_gradio_demos.py
```

## What You'll Learn

- REST API development with FastAPI
- Async programming
- Streaming endpoints
- API documentation
- Interactive UI with Gradio
- Production deployment patterns

## Project Ideas

- Public chatbot API
- Demo showcase
- Integration with existing apps
- Mobile-friendly web interface

## Next Steps

After completing Phase 3, move to [Phase 4: Advanced](../phase4_advanced/) to master advanced techniques.
