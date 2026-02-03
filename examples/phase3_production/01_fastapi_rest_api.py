"""
Phase 3: FastAPI Production Example
====================================
This example demonstrates how to build a production-ready API for LLM applications.

Requirements:
- fastapi>=0.100.0
- uvicorn>=0.23.0
- langchain>=1.2.0
- langchain-google-genai>=4.2.0
- pydantic>=2.0.0
- GEMINI_API_KEY environment variable

Run with:
    uvicorn phase3_fastapi_example:app --reload

Then visit:
    http://localhost:8000/docs for interactive API documentation

Best Practices:
- Use async/await for better performance
- Implement proper error handling
- Add request validation with Pydantic
- Include API documentation
- Add rate limiting in production
"""

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import asyncio
import json


# Pydantic models for request/response validation
class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    temperature: float = Field(0.7, ge=0, le=2, description="Temperature (0-2)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    stream: bool = Field(False, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Response from chat completion."""
    message: str = Field(..., description="Generated message")
    model: str = Field(..., description="Model used")
    usage: dict = Field(..., description="Token usage information")


class SimpleRequest(BaseModel):
    """Simple text completion request."""
    prompt: str = Field(..., description="The prompt text")
    temperature: float = Field(0.7, ge=0, le=2)


class SimpleResponse(BaseModel):
    """Simple text completion response."""
    response: str
    model: str


# Initialize FastAPI app
app = FastAPI(
    title="LLM API",
    description="Production-ready API for LLM applications",
    version="1.0.0"
)

# Initialize LLM (do this once at startup)
llm = None


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global llm
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key
    )
    print("✓ LLM initialized successfully")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM API is running",
        "docs": "/docs",
        "endpoints": {
            "simple": "/api/simple",
            "chat": "/api/chat",
            "stream": "/api/stream"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "gemini-2.0-flash"
    }


@app.post("/api/simple", response_model=SimpleResponse)
async def simple_completion(request: SimpleRequest):
    """
    Simple text completion endpoint.
    
    Example:
        POST /api/simple
        {
            "prompt": "What is Python?",
            "temperature": 0.7
        }
    """
    try:
        # Create message
        message = HumanMessage(content=request.prompt)
        
        # Get response
        response = await llm.ainvoke([message])
        
        return SimpleResponse(
            response=response.content,
            model="gemini-2.0-flash"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Chat completion endpoint with conversation history.
    
    Example:
        POST /api/chat
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"}
            ],
            "temperature": 0.7
        }
    """
    try:
        # Convert to LangChain messages
        messages = []
        for msg in request.messages:
            if msg.role == "system":
                messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        # Get response
        response = await llm.ainvoke(messages)
        
        return ChatResponse(
            message=response.content,
            model="gemini-2.0-flash",
            usage={
                "prompt_tokens": 0,  # Gemini doesn't provide this
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream")
async def stream_completion(request: SimpleRequest):
    """
    Streaming completion endpoint.
    Returns Server-Sent Events (SSE) for real-time streaming.
    
    Example:
        POST /api/stream
        {
            "prompt": "Write a short story",
            "temperature": 0.8
        }
    """
    async def generate():
        """Generator for streaming response."""
        try:
            message = HumanMessage(content=request.prompt)
            
            # Stream response
            async for chunk in llm.astream([message]):
                # Format as SSE
                data = json.dumps({"content": chunk.content})
                yield f"data: {data}\n\n"
                
        except Exception as e:
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/api/batch")
async def batch_completion(prompts: List[str], background_tasks: BackgroundTasks):
    """
    Batch processing endpoint.
    Processes multiple prompts asynchronously.
    
    Example:
        POST /api/batch
        ["What is AI?", "What is ML?", "What is DL?"]
    """
    async def process_prompt(prompt: str) -> str:
        """Process a single prompt."""
        message = HumanMessage(content=prompt)
        response = await llm.ainvoke([message])
        return response.content
    
    try:
        # Process all prompts concurrently
        tasks = [process_prompt(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        return {
            "results": [
                {"prompt": prompt, "response": result}
                for prompt, result in zip(prompts, results)
            ],
            "count": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example usage and testing
if __name__ == "__main__":
    import uvicorn
    
    print("""
    Starting FastAPI server...
    
    API Documentation: http://localhost:8000/docs
    
    Example curl commands:
    
    1. Simple completion:
       curl -X POST http://localhost:8000/api/simple \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "What is Python?"}'
    
    2. Chat completion:
       curl -X POST http://localhost:8000/api/chat \\
         -H "Content-Type: application/json" \\
         -d '{
           "messages": [
             {"role": "user", "content": "Hello!"}
           ]
         }'
    
    3. Streaming:
       curl -X POST http://localhost:8000/api/stream \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "Count from 1 to 5"}'
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8011)
