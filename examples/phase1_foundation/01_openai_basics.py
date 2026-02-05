"""
OpenAI Responses API Examples
==============================
This example demonstrates how to use OpenAI's Responses API - the newest,
most advanced API that simplifies development with a cleaner interface.

Requirements:
- openai>=1.0.0
- OPENAI_API_KEY environment variable

Best Practices:
- Use environment variables for API keys
- Handle errors gracefully
- Use streaming for real-time responses
- Specify model versions explicitly
- Use instructions to guide model behavior
"""

import os
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# ============================================================================
# Responses API Examples
# ============================================================================

def simple_example():
    """
    Simple example using the Responses API.
    The Responses API provides a cleaner interface with instructions and input.
    """
    print("=== Simple Example ===")
    
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful assistant that explains concepts simply.",
        input="What is the capital of France?"
    )
    
    # Clean output - just the text
    print(response.output_text)
    return response


def detailed_instructions_example():
    """
    Example with detailed instructions.
    Shows how to guide the model's behavior using structured instructions.
    """
    print("\n=== Detailed Instructions ===")
    
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="""You are a Python expert. When explaining code:
        1. Keep explanations concise
        2. Use simple language
        3. Provide practical examples
        4. Focus on best practices""",
        input="Explain what a list comprehension is in Python"
    )
    
    print(response.output_text)
    return response


def streaming_example():
    """
    Streaming example for real-time responses.
    Useful for chatbot interfaces where you want to show responses as they're generated.
    """
    print("\n=== Streaming Example ===")
    
    stream = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful assistant.",
        input="Count from 1 to 5 slowly.",
        stream=True
    )
    
    print("Streaming response: ", end="")
    for chunk in stream:
        if getattr(chunk, 'type', None) == 'response.output_text.delta' and hasattr(chunk, 'delta'):
            print(chunk.delta, end="", flush=True)
    print()  # New line


def temperature_example():
    """
    Example showing temperature control.
    Temperature controls randomness: 0 = deterministic, 2 = very creative.
    """
    print("\n=== Temperature Control ===")
    
    # Low temperature (more focused, deterministic)
    print("\nLow temperature (0.2):")
    response_low = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a creative writer.",
        input="Write a one-sentence story about a robot.",
        temperature=0.2
    )
    print(response_low.output_text)
    
    # High temperature (more creative, random)
    print("\nHigh temperature (1.5):")
    response_high = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a creative writer.",
        input="Write a one-sentence story about a robot.",
        temperature=1.5
    )
    print(response_high.output_text)


def multi_turn_conversation():
    """
    Multi-turn conversation example.
    Shows how to maintain context across multiple interactions.
    """
    print("\n=== Multi-turn Conversation ===")
    
    # First turn
    response1 = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful assistant. Remember what the user tells you.",
        input="My favorite programming language is Python."
    )
    print(f"User: My favorite programming language is Python.")
    print(f"Assistant: {response1.output_text}")
    
    # Second turn - reference previous context
    # Note: In production, you'd maintain conversation history
    response2 = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful assistant. The user previously said their favorite programming language is Python.",
        input="What's my favorite programming language?"
    )
    print(f"\nUser: What's my favorite programming language?")
    print(f"Assistant: {response2.output_text}")


def api_overview():
    """
    Overview of the Responses API.
    """
    print("\n=== Responses API Overview ===")
    
    print("""
The Responses API is OpenAI's newest, most advanced API that simplifies
development by providing a cleaner interface.

Key Features:
• Simplified interface with 'instructions' and 'input' parameters
• Clean output via 'output_text' attribute
• Built-in orchestration and tool integration
• Automatic handling of complex workflows
• Easier to learn and use than Chat Completions API

Basic Usage:
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful assistant.",
        input="Your question here"
    )
    print(response.output_text)

Parameters:
• model: The model to use (e.g., "gpt-4o-mini", "gpt-4o")
• instructions: System-level guidance for the model's behavior
• input: The user's input/question
• temperature: Controls randomness (0-2, default ~1)
• stream: Enable streaming responses (True/False)

Best For:
• Quick prototypes and simple applications
• Single-turn or simple interactions
• When you want clean, straightforward output
• Fast development with minimal boilerplate
    """)


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    print("=" * 70)
    print("OpenAI Responses API Examples")
    print("=" * 70)
    
    # Run all examples
    try:
        # simple_example()
        # print("\n" + "=" * 70 + "\n")
        
        # detailed_instructions_example()
        # print("\n" + "=" * 70 + "\n")
        
        streaming_example()
        print("\n" + "=" * 70 + "\n")
        
        # temperature_example()
        # print("\n" + "=" * 70 + "\n")
        
        # multi_turn_conversation()
        # print("\n" + "=" * 70 + "\n")
        
        # api_overview()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure your API key is valid and you have credits available.")

