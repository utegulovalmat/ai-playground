"""
OpenAI API Examples
===================
This example demonstrates how to use OpenAI's APIs:
- Chat Completions API (classic, flexible)
- Responses API (newer, simplified)

Requirements:
- openai>=1.0.0
- OPENAI_API_KEY environment variable

Best Practices:
- Use environment variables for API keys
- Handle errors gracefully
- Use async for better performance when making multiple calls
- Specify model versions explicitly
- Choose the right API for your use case
"""

import os
from openai import OpenAI

# Initialize the OpenAI client
# Best practice: Use environment variables for API keys
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def simple_completion():
    """
    Simple completion example using GPT-4o-mini.
    This is the most basic way to interact with OpenAI's API.
    """
    print("=== Simple Completion ===")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Specify model explicitly
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
        temperature=0.7,  # Controls randomness (0-2)
        max_tokens=100    # Limit response length
    )
    
    print(response.choices[0].message.content)
    return response


def system_message_example():
    """
    Example using system messages to set behavior.
    System messages help guide the AI's personality and response style.
    """
    print("\n=== System Message Example ===")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that explains concepts simply."
            },
            {
                "role": "user",
                "content": "Explain what an API is in one sentence."
            }
        ],
        temperature=0.5
    )
    
    print(response.choices[0].message.content)
    return response


def conversation_example():
    """
    Example maintaining conversation history.
    This shows how to build multi-turn conversations.
    """
    print("\n=== Conversation Example ===")
    
    # Conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My favorite color is blue."}
    ]
    
    # First turn
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    assistant_message = response.choices[0].message.content
    print(f"Assistant: {assistant_message}")
    
    # Add assistant's response to history
    messages.append({"role": "assistant", "content": assistant_message})
    
    # Second turn - reference previous context
    messages.append({"role": "user", "content": "What's my favorite color?"})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    print(f"Assistant: {response.choices[0].message.content}")
    return response


def streaming_example():
    """
    Streaming example for real-time responses.
    Useful for chatbot interfaces where you want to show responses as they're generated.
    """
    print("\n=== Streaming Example ===")
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        stream=True  # Enable streaming
    )
    
    print("Streaming response: ", end="")
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # New line after streaming


def function_calling_example():
    """
    Function calling example (tool use).
    This allows the model to call predefined functions.
    """
    print("\n=== Function Calling Example ===")
    
    # Define available functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "What's the weather in Paris?"}
        ],
        tools=tools,
        tool_choice="auto"  # Let the model decide when to use tools
    )
    
    # Check if the model wants to call a function
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        print(f"Model wants to call: {tool_call.function.name}")
        print(f"With arguments: {tool_call.function.arguments}")
    else:
        print(response.choices[0].message.content)


# ============================================================================
# Responses API Examples (Newer, Simplified API)
# ============================================================================

def responses_api_simple():
    """
    Simple example using the Responses API.
    The Responses API is OpenAI's newest, most advanced API that simplifies
    development by providing a cleaner interface.
    """
    print("\n=== Responses API - Simple Example ===")
    
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful assistant that explains concepts simply.",
        input="What is the capital of France?"
    )
    
    # Cleaner output - just the text
    print(response.output_text)
    return response


def responses_api_with_instructions():
    """
    Responses API with detailed instructions.
    Shows how to guide the model's behavior using the instructions parameter.
    """
    print("\n=== Responses API - With Instructions ===")
    
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


def responses_api_streaming():
    """
    Streaming with the Responses API.
    Get real-time responses as they're generated.
    """
    print("\n=== Responses API - Streaming ===")
    
    stream = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a helpful assistant.",
        input="Count from 1 to 5 slowly.",
        stream=True
    )
    
    print("Streaming response: ", end="")
    for chunk in stream:
        if hasattr(chunk, 'output_text_delta') and chunk.output_text_delta:
            print(chunk.output_text_delta, end="", flush=True)
    print()  # New line


def responses_api_comparison():
    """
    Comparison between Responses API and Chat Completions API.
    Shows when to use each.
    """
    print("\n=== API Comparison ===")
    
    print("""
┌─────────────────────┬──────────────────────┬─────────────────────────┐
│ Feature             │ Responses API        │ Chat Completions API    │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ Interface           │ Simplified           │ More flexible           │
│ Input               │ instructions + input │ messages list           │
│ Output              │ output_text          │ choices[0].message      │
│ Best For            │ Simple tasks         │ Complex conversations   │
│                     │ Fast prototyping     │ Multi-turn chats        │
│                     │ Clean output         │ Advanced control        │
│ Built-in Tools      │ Yes (native)         │ Requires setup          │
│ Orchestration       │ Automatic            │ Manual                  │
│ Learning Curve      │ Easier               │ Steeper                 │
└─────────────────────┴──────────────────────┴─────────────────────────┘

When to use Responses API:
• Quick prototypes and simple applications
• When you want clean, straightforward output
• When using built-in tools (web search, file search)
• For single-turn or simple interactions

When to use Chat Completions API:
• Complex multi-turn conversations
• When you need fine-grained control over message history
• Advanced function calling with custom logic
• When building sophisticated chatbots
    """)


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    print("=" * 70)
    print("OpenAI API Examples")
    print("=" * 70)
    
    # Run all examples
    try:
        # Chat Completions API (Classic)
        print("\n" + "=" * 70)
        print("CHAT COMPLETIONS API (Classic)")
        print("=" * 70)
        
        simple_completion()
        system_message_example()
        conversation_example()
        streaming_example()
        function_calling_example()
        
        # Responses API (Newer)
        print("\n" + "=" * 70)
        print("RESPONSES API (Newer, Simplified)")
        print("=" * 70)
        
        responses_api_simple()
        responses_api_with_instructions()
        responses_api_streaming()
        responses_api_comparison()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure your API key is valid and you have credits available.")

