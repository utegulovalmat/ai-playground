"""
OpenAI API Example
==================
This example demonstrates how to use OpenAI's API directly.

Requirements:
- openai>=1.0.0
- OPENAI_API_KEY environment variable

Best Practices:
- Use environment variables for API keys
- Handle errors gracefully
- Use async for better performance when making multiple calls
- Specify model versions explicitly
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


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Run all examples
    try:
        simple_completion()
        system_message_example()
        conversation_example()
        streaming_example()
        function_calling_example()
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure your API key is valid and you have credits available.")
