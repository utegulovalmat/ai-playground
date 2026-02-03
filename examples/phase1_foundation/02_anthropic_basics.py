"""
Anthropic Claude API Example
=============================
This example demonstrates how to use Anthropic's Claude API directly.

Requirements:
- anthropic>=0.18.0
- ANTHROPIC_API_KEY environment variable

Best Practices:
- Use environment variables for API keys
- Handle errors gracefully
- Use streaming for better UX
- Leverage Claude's strong instruction-following
"""

import os
from anthropic import Anthropic

# Initialize the Anthropic client
# Best practice: Use environment variables for API keys
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def simple_completion():
    """
    Simple completion example using Claude.
    Claude excels at following detailed instructions.
    """
    print("=== Simple Completion ===")
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",  # Latest Claude model
        max_tokens=1024,  # Maximum tokens in response
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ]
    )
    
    print(message.content[0].text)
    return message


def system_prompt_example():
    """
    Example using system prompts.
    Claude is particularly good at following system instructions.
    """
    print("\n=== System Prompt Example ===")
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        system="You are a helpful assistant that explains technical concepts in simple terms suitable for beginners.",
        messages=[
            {
                "role": "user",
                "content": "Explain what machine learning is."
            }
        ]
    )
    
    print(message.content[0].text)
    return message


def conversation_example():
    """
    Multi-turn conversation example.
    Shows how to maintain context across multiple exchanges.
    """
    print("\n=== Conversation Example ===")
    
    # Build conversation history
    messages = [
        {
            "role": "user",
            "content": "I'm learning Python. What should I start with?"
        }
    ]
    
    # First turn
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=messages
    )
    
    assistant_message = response.content[0].text
    print(f"Claude: {assistant_message}\n")
    
    # Add to conversation history
    messages.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    # Second turn
    messages.append({
        "role": "user",
        "content": "Can you give me a simple code example?"
    })
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=messages
    )
    
    print(f"Claude: {response.content[0].text}")
    return response


def streaming_example():
    """
    Streaming example for real-time responses.
    Provides better UX by showing responses as they're generated.
    """
    print("\n=== Streaming Example ===")
    
    print("Claude: ", end="", flush=True)
    
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about coding."
            }
        ]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
    
    print()  # New line after streaming


def vision_example():
    """
    Vision example using Claude's multimodal capabilities.
    Claude can analyze images when provided in base64 format.
    Note: This is a template - you need to provide actual image data.
    """
    print("\n=== Vision Example (Template) ===")
    
    # Example structure for image analysis
    # You would need to encode an actual image to base64
    message_template = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "<base64-encoded-image-data>"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    }
                ]
            }
        ]
    }
    
    print("Vision example template shown above.")
    print("To use: encode an image to base64 and replace <base64-encoded-image-data>")


def tool_use_example():
    """
    Tool use example (function calling).
    Claude can use tools to perform specific tasks.
    """
    print("\n=== Tool Use Example ===")
    
    # Define available tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in Tokyo?"
            }
        ]
    )
    
    # Check if Claude wants to use a tool
    for content_block in message.content:
        if content_block.type == "tool_use":
            print(f"Claude wants to use tool: {content_block.name}")
            print(f"With input: {content_block.input}")
        else:
            print(content_block.text)


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)
    
    # Run all examples
    try:
        simple_completion()
        system_prompt_example()
        conversation_example()
        streaming_example()
        vision_example()
        tool_use_example()
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure your API key is valid and you have credits available.")
