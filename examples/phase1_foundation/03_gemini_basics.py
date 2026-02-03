"""
Google Gemini API Example
==========================
This example demonstrates how to use Google's Gemini API directly.

Requirements:
- google-genai>=1.0.0
- GEMINI_API_KEY environment variable

Best Practices:
- Use environment variables for API keys
- Handle quota limits gracefully
- Leverage Gemini's multimodal capabilities
- Use appropriate models for your use case
"""

import os
from google import genai
from google.genai import types

# Initialize the Gemini client
# Best practice: Use environment variables for API keys
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def simple_completion():
    """
    Simple text generation example.
    Gemini 2.0 Flash is fast and cost-effective for most tasks.
    """
    print("=== Simple Completion ===")
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="What is the capital of France?"
    )
    
    print(response.text)
    return response


def system_instruction_example():
    """
    Example using system instructions.
    System instructions guide the model's behavior.
    """
    print("\n=== System Instruction Example ===")
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Explain quantum computing.",
        config=types.GenerateContentConfig(
            system_instruction="You are a science teacher explaining concepts to high school students. Use simple language and analogies.",
            temperature=0.7,
            max_output_tokens=500
        )
    )
    
    print(response.text)
    return response


def conversation_example():
    """
    Multi-turn conversation example.
    Shows how to maintain chat history.
    """
    print("\n=== Conversation Example ===")
    
    # Create a chat session
    chat = client.chats.create(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            temperature=0.8
        )
    )
    
    # First message
    response1 = chat.send_message("My name is Alice and I love astronomy.")
    print(f"Gemini: {response1.text}\n")
    
    # Second message - references previous context
    response2 = chat.send_message("What's my name and what do I love?")
    print(f"Gemini: {response2.text}")
    
    return chat


def streaming_example():
    """
    Streaming example for real-time responses.
    Useful for interactive applications.
    """
    print("\n=== Streaming Example ===")
    
    print("Gemini: ", end="", flush=True)
    
    for chunk in client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents="Write a short poem about AI."
    ):
        print(chunk.text, end="", flush=True)
    
    print()  # New line after streaming


def multimodal_example():
    """
    Multimodal example (text + image).
    Gemini excels at understanding images alongside text.
    Note: This is a template - you need to provide actual image data.
    """
    print("\n=== Multimodal Example (Template) ===")
    
    # Example structure for multimodal input
    # You would need to provide actual image bytes
    print("To use multimodal capabilities:")
    print("1. Load an image file")
    print("2. Create a Part with the image data")
    print("3. Combine with text in the contents")
    print("\nExample code:")
    print("""
    from pathlib import Path
    
    image_path = Path("path/to/image.jpg")
    image_bytes = image_path.read_bytes()
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            "What's in this image?"
        ]
    )
    print(response.text)
    """)


def function_calling_example():
    """
    Function calling example (tool use).
    Gemini can call functions to perform specific tasks.
    """
    print("\n=== Function Calling Example ===")
    
    # Define available functions
    get_weather_func = types.FunctionDeclaration(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
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
    )
    
    # Create tool with the function
    weather_tool = types.Tool(
        function_declarations=[get_weather_func]
    )
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="What's the weather in London?",
        config=types.GenerateContentConfig(
            tools=[weather_tool]
        )
    )
    
    # Check if function was called
    if response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call'):
                print(f"Function called: {part.function_call.name}")
                print(f"Arguments: {dict(part.function_call.args)}")
            elif hasattr(part, 'text'):
                print(part.text)


def safety_settings_example():
    """
    Example with custom safety settings.
    Control content filtering for different use cases.
    """
    print("\n=== Safety Settings Example ===")
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Tell me about internet safety for kids.",
        config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_MEDIUM_AND_ABOVE"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_MEDIUM_AND_ABOVE"
                )
            ]
        )
    )
    
    print(response.text)
    return response


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
        exit(1)
    
    # Run all examples
    try:
        simple_completion()
        system_instruction_example()
        conversation_example()
        streaming_example()
        multimodal_example()
        function_calling_example()
        safety_settings_example()
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure your API key is valid.")
        print("Note: Free tier has rate limits. Wait a moment and try again if you hit quota limits.")
