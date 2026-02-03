"""
Pydantic AI Example - Enhanced
===============================
This example demonstrates Pydantic AI's type-safe approach to LLM applications.

Requirements:
- pydantic-ai>=1.52.0
- GEMINI_API_KEY or OPENAI_API_KEY environment variable

Best Practices:
- Use type hints for better IDE support
- Leverage Pydantic models for structured outputs
- Use dependency injection for testability
- Take advantage of built-in validation
"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel


def simple_example():
    """
    Simple Pydantic AI example.
    Shows basic agent creation and usage.
    """
    print("=== Simple Example ===")
    
    # Create an agent with instructions
    agent = Agent(
        'gemini-2.0-flash',
        instructions='You are a helpful assistant. Be concise.',
    )
    
    # Run synchronously
    result = agent.run_sync('What is the capital of France?')
    print(result.output)
    return result


def system_prompt_example():
    """
    Example with detailed system prompts.
    Pydantic AI makes it easy to set agent behavior.
    """
    print("\n=== System Prompt Example ===")
    
    agent = Agent(
        'gemini-2.0-flash',
        instructions="""You are a Python programming tutor.
        - Explain concepts clearly and simply
        - Provide code examples when relevant
        - Be encouraging and supportive
        """,
    )
    
    result = agent.run_sync('How do I create a list in Python?')
    print(result.output)
    return result


def structured_output_example():
    """
    Structured output with Pydantic models.
    This is where Pydantic AI really shines - type-safe outputs!
    """
    print("\n=== Structured Output Example ===")
    
    # Define output structure using Pydantic
    class Person(BaseModel):
        """Information about a person."""
        name: str = Field(description="The person's full name")
        age: int = Field(description="The person's age in years")
        occupation: str = Field(description="The person's job or profession")
        hobbies: list[str] = Field(description="List of the person's hobbies")
    
    # Create agent with result type
    agent = Agent(
        'gemini-2.0-flash',
        result_type=Person,
        instructions='Extract person information from the text.',
    )
    
    # The result will be validated and typed!
    result = agent.run_sync(
        'John Smith is a 35-year-old software engineer who enjoys hiking, photography, and cooking.'
    )
    
    # Access with full type safety
    person: Person = result.output
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Occupation: {person.occupation}")
    print(f"Hobbies: {', '.join(person.hobbies)}")
    
    return result


def dependency_injection_example():
    """
    Dependency injection example.
    Pass runtime context to your agent for dynamic behavior.
    """
    print("\n=== Dependency Injection Example ===")
    
    # Define dependencies
    class UserContext(BaseModel):
        user_name: str
        user_level: str  # beginner, intermediate, advanced
    
    # Create agent with dependencies
    agent = Agent(
        'gemini-2.0-flash',
        deps_type=UserContext,
        instructions="""Adjust your explanation based on the user's level:
        - beginner: Use simple terms and analogies
        - intermediate: Include technical details
        - advanced: Discuss advanced concepts and edge cases
        """,
    )
    
    # Run with dependencies
    result = agent.run_sync(
        'Explain what a decorator is in Python.',
        deps=UserContext(user_name='Alice', user_level='beginner')
    )
    
    print(result.output)
    return result


def tool_usage_example():
    """
    Tool usage example (function calling).
    Pydantic AI makes it easy to give agents tools to use.
    """
    print("\n=== Tool Usage Example ===")
    
    # Create agent
    agent = Agent(
        'gemini-2.0-flash',
        instructions='You are a helpful assistant with access to tools.',
    )
    
    # Define a tool using decorator
    @agent.tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location.
        
        Args:
            location: The city name, e.g., 'Paris' or 'Tokyo'
        """
        # In a real app, this would call a weather API
        return f"The weather in {location} is sunny and 22°C"
    
    @agent.tool
    def calculate(expression: str) -> str:
        """Calculate a mathematical expression.
        
        Args:
            expression: A math expression like '2 + 2' or '10 * 5'
        """
        try:
            result = eval(expression)  # Note: eval is unsafe in production!
            return f"The result is {result}"
        except Exception as e:
            return f"Error calculating: {e}"
    
    # The agent can now use these tools
    result = agent.run_sync('What is the weather in Tokyo and what is 15 * 7?')
    print(result.output)
    
    return result


def conversation_example():
    """
    Conversation with message history.
    Pydantic AI handles conversation state elegantly.
    """
    print("\n=== Conversation Example ===")
    
    agent = Agent(
        'gemini-2.0-flash',
        instructions='You are a friendly chatbot. Remember context from the conversation.',
    )
    
    # First message
    result1 = agent.run_sync('My favorite programming language is Python.')
    print(f"AI: {result1.output}\n")
    
    # Continue conversation with history
    result2 = agent.run_sync(
        'What is my favorite language?',
        message_history=result1.new_messages()  # Pass previous messages
    )
    print(f"AI: {result2.output}")
    
    return result2


def validation_example():
    """
    Input validation example.
    Pydantic AI validates inputs automatically.
    """
    print("\n=== Validation Example ===")
    
    class MovieReview(BaseModel):
        """A movie review with rating."""
        title: str = Field(description="Movie title")
        rating: int = Field(ge=1, le=5, description="Rating from 1-5 stars")
        review: str = Field(min_length=10, description="Review text")
        recommended: bool = Field(description="Whether you recommend it")
    
    agent = Agent(
        'gemini-2.0-flash',
        result_type=MovieReview,
        instructions='Extract movie review information. Rating must be 1-5.',
    )
    
    result = agent.run_sync(
        'I watched Inception last night. It was mind-blowing! The plot was complex but amazing. Definitely a 5-star movie. Highly recommend!'
    )
    
    review: MovieReview = result.output
    print(f"Movie: {review.title}")
    print(f"Rating: {'⭐' * review.rating}")
    print(f"Review: {review.review}")
    print(f"Recommended: {'Yes' if review.recommended else 'No'}")
    
    return result


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("Error: No API key found")
        print("Set one of these environment variables:")
        print("- export GEMINI_API_KEY='your-key-here'")
        print("- export OPENAI_API_KEY='your-key-here'")
        print("\nTo use OpenAI instead, change the model to 'openai:gpt-4o-mini'")
        exit(1)
    
    # Run all examples
    try:
        simple_example()
        system_prompt_example()
        structured_output_example()
        dependency_injection_example()
        tool_usage_example()
        conversation_example()
        validation_example()
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure your API key is valid and you have credits available.")
