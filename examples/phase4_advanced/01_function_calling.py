"""
Phase 4: Advanced Function Calling Example
===========================================
This example demonstrates advanced function calling (tool use) patterns.

Requirements:
- langchain>=1.2.0
- langchain-google-genai>=4.2.0
- GEMINI_API_KEY environment variable

What is Function Calling?
Function calling allows LLMs to use external tools and APIs.
The LLM decides when and how to call functions based on the conversation.

Best Practices:
- Provide clear function descriptions
- Use type hints for parameters
- Handle errors gracefully
- Validate function outputs
"""

import os
import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime
import random


# Define tools using the @tool decorator
@tool
def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Get the current weather for a location.
    
    Args:
        location: The city name, e.g., 'Paris' or 'Tokyo'
        unit: Temperature unit, either 'celsius' or 'fahrenheit'
    
    Returns:
        Weather information as a string
    """
    # Simulate weather API call
    weather_data = {
        "temperature": random.randint(15, 30),
        "condition": random.choice(["sunny", "cloudy", "rainy", "partly cloudy"]),
        "humidity": random.randint(40, 80),
        "wind_speed": random.randint(5, 25)
    }
    
    temp = weather_data["temperature"]
    if unit == "fahrenheit":
        temp = (temp * 9/5) + 32
    
    return json.dumps({
        "location": location,
        "temperature": f"{temp}°{unit[0].upper()}",
        "condition": weather_data["condition"],
        "humidity": f"{weather_data['humidity']}%",
        "wind_speed": f"{weather_data['wind_speed']} km/h"
    })


@tool
def search_web(query: str, num_results: int = 3) -> str:
    """
    Search the web for information.
    
    Args:
        query: The search query
        num_results: Number of results to return (1-5)
    
    Returns:
        Search results as a string
    """
    # Simulate web search
    results = [
        {
            "title": f"Result {i+1} for '{query}'",
            "snippet": f"This is a simulated search result about {query}. "
                      f"It contains relevant information that would help answer questions.",
            "url": f"https://example.com/result{i+1}"
        }
        for i in range(min(num_results, 5))
    ]
    
    return json.dumps(results, indent=2)


@tool
def calculate(expression: str) -> str:
    """
    Calculate a mathematical expression.
    
    Args:
        expression: A math expression like '2 + 2' or '10 * 5'
    
    Returns:
        The calculation result
    """
    try:
        # Safe evaluation (in production, use a proper math parser)
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """
    Get the current time.
    
    Args:
        timezone: Timezone name (e.g., 'UTC', 'EST', 'PST')
    
    Returns:
        Current time as a string
    """
    # Simplified timezone handling
    now = datetime.now()
    return f"Current time in {timezone}: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@tool
def create_reminder(task: str, time: str) -> str:
    """
    Create a reminder for a task.
    
    Args:
        task: The task to remember
        time: When to be reminded (e.g., '2pm', 'tomorrow')
    
    Returns:
        Confirmation message
    """
    return f"✓ Reminder created: '{task}' at {time}"


def simple_function_calling():
    """
    Simple function calling example.
    """
    print("=== Simple Function Calling ===\n")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0
    )
    
    # Create tools list
    tools = [get_current_weather, calculate]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    # Ask a question that requires a tool
    query = "What's the weather in Paris and what is 15 * 7?"
    
    print(f"Query: {query}\n")
    
    response = llm_with_tools.invoke(query)
    
    # Check if tools were called
    if response.tool_calls:
        print("Tools called:")
        for tool_call in response.tool_calls:
            print(f"  - {tool_call['name']}")
            print(f"    Args: {tool_call['args']}")
            
            # Execute the tool
            if tool_call['name'] == 'get_current_weather':
                result = get_current_weather.invoke(tool_call['args'])
            elif tool_call['name'] == 'calculate':
                result = calculate.invoke(tool_call['args'])
            
            print(f"    Result: {result}\n")
    else:
        print(f"Response: {response.content}")


def agent_with_tools():
    """
    Create an agent that can use multiple tools.
    """
    print("\n=== Agent with Multiple Tools ===\n")
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0
    )
    
    # Create tools
    tools = [
        get_current_weather,
        search_web,
        calculate,
        get_current_time,
        create_reminder
    ]
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to various tools. Use them when needed."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Test queries
    queries = [
        "What's the weather in Tokyo?",
        "Calculate 234 * 567",
        "What time is it?",
        "Remind me to call John at 3pm"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        result = agent_executor.invoke({"input": query})
        print(f"\nFinal Answer: {result['output']}\n")


def multi_step_reasoning():
    """
    Example of multi-step reasoning with tools.
    """
    print("\n=== Multi-Step Reasoning ===\n")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0
    )
    
    tools = [get_current_weather, calculate]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Break down complex queries into steps and use tools as needed."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Complex query requiring multiple steps
    query = """
    Get the weather in Paris and London, then calculate the average temperature.
    """
    
    print(f"Query: {query}")
    print("\nAgent thinking process:")
    print("-" * 60)
    
    result = agent_executor.invoke({"input": query})
    
    print("\n" + "="*60)
    print(f"Final Answer: {result['output']}")


def custom_tool_example():
    """
    Example of creating custom tools for specific use cases.
    """
    print("\n\n=== Custom Tools Example ===\n")
    
    @tool
    def analyze_sentiment(text: str) -> str:
        """
        Analyze the sentiment of text.
        
        Args:
            text: The text to analyze
        
        Returns:
            Sentiment analysis result
        """
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'happy', 'love']
        negative_words = ['bad', 'terrible', 'awful', 'sad', 'hate']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return json.dumps({
            "sentiment": sentiment,
            "positive_words": pos_count,
            "negative_words": neg_count,
            "confidence": abs(pos_count - neg_count) / max(pos_count + neg_count, 1)
        })
    
    @tool
    def translate_text(text: str, target_language: str) -> str:
        """
        Translate text to another language.
        
        Args:
            text: The text to translate
            target_language: Target language (e.g., 'French', 'Spanish')
        
        Returns:
            Simulated translation
        """
        # Simulated translation
        return f"[Translated to {target_language}]: {text}"
    
    # Create agent with custom tools
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.environ.get("GEMINI_API_KEY")
    )
    
    tools = [analyze_sentiment, translate_text]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with text analysis capabilities."),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    query = "Analyze the sentiment of 'I love this product, it's excellent!' and translate it to French"
    
    print(f"Query: {query}\n")
    result = agent_executor.invoke({"input": query})
    print(f"\nResult: {result['output']}")


if __name__ == "__main__":
    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        exit(1)
    
    try:
        # Run examples
        simple_function_calling()
        agent_with_tools()
        multi_step_reasoning()
        custom_tool_example()
        
        print("\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)
        print("\nKey Takeaways:")
        print("1. Tools extend LLM capabilities")
        print("2. Agents can chain multiple tool calls")
        print("3. Clear tool descriptions are crucial")
        print("4. Custom tools enable domain-specific functionality")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure to install: uv pip install langchain langchain-google-genai")
