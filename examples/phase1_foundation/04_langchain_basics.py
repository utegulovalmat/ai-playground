"""
LangChain Example - Enhanced
============================
This example demonstrates LangChain's powerful abstractions for working with LLMs.

Requirements:
- langchain>=1.2.0
- langchain-google-genai>=4.2.0 (or other provider packages)
- GEMINI_API_KEY environment variable (or other provider keys)

Best Practices:
- Use LangChain for complex workflows and chains
- Leverage built-in memory for conversations
- Use prompt templates for reusability
- Take advantage of LangChain's ecosystem
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory


# Initialize the model
# LangChain supports multiple providers - just swap the import
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.7
)


def simple_example():
    """
    Simple LangChain example.
    Shows basic message invocation.
    """
    print("=== Simple Example ===")
    
    messages = [
        HumanMessage(content="What is the capital of France?")
    ]
    
    response = llm.invoke(messages)
    print(response.content)
    return response


def system_message_example():
    """
    Example with system messages.
    System messages set the AI's behavior and personality.
    """
    print("\n=== System Message Example ===")
    
    messages = [
        SystemMessage(content="You are a helpful coding assistant. Provide concise, practical advice."),
        HumanMessage(content="How do I read a file in Python?")
    ]
    
    response = llm.invoke(messages)
    print(response.content)
    return response


def prompt_template_example():
    """
    Using prompt templates for reusability.
    Templates make it easy to create consistent prompts.
    """
    print("\n=== Prompt Template Example ===")
    
    # Create a reusable prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "{text}")
    ])
    
    # Create a chain: prompt -> model -> output parser
    chain = prompt | llm | StrOutputParser()
    
    # Use the chain
    result = chain.invoke({
        "input_language": "English",
        "output_language": "French",
        "text": "Hello, how are you?"
    })
    
    print(result)
    return result


def conversation_chain_example():
    """
    Conversation with memory.
    LangChain can automatically manage conversation history.
    """
    print("\n=== Conversation Chain Example ===")
    
    # Create a prompt that includes message history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # Create the chain
    chain = prompt | llm | StrOutputParser()
    
    # Maintain conversation history manually
    history = []
    
    # First turn
    response1 = chain.invoke({
        "history": history,
        "input": "My name is Alice."
    })
    print(f"AI: {response1}\n")
    
    # Add to history
    history.append(HumanMessage(content="My name is Alice."))
    history.append(AIMessage(content=response1))
    
    # Second turn - references previous context
    response2 = chain.invoke({
        "history": history,
        "input": "What's my name?"
    })
    print(f"AI: {response2}")
    
    return chain


def streaming_example():
    """
    Streaming responses with LangChain.
    Useful for real-time user interfaces.
    """
    print("\n=== Streaming Example ===")
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    
    print("AI: ", end="", flush=True)
    
    for chunk in chain.stream({"question": "Write a haiku about programming."}):
        print(chunk.content, end="", flush=True)
    
    print()  # New line after streaming


def multi_step_chain_example():
    """
    Multi-step chain example.
    Shows how to chain multiple operations together.
    """
    print("\n=== Multi-Step Chain Example ===")
    
    # Step 1: Generate a topic
    topic_prompt = ChatPromptTemplate.from_template(
        "Generate a random interesting topic about {subject}. Just output the topic, nothing else."
    )
    
    # Step 2: Write about the topic
    writing_prompt = ChatPromptTemplate.from_template(
        "Write a short paragraph about: {topic}"
    )
    
    # Create chains
    topic_chain = topic_prompt | llm | StrOutputParser()
    writing_chain = writing_prompt | llm | StrOutputParser()
    
    # Execute step by step
    topic = topic_chain.invoke({"subject": "space exploration"})
    print(f"Generated topic: {topic}\n")
    
    paragraph = writing_chain.invoke({"topic": topic})
    print(f"Generated paragraph:\n{paragraph}")
    
    return paragraph


def structured_output_example():
    """
    Structured output example.
    Parse LLM responses into structured data.
    """
    print("\n=== Structured Output Example ===")
    
    from langchain_core.pydantic_v1 import BaseModel, Field
    
    # Define output structure
    class Person(BaseModel):
        """Information about a person."""
        name: str = Field(description="The person's name")
        age: int = Field(description="The person's age")
        occupation: str = Field(description="The person's occupation")
    
    # Create a chain that outputs structured data
    prompt = ChatPromptTemplate.from_template(
        "Extract information about the person from this text: {text}"
    )
    
    # Note: Structured output support varies by model
    # This is a simplified example
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "text": "John Smith is a 35-year-old software engineer."
    })
    
    print(result)
    return result


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
        print("\nNote: You can use other providers by changing the import:")
        print("- langchain_openai.ChatOpenAI (requires OPENAI_API_KEY)")
        print("- langchain_anthropic.ChatAnthropic (requires ANTHROPIC_API_KEY)")
        exit(1)
    
    # Run all examples
    try:
        simple_example()
        system_message_example()
        prompt_template_example()
        conversation_chain_example()
        streaming_example()
        multi_step_chain_example()
        structured_output_example()
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure your API key is valid and you have credits available.")
