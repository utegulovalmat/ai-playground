"""
Phase 4: Prompt Optimization Example
=====================================
This example demonstrates techniques for optimizing prompts.

Requirements:
- langchain>=1.2.0
- langchain-google-genai>=4.2.0
- GEMINI_API_KEY environment variable

What is Prompt Optimization?
Prompt optimization involves iteratively improving prompts to get better results.
This includes techniques like few-shot learning, chain-of-thought, and structured outputs.

Best Practices:
- Start simple, iterate based on results
- Use examples (few-shot learning)
- Break complex tasks into steps
- Test with diverse inputs
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field
from typing import List


# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.3
)


def basic_vs_optimized():
    """
    Compare basic vs optimized prompts.
    """
    print("=== Basic vs Optimized Prompts ===\n")
    
    # Basic prompt (vague)
    basic_prompt = "Tell me about Python"
    
    # Optimized prompt (specific, structured)
    optimized_prompt = """Explain Python programming language in the following structure:
1. Brief definition (1 sentence)
2. Key features (3 bullet points)
3. Common use cases (3 examples)
4. One code example

Keep it concise and beginner-friendly."""
    
    print("BASIC PROMPT:")
    print(f"'{basic_prompt}'\n")
    basic_response = llm.invoke([HumanMessage(content=basic_prompt)])
    print(f"Response: {basic_response.content[:200]}...\n")
    
    print("\n" + "="*60 + "\n")
    
    print("OPTIMIZED PROMPT:")
    print(f"'{optimized_prompt}'\n")
    optimized_response = llm.invoke([HumanMessage(content=optimized_prompt)])
    print(f"Response:\n{optimized_response.content}")


def few_shot_learning():
    """
    Few-shot learning example.
    Provide examples to guide the model's behavior.
    """
    print("\n\n=== Few-Shot Learning ===\n")
    
    # Define examples
    examples = [
        {
            "input": "Python",
            "output": "🐍 Python: High-level, interpreted language known for simplicity and readability. Popular in data science, web dev, and AI."
        },
        {
            "input": "JavaScript",
            "output": "⚡ JavaScript: Dynamic language for web development. Runs in browsers and on servers (Node.js). Essential for interactive websites."
        }
    ]
    
    # Create few-shot prompt template
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a tech educator. Describe programming languages concisely with an emoji, name, and brief description."),
        few_shot_prompt,
        ("human", "{input}")
    ])
    
    # Test with new input
    chain = final_prompt | llm
    
    test_inputs = ["Java", "Rust", "Go"]
    
    print("Examples provided to the model:")
    for ex in examples:
        print(f"  Input: {ex['input']}")
        print(f"  Output: {ex['output']}\n")
    
    print("Model's responses to new inputs:\n")
    for lang in test_inputs:
        response = chain.invoke({"input": lang})
        print(f"{response.content}\n")


def chain_of_thought():
    """
    Chain-of-thought prompting.
    Ask the model to show its reasoning process.
    """
    print("\n=== Chain-of-Thought Prompting ===\n")
    
    # Without CoT
    basic_prompt = "If a store has 23 apples and sells 17, then receives 45 more, how many apples does it have?"
    
    # With CoT
    cot_prompt = """If a store has 23 apples and sells 17, then receives 45 more, how many apples does it have?

Let's solve this step by step:
1. First, calculate how many apples remain after selling
2. Then, add the newly received apples
3. Finally, state the total

Show your work:"""
    
    print("WITHOUT Chain-of-Thought:")
    response1 = llm.invoke([HumanMessage(content=basic_prompt)])
    print(f"{response1.content}\n")
    
    print("\n" + "="*60 + "\n")
    
    print("WITH Chain-of-Thought:")
    response2 = llm.invoke([HumanMessage(content=cot_prompt)])
    print(f"{response2.content}")


def structured_output():
    """
    Structured output with Pydantic models.
    """
    print("\n\n=== Structured Output ===\n")
    
    # Define output structure
    class MovieReview(BaseModel):
        """A structured movie review."""
        title: str = Field(description="Movie title")
        rating: int = Field(description="Rating from 1-5 stars", ge=1, le=5)
        genre: str = Field(description="Movie genre")
        pros: List[str] = Field(description="List of positive aspects")
        cons: List[str] = Field(description="List of negative aspects")
        recommendation: str = Field(description="Who would enjoy this movie")
    
    # Create structured LLM
    structured_llm = llm.with_structured_output(MovieReview)
    
    review_text = """
    I watched Inception last night. It's a mind-bending sci-fi thriller.
    The plot is complex but fascinating, with amazing visual effects.
    However, it can be confusing at times and requires full attention.
    Great for people who enjoy thought-provoking films.
    I'd give it 5 stars.
    """
    
    prompt = f"Extract a structured review from this text:\n\n{review_text}"
    
    print("Input text:")
    print(review_text)
    print("\nStructured output:")
    
    result = structured_llm.invoke(prompt)
    print(f"Title: {result.title}")
    print(f"Rating: {'⭐' * result.rating}")
    print(f"Genre: {result.genre}")
    print(f"Pros: {', '.join(result.pros)}")
    print(f"Cons: {', '.join(result.cons)}")
    print(f"Recommendation: {result.recommendation}")


def role_prompting():
    """
    Role-based prompting.
    Assign specific roles to get better responses.
    """
    print("\n\n=== Role-Based Prompting ===\n")
    
    roles = {
        "Teacher": "You are an experienced teacher explaining concepts to high school students. Use simple language and analogies.",
        "Expert": "You are a senior software engineer with 15 years of experience. Provide technical, detailed explanations.",
        "Storyteller": "You are a creative storyteller. Explain concepts through engaging narratives and examples."
    }
    
    question = "What is recursion in programming?"
    
    for role_name, role_prompt in roles.items():
        print(f"\n{role_name.upper()} PERSPECTIVE:")
        print("-" * 60)
        
        messages = [
            SystemMessage(content=role_prompt),
            HumanMessage(content=question)
        ]
        
        response = llm.invoke(messages)
        print(f"{response.content[:300]}...\n")


def iterative_refinement():
    """
    Iterative prompt refinement example.
    """
    print("\n\n=== Iterative Refinement ===\n")
    
    # Version 1: Too vague
    v1 = "Write code"
    
    # Version 2: More specific
    v2 = "Write Python code for a function"
    
    # Version 3: Very specific
    v3 = """Write a Python function that:
- Takes a list of numbers as input
- Returns the average of the numbers
- Handles empty lists gracefully
- Includes docstring and type hints
- Add 2-3 test cases"""
    
    prompts = [
        ("Version 1 (Vague)", v1),
        ("Version 2 (Better)", v2),
        ("Version 3 (Optimized)", v3)
    ]
    
    for version, prompt in prompts:
        print(f"\n{version}:")
        print(f"Prompt: '{prompt}'\n")
        
        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"Response:\n{response.content[:200]}...")
        print("\n" + "="*60)


def temperature_comparison():
    """
    Compare different temperature settings.
    """
    print("\n\n=== Temperature Comparison ===\n")
    
    prompt = "Write a creative tagline for a coffee shop"
    
    temperatures = [0.0, 0.5, 1.0, 1.5]
    
    print(f"Prompt: '{prompt}'\n")
    
    for temp in temperatures:
        custom_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            temperature=temp
        )
        
        response = custom_llm.invoke([HumanMessage(content=prompt)])
        print(f"Temperature {temp}: {response.content}")


if __name__ == "__main__":
    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        exit(1)
    
    try:
        # Run all examples
        basic_vs_optimized()
        few_shot_learning()
        chain_of_thought()
        structured_output()
        role_prompting()
        iterative_refinement()
        temperature_comparison()
        
        print("\n\n" + "="*60)
        print("✓ All examples completed!")
        print("="*60)
        print("\nPrompt Optimization Techniques:")
        print("1. Be specific and structured")
        print("2. Use few-shot examples")
        print("3. Request step-by-step reasoning (CoT)")
        print("4. Define structured outputs")
        print("5. Assign appropriate roles")
        print("6. Iterate and refine")
        print("7. Adjust temperature for creativity vs consistency")
        
    except Exception as e:
        print(f"\nError: {e}")
