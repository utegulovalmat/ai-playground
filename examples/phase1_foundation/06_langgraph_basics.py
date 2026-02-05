"""
LangGraph Example - Graph-Based Workflows
==========================================
This example demonstrates LangGraph's powerful graph-based approach to building
stateful, multi-actor applications with LLMs.

Requirements:
- langgraph>=0.2.0
- langchain>=1.2.0
- langchain-google-genai>=4.2.0 (or other provider packages)
- GEMINI_API_KEY environment variable (or other provider keys)

Best Practices:
- Use graphs for complex, stateful workflows
- Leverage conditional edges for decision-making
- Implement checkpointing for long-running processes
- Use state to maintain context across nodes
"""

import os
from typing import TypedDict, Annotated, Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.7
)


# ============================================================================
# Example 1: Simple Sequential Chain
# ============================================================================

def simple_sequential_example():
    """
    Simple sequential chain example.
    Shows basic linear workflow: input -> process -> output
    """
    print("=== Simple Sequential Chain ===")
    
    # Define state
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    # Define nodes
    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}
    
    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    
    graph = graph_builder.compile()
    
    # Run the graph
    result = graph.invoke({
        "messages": [HumanMessage(content="What is LangGraph?")]
    })
    
    print(f"AI: {result['messages'][-1].content}\n")
    return graph


# ============================================================================
# Example 2: Conditional Branching
# ============================================================================

def conditional_branching_example():
    """
    Conditional branching example.
    Routes to different nodes based on conditions.
    """
    print("=== Conditional Branching ===")
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        category: str
    
    # Categorizer node
    def categorize(state: State):
        response = llm.invoke([
            SystemMessage(content="Categorize the user's question as 'technical', 'general', or 'creative'. Respond with only one word."),
            state["messages"][-1]
        ])
        category = response.content.lower().strip()
        print(f"Category: {category}")
        return {"category": category}
    
    # Specialized responder nodes
    def technical_response(state: State):
        response = llm.invoke([
            SystemMessage(content="You are a technical expert. Provide detailed, technical answers."),
            state["messages"][-1]
        ])
        return {"messages": [response]}
    
    def general_response(state: State):
        response = llm.invoke([
            SystemMessage(content="You are a friendly assistant. Provide clear, simple answers."),
            state["messages"][-1]
        ])
        return {"messages": [response]}
    
    def creative_response(state: State):
        response = llm.invoke([
            SystemMessage(content="You are a creative writer. Provide imaginative, engaging answers."),
            state["messages"][-1]
        ])
        return {"messages": [response]}
    
    # Router function
    def route_question(state: State) -> Literal["technical", "general", "creative"]:
        category = state.get("category", "general")
        if "technical" in category:
            return "technical"
        elif "creative" in category:
            return "creative"
        else:
            return "general"
    
    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("categorize", categorize)
    graph_builder.add_node("technical", technical_response)
    graph_builder.add_node("general", general_response)
    graph_builder.add_node("creative", creative_response)
    
    graph_builder.add_edge(START, "categorize")
    graph_builder.add_conditional_edges(
        "categorize",
        route_question,
        {
            "technical": "technical",
            "general": "general",
            "creative": "creative"
        }
    )
    graph_builder.add_edge("technical", END)
    graph_builder.add_edge("general", END)
    graph_builder.add_edge("creative", END)
    
    graph = graph_builder.compile()
    
    # Test with different types of questions
    questions = [
        "Explain how neural networks work",
        "What's the weather like?",
        "Write a poem about coding"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "category": ""
        })
        print(f"AI: {result['messages'][-1].content[:150]}...\n")
    
    return graph


# ============================================================================
# Example 3: Cyclic Graph (Loops)
# ============================================================================

def cyclic_graph_example():
    """
    Cyclic graph with loops.
    Demonstrates iterative refinement until a condition is met.
    """
    print("=== Cyclic Graph (Iterative Refinement) ===")
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        iteration: int
        is_good: bool
    
    # Generator node
    def generate_idea(state: State):
        iteration = state.get("iteration", 0) + 1
        response = llm.invoke([
            SystemMessage(content=f"Generate a creative product idea (iteration {iteration}). Be brief."),
            HumanMessage(content="Generate a unique tech product idea")
        ])
        print(f"\nIteration {iteration}: {response.content}")
        return {"messages": [response], "iteration": iteration}
    
    # Critic node
    def critique_idea(state: State):
        last_idea = state["messages"][-1].content
        response = llm.invoke([
            SystemMessage(content="Rate this idea from 1-10. If 8 or higher, say 'APPROVED: [score]'. Otherwise say 'NEEDS_WORK: [score]' and explain why."),
            HumanMessage(content=f"Idea: {last_idea}")
        ])
        
        is_approved = "APPROVED" in response.content
        print(f"Critique: {response.content}")
        
        return {"messages": [response], "is_good": is_approved}
    
    # Router: continue or end
    def should_continue(state: State) -> Literal["generate", "end"]:
        # Stop after 3 iterations or if approved
        if state.get("is_good", False) or state.get("iteration", 0) >= 3:
            return "end"
        return "generate"
    
    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("generate", generate_idea)
    graph_builder.add_node("critique", critique_idea)
    
    graph_builder.add_edge(START, "generate")
    graph_builder.add_edge("generate", "critique")
    graph_builder.add_conditional_edges(
        "critique",
        should_continue,
        {
            "generate": "generate",
            "end": END
        }
    )
    
    graph = graph_builder.compile()
    
    # Run the iterative process
    result = graph.invoke({
        "messages": [],
        "iteration": 0,
        "is_good": False
    })
    
    print(f"\nFinal iteration count: {result['iteration']}")
    return graph


# ============================================================================
# Example 4: State Management
# ============================================================================

def state_management_example():
    """
    Advanced state management example.
    Shows how to maintain and update complex state across nodes.
    """
    print("=== State Management ===")
    
    class ResearchState(TypedDict):
        topic: str
        messages: Annotated[list, add_messages]
        facts: list[str]
        summary: str
    
    # Research node
    def research_topic(state: ResearchState):
        topic = state["topic"]
        response = llm.invoke([
            SystemMessage(content="List 3 interesting facts about the topic. Format as numbered list."),
            HumanMessage(content=f"Topic: {topic}")
        ])
        
        # Extract facts (simplified)
        facts = [line.strip() for line in response.content.split('\n') if line.strip() and any(c.isdigit() for c in line[:3])]
        
        return {
            "messages": [response],
            "facts": facts
        }
    
    # Summarize node
    def summarize_research(state: ResearchState):
        facts_text = "\n".join(state["facts"])
        response = llm.invoke([
            SystemMessage(content="Create a brief summary paragraph from these facts."),
            HumanMessage(content=f"Facts:\n{facts_text}")
        ])
        
        return {
            "messages": [response],
            "summary": response.content
        }
    
    # Build graph
    graph_builder = StateGraph(ResearchState)
    graph_builder.add_node("research", research_topic)
    graph_builder.add_node("summarize", summarize_research)
    
    graph_builder.add_edge(START, "research")
    graph_builder.add_edge("research", "summarize")
    graph_builder.add_edge("summarize", END)
    
    graph = graph_builder.compile()
    
    # Run research
    result = graph.invoke({
        "topic": "Quantum Computing",
        "messages": [],
        "facts": [],
        "summary": ""
    })
    
    print(f"\nTopic: {result['topic']}")
    print(f"\nFacts found: {len(result['facts'])}")
    for fact in result['facts']:
        print(f"  - {fact}")
    print(f"\nSummary:\n{result['summary']}")
    
    return graph


# ============================================================================
# Example 5: Multi-Agent Collaboration
# ============================================================================

def multi_agent_example():
    """
    Multi-agent collaboration example.
    Different specialized agents work together on a task.
    """
    print("=== Multi-Agent Collaboration ===")
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        code: str
        review: str
        final_code: str
    
    # Coder agent
    def coder_agent(state: State):
        response = llm.invoke([
            SystemMessage(content="You are a Python developer. Write clean, simple code."),
            state["messages"][-1]
        ])
        
        # Extract code (simplified)
        code = response.content
        print(f"\nCoder wrote:\n{code[:200]}...")
        
        return {"messages": [response], "code": code}
    
    # Reviewer agent
    def reviewer_agent(state: State):
        code = state["code"]
        response = llm.invoke([
            SystemMessage(content="You are a code reviewer. Provide constructive feedback."),
            HumanMessage(content=f"Review this code:\n\n{code}")
        ])
        
        print(f"\nReviewer says:\n{response.content[:200]}...")
        
        return {"messages": [response], "review": response.content}
    
    # Refactor agent
    def refactor_agent(state: State):
        code = state["code"]
        review = state["review"]
        response = llm.invoke([
            SystemMessage(content="You are a refactoring expert. Improve the code based on feedback."),
            HumanMessage(content=f"Original code:\n{code}\n\nFeedback:\n{review}\n\nProvide improved code.")
        ])
        
        print(f"\nRefactored code:\n{response.content[:200]}...")
        
        return {"messages": [response], "final_code": response.content}
    
    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("coder", coder_agent)
    graph_builder.add_node("reviewer", reviewer_agent)
    graph_builder.add_node("refactor", refactor_agent)
    
    graph_builder.add_edge(START, "coder")
    graph_builder.add_edge("coder", "reviewer")
    graph_builder.add_edge("reviewer", "refactor")
    graph_builder.add_edge("refactor", END)
    
    graph = graph_builder.compile()
    
    # Run collaboration
    result = graph.invoke({
        "messages": [HumanMessage(content="Write a function to calculate fibonacci numbers")],
        "code": "",
        "review": "",
        "final_code": ""
    })
    
    print(f"\n=== Final Result ===")
    print(result["final_code"][:300])
    
    return graph


# ============================================================================
# Example 6: Checkpointing and Persistence
# ============================================================================

def checkpointing_example():
    """
    Checkpointing example.
    Shows how to save and resume graph state.
    """
    print("=== Checkpointing and Persistence ===")
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        step: int
    
    def step_1(state: State):
        print("Executing Step 1...")
        response = llm.invoke([
            HumanMessage(content="Say 'Step 1 complete' and mention you're ready for step 2")
        ])
        return {"messages": [response], "step": 1}
    
    def step_2(state: State):
        print("Executing Step 2...")
        response = llm.invoke([
            HumanMessage(content="Say 'Step 2 complete' and mention you're ready for step 3")
        ])
        return {"messages": [response], "step": 2}
    
    def step_3(state: State):
        print("Executing Step 3...")
        response = llm.invoke([
            HumanMessage(content="Say 'All steps complete!'")
        ])
        return {"messages": [response], "step": 3}
    
    # Build graph with checkpointing
    graph_builder = StateGraph(State)
    graph_builder.add_node("step_1", step_1)
    graph_builder.add_node("step_2", step_2)
    graph_builder.add_node("step_3", step_3)
    
    graph_builder.add_edge(START, "step_1")
    graph_builder.add_edge("step_1", "step_2")
    graph_builder.add_edge("step_2", "step_3")
    graph_builder.add_edge("step_3", END)
    
    # Add memory checkpointer
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    
    # Run with thread ID for persistence
    config = {"configurable": {"thread_id": "example_thread"}}
    
    print("\nRunning complete workflow...")
    result = graph.invoke({
        "messages": [],
        "step": 0
    }, config)
    
    print(f"\nCompleted step: {result['step']}")
    print(f"Last message: {result['messages'][-1].content}")
    
    # You can resume from checkpoint by using the same thread_id
    print("\n✓ State saved to checkpoint (can be resumed later)")
    
    return graph


# ============================================================================
# Example 7: Human-in-the-Loop
# ============================================================================

def human_in_loop_example():
    """
    Human-in-the-loop example.
    Shows how to pause for human input during execution.
    Note: This is a simplified version. Full implementation requires
    interrupt handling.
    """
    print("=== Human-in-the-Loop (Simplified) ===")
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        user_approved: bool
    
    def generate_draft(state: State):
        response = llm.invoke([
            SystemMessage(content="Write a short email draft about a meeting."),
            HumanMessage(content="Draft an email to schedule a team meeting")
        ])
        print(f"\nDraft email:\n{response.content}")
        return {"messages": [response]}
    
    def send_email(state: State):
        print("\n✓ Email sent!")
        return {"messages": [AIMessage(content="Email has been sent successfully.")]}
    
    # Build graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("draft", generate_draft)
    graph_builder.add_node("send", send_email)
    
    graph_builder.add_edge(START, "draft")
    graph_builder.add_edge("draft", "send")
    graph_builder.add_edge("send", END)
    
    graph = graph_builder.compile()
    
    # In a real implementation, you would:
    # 1. Use interrupt_before=["send"] in compile()
    # 2. Get user approval
    # 3. Resume with graph.invoke(None, config) if approved
    
    print("\nNote: In production, you would pause here for user approval")
    print("before sending. This is a simplified demonstration.\n")
    
    graph.invoke({
        "messages": [],
        "user_approved": True
    })
    
    return graph


if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
        exit(1)
    
    print("=" * 70)
    print("LangGraph Examples - Graph-Based Workflows")
    print("=" * 70)
    
    # Run all examples
    try:
        simple_sequential_example()
        print("\n" + "=" * 70 + "\n")
        
        conditional_branching_example()
        print("\n" + "=" * 70 + "\n")
        
        cyclic_graph_example()
        print("\n" + "=" * 70 + "\n")
        
        state_management_example()
        print("\n" + "=" * 70 + "\n")
        
        multi_agent_example()
        print("\n" + "=" * 70 + "\n")
        
        checkpointing_example()
        print("\n" + "=" * 70 + "\n")
        
        human_in_loop_example()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure your API key is valid and you have credits available.")
