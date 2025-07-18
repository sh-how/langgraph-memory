"""
Episodic Planner Agent with Episodic Memory

This script demonstrates a planning agent that stores each planning event (plan proposal, approval, feedback, etc.) as an episode in episodic memory.
Uses Redis for persistence and local LLM for processing.

Features:
- Each planning event is stored as an episode (timestamp + content)
- Planner can recall/search past planning episodes
- Redis-based persistence for long-term memory
- User-specific namespaces for planning episodes
- Automated and interactive demo modes
"""

import uuid
from datetime import datetime
from langgraph.store.redis import RedisStore
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langmem import create_manage_memory_tool, create_search_memory_tool
from config.llm import create_local_embeddings, create_local_llm

# Helper to format a planning episode
def make_planning_episode(plan_content: str, approved: bool, feedback: str = "") -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "plan": plan_content,
        "approved": approved,
        "feedback": feedback,
        "outcome": "approved" if approved else "rejected"
    }

def setup_episodic_planner():
    """Set up the episodic planner system with local LLM and Redis storage."""
    
    # Initialize embeddings and LLM using your configuration
    embeddings = create_local_embeddings()
    llm = create_local_llm()
    
    # Redis configuration
    REDIS_URI = "redis://localhost:6379"
    
    
    checkpointer = InMemorySaver()

    # Create Redis store with embeddings using context manager
    store_context = RedisStore.from_conn_string(REDIS_URI)
    store = store_context.__enter__()
    store.setup()
    store.index = {
        "dims": 1536,
        "embed": embeddings,
    }
    
    # Create episodic planner agent
    planner_agent = create_react_agent(
        llm,
        prompt="""You are a Planning Agent with episodic memory for plan generation and approval tracking.

Your episodic memory stores:
- Plans you generate
- Whether each plan was approved or rejected  
- Feedback received on rejected plans
- Timestamps of all planning events

This memory acts as few-shot examples to help you create better plans. When generating new plans:
1. Search your memory for similar planning scenarios
2. Learn from approved plans (what worked)
3. Learn from rejected plans (what to avoid)
4. Use this knowledge to create better, more detailed plans

When asked to store a plan outcome:
1. Use manage_memory to store the planning episode
2. Include timestamp, plan content, approval status, and feedback
3. Respond with "Plan outcome stored successfully."

When asked to create a plan:
1. First search your memory for relevant past planning episodes
2. Learn from the patterns in approved vs rejected plans
3. Provide a detailed, step-by-step plan based on your learning

When asked about your planning history:
1. Search your memory for relevant episodes
2. Summarize patterns and learnings from past plans""",
        tools=[
            create_manage_memory_tool(namespace=("planner_episodes",)),
            create_search_memory_tool(namespace=("planner_episodes",)),
        ],
        store=store,
        checkpointer=checkpointer,
    )
    
    return planner_agent, store

def run_planner_episodic_demo():
    """Run a demonstration of episodic planner capabilities."""
    
    print("=== Episodic Planner Agent Demo ===")
    print("Stores plans and their approval status for few-shot learning")
    print("Using local LLM and Redis storage")
    print("-" * 50)
    
    # Set up the planner system
    planner_agent, store = setup_episodic_planner()
    config = {"configurable": {"thread_id": "planner-episodic-demo"}}

    # Demo: Store some example plans with outcomes
    plan_examples = [
        ("Task 1: Write Python script for data processing\nTask 2: Test the script\nTask 3: Deploy to production", 
         False, 
         "Missing documentation step and error handling"),
        ("Task 1: Write Python script for data processing\nTask 2: Add error handling\nTask 3: Write documentation\nTask 4: Test the script\nTask 5: Deploy to production", 
         True, 
         ""),
        ("Task 1: Create database schema\nTask 2: Implement API endpoints\nTask 3: Deploy", 
         False, 
         "Missing testing and security considerations"),
        ("Task 1: Design database schema\nTask 2: Implement API endpoints with authentication\nTask 3: Write unit tests\nTask 4: Security review\nTask 5: Deploy to staging\nTask 6: Deploy to production", 
         True, 
         ""),
    ]
    
    for i, (plan, approved, feedback) in enumerate(plan_examples, 1):
        print(f"\nStoring Plan {i} ({'APPROVED' if approved else 'REJECTED'}):")
        print(f"Plan: {plan}")
        if feedback:
            print(f"Feedback: {feedback}")
        
        # Format as planning episode
        episode_data = make_planning_episode(plan, approved, feedback)
        
        response = planner_agent.invoke({
            "messages": [
                {"role": "user", "content": f"Store this plan outcome in your memory: {episode_data}"}
            ]
        }, config=config) # type: ignore
        print(f"AI: {response['messages'][-1].content}")

    # Test few-shot learning by asking for a new plan
    print("\n" + "="*50)
    print("TESTING FEW-SHOT LEARNING")
    print("="*50)
    
    test_queries = [
        "Create a plan for building a web application with user authentication",
        "Search your memory for patterns in approved vs rejected plans", 
        "Based on your planning episodes, what makes a good plan?",
        "Show me examples of plans that were rejected and why"
    ]
    
    for q in test_queries:
        print(f"\nUser: {q}")
        response = planner_agent.invoke({
            "messages": [
                {"role": "user", "content": q}
            ]
        }, config=config) # type: ignore
        print(f"AI: {response['messages'][-1].content}")

    print("\n=== Demo Complete ===")

def run_interactive_planner_episodic():
    """Run an interactive episodic planner session."""
    
    print("=== Interactive Episodic Planner Agent ===")
    print("Store plans and their approval status for few-shot learning")
    print("Using local LLM and Redis storage")
    print("Commands: 'quit' to exit, 'memory' to search episodes, 'create' to generate a plan")
    print("-" * 60)
    
    # Set up the planner system
    planner_agent, store = setup_episodic_planner()
    
    user_id = input("Enter your user ID (or press Enter for 'interactive_user'): ").strip()
    if not user_id:
        user_id = "interactive_user"
    
    config = {"configurable": {"thread_id": f"planner-episodic-{user_id}"}}
    
    while True:
        try:
            user_input = input("\nCommand (create/memory/query/quit): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if user_input.lower() == 'memory':
                # Search planning episodes
                query = input("Search your planning episodes for: ").strip()
                if query:
                    response = planner_agent.invoke({
                        "messages": [
                            {"role": "user", "content": f"Search your memory for planning episodes related to: {query}"}
                        ]
                    }, config=config) # type: ignore
                    print(f"AI: {response['messages'][-1].content}")
                continue
                
            if user_input.lower() == 'create':
                # Generate a new plan
                task = input("Describe the task you need a plan for: ").strip()
                if task:
                    print(f"\nGenerating plan for: {task}")
                    response = planner_agent.invoke({
                        "messages": [
                            {"role": "user", "content": f"Create a detailed plan for: {task}. First search your memory for similar plans and learn from them."}
                        ]
                    }, config=config) # type: ignore
                    print(f"AI: {response['messages'][-1].content}")
                    
                    # Ask for approval status
                    approved = input("\nWas this plan approved? (y/n): ").strip().lower()
                    feedback = ""
                    if approved != 'y':
                        feedback = input("What feedback would you give? ").strip()
                    
                    # Store the plan outcome
                    plan_content = response['messages'][-1].content
                    episode_data = make_planning_episode(plan_content, approved == 'y', feedback)
                    
                    store_response = planner_agent.invoke({
                        "messages": [
                            {"role": "user", "content": f"Store this plan outcome in your memory: {episode_data}"}
                        ]
                    }, config=config) # type: ignore
                    print(f"Plan outcome stored: {store_response['messages'][-1].content}")
                continue
                
            if user_input.lower() == 'query':
                # Query the agent about planning
                query = input("Ask about planning (e.g., 'What makes a good plan?'): ").strip()
                if query:
                    response = planner_agent.invoke({
                        "messages": [
                            {"role": "user", "content": query}
                        ]
                    }, config=config) # type: ignore
                    print(f"AI: {response['messages'][-1].content}")
                continue
                
            if not user_input:
                continue
            print("Unknown command. Use: create, memory, query, or quit")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        run_interactive_planner_episodic()
    else:
        run_planner_episodic_demo() 