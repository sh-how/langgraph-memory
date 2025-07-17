"""
Episodic Planner Agent with Episodic Memory

This script demonstrates a planning agent that stores each planning event (plan proposal, approval, feedback, etc.) as an episode in episodic memory.
Uses your local LLM and embedding setup.

Features:
- Each planning event is stored as an episode (timestamp + content)
- Planner can recall/search past planning episodes
- Automated and interactive demo modes
"""

from datetime import datetime
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.checkpoint.memory import MemorySaver
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

def setup_planner_with_episodic_memory():
    embeddings = create_local_embeddings()
    llm = create_local_llm()
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": embeddings,
        }
    )
    checkpointer = MemorySaver()
    planner_agent = create_react_agent(
        llm,
        prompt="""You are a Planning Agent with episodic memory for plan generation and approval tracking.

Your episodic memory stores:
- Plans you generate
- Whether each plan was approved or rejected
- Feedback received on rejected plans

This memory acts as few-shot examples to help you create better plans. When generating new plans:
1. Search your memory for similar past plans
2. Learn from approved plans (what worked)
3. Learn from rejected plans (what to avoid)
4. Use this knowledge to create better plans

Use the memory tools to:
- Store new plans and their outcomes in a structured format
- Search for relevant past plans when creating new ones
- Recall patterns from successful vs failed plans

When storing plan outcomes, use this format:
"Plan: [plan content] | Approved: [true/false] | Feedback: [feedback if any]"

Focus on learning from your planning history to improve future plan quality.""",
        tools=[
            create_manage_memory_tool(namespace=("planner_episodes",)),
            create_search_memory_tool(namespace=("planner_episodes",)),
        ],
        store=store,
        checkpointer=checkpointer,
    )
    return planner_agent, store

def run_planner_episodic_demo():
    print("=== Episodic Planner Agent Demo ===")
    print("Stores plans and their approval status for few-shot learning")
    print("-" * 50)
    agent, store = setup_planner_with_episodic_memory()
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
        
        episode = make_planning_episode(plan, approved, feedback)
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": f"Store this plan outcome in your memory: Plan: {plan} | Approved: {approved} | Feedback: {feedback}"}
            ]
        }, config=config)
        print(f"AI: {response['messages'][-1].content}")

    # Test few-shot learning by asking for a new plan
    print("\n" + "="*50)
    print("TESTING FEW-SHOT LEARNING")
    print("="*50)
    
    test_queries = [
        "Create a plan for building a web application with user authentication",
        "What patterns do you notice in your approved vs rejected plans?",
        "Based on your memory, what makes a good plan?",
        "Show me examples of plans that were rejected and why"
    ]
    
    for q in test_queries:
        print(f"\nUser: {q}")
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": q}
            ]
        }, config=config)
        print(f"AI: {response['messages'][-1].content}")

    # Show all stored planning episodes
    print("\n" + "="*50)
    print("ALL STORED PLAN OUTCOMES")
    print("="*50)
    memories = store.search(("planner_episodes",))
    if memories:
        for i, memory in enumerate(memories, 1):
            value = getattr(memory, 'value', {})
            if isinstance(value, dict):
                content = value.get('content', str(value))
            else:
                content = str(value)
            
            # Parse the stored content to extract plan, approval, and feedback
            if '|' in content:
                parts = content.split('|')
                plan_part = parts[0].replace('Plan:', '').strip()
                approved_part = parts[1].replace('Approved:', '').strip()
                feedback_part = parts[2].replace('Feedback:', '').strip() if len(parts) > 2 else ""
                
                status = "APPROVED" if approved_part.lower() == 'true' else "REJECTED"
                print(f"\n{i}. {status}")
                print(f"   Plan: {plan_part[:100]}{'...' if len(plan_part) > 100 else ''}")
                if feedback_part and feedback_part.lower() != 'none':
                    print(f"   Feedback: {feedback_part}")
            else:
                # Fallback for other formats
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"  {i}. {content}")
    else:
        print("No planning episodes found.")
    print("\n=== Demo Complete ===")

def run_interactive_planner_episodic():
    print("=== Interactive Episodic Planner Agent Demo ===")
    print("Store plans and their approval status for few-shot learning")
    print("Commands: 'quit' to exit, 'episodes' to see stored plans, 'create' to generate a plan")
    print("-" * 60)
    agent, store = setup_planner_with_episodic_memory()
    config = {"configurable": {"thread_id": "planner-episodic-interactive"}}
    while True:
        try:
            user_input = input("\nCommand (create/query/quit/episodes): ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if user_input.lower() == 'episodes':
                memories = store.search(("planner_episodes",))
                if memories:
                    print(f"\nStored Plan Outcomes ({len(memories)} found):")
                    for i, memory in enumerate(memories, 1):
                        value = getattr(memory, 'value', {})
                        if isinstance(value, dict):
                            content = value.get('content', str(value))
                        else:
                            content = str(value)
                        
                        # Parse the stored content to extract plan, approval, and feedback
                        if '|' in content:
                            parts = content.split('|')
                            plan_part = parts[0].replace('Plan:', '').strip()
                            approved_part = parts[1].replace('Approved:', '').strip()
                            feedback_part = parts[2].replace('Feedback:', '').strip() if len(parts) > 2 else ""
                            
                            status = "APPROVED" if approved_part.lower() == 'true' else "REJECTED"
                            print(f"\n{i}. {status}")
                            print(f"   Plan: {plan_part[:100]}{'...' if len(plan_part) > 100 else ''}")
                            if feedback_part and feedback_part.lower() != 'none':
                                print(f"   Feedback: {feedback_part}")
                        else:
                            # Fallback for other formats
                            if len(content) > 100:
                                content = content[:100] + "..."
                            print(f"  {i}. {content}")
                else:
                    print("\nNo planning episodes stored yet.")
                continue
            if user_input.lower() == 'create':
                # Generate a new plan
                task = input("Describe the task you need a plan for: ").strip()
                if task:
                    print(f"\nGenerating plan for: {task}")
                    response = agent.invoke({
                        "messages": [
                            {"role": "user", "content": f"Create a plan for: {task}. Use your memory of past plans to make this a good plan."}
                        ]
                    }, config=config)
                    print(f"AI: {response['messages'][-1].content}")
                    
                    # Ask for approval status
                    approved = input("\nWas this plan approved? (y/n): ").strip().lower()
                    feedback = ""
                    if approved != 'y':
                        feedback = input("What feedback would you give? ").strip()
                    
                    # Store the plan outcome
                    plan_content = response['messages'][-1].content
                    store_response = agent.invoke({
                        "messages": [
                            {"role": "user", "content": f"Store this plan outcome in your memory: Plan: {plan_content} | Approved: {approved == 'y'} | Feedback: {feedback}"}
                        ]
                    }, config=config)
                    print(f"Plan outcome stored: {store_response['messages'][-1].content}")
                continue
            if user_input.lower() == 'query':
                # Query the agent about planning
                query = input("Ask about planning (e.g., 'What makes a good plan?'): ").strip()
                if query:
                    response = agent.invoke({
                        "messages": [
                            {"role": "user", "content": query}
                        ]
                    }, config=config)
                    print(f"AI: {response['messages'][-1].content}")
                continue
            if not user_input:
                continue
            print("Unknown command. Use: create, query, episodes, or quit")
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