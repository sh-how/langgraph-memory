"""
Simple Semantic Memory Implementation

This is a basic implementation of semantic memory using langmem, demonstrating:
1. Memory storage and retrieval
2. Conversation threading
3. Memory search and management
4. Cross-conversation memory recall

Based on langmem documentation examples.
"""

from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.checkpoint.memory import MemorySaver
from config.llm import create_local_embeddings, create_local_llm

def setup_semantic_memory():
    """Set up the semantic memory system with local LLM and embeddings."""
    
    # Initialize embeddings and LLM using your configuration
    embeddings = create_local_embeddings()
    llm = create_local_llm()
    
    # Create memory store with embeddings
    store = InMemoryStore(
        index={
            "dims": 1536,  # Adjust based on your embedding model
            "embed": embeddings,
        }
    )
    
    # Create checkpointer for conversation state
    checkpointer = MemorySaver()
    
    # Create memory management agent
    memory_agent = create_react_agent(
        llm,
        prompt="""You are a helpful assistant with semantic memory capabilities.

Your role is to:
1. Remember important information from conversations
2. Store user preferences, facts, and context
3. Retrieve relevant memories when needed
4. Update and manage existing memories

Use the memory tools to:
- Store new information using manage_memory
- Search existing memories using search_memory
- Update or delete outdated information

Always be helpful and remember user preferences across conversations.""",
        tools=[
            create_manage_memory_tool(namespace=("memories",)),
            create_search_memory_tool(namespace=("memories",)),
        ],
        store=store,
        checkpointer=checkpointer,
    )
    
    return memory_agent, store

def run_memory_demo():
    """Run a demonstration of semantic memory capabilities."""
    
    print("=== Semantic Memory Demo ===")
    print("Using local LLM and embeddings")
    print("-" * 40)
    
    # Set up the memory system
    memory_agent, store = setup_semantic_memory()
    
    # Demo 1: Store user preferences
    print("\n1. Storing user preferences...")
    config_a = {"configurable": {"thread_id": "user-preferences"}}
    
    response = memory_agent.invoke({
        "messages": [
            {"role": "user", "content": "I prefer dark mode for my applications and I'm a software developer."}
        ]
    }, config=config_a)
    
    print(f"Response: {response['messages'][-1].content}")
    
    # Demo 2: Continue conversation in same thread
    print("\n2. Continuing conversation (same thread)...")
    
    response = memory_agent.invoke({
        "messages": [
            {"role": "user", "content": "What do you remember about my preferences?"}
        ]
    }, config=config_a)
    
    print(f"Response: {response['messages'][-1].content}")
    
    # Demo 3: New conversation thread - test memory recall
    print("\n3. New conversation thread - testing memory recall...")
    config_b = {"configurable": {"thread_id": "new-conversation"}}
    
    response = memory_agent.invoke({
        "messages": [
            {"role": "user", "content": "Hello! Do you remember anything about me?"}
        ]
    }, config=config_b)
    
    print(f"Response: {response['messages'][-1].content}")
    
    # Demo 4: Store more information
    print("\n4. Storing additional information...")
    
    response = memory_agent.invoke({
        "messages": [
            {"role": "user", "content": "I also love coffee and work with Python and JavaScript."}
        ]
    }, config=config_b)
    
    print(f"Response: {response['messages'][-1].content}")
    
    # Demo 5: Query specific information
    print("\n5. Querying specific information...")
    
    response = memory_agent.invoke({
        "messages": [
            {"role": "user", "content": "What programming languages do I work with?"}
        ]
    }, config=config_b)
    
    print(f"Response: {response['messages'][-1].content}")
    
    # Demo 6: Show all stored memories
    print("\n6. Displaying all stored memories:")
    memories = store.search(("memories",))
    
    if memories:
        print(f"Found {len(memories)} memories:")
        for i, memory in enumerate(memories, 1):
            # Extract content from memory
            value = getattr(memory, 'value', {})
            if isinstance(value, dict):
                content = value.get('content', str(value))
            else:
                content = str(value)
            
            # Truncate for display
            if len(content) > 100:
                content = content[:100] + "..."
            
            print(f"  {i}. {content}")
    else:
        print("No memories found.")
    
    print("\n=== Demo Complete ===")

def run_interactive_demo():
    """Run an interactive demo where user can chat with the memory system."""
    
    print("=== Interactive Semantic Memory Demo ===")
    print("Chat with the AI and see how it remembers information across messages.")
    print("Type 'quit' to exit, 'memories' to see stored memories.")
    print("-" * 50)
    
    # Set up the memory system
    memory_agent, store = setup_semantic_memory()
    
    # Use a single thread for the interactive session
    config = {"configurable": {"thread_id": "interactive-session"}}
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'memories':
                # Show stored memories
                memories = store.search(("memories",))
                if memories:
                    print(f"\nStored Memories ({len(memories)} found):")
                    for i, memory in enumerate(memories, 1):
                        value = getattr(memory, 'value', {})
                        if isinstance(value, dict):
                            content = value.get('content', str(value))
                        else:
                            content = str(value)
                        
                        if len(content) > 100:
                            content = content[:100] + "..."
                        
                        print(f"  {i}. {content}")
                else:
                    print("\nNo memories stored yet.")
                continue
            
            if not user_input:
                continue
            
            # Get response from memory agent
            response = memory_agent.invoke({
                "messages": [
                    {"role": "user", "content": user_input}
                ]
            }, config=config)
            
            print(f"AI: {response['messages'][-1].content}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        run_interactive_demo()
    else:
        run_memory_demo() 