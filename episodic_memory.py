"""
Simple Episodic Memory Implementation

This script demonstrates episodic memory using langmem, with your own LLM and embedding setup.
Each memory is an episode (event) with a timestamp and content.

Features:
- Stores each user interaction as a distinct episode
- Allows querying and retrieval of past episodes
- Automated and interactive demo modes

Based on langmem documentation patterns.
"""

from datetime import datetime
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.checkpoint.memory import MemorySaver
from config.llm import create_local_embeddings, create_local_llm

# Helper to format an episode
def make_episode(content: str) -> dict:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": content
    }

def setup_episodic_memory():
    """Set up the episodic memory system with local LLM and embeddings."""
    embeddings = create_local_embeddings()
    llm = create_local_llm()
    store = InMemoryStore(
        index={
            "dims": 1536,
            "embed": embeddings,
        }
    )
    checkpointer = MemorySaver()
    episodic_agent = create_react_agent(
        llm,
        prompt="""You are an assistant with episodic memory. Each user interaction is stored as a separate episode, with a timestamp and event description. Use the memory tools to store, search, and recall episodes. When asked about the past, retrieve and summarize relevant episodes.""",
        tools=[
            create_manage_memory_tool(namespace=("episodes",)),
            create_search_memory_tool(namespace=("episodes",)),
        ],
        store=store,
        checkpointer=checkpointer,
    )
    return episodic_agent, store

def run_episodic_demo():
    print("=== Episodic Memory Demo ===")
    print("Using local LLM and embeddings")
    print("-" * 40)
    agent, store = setup_episodic_memory()
    config = {"configurable": {"thread_id": "episodic-demo"}}

    # Demo: Store a sequence of episodes
    episodes = [
        "I had coffee this morning.",
        "I went for a run in the park.",
        "I attended a meeting about project X.",
        "I had lunch with Sarah.",
        "I finished reading a book on AI."
    ]
    for i, event in enumerate(episodes, 1):
        print(f"\nEpisode {i}: {event}")
        episode = make_episode(event)
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": f"Remember this event: {episode['event']} (at {episode['timestamp']})"}
            ]
        }, config=config)
        print(f"AI: {response['messages'][-1].content}")

    # Query about the past
    print("\nQuerying episodic memory...")
    queries = [
        "What did I do this morning?",
        "Who did I have lunch with?",
        "What meetings did I attend?",
        "Summarize my recent activities."
    ]
    for q in queries:
        print(f"\nUser: {q}")
        response = agent.invoke({
            "messages": [
                {"role": "user", "content": q}
            ]
        }, config=config)
        print(f"AI: {response['messages'][-1].content}")

    # Show all stored episodes
    print("\nAll stored episodes:")
    memories = store.search(("episodes",))
    if memories:
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
        print("No episodes found.")
    print("\n=== Demo Complete ===")

def run_interactive_episodic():
    print("=== Interactive Episodic Memory Demo ===")
    print("Each message is stored as an episode with a timestamp.")
    print("Type 'quit' to exit, 'episodes' to see stored episodes.")
    print("-" * 50)
    agent, store = setup_episodic_memory()
    config = {"configurable": {"thread_id": "episodic-interactive"}}
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if user_input.lower() == 'episodes':
                memories = store.search(("episodes",))
                if memories:
                    print(f"\nStored Episodes ({len(memories)} found):")
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
                    print("\nNo episodes stored yet.")
                continue
            if not user_input:
                continue
            # Store the episode
            episode = make_episode(user_input)
            response = agent.invoke({
                "messages": [
                    {"role": "user", "content": f"Remember this event: {episode['event']} (at {episode['timestamp']})"}
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
        run_interactive_episodic()
    else:
        run_episodic_demo() 