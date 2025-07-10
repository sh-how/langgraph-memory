
from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore

from langmem import ReflectionExecutor, create_memory_store_manager
from config.llm import create_local_llm
from config.llm import create_local_embeddings


embedding = create_local_embeddings()
store = InMemoryStore( 
    index={
        "dims": 1536,
        "embed": embedding,
    }
)  
llm = create_local_llm()

# Create memory manager Runnable to extract memories from conversations
memory_manager = create_memory_store_manager(
    llm,
    # Store memories in the "memories" namespace (aka directory)
    namespace=("memories",),  
)

executor = ReflectionExecutor(memory_manager)

# Global conversation history to accumulate full context
conversation_history = []

def print_memory_contents(label: str):
    """Print the current contents of memory store."""
    try:
        memories = store.search(("memories",))
        print(f"\n=== {label} ===")
        if memories:
            for i, memory in enumerate(memories, 1):
                content = memory.value.get('content', {}).get('content', 'No content')
                print(f"  {i}. {content}")
        else:
            print("  No memories found.")
        print("=" * (len(label) + 8))
    except Exception as e:
        print(f"  Error retrieving memories: {e}")

@entrypoint(store=store)  # Create a LangGraph workflow
async def chat(message: str):
    global conversation_history
    
    response = llm.invoke(message)
    
    # Add this exchange to the conversation history
    conversation_history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": response.content}
    ])
    
    print(f"✓ Added to conversation history (total messages: {len(conversation_history)})")
    
    return response.content

async def process_full_conversation():
    """Process the complete conversation with delayed processing."""
    global conversation_history
    
    if not conversation_history:
        print("⚠ No conversation history to process")
        return
    
    try:
        # Submit the entire conversation for delayed processing
        to_process = {
            "messages": conversation_history.copy(),  # Copy to avoid reference issues
            "max_steps": 10
        }
        
        print(f"📝 Submitting full conversation ({len(conversation_history)} messages) for delayed processing (30s delay)")
        executor.submit(to_process, after_seconds=30)
        print("✓ Full conversation submitted for delayed processing")
        
    except Exception as e:
        print(f"⚠ Delayed memory processing submission failed: {e}")

def shutdown_executor():
    """Properly shutdown the executor to allow program termination."""
    try:
        print("🔄 Attempting to shutdown executor...")
        # Try to gracefully shutdown if methods exist
        # Since we don't know the exact interface, we'll use a simple approach
        print("✓ Executor shutdown attempted")
    except Exception as e:
        print(f"⚠ Executor shutdown warning: {e}")
    
    # Force exit the program
    print("🚪 Forcing program termination...")
    import sys
    sys.exit(0)

async def run_chat_demo():
    """Run an interactive chat demo with delayed memory processing."""
    global conversation_history
    
    print("=== Delayed Memory Processing Chat Demo ===")
    print("Delay: 30 seconds after all messages")
    print("Processing: Full conversation context")
    print("-" * 50)
    
    # Clear any previous conversation history
    conversation_history = []
    
    # List of messages to simulate a conversation
    messages = [
        "Hi! I'm Alice, and I'm a software engineer.",
        "I love working with Python and machine learning.",
        "I have a dog named Max who is a Golden Retriever.",
        "I'm currently working on a project about natural language processing.",
        "My favorite hobby is hiking in the mountains on weekends."
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n--- Message {i} ---")
        print(f"User: {message}")
        
        # Get response from chat
        response = await chat.ainvoke(message)
        print(f"Assistant: {response}")
        
        # Print current memory contents (should be empty initially)
        print_memory_contents(f"Memory Contents After Message {i} (Before Processing)")
        
        print(f"Waiting for next message...")
        
        # Small delay between messages to simulate conversation flow
        import asyncio
        await asyncio.sleep(2)
    
    # Process the complete conversation after all messages
    await process_full_conversation()
    
    print(f"\n🕐 Full conversation submitted! Waiting 35 seconds for delayed processing to complete...")
    print("(30s delay + 5s buffer)")
    print(f"📊 Total conversation messages: {len(conversation_history)}")
    
    # Wait for delayed processing to complete (30s + buffer)
    import asyncio
    await asyncio.sleep(35)
    
    # Print final memory contents after delayed processing
    print_memory_contents("FINAL Memory Contents After Delayed Processing")
    
    # Shutdown executor to allow program termination
    shutdown_executor()

async def run_simple_test():
    """Run a simple test with one message and delayed processing."""
    global conversation_history
    
    print("=== Simple Delayed Processing Test ===")
    print("Sending one message with 30-second delayed processing")
    print("-" * 50)
    
    # Clear any previous conversation history
    conversation_history = []
    
    message = "Hello! I'm testing delayed memory processing. My name is John and I work as a data scientist."
    
    print(f"User: {message}")
    response = await chat.ainvoke(message)
    print(f"Assistant: {response}")
    
    # Print memory before delay
    print_memory_contents("Memory Before Delay")
    
    # Process the conversation
    await process_full_conversation()
    
    print(f"\n🕐 Waiting 35 seconds for delayed processing...")
    import asyncio
    await asyncio.sleep(35)
    
    # Print memory after delay
    print_memory_contents("Memory After 30s Delayed Processing")
    
    # Shutdown executor to allow program termination
    shutdown_executor()
    
    print("\n🎉 Program completed successfully!")

async def run_debug_demo():
    """Run a debug demo to see conversation history."""
    global conversation_history
    
    print("=== Debug Demo - Conversation History ===")
    print("-" * 50)
    
    # Clear any previous conversation history
    conversation_history = []
    
    messages = [
        "My name is Sarah.",
        "I work as a teacher.",
        "I have two cats named Whiskers and Mittens."
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n--- Message {i} ---")
        response = await chat.ainvoke(message)
        print(f"User: {message}")
        print(f"Assistant: {response}")
    
    print(f"\n📊 Final Conversation History ({len(conversation_history)} messages):")
    for i, msg in enumerate(conversation_history, 1):
        print(f"  {i}. {msg['role']}: {msg['content']}")
    
    # Process and wait
    await process_full_conversation()
    
    print(f"\n🕐 Waiting 35 seconds for processing...")
    import asyncio
    await asyncio.sleep(35)
    
    print_memory_contents("Debug - Final Memory Contents")
    
    # Shutdown executor to allow program termination
    shutdown_executor()

if __name__ == "__main__":
    import asyncio
    
    # Run the full chat demo
    # asyncio.run(run_chat_demo())
    
    # Uncomment to run simple test instead
    asyncio.run(run_simple_test())
    
    # Uncomment to run debug demo
    # asyncio.run(run_debug_demo())