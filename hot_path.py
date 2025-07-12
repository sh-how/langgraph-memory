from typing import Dict, Any, Union
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store
from langchain_core.runnables.config import RunnableConfig
from langmem import create_manage_memory_tool
from langgraph_supervisor import create_supervisor
from config.llm import create_local_llm, create_local_embeddings
from dotenv import load_dotenv
import os
import json
import pickle
from pathlib import Path
from datetime import datetime
load_dotenv()

"""
LangGraph Multi-Agent Memory Architecture

This implementation provides separate memory namespaces for each agent to prevent knowledge mixing:

Memory Namespaces:
- math_memories: Dedicated to math_expert agent for calculations and mathematical knowledge
- research_memories: Dedicated to research_expert agent for research findings and company data
- writing_memories: Dedicated to writing_expert agent for content creation and writing tasks
- supervisor_memories: Dedicated to supervisor for coordination and routing decisions

Benefits:
1. Prevents confusion between agent domains
2. Allows specialized knowledge storage
3. Enables better agent autonomy
4. Supervisor can view all agent memories for informed routing decisions
5. External persistence maintains separation across sessions
"""

# Initialize LLM and embeddings
llm = create_local_llm()
embeddings = create_local_embeddings()

# ============================================================================
# EXTERNAL STORAGE UTILITIES
# ============================================================================

class MemoryPersistence:
    """Utility class to save/load memories to external storage."""
    
    def __init__(self, storage_dir="./memory_storage"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.memories_file = self.storage_dir / "memories.json"
        self.backup_dir = self.storage_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
    def save_memories(self, store):
        """Save current memories from all namespaces to external file."""
        try:
            # Define all memory namespaces
            namespaces = [
                ("math_memories",),
                ("research_memories",),
                ("writing_memories",),
                ("supervisor_memories",)
            ]
            
            all_memories = []
            
            # Collect memories from all namespaces
            for namespace in namespaces:
                try:
                    memories = store.search(namespace)
                    namespace_name = namespace[0]
                    print(f"Found {len(memories)} memories in {namespace_name}")
                    
                    for memory in memories:
                        memory_data = {
                            "key": getattr(memory, 'key', None),
                            "value": getattr(memory, 'value', {}),
                            "namespace": namespace,
                            "namespace_name": namespace_name,
                            "created_at": getattr(memory, 'created_at', None),
                            "updated_at": getattr(memory, 'updated_at', None),
                            "saved_timestamp": datetime.now().isoformat()
                        }
                        all_memories.append(memory_data)
                except Exception as ns_error:
                    print(f"⚠ Error accessing {namespace}: {ns_error}")
            
            # Save to JSON file
            with open(self.memories_file, 'w') as f:
                json.dump(all_memories, f, indent=2, default=str)
            
            # Create backup
            backup_file = self.backup_dir / f"memories_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(backup_file, 'w') as f:
                json.dump(all_memories, f, indent=2, default=str)
            
            print(f"✓ Saved {len(all_memories)} total memories to {self.memories_file}")
            print(f"✓ Backup created: {backup_file}")
            
            # Show breakdown by namespace
            namespace_counts = {}
            for memory in all_memories:
                ns_name = memory.get('namespace_name', 'unknown')
                namespace_counts[ns_name] = namespace_counts.get(ns_name, 0) + 1
            
            print("Memory breakdown by agent:")
            for ns_name, count in namespace_counts.items():
                print(f"  - {ns_name}: {count} memories")
            
        except Exception as e:
            print(f"⚠ Failed to save memories: {e}")
    
    def load_memories(self):
        """Load memories from external file."""
        try:
            if self.memories_file.exists():
                with open(self.memories_file, 'r') as f:
                    memories = json.load(f)
                print(f"✓ Loaded {len(memories)} memories from {self.memories_file}")
                return memories
            else:
                print("No existing memory file found.")
                return []
        except Exception as e:
            print(f"⚠ Failed to load memories: {e}")
            return []
    
    def export_to_database(self, memories, db_type="sqlite"):
        """Export memories to a database (SQLite example)."""
        try:
            if db_type == "sqlite":
                import sqlite3
                
                db_path = self.storage_dir / "memories.sqlite"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Create table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        key TEXT,
                        content TEXT,
                        namespace TEXT,
                        created_at TEXT,
                        updated_at TEXT,
                        saved_timestamp TEXT
                    )
                ''')
                
                # Insert memories
                for memory in memories:
                    cursor.execute('''
                        INSERT INTO memories (key, content, namespace, created_at, updated_at, saved_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        memory.get('key'),
                        json.dumps(memory.get('value')),
                        json.dumps(memory.get('namespace')),
                        memory.get('created_at'),
                        memory.get('updated_at'),
                        memory.get('saved_timestamp')
                    ))
                
                conn.commit()
                conn.close()
                
                print(f"✓ Exported {len(memories)} memories to SQLite database: {db_path}")
                
        except Exception as e:
            print(f"⚠ Failed to export to database: {e}")

# Initialize memory persistence
memory_persistence = MemoryPersistence()

def create_memory_prompt(state, agent_role: str = "assistant", memory_namespace: tuple = ("memories",)):
    """Create a memory-aware prompt for agents with dedicated memory namespace."""
    store = get_store()
    memories = store.search(
        memory_namespace,
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful {agent_role}.

## Your Personal Memories
<memories>
{memories}
</memories>

Always use your memory tool to save important information that might be useful for future conversations.
Your memories are private and separate from other agents.
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


# ============================================================================
# STORAGE CONFIGURATION
# ============================================================================

# Use InMemoryStore with external persistence utilities
store = InMemoryStore(
    index={
        "embed": embeddings
    }
)

# Load any existing memories from external storage
print("Loading existing memories from external storage...")
existing_memories = memory_persistence.load_memories()
if existing_memories:
    print(f"Found {len(existing_memories)} existing memories")
    # Note: You would need to restore these to the store if needed
    # This is a simplified example showing the concept

checkpointer = MemorySaver()

# Math tools
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def divide(a: float, b: float) -> Union[float, str]:
    """Divide two numbers."""
    if b == 0:
        return "Error: Division by zero"
    return a / b

# Research tools
def web_search(query: str) -> str:
    """Search the web for information."""
    # Mock web search - in real implementation, use actual web search
    return f"Mock search results for: {query}\n" + \
           "Here are some relevant findings based on current knowledge..."

def company_info(company: str) -> str:
    """Get company information."""
    # Mock company data
    companies = {
        "meta": "Meta (formerly Facebook): 67,317 employees, headquarters in Menlo Park, CA",
        "apple": "Apple: 164,000 employees, headquarters in Cupertino, CA",
        "amazon": "Amazon: 1,551,000 employees, headquarters in Seattle, WA",
        "netflix": "Netflix: 14,000 employees, headquarters in Los Gatos, CA",
        "google": "Google (Alphabet): 181,269 employees, headquarters in Mountain View, CA"
    }
    return companies.get(company.lower(), f"No information found for {company}")

# Writing tools
def create_outline(topic: str) -> str:
    """Create an outline for a given topic."""
    return f"Outline for {topic}:\n1. Introduction\n2. Main Points\n3. Conclusion"

def grammar_check(text: str) -> str:
    """Check grammar of the given text."""
    return f"Grammar check complete for: {text[:50]}... (Text appears to be well-structured)"

# Create memory-enabled specialized agents with dedicated namespaces
def create_math_prompt(state):
    return create_memory_prompt(
        state, 
        "math expert specializing in calculations and problem-solving",
        memory_namespace=("math_memories",)
    )

def create_research_prompt(state):
    return create_memory_prompt(
        state, 
        "research expert specializing in information gathering and analysis",
        memory_namespace=("research_memories",)
    )

def create_writing_prompt(state):
    return create_memory_prompt(
        state, 
        "writing expert specializing in content creation and editing",
        memory_namespace=("writing_memories",)
    )

def create_supervisor_prompt(state):
    return create_memory_prompt(
        state, 
        "supervisor specializing in coordination and routing decisions",
        memory_namespace=("supervisor_memories",)
    )

# Create specialized agents with dedicated memory namespaces
math_agent = create_react_agent(
    llm,
    prompt=create_math_prompt,
    tools=[
        add, multiply, divide,
        create_manage_memory_tool(namespace=("math_memories",))
    ],
    name="math_expert",
    store=store,
    checkpointer=checkpointer,
)

research_agent = create_react_agent(
    llm,
    prompt=create_research_prompt,
    tools=[
        web_search, company_info,
        create_manage_memory_tool(namespace=("research_memories",))
    ],
    name="research_expert",
    store=store,
    checkpointer=checkpointer,
)

writing_agent = create_react_agent(
    llm,
    prompt=create_writing_prompt,
    tools=[
        create_outline, grammar_check,
        create_manage_memory_tool(namespace=("writing_memories",))
    ],
    name="writing_expert",
    store=store,
    checkpointer=checkpointer,
)

# Create supervisor workflow with its own memory namespace
supervisor_workflow = create_supervisor(
    [math_agent, research_agent, writing_agent],
    model=llm,
    prompt=create_supervisor_prompt,
    # Add memory tool to supervisor with its own namespace
    tools=[create_manage_memory_tool(namespace=("supervisor_memories",))],
)

# Compile the supervisor with memory capabilities
supervisor_app = supervisor_workflow.compile(
    checkpointer=checkpointer,
    store=store
)

def run_supervisor_demo():
    """Run a demonstration of the supervisor system with memory."""
    
    # Configuration for conversation threading
    config: RunnableConfig = {"configurable": {"thread_id": "supervisor-demo"}}
    
    print("=== LangGraph Supervisor with External Memory Storage Demo ===\n")
    print(f"Storage type: {type(store).__name__} + External Persistence")
    print(f"Storage location: {memory_persistence.storage_dir}")
    print("-" * 60)
    
    # Test 1: Math problem
    print("1. Testing Math Agent through Supervisor:")
    response = supervisor_app.invoke({
        "messages": [
            {"role": "user", "content": "What is 15 * 23 + 47?"}
        ]
    }, config=config)
    print(f"Response: {response['messages'][-1].content}\n")
    
    # Test 2: Research query
    print("2. Testing Research Agent through Supervisor:")
    response = supervisor_app.invoke({
        "messages": [
            {"role": "user", "content": "Can you research information about Meta's employee count?"}
        ]
    }, config=config)
    print(f"Response: {response['messages'][-1].content}\n")
    
    # Test 3: Writing task
    print("3. Testing Writing Agent through Supervisor:")
    response = supervisor_app.invoke({
        "messages": [
            {"role": "user", "content": "Create an outline for a presentation about AI in business."}
        ]
    }, config=config)
    print(f"Response: {response['messages'][-1].content}\n")
    
    # Save memories to external storage
    print("4. Saving Memories to External Storage:")
    memory_persistence.save_memories(store)
    
    # Test 4: Memory recall across different conversation
    print("\n5. Testing Memory Recall in New Conversation:")
    new_config: RunnableConfig = {"configurable": {"thread_id": "supervisor-recall"}}
    response = supervisor_app.invoke({
        "messages": [
            {"role": "user", "content": "Do you remember any calculations I asked about? And what was my presentation topic?"}
        ]
    }, config=new_config)
    print(f"Response: {response['messages'][-1].content}\n")
    
    # Test 5: Multi-agent coordination
    print("6. Testing Multi-Agent Coordination:")
    response = supervisor_app.invoke({
        "messages": [
            {"role": "user", "content": "Calculate the total employees of Meta and Google, then write a brief summary about these companies."}
        ]
    }, config=config)
    print(f"Response: {response['messages'][-1].content}\n")
    
    # Final save and export
    print("7. Final Memory Export:")
    memory_persistence.save_memories(store)
    
    # Export to database (optional)
    memories = memory_persistence.load_memories()
    if memories:
        memory_persistence.export_to_database(memories, "sqlite")
    
    # Show stored memories from all namespaces
    print("\n8. Checking Stored Memories by Agent:")
    try:
        namespaces = [
            ("math_memories", "Math Agent"),
            ("research_memories", "Research Agent"),
            ("writing_memories", "Writing Agent"),
            ("supervisor_memories", "Supervisor")
        ]
        
        total_memories = 0
        for namespace_tuple, agent_name in namespaces:
            try:
                memories = store.search((namespace_tuple,))
                if memories:
                    print(f"\n{agent_name} Memories ({len(memories)} found):")
                    for i, memory in enumerate(memories, 1):
                        try:
                            # Extract content from memory based on LangGraph memory structure
                            value = getattr(memory, 'value', '')
                            
                            # Handle different memory value formats
                            if isinstance(value, str):
                                content = value
                            elif isinstance(value, dict):
                                content = (value.get('content') or 
                                         value.get('text') or 
                                         value.get('message') or 
                                         str(value))
                            else:
                                content = str(value)
                            
                            # Truncate long content for display
                            if len(content) > 100:
                                content = content[:100] + "..."
                            
                            print(f"  {i}. {content}")
                            
                        except Exception as mem_error:
                            print(f"  {i}. Error processing memory: {mem_error}")
                    
                    total_memories += len(memories)
                else:
                    print(f"\n{agent_name} Memories: No memories found")
                    
            except Exception as ns_error:
                print(f"\n{agent_name} Memories: Error accessing namespace - {ns_error}")
        
        if total_memories == 0:
            print("\nNo memories found in any agent namespace.")
            
        # Also show external storage
        external_memories = memory_persistence.load_memories()
        if external_memories:
            print(f"\nFound {len(external_memories)} memories in external storage:")
            print(f"Storage location: {memory_persistence.memories_file}")
            
            # Show some external memory examples
            print("External memory examples:")
            for i, ext_mem in enumerate(external_memories, 1):  # Show all memories
                try:
                    value = ext_mem.get('value', {})
                    if isinstance(value, str):
                        content = value
                    elif isinstance(value, dict):
                        content = (value.get('content') or 
                                 value.get('text') or 
                                 value.get('message') or 
                                 str(value))
                    else:
                        content = str(value)
                    
                    # Truncate long content for display
                    if len(content) > 100:
                        content = content[:100] + "..."
                        
                    print(f"  External {i}. {content}")
                except Exception as ext_error:
                    print(f"  External {i}. Error: {ext_error}")
            
    except Exception as e:
        print(f"Error retrieving memories: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    run_supervisor_demo()