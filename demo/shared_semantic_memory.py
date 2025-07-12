"""
Demo: Collaborative Agents with Private and Shared Memory

This demo shows two agents collaborating:
- Research Agent: Finds and stores information in private + shared memory
- Writing Agent: Creates content using shared information + private writing preferences

Each agent maintains:
1. Private memory (their own namespace)
2. Shared memory (common workspace namespace)
"""

from langchain_core.language_models import llms
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import Union, Dict, Any

from config.llm import create_local_embeddings
from config.llm import create_local_llm

# Pydantic schema for shared workspace entries
class MemoryEntry(BaseModel):
    """Schema for structured data in shared workspace memory."""
    task: str = Field(..., description="The task or objective being worked on")
    action: str = Field(..., description="The specific action taken or being requested")
    result: Union[str, Dict[str, Any]] = Field(..., description="The outcome, data, or result of the action")
    
# Set up embeddings model
embeddings = create_local_embeddings()

# Set up LLM
llm = create_local_llm()

# Set up store and checkpointer
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": embeddings,
    }
)

checkpointer = MemorySaver()

# Mock research tool
def research_company(company_name: str) -> str:
    """Mock research function for demo purposes."""
    company_data = {
        "meta": "Meta Platforms Inc. has 67,317 employees as of 2023, founded in 2004, headquarters in Menlo Park, CA. Revenue: $117.9 billion (2022).",
        "apple": "Apple Inc. has 164,000 employees as of 2023, founded in 1976, headquarters in Cupertino, CA. Revenue: $394.3 billion (2022).",
        "google": "Google (Alphabet) has 181,269 employees as of 2023, founded in 1998, headquarters in Mountain View, CA. Revenue: $282.8 billion (2022)."
    }
    return company_data.get(company_name.lower(), f"No data found for {company_name}")

# Create Research Agent with private + shared memory access
research_agent = create_react_agent(
    llm,
    prompt="""You are a Research Agent specialized in finding and organizing information.

You have access to:
1. Your PRIVATE memory (research_private) - for your research methods, sources, and personal notes
2. SHARED memory (shared_workspace) - for information that other agents can use

When you find information:
- Store research methodology and sources in your private memory
- Store the actual findings in shared memory for other agents to access
- Always be clear about what you're storing where

Use the manage_memory tool to save information and the search_memory tool to find existing data.""",
    tools=[
        research_company,
        create_manage_memory_tool(namespace=("research_private"), schema=MemoryEntry),
        create_manage_memory_tool(namespace=("shared_workspace"), schema=MemoryEntry),
        create_search_memory_tool(namespace=("shared_workspace")),
        create_search_memory_tool(namespace=("research_private"))
    ],
    store=store,
    checkpointer=checkpointer,
)

# Create Writing Agent with private + shared memory access
writing_agent = create_react_agent(
    llm,
    prompt="""You are a Writing Agent specialized in creating content and documents.

You have access to:
1. Your PRIVATE memory (writing_private) - for your writing preferences, templates, and style guides
2. SHARED memory (shared_workspace) - for information shared by other agents (like research findings)

When creating content:
- Check shared memory first for any relevant information from other agents
- Store your writing preferences and templates in private memory
- Share final drafts or useful writing resources in shared memory if others might need them

Use the manage_memory tool to save information and the search_memory tool to find existing data.""",
    tools=[
        create_manage_memory_tool(namespace=("writing_private"), schema=MemoryEntry),
        create_search_memory_tool(namespace=("shared_workspace")),
        create_manage_memory_tool(namespace=("writing_private"), schema=MemoryEntry),
        create_search_memory_tool(namespace=("shared_workspace"))
    ],
    store=store,
    checkpointer=checkpointer,
)

def run_collaboration_demo():
    """Run a demo showing agent collaboration through shared memory."""
    print("=== COLLABORATIVE AGENTS DEMO ===")
    print("Research Agent + Writing Agent with Private + Shared Memory")
    print("-" * 60)
    
    # Configurations for different threads
    research_config = {"configurable": {"thread_id": "research-thread"}}  # type: ignore
    writing_config = {"configurable": {"thread_id": "writing-thread"}}    # type: ignore
    
    # Step 1: Research Agent gathers information
    print("\n🔍 STEP 1: Research Agent gathers company information")
    print("=" * 50)
    
    research_response = research_agent.invoke({
        "messages": [
            {"role": "user", "content": "Research Meta and Apple. Store your research methodology in your private memory and the company facts in shared memory so other agents can use them."}
        ]
    }, research_config)  # type: ignore
    
    print("Research Agent Response:")
    print(research_response["messages"][-1].content)
    
    # Step 2: Writing Agent creates content using shared information
    print("\n✍️ STEP 2: Writing Agent creates content using shared research")
    print("=" * 50)
    
    writing_response = writing_agent.invoke({
        "messages": [
            {"role": "user", "content": "Create a brief comparison report between Meta and Apple using any information available in shared memory. Store your writing preferences in private memory."}
        ]
    }, writing_config)  # type: ignore
    
    print("Writing Agent Response:")
    print(writing_response["messages"][-1].content)
    
    # Step 3: Research Agent adds more data to shared memory
    print("\n🔍 STEP 3: Research Agent adds more data to shared workspace")
    print("=" * 50)
    
    research_response2 = research_agent.invoke({
        "messages": [
            {"role": "user", "content": "Also research Google and add it to our shared workspace. Check if there's any existing shared information first."}
        ]
    }, research_config)  # type: ignore
    
    print("Research Agent Response:")
    print(research_response2["messages"][-1].content)
    
    # Step 4: Writing Agent updates content with new shared information
    print("\n✍️ STEP 4: Writing Agent updates content with new shared data")
    print("=" * 50)
    
    writing_response2 = writing_agent.invoke({
        "messages": [
            {"role": "user", "content": "Update the comparison report to include Google. Check shared memory for the latest information."}
        ]
    }, writing_config)  # type: ignore
    
    print("Writing Agent Response:")
    print(writing_response2["messages"][-1].content)
    
    # Step 5: Show memory contents
    print("\n📋 STEP 5: Memory Analysis")
    print("=" * 50)
    
    # Check each memory namespace
    namespaces = [
        ("research_private", "Research Agent Private Memory"),
        ("writing_private", "Writing Agent Private Memory"),
        ("shared_workspace", "Shared Memory (Both Agents)")
    ]
    
    for namespace, description in namespaces:
        memories = store.search((namespace,))
        print(f"\n{description}:")
        if memories:
            for i, memory in enumerate(memories, 1):
                content = getattr(memory, 'value', '')
                if isinstance(content, dict):
                    content = content.get('content', str(content))
                print(f"  {i}. {str(content)[:100]}...")
        else:
            print("  No memories found")
    
    # Step 6: Demonstrate cross-agent memory access
    print("\n🤝 STEP 6: Cross-Agent Memory Access Demonstration")
    print("=" * 50)

    # Writing agent provides feedback via shared memory
    writing_response3 = writing_agent.invoke({
        "messages": [
            {"role": "user", "content": "Leave a note in shared memory about what additional information would be helpful for future reports."}
        ]
    }, writing_config)  # type: ignore
    
    print("\nWriting Agent leaving feedback in shared memory:")
    print(writing_response3["messages"][-1].content)
    # Research agent checks what writing agent might need
    research_response3 = research_agent.invoke({
        "messages": [
            {"role": "user", "content": "Check the shared workspace to see what information is available for other agents. What else should I research?"}
        ]
    }, research_config)  # type: ignore
    
    print("Research Agent checking shared workspace:")
    print(research_response3["messages"][-1].content)
    
    print("\n" + "=" * 60)
    print("🎉 COLLABORATION DEMO COMPLETE!")
    print("✅ Agents successfully collaborated using shared memory")
    print("✅ Each agent maintained private memory for their specific needs")
    print("✅ Shared memory enabled cross-agent information exchange")

if __name__ == "__main__":
    run_collaboration_demo()