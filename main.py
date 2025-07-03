from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store 
from langmem import (
    # Lets agent create, update, and delete memories 
    create_manage_memory_tool,
)
from config.llm import create_local_llm
from config.llm import create_local_embeddings

llm = create_local_llm()
embeddings = create_local_embeddings()

def prompt(state):
    """Prepare the messages for the LLM."""
    # Get store from configured contextvar; 
    store = get_store() # Same as that provided to `create_react_agent`
    memories = store.search(
        # Search within the same namespace as the one
        # we've configured for the agent
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = f"""You are a helpful assistant.

## Memories
<memories>
{memories}
</memories>
"""
    return [{"role": "system", "content": system_msg}, *state["messages"]]


store = InMemoryStore(
    index={
        "embed": embeddings
    }
) 
checkpointer = MemorySaver() # Checkpoint graph state 

agent = create_react_agent( 
    llm,
    prompt=prompt,
    tools=[ # Add memory tools 
        # The agent can call "manage_memory" to
        # create, update, and delete memories by ID
        # Namespaces add scope to memories. To
        # scope memories per-user, do ("memories", "{user_id}"): 
        create_manage_memory_tool(namespace=("memories",)),
    ],
    # Our memories will be stored in this provided BaseStore instance
    store=store,
    # And the graph "state" will be checkpointed after each node
    # completes executing for tracking the chat history and durable execution
    checkpointer=checkpointer, 
)

config = {"configurable": {"thread_id": "thread-a"}}

# Use the agent. The agent hasn't saved any memories,
# so it doesn't know about us
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "Know which display mode I prefer?"}
        ]
    },
    config=config,
)
print(response["messages"][-1].content)
# Output: "I don't seem to have any stored memories about your display mode preferences..."

agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "dark. Remember that."}
        ]
    },
    # We will continue the conversation (thread-a) by using the config with
    # the same thread_id
    config=config,
)

# New thread = new conversation!
new_config = {"configurable": {"thread_id": "thread-b"}}
# The agent will only be able to recall
# whatever it explicitly saved using the manage_memories tool
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Hey there. Do you remember me? What are my preferences?"}]},
    config=new_config,
)
print(response["messages"][-1].content)
# Output: "Based on my memory search, I can see that you've previously indicated a preference for dark display mode..."