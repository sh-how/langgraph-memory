from langchain_core.language_models import llms
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import create_react_agent
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.checkpoint.memory import MemorySaver

from langgraph.func import entrypoint

from config.llm import create_local_embeddings
from config.llm import create_local_llm

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

# Create the memory extraction agent
manager = create_react_agent(
    llm,
    prompt=f"""You are a memory manager. Extract and manage all important knowledge, rules, and events using the provided tools.


    Use the manage_memory tool to update and contextualize existing memories, create new ones, or delete old ones that are no longer valid.
    You can also expand your search of existing memories to augment using the search tool.""",
    tools=[
        # Agent can create/update/delete memories
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
    store=store,
    checkpointer=checkpointer,
    )

config = {"configurable": {"thread_id": "thread-a"}}  # type: ignore

response = manager.invoke(
    {
        "messages": [
            {"role": "user", "content": "Know which display mode I prefer?"}
        ]
    },
    config=config,  # type: ignore
)

manager.invoke(
    {
        "messages": [
            {"role": "user", "content": "light."}
        ]
    },
    # We will continue the conversation (thread-a) by using the config with
    # the same thread_id
    config=config,  # type: ignore
)

new_config = {"configurable": {"thread_id": "thread-b"}}

response = manager.invoke(
    {"messages": [{"role": "user", "content": "Hey there. Do you remember me? What are my preferences?"}]},
    config=new_config,  # type: ignore
)
print(response["messages"][-1].content)

print(store.search(("memories",)))