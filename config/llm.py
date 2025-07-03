import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()

def create_llm(
    deployment_name: str = "gpt-4o",
    temperature: float = 0,
    max_tokens: int = 4096,
    api_version: str = "2024-08-01-preview",
    azure_endpoint: str = "",
    api_key: str = ""
) -> AzureChatOpenAI:
    resolved_api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    return AzureChatOpenAI(
        azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key        = SecretStr(resolved_api_key) if resolved_api_key else None,
        api_version    = api_version,
        azure_deployment = deployment_name,
        temperature    = temperature,
        max_tokens     = max_tokens
    )

def create_local_llm(
    model_name: str = "qwen2.5:32b",
    base_url: str = "http://localhost:11434",
) -> ChatOllama:
    """
    Create a ChatOllama instance with local settings from environment variables.
    
    Args:
        model_name: Override the model name from environment
        base_url: Override the base URL from environment
    
    Returns:
        ChatOllama instance configured with local settings
    """
    # Get values from environment variables with fallbacks
    model = model_name
    url = os.getenv("OLLAMA_BASE_URL", base_url) 
    api_key = os.getenv("API_KEY")
    
    # Build client_kwargs only if API key is provided
    client_kwargs = {}
    if api_key:
        client_kwargs["headers"] = {"X-Api-Key": api_key}
    
    return ChatOllama(
        model=model,
        base_url=url,
        client_kwargs=client_kwargs if client_kwargs else None
    )


def create_local_embeddings(
    model_name: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> OllamaEmbeddings:
    url = os.getenv("OLLAMA_BASE_URL", base_url) 
    api_key = os.getenv("API_KEY")
    
    # Build client_kwargs only if API key is provided
    client_kwargs = {}
    if api_key:
        client_kwargs["headers"] = {"X-Api-Key": api_key}
        
    return OllamaEmbeddings(
        model=model_name,
        base_url=url,
        client_kwargs=client_kwargs if client_kwargs else None
    )

