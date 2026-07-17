import os
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = "data/"
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
API_ENDPOINT = os.getenv("API_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
CHAT_MODEL = os.getenv("CHAT_MODEL")
IMAGE_MODEL = os.getenv("IMAGE_MODEL")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

# Completion model wrapper for Azure-deployed legacy models
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=API_VERSION,
    azure_endpoint=API_ENDPOINT
)

# Chat model wrapper for Azure-deployed OpenAI endpoints
llm = AzureChatOpenAI(
    azure_deployment=CHAT_MODEL,
    azure_endpoint=API_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=API_VERSION,
    temperature=0,
    # timeout=3
)

embedding_LLM = AzureOpenAIEmbeddings(
    azure_deployment=EMBED_MODEL,
    azure_endpoint=API_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=API_VERSION,
    temperature=0,
    # timeout=3
)