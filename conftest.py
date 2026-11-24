"""Test settings: placeholder values so that the Azure clients built at import time
(config/azure_config.py, agents/hybrid_rag_agent.py) can be created without real
credentials. Tests never call Azure; they patch the clients. Real values from a .env file
are not used by the tests, because load_dotenv() never overrides a variable already set."""

import os

for _name, _value in {
    "AZURE_API_KEY": "test-placeholder",
    "API_ENDPOINT": "https://placeholder.invalid",
    "API_VERSION": "2024-06-01",
    "AZURE_SEARCH_ENDPOINT": "https://placeholder.invalid",
    "AZURE_SEARCH_INDEX": "test-index",
    "AZURE_SEARCH_KEY": "test-placeholder",
}.items():
    os.environ.setdefault(_name, _value)
