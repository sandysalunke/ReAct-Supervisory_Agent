import os
from threading import Lock
from typing import Any

from azure.cosmos import CosmosClient
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient


_client: Any = None
_container: Any = None
_async_client: Any = None
_async_container: Any = None
_sync_lock = Lock()
_async_lock = Lock()


def get_container() -> Any:
    global _client, _container

    if _container is None:
        with _sync_lock:
            if _container is None:
                client = CosmosClient(
                    os.getenv("COSMOS_ENDPOINT"),
                    credential=os.getenv("COSMOS_KEY"),
                )
                container = client.get_database_client(
                    os.getenv("COSMOS_DATABASE")
                ).get_container_client(os.getenv("COSMOS_CONTAINER"))
                _client = client
                _container = container

    return _container


def get_async_container() -> Any:
    global _async_client, _async_container

    if _async_container is None:
        with _async_lock:
            if _async_container is None:
                client = AsyncCosmosClient(
                    os.getenv("COSMOS_ENDPOINT"),
                    credential=os.getenv("COSMOS_KEY"),
                )
                container = client.get_database_client(
                    os.getenv("COSMOS_DATABASE")
                ).get_container_client(os.getenv("COSMOS_CONTAINER"))
                _async_client = client
                _async_container = container

    return _async_container


async def query_items(query: str):
    results = get_async_container().query_items(
        query=query,
    )
    return [item async for item in results]
