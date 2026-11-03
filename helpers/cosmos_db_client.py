from azure.cosmos import CosmosClient
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
import os

client = CosmosClient(
    os.getenv("COSMOS_ENDPOINT"),
    credential=os.getenv("COSMOS_KEY")
)

database = client.get_database_client(
    os.getenv("COSMOS_DATABASE")
)

container = database.get_container_client(
    os.getenv("COSMOS_CONTAINER")
)

async_client = AsyncCosmosClient(
    os.getenv("COSMOS_ENDPOINT"),
    credential=os.getenv("COSMOS_KEY")
)

async_database = async_client.get_database_client(
    os.getenv("COSMOS_DATABASE")
)

async_container = async_database.get_container_client(
    os.getenv("COSMOS_CONTAINER")
)


async def query_items(query: str):
    results = async_container.query_items(
        query=query,
        enable_cross_partition_query=True,
    )
    return [item async for item in results]