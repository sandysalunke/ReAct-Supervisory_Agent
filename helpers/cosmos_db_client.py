from azure.cosmos import CosmosClient
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