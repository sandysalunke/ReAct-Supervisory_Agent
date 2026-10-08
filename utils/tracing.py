import os
from dotenv import load_dotenv
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace

load_dotenv()

connection_string = os.getenv(
    "APPLICATIONINSIGHTS_CONNECTION_STRING"
)

if not connection_string:
    raise ValueError(
        "APPLICATIONINSIGHTS_CONNECTION_STRING is not configured"
    )

configure_azure_monitor(
    connection_string=connection_string
)

tracer = trace.get_tracer("multi-agent-system")